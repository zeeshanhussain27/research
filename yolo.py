from collections import defaultdict
import re

from ultralytics import YOLO
from ocsort.ocsort import OCSort, convert_bbox_to_z, convert_x_to_bbox
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

OCR_AVAILABLE = True

VIDEO_PATH = "clips/tight/panning.mp4"
OUTPUT_PATH = "annotatedpanning.mp4"

TOP_CROP = 90

# Detection / OCSort settings
YOLO_LOW_CONF = 0.15
TRACK_HIGH_CONF = 0.15
TRACK_MAX_AGE = 180
TRACK_IOU_THRESHOLD = 0.20

# Automatic team-color calibration
TEAM_NAMES = ("Team 1", "Team 2")
CALIBRATION_FRAMES = 180
MAX_CALIBRATION_SAMPLES = 300
TEAM_MATCH_THRESHOLD = 0.40

# HSV individual re-identification
HSV_REID_THRESHOLD = 0.30
HSV_PROFILE_UPDATE_ALPHA = 0.10

# Jersey-number OCR
OCR_INTERVAL = 12
MIN_OCR_BOX_HEIGHT = 100
MIN_OCR_CONFIDENCE = 0.40
MAX_OCR_OVERLAP = 0.10
MIN_NUMBER_VOTE_SCORE = 2.0

CMC_ENABLED = True

IDENTITY_AFFINE = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


class GlobalMotionCompensator:
    """
    Estimates background camera motion from frame-to-frame optical flow.

    Player boxes from the prior frame are masked so moving players do not
    dominate the camera-motion estimate.
    """

    def __init__(
        self,
        max_corners=300,
        min_inliers=12,
        # This clip uses a fixed camera. A 3 px/frame gate rejects
        # optical-flow jitter while still allowing meaningful pans later.
        min_motion_pixels=3.0,
        max_translation_pixels=30.0,
    ):
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self.min_motion_pixels = min_motion_pixels
        self.max_translation_pixels = max_translation_pixels
        self.previous_gray = None

    def estimate(self, frame, previous_boxes):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.previous_gray is None:
            self.previous_gray = gray
            return IDENTITY_AFFINE.copy()

        mask = np.full(self.previous_gray.shape, 255, dtype=np.uint8)

        # Ignore previously tracked people; use field markings/background instead.
        for box in previous_boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            cv2.rectangle(
                mask,
                (max(0, x1 - 10), max(0, y1 - 10)),
                (
                    min(mask.shape[1] - 1, x2 + 10),
                    min(mask.shape[0] - 1, y2 + 10),
                ),
                0,
                -1,
            )

        previous_points = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
            mask=mask,
        )

        if previous_points is None:
            self.previous_gray = gray
            return IDENTITY_AFFINE.copy()

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )

        self.previous_gray = gray

        if current_points is None or status is None:
            return IDENTITY_AFFINE.copy()

        valid = status.reshape(-1) == 1
        previous_points = previous_points.reshape(-1, 2)[valid]
        current_points = current_points.reshape(-1, 2)[valid]

        if len(previous_points) < self.min_inliers:
            return IDENTITY_AFFINE.copy()

        matrix, inliers = cv2.estimateAffinePartial2D(
            previous_points,
            current_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )

        if matrix is None or inliers is None:
            return IDENTITY_AFFINE.copy()

        inlier_count = int(inliers.sum())

        if inlier_count < self.min_inliers:
            return IDENTITY_AFFINE.copy()

        matrix = matrix.astype(np.float32)

        translation = np.linalg.norm(matrix[:, 2])
        scale = np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
        rotation_degrees = abs(
            np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
        )

        # Reject implausible transforms caused by player/camera noise.
        if (
            translation > self.max_translation_pixels
            or not 0.95 <= scale <= 1.05
            or rotation_degrees > 5.0
        ):
            return IDENTITY_AFFINE.copy()

        # Fixed-camera footage should remain effectively unchanged.
        if translation < self.min_motion_pixels:
            return IDENTITY_AFFINE.copy()

        return matrix


def warp_box(box, affine_matrix):
    """Apply a 2x3 affine transform to an x1,y1,x2,y2 bounding box."""
    x1, y1, x2, y2 = map(float, box[:4])

    corners = np.array(
        [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
        dtype=np.float32,
    )

    warped = cv2.transform(corners, affine_matrix)[0]

    return np.array(
        [
            warped[:, 0].min(),
            warped[:, 1].min(),
            warped[:, 0].max(),
            warped[:, 1].max(),
        ],
        dtype=np.float32,
    )


def warp_observation(observation, affine_matrix):
    """Transform OCSort's saved [x1, y1, x2, y2, confidence] observation."""
    observation = np.asarray(observation, dtype=np.float32).copy()

    if observation.size < 4 or observation[:4].sum() < 0:
        return observation

    observation[:4] = warp_box(observation[:4], affine_matrix)
    return observation


def apply_cmc_to_ocsort(tracker, affine_matrix):
    """
    Shift every OCSort Kalman state and its observation history into the
    current camera coordinate system before tracker.update().
    """
    if np.allclose(affine_matrix, IDENTITY_AFFINE):
        return

    linear = affine_matrix[:, :2]
    area_scale = abs(np.linalg.det(linear))

    for track in tracker.trackers:
        # Warp Kalman bounding-box state.
        current_box = convert_x_to_bbox(track.kf.x)[0]
        warped_box = warp_box(current_box, affine_matrix)
        track.kf.x[:4] = convert_bbox_to_z(warped_box)

        # Warp x/y velocity stored inside the Kalman state.
        state = track.kf.x.reshape(-1)
        velocity_xy = linear @ np.array([state[4], state[5]])
        state[4] = velocity_xy[0]
        state[5] = velocity_xy[1]
        state[6] *= area_scale

        # Warp all OCSort observation history used for velocity matching.
        track.last_observation = warp_observation(
            track.last_observation,
            affine_matrix,
        )

        track.observations = {
            age: warp_observation(observation, affine_matrix)
            for age, observation in track.observations.items()
        }

        track.history_observations = [
            warp_observation(observation, affine_matrix)
            for observation in track.history_observations
        ]

        # OCSort stores direction as [dy, dx].
        if track.velocity is not None:
            velocity_xy = linear @ np.array(
                [track.velocity[1], track.velocity[0]]
            )
            magnitude = np.linalg.norm(velocity_xy) + 1e-6
            track.velocity = np.array(
                [velocity_xy[1], velocity_xy[0]]
            ) / magnitude


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    return intersection / (area_a + area_b - intersection + 1e-6)


def max_track_overlaps(tracks):
    overlaps = []

    for i, track in enumerate(tracks):
        highest_iou = 0.0

        for j, other_track in enumerate(tracks):
            if i != j:
                highest_iou = max(highest_iou, box_iou(track, other_track))

        overlaps.append(highest_iou)

    return overlaps


def get_jersey_histogram(frame, box):
    """Get an HSV color histogram from the central upper-body region."""
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box[:4])

    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(0, min(x2, frame_w))
    y2 = max(0, min(y2, frame_h))

    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1

    torso_x1 = x1 + int(box_w * 0.20)
    torso_x2 = x2 - int(box_w * 0.20)
    torso_y1 = y1 + int(box_h * 0.15)
    torso_y2 = y1 + int(box_h * 0.65)

    if torso_x2 <= torso_x1 or torso_y2 <= torso_y1:
        return None

    torso = frame[torso_y1:torso_y2, torso_x1:torso_x2]

    if torso.size == 0:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 8],
        [0, 180, 0, 256],
    )

    histogram = cv2.normalize(
        histogram,
        None,
        alpha=1,
        norm_type=cv2.NORM_L1,
    )

    return histogram.flatten()


def color_distance(histogram_a, histogram_b):
    """0 means identical; values closer to 1 mean less similar."""
    if histogram_a is None or histogram_b is None:
        return float("inf")

    return cv2.compareHist(
        histogram_a.astype(np.float32),
        histogram_b.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA,
    )


class RefereeFilter:
    """
    Referee profile:
    - black/white striped upper torso
    - dark neutral-colored lower body (black pants)
    - optional OCR "U" bonus on the shirt back

    A stable ID needs 3 votes before its annotation is hidden.
    """

    def __init__(
        self,
        required_votes=3,
        vote_window=5,
        black_threshold=70,
        white_value_threshold=180,
        white_saturation_threshold=60,
        pants_dark_threshold=80,
        pants_neutral_saturation=90,
        min_black_ratio=0.18,
        min_white_ratio=0.12,
        min_transition_ratio=0.002,
        min_dark_pants_ratio=0.18,
        u_check_interval=15,
    ):
        self.required_votes = required_votes
        self.vote_window = vote_window

        self.black_threshold = black_threshold
        self.white_value_threshold = white_value_threshold
        self.white_saturation_threshold = white_saturation_threshold

        self.pants_dark_threshold = pants_dark_threshold
        self.pants_neutral_saturation = pants_neutral_saturation

        self.min_black_ratio = min_black_ratio
        self.min_white_ratio = min_white_ratio
        self.min_transition_ratio = min_transition_ratio
        self.min_dark_pants_ratio = min_dark_pants_ratio

        self.u_check_interval = u_check_interval

        self.votes = defaultdict(list)
        self.check_counts = defaultdict(int)
        self.confirmed_referees = set()

    def _upper_torso_crop(self, frame, box):
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box[:4])

        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(0, min(x2, frame_w))
        y2 = max(0, min(y2, frame_h))

        if x2 <= x1 or y2 <= y1:
            return None

        box_w = x2 - x1
        box_h = y2 - y1

        return frame[
            y1 + int(box_h * 0.15): y1 + int(box_h * 0.65),
            x1 + int(box_w * 0.20): x2 - int(box_w * 0.20),
        ]

    def _pants_crop(self, frame, box):
        """Central lower region, where referee black pants should appear."""
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box[:4])

        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(0, min(x2, frame_w))
        y2 = max(0, min(y2, frame_h))

        if x2 <= x1 or y2 <= y1:
            return None

        box_w = x2 - x1
        box_h = y2 - y1

        return frame[
            y1 + int(box_h * 0.62): y2,
            x1 + int(box_w * 0.15): x2 - int(box_w * 0.15),
        ]

    def _shirt_metrics(self, torso):
        if torso is None or torso.size == 0:
            return 0.0, 0.0, 0.0

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        _, saturation, value = cv2.split(hsv)

        black = value < self.black_threshold
        white = (
            (value > self.white_value_threshold)
            & (saturation < self.white_saturation_threshold)
        )

        black_ratio = black.mean()
        white_ratio = white.mean()

        stripe_pixels = np.full(black.shape, -1, dtype=np.int8)
        stripe_pixels[black] = 0
        stripe_pixels[white] = 1

        left = stripe_pixels[:, :-1]
        right = stripe_pixels[:, 1:]

        valid_pairs = (left >= 0) & (right >= 0)
        transitions = (left != right) & valid_pairs
        transition_ratio = transitions.mean()

        return black_ratio, white_ratio, transition_ratio

    def _dark_pants_ratio(self, pants):
        if pants is None or pants.size == 0:
            return 0.0

        hsv = cv2.cvtColor(pants, cv2.COLOR_BGR2HSV)
        _, saturation, value = cv2.split(hsv)

        dark_neutral = (
            (value < self.pants_dark_threshold)
            & (saturation < self.pants_neutral_saturation)
        )

        return dark_neutral.mean()

    def _has_u_on_back(self, torso):
        """
        Optional OCR hint. It only runs periodically for an already likely ref.

        Requires:
            brew install tesseract
            python3 -m pip install pytesseract
        """
        if not OCR_AVAILABLE or torso is None or torso.size == 0:
            return False

        torso_h, torso_w = torso.shape[:2]

        # Central shirt stripe / back-number area.
        center = torso[
            int(torso_h * 0.10):int(torso_h * 0.85),
            int(torso_w * 0.25):int(torso_w * 0.75),
        ]

        if center.size == 0:
            return False

        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC,
        )

        gray = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(gray)

        try:
            data = pytesseract.image_to_data(
                gray,
                config="--oem 1 --psm 10",
                output_type=Output.DICT,
            )
        except pytesseract.TesseractNotFoundError:
            return False

        for text, confidence in zip(data["text"], data["conf"]):
            try:
                confidence = float(confidence)
            except ValueError:
                continue

            if text.strip().upper() == "U" and confidence >= 50:
                return True

        return False

    def looks_like_referee(self, frame, box):
        """
        Used during team calibration: requires both striped shirt and dark pants.
        """
        torso = self._upper_torso_crop(frame, box)
        pants = self._pants_crop(frame, box)

        black_ratio, white_ratio, transition_ratio = self._shirt_metrics(torso)
        dark_pants_ratio = self._dark_pants_ratio(pants)

        stripe_checks = sum([
            black_ratio >= self.min_black_ratio,
            white_ratio >= self.min_white_ratio,
            transition_ratio >= self.min_transition_ratio,
        ])

        return (
            stripe_checks >= 2
            and dark_pants_ratio >= self.min_dark_pants_ratio
        )

    def update(self, stable_id, frame, box):
        """
        Add one observation. A visible U contributes an extra vote, but is never
        enough by itself to classify someone as a referee.
        """
        torso = self._upper_torso_crop(frame, box)
        pants = self._pants_crop(frame, box)

        black_ratio, white_ratio, transition_ratio = self._shirt_metrics(torso)
        dark_pants_ratio = self._dark_pants_ratio(pants)

        stripe_checks = sum([
            black_ratio >= self.min_black_ratio,
            white_ratio >= self.min_white_ratio,
            transition_ratio >= self.min_transition_ratio,
        ])

        likely_referee = (
            stripe_checks >= 2
            and dark_pants_ratio >= self.min_dark_pants_ratio
        )

        self.check_counts[stable_id] += 1
        sees_u = False

        # OCR is only attempted on likely referees, once every 15 checks.
        if (
            likely_referee
            and self.check_counts[stable_id] % self.u_check_interval == 0
        ):
            sees_u = self._has_u_on_back(torso)

        # Normal referee appearance contributes one vote.
        vote = 1 if likely_referee else 0

        # A U is only a bonus if pants are dark and at least one shirt cue matches.
        if (
            sees_u
            and dark_pants_ratio >= self.min_dark_pants_ratio
            and stripe_checks >= 1
        ):
            vote += 1

        votes = self.votes[stable_id]
        votes.append(vote)
        votes[:] = votes[-self.vote_window:]

        if sum(votes) >= self.required_votes:
            self.confirmed_referees.add(stable_id)

        return stable_id in self.confirmed_referees

    def is_referee(self, stable_id):
        return stable_id in self.confirmed_referees


def kmeans(features, clusters=2, iterations=40):
    """Small NumPy-only K-means implementation."""
    random = np.random.default_rng(42)

    start_indexes = random.choice(
        len(features),
        size=clusters,
        replace=False,
    )
    centers = features[start_indexes].copy()

    for _ in range(iterations):
        distances = np.sum(
            (features[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        labels = np.argmin(distances, axis=1)

        new_centers = []

        for cluster_index in range(clusters):
            members = features[labels == cluster_index]

            if len(members) == 0:
                new_centers.append(
                    features[random.integers(len(features))]
                )
            else:
                center = members.mean(axis=0)
                new_centers.append(center / (center.sum() + 1e-6))

        new_centers = np.asarray(new_centers)

        if np.allclose(centers, new_centers, atol=1e-5):
            break

        centers = new_centers

    return centers


def calibrate_team_prototypes(cap, model, referee_filter):
    """
    Learn two jersey-color groups from early video frames.
    Referee detections are excluded before K-means.
    """
    print("Calibrating jersey-color groups...")

    samples = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for _ in range(CALIBRATION_FRAMES):
        success, frame = cap.read()
        if not success:
            break

        frame = frame[TOP_CROP:, :]

        results = model(
            frame,
            classes=[0],
            device="mps",
            conf=YOLO_LOW_CONF,
            imgsz=960,
            verbose=False,
        )

        detections = results[0].boxes.data.cpu().numpy()

        for detection in detections:
            if referee_filter.looks_like_referee(frame, detection[:4]):
                continue

            histogram = get_jersey_histogram(frame, detection[:4])

            if histogram is not None:
                samples.append(histogram)

            if len(samples) >= MAX_CALIBRATION_SAMPLES:
                break

        if len(samples) >= MAX_CALIBRATION_SAMPLES:
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if len(samples) < 2:
        raise RuntimeError(
            "Not enough non-referee player detections for team calibration."
        )

    centers = kmeans(np.asarray(samples), clusters=2)

    print(
        f"Calibrated {len(samples)} jersey samples into "
        f"{TEAM_NAMES[0]} and {TEAM_NAMES[1]}."
    )

    return {
        TEAM_NAMES[0]: centers[0],
        TEAM_NAMES[1]: centers[1],
    }


def classify_team(histogram, team_prototypes):
    if histogram is None:
        return "unknown"

    distances = {
        team: color_distance(histogram, prototype)
        for team, prototype in team_prototypes.items()
    }

    best_team = min(distances, key=distances.get)

    if distances[best_team] > TEAM_MATCH_THRESHOLD:
        return "unknown"

    return best_team


def get_number_crop(frame, box):
    """Get a crop where a jersey number is likely to appear."""
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box[:4])

    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(0, min(x2, frame_w))
    y2 = max(0, min(y2, frame_h))

    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1

    number_x1 = x1 + int(box_w * 0.20)
    number_x2 = x2 - int(box_w * 0.20)
    number_y1 = y1 + int(box_h * 0.20)
    number_y2 = y1 + int(box_h * 0.80)

    if number_x2 <= number_x1 or number_y2 <= number_y1:
        return None

    return frame[number_y1:number_y2, number_x1:number_x2]


def read_jersey_number(frame, box):
    """Return (number, confidence), or None if no reliable read is available."""
    if not OCR_AVAILABLE:
        return None

    crop = get_number_crop(frame, box)

    if crop is None or crop.shape[0] < 20 or crop.shape[1] < 15:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Skip blurry crops.
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharpness < 80:
        return None

    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    config = (
        "--oem 1 --psm 8 "
        "-c tessedit_char_whitelist=0123456789"
    )

    try:
        data = pytesseract.image_to_data(
            enhanced,
            config=config,
            output_type=Output.DICT,
        )
    except pytesseract.TesseractNotFoundError:
        return None

    best_result = None

    for text, confidence in zip(data["text"], data["conf"]):
        digits = re.sub(r"\D", "", text)

        try:
            confidence = float(confidence) / 100.0
        except ValueError:
            continue

        if (
            1 <= len(digits) <= 2
            and 0 <= int(digits) <= 99
            and confidence >= MIN_OCR_CONFIDENCE
        ):
            if best_result is None or confidence > best_result[1]:
                best_result = (digits, confidence)

    return best_result


class PlayerIdentityManager:
    """Adds stable IDs, team gating, HSV recovery, and jersey-number votes."""

    def __init__(self, max_age=TRACK_MAX_AGE):
        self.max_age = max_age
        self.frame_number = 0
        self.next_stable_id = 1

        self.raw_to_stable = {}
        self.raw_last_seen = {}

        self.stable_histograms = {}
        self.stable_teams = {}
        self.stable_last_seen = {}

        self.number_votes = defaultdict(lambda: defaultdict(float))
        self.confirmed_numbers = {}

    def _create_stable_id(self, histogram, team):
        stable_id = self.next_stable_id
        self.next_stable_id += 1

        if histogram is not None:
            self.stable_histograms[stable_id] = histogram.copy()

        self.stable_teams[stable_id] = team
        self.stable_last_seen[stable_id] = self.frame_number

        return stable_id

    def _find_number_match(self, team, number_read, used_stable_ids):
        if team == "unknown" or number_read is None:
            return None

        number, confidence = number_read

        if confidence < 0.75:
            return None

        matches = [
            stable_id
            for stable_id, confirmed_number in self.confirmed_numbers.items()
            if stable_id not in used_stable_ids
            and self.stable_teams.get(stable_id) == team
            and confirmed_number == number
        ]

        return matches[0] if len(matches) == 1 else None

    def _find_hsv_match(self, histogram, team, used_stable_ids):
        best_id = None
        best_distance = float("inf")

        for stable_id, saved_histogram in self.stable_histograms.items():
            if stable_id in used_stable_ids:
                continue

            frames_missing = (
                self.frame_number - self.stable_last_seen.get(stable_id, 0)
            )

            if frames_missing > self.max_age:
                continue

            saved_team = self.stable_teams.get(stable_id, "unknown")

            # Never reconnect across teams.
            if (
                team != "unknown"
                and saved_team != "unknown"
                and team != saved_team
            ):
                continue

            distance = color_distance(histogram, saved_histogram)

            if distance < best_distance:
                best_distance = distance
                best_id = stable_id

        if best_distance <= HSV_REID_THRESHOLD:
            return best_id

        return None

    def _update_hsv_profile(
        self,
        stable_id,
        histogram,
        detection_confidence,
        overlap,
    ):
        if histogram is None:
            return

        # Do not update color profiles while players overlap.
        if detection_confidence < TRACK_HIGH_CONF or overlap > MAX_OCR_OVERLAP:
            return

        old_histogram = self.stable_histograms.get(stable_id)

        if old_histogram is None:
            self.stable_histograms[stable_id] = histogram.copy()
            return

        if color_distance(old_histogram, histogram) > HSV_REID_THRESHOLD:
            return

        updated = (
            (1.0 - HSV_PROFILE_UPDATE_ALPHA) * old_histogram
            + HSV_PROFILE_UPDATE_ALPHA * histogram
        )

        self.stable_histograms[stable_id] = updated / (updated.sum() + 1e-6)

    def _add_number_vote(self, stable_id, team, number_read):
        if team == "unknown" or number_read is None:
            return

        number, confidence = number_read
        self.number_votes[stable_id][number] += confidence

        ranked = sorted(
            self.number_votes[stable_id].items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_number, top_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if (
            top_score >= MIN_NUMBER_VOTE_SCORE
            and (
                runner_up_score == 0.0
                or top_score >= runner_up_score * 2.0
            )
        ):
            existing = self.confirmed_numbers.get(stable_id)

            if existing is None or existing == top_number:
                self.confirmed_numbers[stable_id] = top_number

    def assign(
        self,
        tracks,
        histograms,
        predicted_teams,
        number_reads,
        overlaps,
    ):
        self.frame_number += 1

        stable_ids = [None] * len(tracks)
        resolved_teams = ["unknown"] * len(tracks)
        used_stable_ids = set()

        # Preserve OCSort IDs where possible.
        for index, track in enumerate(tracks):
            raw_id = int(track[4])
            self.raw_last_seen[raw_id] = self.frame_number

            if raw_id in self.raw_to_stable:
                stable_id = self.raw_to_stable[raw_id]

                stable_ids[index] = stable_id
                resolved_teams[index] = self.stable_teams.get(
                    stable_id,
                    predicted_teams[index],
                )
                used_stable_ids.add(stable_id)

        # New OCSort ID: use jersey number first, then HSV.
        for index, track in enumerate(tracks):
            if stable_ids[index] is not None:
                continue

            raw_id = int(track[4])
            histogram = histograms[index]
            team = predicted_teams[index]
            number_read = number_reads[index]

            stable_id = self._find_number_match(
                team,
                number_read,
                used_stable_ids,
            )

            if stable_id is None:
                stable_id = self._find_hsv_match(
                    histogram,
                    team,
                    used_stable_ids,
                )

            if stable_id is None:
                stable_id = self._create_stable_id(histogram, team)

            self.raw_to_stable[raw_id] = stable_id
            stable_ids[index] = stable_id
            resolved_teams[index] = self.stable_teams.get(stable_id, team)
            used_stable_ids.add(stable_id)

        # Update HSV, team, and jersey-number evidence.
        for index, track in enumerate(tracks):
            stable_id = stable_ids[index]
            histogram = histograms[index]
            predicted_team = predicted_teams[index]
            confidence = float(track[6])

            if (
                self.stable_teams.get(stable_id) == "unknown"
                and predicted_team != "unknown"
            ):
                self.stable_teams[stable_id] = predicted_team

            resolved_teams[index] = self.stable_teams.get(
                stable_id,
                predicted_team,
            )
            self.stable_last_seen[stable_id] = self.frame_number

            self._update_hsv_profile(
                stable_id,
                histogram,
                confidence,
                overlaps[index],
            )

            self._add_number_vote(
                stable_id,
                resolved_teams[index],
                number_reads[index],
            )

        stale_raw_ids = [
            raw_id
            for raw_id, last_seen in self.raw_last_seen.items()
            if self.frame_number - last_seen > self.max_age * 3
        ]

        for raw_id in stale_raw_ids:
            self.raw_last_seen.pop(raw_id, None)
            self.raw_to_stable.pop(raw_id, None)

        confirmed_numbers = [
            self.confirmed_numbers.get(stable_id)
            for stable_id in stable_ids
        ]

        return stable_ids, resolved_teams, confirmed_numbers


model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if frame_h <= TOP_CROP:
    raise ValueError("TOP_CROP is larger than the video height.")

referee_filter = RefereeFilter()

team_prototypes = calibrate_team_prototypes(
    cap,
    model,
    referee_filter,
)

tracker = OCSort(
    det_thresh=TRACK_HIGH_CONF,
    max_age=TRACK_MAX_AGE,
    min_hits=2,
    iou_threshold=TRACK_IOU_THRESHOLD,
    delta_t=3,
    inertia=0.2,
    use_byte=True,
)

cmc = GlobalMotionCompensator()
previous_track_boxes = []

identity_manager = PlayerIdentityManager()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (frame_w, frame_h - TOP_CROP),
)

if not out.isOpened():
    raise RuntimeError(f"Could not create output: {OUTPUT_PATH}")

if not OCR_AVAILABLE:
    print(
        "Jersey-number OCR disabled. Install Tesseract and pytesseract "
        "to enable it."
    )

frame_index = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_index += 1
    frame = frame[TOP_CROP:, :]

    if CMC_ENABLED:
        camera_transform = cmc.estimate(frame, previous_track_boxes)
        apply_cmc_to_ocsort(tracker, camera_transform)

    results = model(
        frame,
        classes=[0],
        device="mps",
        conf=YOLO_LOW_CONF,
        imgsz=960,
        verbose=False,
    )

    # OCSort input: x1, y1, x2, y2, confidence, class_id.
    detections = results[0].boxes.data.cpu()

    # Run every frame, including when no detections are found.
    tracks = tracker.update(detections, frame)
    previous_track_boxes = [track[:4] for track in tracks]

    histograms = [
        get_jersey_histogram(frame, track[:4])
        for track in tracks
    ]

    predicted_teams = [
        classify_team(histogram, team_prototypes)
        for histogram in histograms
    ]

    overlaps = max_track_overlaps(tracks)
    number_reads = [None] * len(tracks)

    # OCR only runs periodically on clear player boxes.
    if OCR_AVAILABLE and frame_index % OCR_INTERVAL == 0:
        for index, track in enumerate(tracks):
            box_height = track[3] - track[1]
            confidence = float(track[6])

            if (
                box_height >= MIN_OCR_BOX_HEIGHT
                and confidence >= TRACK_HIGH_CONF
                and overlaps[index] <= MAX_OCR_OVERLAP
            ):
                number_reads[index] = read_jersey_number(
                    frame,
                    track[:4],
                )

    stable_ids, teams, confirmed_numbers = identity_manager.assign(
        tracks,
        histograms,
        predicted_teams,
        number_reads,
        overlaps,
    )

    # Update referee votes. Refs stay in OCSort but will not be drawn.
    for track, stable_id in zip(tracks, stable_ids):
        referee_filter.update(stable_id, frame, track[:4])

    for track, stable_id, team, jersey_number in zip(
        tracks,
        stable_ids,
        teams,
        confirmed_numbers,
    ):
        # Hide confirmed referees.
        if referee_filter.is_referee(stable_id):
            continue

        x1, y1, x2, y2 = map(int, track[:4])

        color = (
            (255, 100, 0)
            if team == TEAM_NAMES[0]
            else (0, 200, 255)
            if team == TEAM_NAMES[1]
            else (0, 255, 0)
        )

        label = f"ID: {stable_id} | {team}"

        if jersey_number is not None:
            label += f" | #{jersey_number}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 5, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
        )

    out.write(frame)

cap.release()
out.release()

print(f"Saved {OUTPUT_PATH}")
