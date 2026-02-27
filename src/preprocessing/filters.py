import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

class FrameFilters:
    @staticmethod
    def is_blurry(frame, threshold=50.0):
        """
        Check if a frame is blurry using the variance of the Laplacian.
        Returns: (bool is_blurry, float variance_score)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold, variance

    @staticmethod
    def get_green_ratio(frame):
        """
        Calculate the ratio of green pixels in the frame, indicating the pitch.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Broader green range for grass variations under different lighting
        lower_green = np.array([30, 30, 30])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = cv2.countNonZero(mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        return green_pixels / total_pixels if total_pixels > 0 else 0

    @staticmethod
    def is_pitch_frame(frame, min_ratio=0.20):
        """
        Check if the frame contains enough green pitch.
        Returns: (bool is_pitch, float green_ratio)
        """
        ratio = FrameFilters.get_green_ratio(frame)
        return ratio >= min_ratio, ratio

    @staticmethod
    def compute_histogram(frame):
        """
        Compute an HSV histogram for the frame to be used in scene comparison.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    @staticmethod
    def is_scene_transition(prev_hist, curr_hist, threshold=0.70):
        """
        Detect abrupt scene changes (e.g., cuts to replays).
        """
        if prev_hist is None or curr_hist is None:
            return False
        score = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
        return score < threshold

    @staticmethod
    def get_brightness(frame):
        """
        Calculate the mean brightness of the frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.mean(gray)[0]

    @staticmethod
    def is_exposed_properly(frame, min_val=30.0, max_val=220.0):
        """
        Check if the frame is too dark or too bright.
        Returns: (bool is_valid, float brightness_score)
        """
        brightness = FrameFilters.get_brightness(frame)
        is_valid = min_val <= brightness <= max_val
        return is_valid, brightness

    @staticmethod
    def is_slow_motion(prev_frame, frame, threshold=15.0):
        """
        Detect slow-motion by low inter-frame pixel differences.
        Returns: bool is_slow_mo
        """
        if prev_frame is None:
            return False
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_prev, gray_curr)
        mean_diff = np.mean(diff)
        return mean_diff < threshold

    @staticmethod
    def detect_players_and_ball(frame, min_player_conf=0.5, min_ball_conf=0.4):
        """
        Use YOLO to detect players (persons) and ball.
        Returns: (int num_players, bool has_ball)
        """
        results = model(frame, verbose=False)
        persons = [
            det
            for det in results[0].boxes
            if int(det.cls) == 0 and det.conf > min_player_conf
        ]
        balls = [
            det
            for det in results[0].boxes
            if int(det.cls) == 32 and det.conf > min_ball_conf
        ]
        return len(persons), len(balls) > 0

    @staticmethod
    def is_live_play_frame(
        frame, min_players=1, green_ratio_threshold=0.15, require_ball=False
    ):
        """
        Check if frame is a valid live play scene (wide or close-up).
        - Allows >=1 player for close-ups (e.g., GK shots).
        - Optional ball requirement (disable for close-ups without ball).
        - Lower green threshold for sideline/close shots.
        Returns: (bool is_good, dict metrics)
        """
        num_players, has_ball = FrameFilters.detect_players_and_ball(frame)
        green_ratio = FrameFilters.get_green_ratio(frame)

        is_good = (
            num_players >= min_players
            and green_ratio >= green_ratio_threshold
            and (has_ball if require_ball else True)
        )

        metrics = {
            "num_players": num_players,
            "has_ball": has_ball,
            "green_ratio": green_ratio,
        }
        return is_good, metrics

    @staticmethod
    def detect_replay_heuristic(
        is_transition,
        is_slow_mo,
        num_players,
        prev_num_players=None,
        green_ratio=0.0,
        min_green_threshold=0.15,
        max_players_for_close_replay=4,
    ):
        """
        Detect potential replays:
        - Transition + slow-motion (hallmark of replays).
        - Sudden drop in player count (zoom to close-up replay).
        - Low green in post-transition frame (off-pitch graphic).
        - Close-ups (low players) after transition are likely replays, but allow live close-ups without transition.
        Returns: bool is_replay
        """
        sudden_player_drop = (
            prev_num_players is not None and num_players < prev_num_players * 0.5
        )

        return is_transition and (
            is_slow_mo
            or green_ratio < min_green_threshold
            or sudden_player_drop
            or (num_players <= max_players_for_close_replay and is_slow_mo)
        )

# # Example usage in video loop (stateful for transitions, slow-mo, replays)
# cap = cv2.VideoCapture("your_match.mp4")
# good_frames = []  # Collect good frames or write to output video
# prev_frame = None
# prev_hist = None
# prev_num_players = None
# is_replay_sequence = False  # Skip entire replay until back to live

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     curr_hist = FrameFilters.compute_histogram(frame)
#     is_transition = FrameFilters.is_scene_transition(prev_hist, curr_hist)
#     is_slow_mo = FrameFilters.is_slow_motion(prev_frame, frame)

#     num_players, has_ball = FrameFilters.detect_players_and_ball(frame)
#     green_ratio = FrameFilters.get_green_ratio(frame)

#     is_replay = FrameFilters.detect_replay_heuristic(
#         is_transition, is_slow_mo, num_players, prev_num_players, green_ratio
#     )

#     if is_replay:
#         is_replay_sequence = True
#     elif num_players >= 1 and green_ratio > 0.30 and not is_slow_mo:  # Recovery: back to live (adjust thresholds)
#         is_replay_sequence = False

#     is_good, metrics = FrameFilters.is_live_play_frame(frame, min_players=1, require_ball=False)  # Allow no ball for GK/close

#     if (
#         not is_replay_sequence and
#         is_good and
#         not FrameFilters.is_blurry(frame)[0] and
#         FrameFilters.is_exposed_properly(frame)[0]
#     ):
#         good_frames.append(frame)  # or process/save

#     # Update previous states
#     prev_frame = frame.copy()
#     prev_hist = curr_hist
#     prev_num_players = num_players

# cap.release()
