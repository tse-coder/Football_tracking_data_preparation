import cv2
import numpy as np

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
        
        # Define ranges for green color (pitch)
        # Hue: 35-85 captures most grass greens
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate ratio of green pixels
        green_pixels = cv2.countNonZero(mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        
        return green_pixels / total_pixels if total_pixels > 0 else 0

    @staticmethod
    def is_pitch_frame(frame, min_ratio=0.20):
        """
        Check if the frame contains enough green pitch to be considered a game view.
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
        # Calculate histogram for Hue and Saturation
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    @staticmethod
    def is_scene_transition(prev_hist, curr_hist, threshold=0.75):
        """
        Compare current frame histogram with previous to detect abrupt cuts 
        (like transitions to replays or ad wipes).
        """
        if prev_hist is None or curr_hist is None:
            return False
        
        # cv2.HISTCMP_CORREL returns 1 for perfect match, -1 for complete mismatch
        score = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
        
        # If correlation is below threshold, it's a significant change/transition
        return score < threshold

    @staticmethod
    def get_brightness(frame):
        """
        Calculate the mean brightness of the frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.mean(gray)[0]

    @staticmethod
    def is_exposed_properly(frame, min_val=20.0, max_val=230.0):
        """
        Check if the frame is too dark or too bright (overexposed).
        Returns: (bool is_valid, float brightness_score)
        """
        brightness = FrameFilters.get_brightness(frame)
        is_valid = min_val <= brightness <= max_val
        return is_valid, brightness

    @staticmethod
    def detect_replay_heuristic(is_transition, green_ratio, min_green_threshold=0.20):
        """
        Lightweight heuristic: if there's an abrupt scene transition AND the resulting
        scene has very little green pitch, it's highly likely a cut to a replay graphic,
        close-up of a player, or an ad wipe.
        
        Returns True if a replay/break is suspected.
        """
        if is_transition and green_ratio < min_green_threshold:
            return True
        return False
