import os
import sys
import cv2
import matplotlib.pyplot as plt

# Ensure parent directory is in sys.path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FRAMES_DIR,
    MIN_GREEN_RATIO,
    BLUR_THRESHOLD,
    SCENE_CHANGE_THRESHOLD,
    MIN_BRIGHTNESS,
    MAX_BRIGHTNESS,
    REPLAY_SKIP_SECONDS,
    LOG_FILE_DUMP_INTERVAL,
)
from preprocessing.filters import FrameFilters
from preprocessing.logger import FrameLogger


class FrameExtractor:
    def __init__(
        self,
        vid_capture,
        target_fps=10,
        target_width=640,
        target_height=360,
        display_every_n=10,
    ):
        self.vid_capture = vid_capture
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        self.display_every_n = display_every_n
        self.original_fps = vid_capture.original_fps
        self.time_interval_sec = 1.0 / self.target_fps if self.target_fps > 0 else 1.0

    def extract_frames(
        self,
        start_minute=0,
        end_minute=None,
        frame_limit=1000,
        output_dir=FRAMES_DIR,
        display=False,
        filter_blurry=True,
        filter_crowd=True,
        filter_transitions=True,
        filter_brightness=True,
        filter_replays=True,
        log_csv_path=None,
    ):
        """
        Extract frames from the video within a specified time range.

        Args:
            start_minute (int): The minute to start extraction from.
            end_minute (int, optional): The minute to stop extraction. If None, extracts until the end or frame_limit is reached.
            frame_limit (int): Maximum number of frames to save. Defaults to 1000.
            output_dir (str): Directory where frames will be saved.
            display (bool): to display in the notebook if called in notebook.
        """
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if log_csv_path is None:
            log_csv_path = os.path.join(output_dir, "frames_log.csv")

        logger = FrameLogger(log_csv_path, dump_interval=LOG_FILE_DUMP_INTERVAL)

        # set frame positions (start and end)
        start_frame = int(start_minute * 60 * self.original_fps)
        self.vid_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        end_time_sec = end_minute * 60.0 if end_minute is not None else float("inf")

        saved_count = 0
        frames_read = 0
        display_count = 0
        last_filename = None

        # Track statistics for filtering
        stats = {
            "saved": 0,
            "blurry_skipped": 0,
            "crowd_skipped": 0,
            "transition_skipped": 0,
            "brightness_skipped": 0,
            "replay_skipped": 0,
            "sum_blur": 0.0,
            "sum_green": 0.0,
            "frames_analyzed": 0,
        }
        prev_hist = None
        prev_frame = None
        prev_num_players = None

        print(
            f"Starting extraction from {start_minute}m to {end_minute if end_minute else 'end'}m..."
        )
        print(
            f"Targeting {self.target_fps} FPS (Interval: {self.time_interval_sec:.3f}s), max {frame_limit} frames."
        )

        plt.figure(figsize=(15, 10))

        next_target_time_sec = start_minute * 60.0
        skip_until_time_sec = 0.0

        try:
            while self.vid_capture.is_opened():
                ret, frame = self.vid_capture.read()
                frames_read += 1

                if not ret or saved_count >= frame_limit:
                    break

                current_time_sec = (
                    self.vid_capture.video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                )

                if current_time_sec > end_time_sec:
                    break

                # 1. Replay skip logic (time-based skip)
                if filter_replays and current_time_sec < skip_until_time_sec:
                    continue

                # 2. Time-aware sampling
                if current_time_sec < next_target_time_sec:
                    continue

                # We reached our target time. Advance target for the next frame.
                next_target_time_sec += self.time_interval_sec
                stats["frames_analyzed"] += 1

                resized = cv2.resize(
                    frame,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_LINEAR,
                )

                skip_frame = False
                status = "SAVED"

                # Compute core metrics for logging
                is_blurry, blur_score = FrameFilters.is_blurry(resized, BLUR_THRESHOLD)
                is_pitch, green_ratio = FrameFilters.is_pitch_frame(
                    resized, MIN_GREEN_RATIO
                )
                is_exposed, brightness_score = FrameFilters.is_exposed_properly(
                    resized, MIN_BRIGHTNESS, MAX_BRIGHTNESS
                )

                curr_hist = (
                    FrameFilters.compute_histogram(resized)
                    if (filter_transitions or filter_replays)
                    else None
                )
                is_transition = False

                if prev_hist is not None and curr_hist is not None:
                    is_transition = FrameFilters.is_scene_transition(
                        prev_hist, curr_hist, SCENE_CHANGE_THRESHOLD
                    )

                is_slow_mo = False
                num_players = 0
                has_ball = False
                if filter_replays:
                    is_slow_mo = FrameFilters.is_slow_motion(prev_frame, resized)
                    num_players, has_ball = FrameFilters.detect_players_and_ball(
                        resized
                    )

                is_replay = False
                if filter_replays:
                    is_replay = FrameFilters.detect_replay_heuristic(
                        is_transition=is_transition,
                        is_slow_mo=is_slow_mo,
                        num_players=num_players,
                        prev_num_players=prev_num_players,
                        green_ratio=green_ratio,
                        min_green_threshold=MIN_GREEN_RATIO,
                    )

                stats["sum_blur"] += blur_score
                stats["sum_green"] += green_ratio

                # Apply filters hierarchically
                if filter_replays and is_replay:
                    stats["replay_skipped"] += 1
                    skip_until_time_sec = current_time_sec + REPLAY_SKIP_SECONDS
                    status = "SKIPPED_REPLAY_START"
                    skip_frame = True

                elif filter_blurry and is_blurry:
                    stats["blurry_skipped"] += 1
                    status = "SKIPPED_BLUR"
                    skip_frame = True

                elif filter_brightness and not is_exposed:
                    stats["brightness_skipped"] += 1
                    status = "SKIPPED_BRIGHTNESS"
                    skip_frame = True

                elif filter_crowd and not is_pitch:
                    stats["crowd_skipped"] += 1
                    status = "SKIPPED_CROWD"
                    skip_frame = True

                elif filter_transitions and is_transition:
                    stats["transition_skipped"] += 1
                    status = "SKIPPED_TRANSITION"
                    skip_frame = True

                if filter_transitions or filter_replays:
                    prev_hist = curr_hist
                if filter_replays:
                    prev_frame = resized.copy()
                    prev_num_players = num_players

                # Write to Log
                logger.log_frame(
                    frame_idx=frames_read,
                    timestamp_sec=current_time_sec,
                    blur=blur_score,
                    green=green_ratio,
                    bright=brightness_score,
                    transition_flag=is_transition,
                    status=status,
                )

                if skip_frame:
                    continue

                last_filename = (
                    f"frame_{saved_count:06d}_sec_{current_time_sec:.2f}.jpg"
                )
                cv2.imwrite(os.path.join(output_dir, last_filename), resized)

                # Display progress in notebook if display is true
                if (
                    display
                    and saved_count % self.display_every_n == 0
                    and display_count < 12
                ):
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    plt.subplot(3, 4, display_count + 1)
                    plt.imshow(rgb)
                    plt.title(f"{current_time_sec:.1f}s")
                    plt.axis("off")
                    display_count += 1

                saved_count += 1
                stats["saved"] += 1

        finally:
            logger.close()
            # We don't release self.vid_capture here because the VideoLoader manages its own lifecycle out of scope

        if display:
            plt.tight_layout()
            plt.show()

        analyzed = max(1, stats["frames_analyzed"])
        print("\n=== FRAME PROCESSING SUMMARY ===")
        print(f"Frames Read from Video: {frames_read}")
        print(f"Frames Analyzed (Target FPS): {stats['frames_analyzed']}")
        print(f"Frames Saved: {stats['saved']}")
        print("-------------")
        print(f"Skipped - Blurry: {stats['blurry_skipped']}")
        print(f"Skipped - Too Dark/Bright: {stats['brightness_skipped']}")
        print(f"Skipped - Crowd/No Pitch: {stats['crowd_skipped']}")
        print(f"Skipped - Scene Transition: {stats['transition_skipped']}")
        print(f"Skipped - Replay Detected: {stats['replay_skipped']}")
        print("-------------")
        print(f"Average Blur Score: {stats['sum_blur'] / analyzed:.2f}")
        print(f"Average Green Ratio: {stats['sum_green'] / analyzed:.3f}")
        print("================================")
        print(f"Saved frames to {output_dir}")
        print(f"Saved logs to {log_csv_path}")

        return [
            os.path.join(output_dir, last_filename) if last_filename else None,
            stats,
        ]
