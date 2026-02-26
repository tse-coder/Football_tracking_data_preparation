import csv
import os


class FrameLogger:
    def __init__(self, log_path, dump_interval=100):
        self.log_path = log_path
        self.dump_interval = dump_interval
        self.buffer = []

        # Write headers if file doesn't exist
        is_new_file = not os.path.exists(log_path)

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)

        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(
                    [
                        "frame_index",
                        "timestamp_sec",
                        "blur_score",
                        "green_ratio",
                        "brightness",
                        "scene_transition_flag",
                        "status",
                    ]
                )

    def log_frame(
        self, frame_idx, timestamp_sec, blur, green, bright, transition_flag, status
    ):
        """
        Buffer the log entry. If buffer reaches dump_interval, write to disk to save memory.
        """
        self.buffer.append(
            [
                frame_idx,
                round(timestamp_sec, 2),
                round(blur, 2) if blur is not None else -1,
                round(green, 3) if green is not None else -1,
                round(bright, 2) if bright is not None else -1,
                int(transition_flag) if transition_flag is not None else -1,
                status,
            ]
        )

        if len(self.buffer) >= self.dump_interval:
            self.flush()

    def flush(self):
        """
        Write all buffered logs to disk and clear buffer.
        """
        if not self.buffer:
            return

        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.buffer)

        self.buffer.clear()

    def close(self):
        self.flush()
