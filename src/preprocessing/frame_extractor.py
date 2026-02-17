import os
import sys
import cv2
import matplotlib.pyplot as plt

# Ensure parent directory is in sys.path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FRAMES_DIR

class FrameExtractor:
    def __init__(self, vid_capture, target_fps=10, target_width=640, target_height=360, display_every_n=10):
        self.vid_capture = vid_capture
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        self.display_every_n = display_every_n
        self.original_fps = vid_capture.original_fps
        self.frame_interval = max(1, self.original_fps // self.target_fps)

    def extract_frames(
        self, 
        start_minute=0, 
        end_minute=None, 
        frame_limit=1000, 
        output_dir=FRAMES_DIR,
        display=False
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

        # set frame positions (start and end)
        start_frame = int(start_minute * 60 * self.original_fps)
        self.vid_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        end_frame = int(end_minute * 60 * self.original_fps) if end_minute is not None else float('inf')
        
        saved_count = 0
        current_frame_idx = start_frame
        display_count = 0
        last_filename = None

        print(f"Starting extraction from {start_minute}m to {end_minute if end_minute else 'end'}m...")
        print(f"Targeting {self.target_fps} FPS, max {frame_limit} frames.")

        plt.figure(figsize=(15, 10))

        while self.vid_capture.is_opened():
            ret, frame = self.vid_capture.read()
            if not ret or current_frame_idx >= end_frame or saved_count >= frame_limit:
                break

            # process frames at the specified interval relative to start_frame
            if (current_frame_idx - start_frame) % self.frame_interval == 0:
                resized = cv2.resize(
                    frame,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_LINEAR
                )

                timestamp = current_frame_idx / self.original_fps
                last_filename = f"frame_{saved_count:06d}_sec_{timestamp:.2f}.jpg"
                cv2.imwrite(os.path.join(output_dir, last_filename), resized)

                # Display progress in notebook if display is true
                if display and saved_count % self.display_every_n == 0 and display_count < 12:
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    plt.subplot(3, 4, display_count + 1)
                    plt.imshow(rgb)
                    plt.title(f"{timestamp:.1f}s")
                    plt.axis("off")
                    display_count += 1

                saved_count += 1

            current_frame_idx += 1

        if display:
            plt.tight_layout()
            plt.show()

        print(f"Successfully saved {saved_count} frames to {output_dir}")
        
        return [
            os.path.join(output_dir, last_filename) if last_filename else None,
            saved_count
        ]
