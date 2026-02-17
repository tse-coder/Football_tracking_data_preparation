import os
import sys
import cv2

# Ensure parent directory is in sys.path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VIDEOS_DIR

class VideoLoader:
    def __init__(self, video_path=None):
        """
        Initializes the VideoLoader with the path to the video file.
        If video_path is None, it defaults to the first video found in VIDEOS_DIR.
        """
        if video_path is None:
            video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(('.mp4', '.avi', '.mov','.mkv'))]
            if not video_files:
                raise FileNotFoundError(f"No video files found in {VIDEOS_DIR}")
            video_path = os.path.join(VIDEOS_DIR, video_files[0]) # pick the first file in the folder
            print(f"No video path provided. Defaulting to: {video_path}")
        
        self.file_path = video_path
        self.video = cv2.VideoCapture(self.file_path)
        if not self.video.isOpened():
            raise ValueError(f"Cannot open video: {self.file_path}")
        
        # Initialize metadata
        self.get_video_info()

    def get_video_info(self):
        """
        Extract and store video metadata.
        """
        fps = self.video.get(cv2.CAP_PROP_FPS)
        self.original_fps = fps if fps > 0 else 15.0  # Fallback to 15 FPS if unavailable
        
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.original_fps if self.original_fps > 0 else 0
        
        if fps <= 0:
            print(f"Warning: Could not detect FPS for {self.file_path}. Defaulting to {self.original_fps}.")

        return {
            "video": self.file_path,
            "fps": self.original_fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "resolution": f"{self.width}x{self.height}"
        }

    def is_opened(self):
        return self.video.isOpened()

    def read(self):
        return self.video.read()

    def set(self, propId, value):
        return self.video.set(propId, value)

    def release(self):
        """
        Release the video capture object.
        """
        self.video.release()