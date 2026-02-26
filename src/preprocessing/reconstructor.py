import os
import cv2
import re
from pathlib import Path


class VideoReconstructor:
    def __init__(self, output_fps=10):
        """
        Initializes the VideoReconstructor.

        Args:
            output_fps (int): The frames per second for the output video.
        """
        self.output_fps = output_fps

    def _get_frame_number(self, filename):
        """
        Extracts the frame number from the filename for sorting.
        Assumes format like 'frame_000001_sec_0.10.jpg' or 'frame_000001.jpg'.
        """
        match = re.search(r"frame_(\d+)", filename)
        if match:
            return int(match.group(1))
        return filename

    def reconstruct_from_frames(self, frames_dir, output_path):
        """
        Assembles frames from a directory into a video file.

        Args:
            frames_dir (str): Path to the directory containing image frames.
            output_path (str): Path where the output video will be saved.
        """
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            print(f"Error: Frames directory {frames_dir} does not exist.")
            return False

        # Get and sort image files
        images = [
            f for f in os.listdir(frames_dir) if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        images.sort(key=self._get_frame_number)

        if not images:
            print(f"No images found in {frames_dir}")
            return False

        # Read the first image to get dimensions
        first_image_path = os.path.join(frames_dir, images[0])
        frame = cv2.imread(first_image_path)
        if frame is None:
            print(f"Could not read first image: {first_image_path}")
            return False

        height, width, layers = frame.shape
        size = (width, height)

        # Initialize VideoWriter
        # Use mp4v for .mp4 or XVID for .avi
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.output_fps, size)

        print(f"Reconstructing video to {output_path} from {len(images)} frames...")

        for i, filename in enumerate(images):
            img_path = os.path.join(frames_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                out.write(img)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(images)} frames...")

        out.release()
        print(f"Successfully saved reconstructed video to {output_path}")
        return True

