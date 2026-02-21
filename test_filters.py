import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from preprocessing.video_loader import VideoLoader
from preprocessing.frame_extractor import FrameExtractor

def main():
    print("Testing Video Loader and Filters...")
    try:
        video_path = os.path.join("data", "raw_videos", "kaggle", "100.mp4")
        vid_loader = VideoLoader(video_path)
        print("Video Loaded:", vid_loader.get_video_info())
        
        extractor = FrameExtractor(vid_loader, target_fps=2) # Extract 2 frames per second
        print("\nStarting extraction for first 1 minute with filters...")
        
        last_file, stats = extractor.extract_frames(
            start_minute=0, 
            end_minute=1, 
            display=False,
            filter_blurry=True,
            filter_crowd=True,
            filter_transitions=True,
            filter_brightness=True,
            filter_replays=True,
            log_csv_path="frames_v2_log.csv"
        )
        
        print("\nTest completed.")
        print("Final Stats:", stats)
    except Exception as e:
        print("Error during test:", e)

if __name__ == "__main__":
    main()
