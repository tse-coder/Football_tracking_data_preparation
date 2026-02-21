from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/config.py -> parent is src -> parent is root

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "raw_videos"
FRAMES_DIR = DATA_DIR / "interim" / "frames"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Aliases for backward compatibility or clarity if needed
RAW_VIDEOS_DIR = VIDEOS_DIR

# Preprocessing Filter Thresholds
MIN_GREEN_RATIO = 0.20  # Minimum percentage of green pitch required (0.0 to 1.0)
BLUR_THRESHOLD = 50.0   # Variance of Laplacian threshold (lower means blurrier)
SCENE_CHANGE_THRESHOLD = 0.75 # Histogram correlation threshold for scene cuts

# V2 Preprocessing
MIN_BRIGHTNESS = 20.0
MAX_BRIGHTNESS = 230.0
REPLAY_SKIP_SECONDS = 5.0
LOG_FILE_DUMP_INTERVAL = 100

# Make sure output folders exist
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
