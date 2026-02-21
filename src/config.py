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

# Make sure output folders exist
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
