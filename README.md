# Football Data Generation Pipeline

A robust pipeline for extracting, filtering, and processing football video frames. This project focuses on generating high-quality datasets from football match footage by intelligently filtering out non-play frames (replays, crowd shots, transitions) and reconstructing clean sequences.

## Features

- **Intelligent Frame Extraction**: Time-aware sampling and quality filtering.
- **Advanced Filtering (YOLOv8)**:
  - Blurry frame detection (Laplacian variance).
  - Pitch detection (Green ratio analysis).
  - Scene transition detection (Histogram correlation).
  - Brightness/Exposure validation.
  - Live play detection using player and ball presence.
  - Replay detection heuristics (Transition + Slow-motion + Heuristics).
- **Logging**: Frame-by-frame quality metrics exported to CSV.
- **Video Reconstruction**: Assemble processed frames back into standardized video sequences.

## Project Structure

```text
├── data/
│   ├── interim/            # Processed frames and intermediate data
│   └── raw_videos/         # Source video files
├── src/
│   ├── preprocessing/
│   │   ├── filters.py        # Core filtering logic (YOLO, CV2)
│   │   ├── frame_extractor.py # Logic for extracting frames from videos
│   │   ├── logger.py         # CSV logging for frame metrics
│   │   ├── reconstructor.py  # Rebuilds videos from image sequences
│   │   └── video_loader.py   # Robust video reading utilities
│   └── config.py           # Project-wide configurations
├── notebooks/
│   ├── 00_video_downloader.ipynb
│   └── 01_video_preprocessing.ipynb
├── test_filters.py         # Test suite for filtering logic
├── requirements.txt
└── README.md
```

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd football_data
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- **Preprocessing**: Use the `notebooks/01_video_preprocessing.ipynb` for an end-to-end walkthrough of the extraction and filtering pipeline.
- **Custom Logic**: Integrate specific filters from `src/preprocessing/filters.py` into your own scripts.
- **Testing**: Run `python test_filters.py` to validate the filtering logic against sample frames.
