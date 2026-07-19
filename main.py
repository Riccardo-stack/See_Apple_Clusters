"""
Apple Clusters Detector — Real-time apple cluster detection using YOLO.

Usage:
    python main.py                      # Run on webcam (default)
    python main.py --source samples/    # Run on sample images
    python main.py --source video.mp4   # Run on a video file
    python main.py --confidence 0.5     # Custom confidence threshold
"""

import argparse
import sys
import urllib.request
from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'best.pt'
MODEL_URL = "https://github.com/Riccardo-stack/Apple_Clusters_Detector/releases/download/v1.0/best.pt"


def download_progress(count, block_size, total_size):
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        percent = min(percent, 100)
        sys.stdout.write(f"\rDownloading model weights: {percent}%")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Apple Clusters Detector")
    parser.add_argument("--source", default="0", help="Input source: webcam index (0), image path, directory, or video.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", help="Flag to save results")
    parser.add_argument("--no-display", action="store_true", help="Flag to disable display window")
    args = parser.parse_args()

    # Download model if it doesn't exist
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"Downloading from {MODEL_URL}...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=download_progress)
            print("\nDownload complete.")
        except Exception as e:
            print(f"\nFailed to download model: {e}")
            sys.exit(1)

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    source = args.source
    if source.isdigit():
        source = int(source)
    else:
        if not Path(source).exists():
            print(f"Error: Source '{source}' does not exist.")
            sys.exit(1)

    try:
        # Run inference
        model.predict(
            source=source,
            conf=args.confidence,
            save=args.save,
            show=not args.no_display
        )
    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Exiting cleanly.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")


if __name__ == '__main__':
    main()
