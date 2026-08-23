"""
Apple Clusters Detector — Real-time apple cluster detection using YOLO.

Launch the app and choose between live webcam detection or
running inference on a photo / video selected through a file dialog.
"""

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import cv2
from ultralytics import YOLO

from core import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, ensure_model
from settings import Settings

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
_settings = Settings()
MODEL_PATH = _settings.resolved_model_path
MODEL_URL = _settings.model_url

RESULTS_DIR = Path(__file__).resolve().parent / "results"



def pick_file() -> str | None:
    """Open a native file‑browser dialog and return the selected path."""
    root = tk.Tk()
    root.withdraw()
    # Raise the dialog above other windows
    root.attributes("-topmost", True)

    filetypes = [
        ("Images & Videos", " ".join(
            f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
        )),
        ("Images", " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS))),
        ("Videos", " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS))),
        ("All files", "*.*"),
    ]

    path = filedialog.askopenfilename(
        title="Select a photo or video",
        filetypes=filetypes,
    )
    root.destroy()
    return path if path else None


def print_menu():
    """Print the interactive menu."""
    print()
    print("╔══════════════════════════════════════╗")
    print("║       Apple Clusters Detector        ║")
    print("╠══════════════════════════════════════╣")
    print("║                                      ║")
    print("║  [1]  Use Webcam                     ║")
    print("║  [2]  Open a Photo / Video           ║")
    print("║  [q]  Quit                           ║")
    print("║                                      ║")
    print("╚══════════════════════════════════════╝")


# ──────────────────────────────────────────────────────────────
# Detection modes
# ──────────────────────────────────────────────────────────────
def run_webcam(model: YOLO, confidence: float):
    """Run live webcam detection with OpenCV display."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam started — press 'q' to stop.\n")

    while True:
        success, frame = cap.read()
        if not success:
            continue

        results = model.predict(frame, conf=confidence, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("Apple Clusters Detector — Webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stopped.")


def run_photo(model: YOLO, confidence: float, file_path: str):
    """Run detection on a single image, display result, and optionally save."""
    image = cv2.imread(file_path)
    if image is None:
        print(f"Could not read image: {file_path}")
        return

    results = model.predict(image, conf=confidence, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("Apple Clusters Detector — Photo", annotated)
    print("Showing detection result — press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Ask if the user wants to save
    answer = input("Save the result image? (y/n): ").strip().lower()
    if answer in ("y", "yes"):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"result_{Path(file_path).stem}.jpg"
        save_path = RESULTS_DIR / filename
        cv2.imwrite(str(save_path), annotated)
        print(f"Saved to {save_path}")
    else:
        print("Result not saved.")


def run_video(model: YOLO, confidence: float, file_path: str):
    """Run detection on a video file with OpenCV display."""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Could not open video: {file_path}")
        return

    print(f"Playing video — press 'q' to stop.\n")

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, conf=confidence, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("Apple Clusters Detector — Video", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video playback finished.")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Apple Clusters Detector")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    args = parser.parse_args()

    # Ensure model weights are available
    ensure_model(MODEL_PATH, MODEL_URL)

    # Load model once
    print("Loading model...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    print("Model loaded!\n")

    # Interactive loop
    while True:
        print_menu()
        choice = input("\nSelect an option: ").strip().lower()

        if choice == "1":
            run_webcam(model, args.confidence)

        elif choice == "2":
            file_path = pick_file()
            if file_path is None:
                print("No file selected.")
                continue

            ext = Path(file_path).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                run_photo(model, args.confidence, file_path)
            elif ext in VIDEO_EXTENSIONS:
                run_video(model, args.confidence, file_path)
            else:
                print(f"Unsupported file type: {ext}")

        elif choice in ("q", "quit", "exit"):
            print("Goodbye!")
            break

        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
