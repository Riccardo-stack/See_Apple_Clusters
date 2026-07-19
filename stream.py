"""
Live Apple Detection Stream
============================
Captures frames from the camera, runs YOLO detection,
and streams the annotated video as MJPEG over HTTP.

Open the stream URL in Safari on your iPhone to watch live.

Usage:
    uv run python stream.py
    uv run python stream.py --camera 1          # use a specific camera index
    uv run python stream.py --conf 0.4          # set confidence threshold
    uv run python stream.py --weights path.pt   # use different weights
"""

import argparse
import socket
import threading
import time
import cv2
import numpy as np
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "models" / "best.pt"
DEFAULT_CAMERA = 0
DEFAULT_CONF = 0.25
PORT = 8080

# ──────────────────────────────────────────────────────────────
# HTML page served at /
# ──────────────────────────────────────────────────────────────
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🍎 Apple Detector</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #111;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            font-family: -apple-system, sans-serif;
            color: #fff;
        }
        h1 {
            font-size: 1.4rem;
            margin: 16px 0;
            opacity: 0.9;
        }
        img {
            width: 100%;
            max-width: 720px;
            border-radius: 12px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.5);
            background-color: #222;
        }
        .info {
            margin-top: 12px;
            font-size: 0.85rem;
            opacity: 0.5;
        }
    </style>
</head>
<body>
    <h1>🍎 Live Apple Cluster Detection</h1>
    <img src="/video" alt="Live stream">
    <p class="info">Confidence threshold: {{ conf }}</p>
</body>
</html>
"""

app = Flask(__name__)

# These will be set in main()
model = None
camera_index = 0
confidence = 0.25

# ──────────────────────────────────────────────────────────────
# Shared frame buffer — one camera, many clients
# ──────────────────────────────────────────────────────────────
latest_frame_bytes = None
frame_lock = threading.Lock()
frame_count = 0


def get_local_ip():
    """Get the Mac's local IP address on the network."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def camera_loop():
    """
    Background thread: continuously captures frames, runs YOLO,
    and stores the latest annotated JPEG in `latest_frame_bytes`.
    """
    global latest_frame_bytes, frame_count

    print(f"🎬 Camera thread starting, trying index {camera_index}...")

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ Could not open camera index {camera_index}")
        if camera_index != 0:
            print("🔄 Trying fallback to camera 0...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Fallback failed. No camera available.")
                return
            else:
                print("✅ Fallback to camera 0 successful!")
        else:
            print("❌ No camera available.")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"📷 Camera opened! Reading frames...")

    # Warm up: read a few frames to let the camera settle
    for _ in range(5):
        cap.read()

    print("🔥 Camera warmed up, starting inference loop...")

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue

        # Run YOLO inference
        results = model.predict(frame, conf=confidence, verbose=False)

        # Draw bounding boxes
        annotated = results[0].plot()

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        # Store the latest frame for all clients to read
        with frame_lock:
            latest_frame_bytes = buffer.tobytes()
            frame_count += 1

        # Log every 100 frames so we know it's working
        if frame_count % 100 == 0:
            print(f"📊 Processed {frame_count} frames")

    cap.release()


def generate_frames():
    """Yield the latest annotated frame as MJPEG. Safe for multiple clients."""
    last_served_count = -1

    while True:
        with frame_lock:
            frame = latest_frame_bytes
            current_count = frame_count

        if frame is None:
            # Camera hasn't produced a frame yet, wait
            time.sleep(0.1)
            continue

        # Only yield when we have a new frame
        if current_count != last_served_count:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
            last_served_count = current_count

        # Small sleep to avoid busy-waiting
        time.sleep(0.03)


@app.route('/')
def index():
    return render_template_string(HTML_PAGE, conf=confidence)


@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def main():
    global model, camera_index, confidence

    parser = argparse.ArgumentParser(description="Live Apple Detection Stream")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to YOLO weights")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA, help="Camera index (0=default, 1,2...)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold (0-1)")
    parser.add_argument("--port", type=int, default=PORT, help="Server port")
    args = parser.parse_args()

    camera_index = args.camera
    confidence = args.conf

    # Load model
    print(f"🔄 Loading model from {args.weights}...")
    model = YOLO(args.weights)
    print("✅ Model loaded!")

    # Warm up the model with a dummy image so first real frame is fast
    print("🔥 Warming up model...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, conf=confidence, verbose=False)
    print("✅ Model warm-up complete!")

    # Start camera capture in background thread
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    # Give camera thread a moment to initialize
    time.sleep(2)

    # Print connection info
    local_ip = get_local_ip()
    print()
    print("=" * 50)
    print("🍎 Apple Detector Stream Ready!")
    print("=" * 50)
    print(f"  Mac:    http://localhost:{args.port}")
    print(f"  iPhone: http://{local_ip}:{args.port}")
    print(f"  Camera: {args.camera}")
    print(f"  Conf:   {args.conf}")
    print("=" * 50)
    print("  Press Ctrl+C to stop")
    print()

    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
