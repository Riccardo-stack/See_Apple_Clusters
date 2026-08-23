"""
Core inference logic and model lifecycle management.

This module is used by both the API (api.py) and the desktop app (main.py).
It contains NO HTTP or framework concepts — only pure model logic.
"""

from __future__ import annotations

import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
API_ACCEPTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_DIMENSION = 10_000  # pixels — reject images larger than this


# ──────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────
class ModelError(Exception):
    """Base exception for model-related errors."""


class ModelLoadError(ModelError):
    """Raised when the model fails to load or weights are missing."""


class ModelUnavailableError(ModelError):
    """Raised when the model is busy loading (cold start in progress)."""


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _download_progress(count: int, block_size: int, total_size: int) -> None:
    """Report download progress to stdout."""
    if total_size > 0:
        percent = min(int(count * block_size * 100 / total_size), 100)
        sys.stdout.write(f"\rDownloading model weights: {percent}%")
        sys.stdout.flush()


def ensure_model(model_path: Path, model_url: str) -> Path:
    """Download model weights from *model_url* if *model_path* doesn't exist.

    Returns the resolved *model_path* on success.
    Raises ``ModelLoadError`` if the download fails.
    """
    if model_path.exists():
        return model_path

    print(f"Model not found at {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading from {model_url}...")
        urllib.request.urlretrieve(
            model_url, model_path, reporthook=_download_progress
        )
        print("\nDownload complete.")
    except Exception as e:
        raise ModelLoadError(f"Failed to download model: {e}") from e

    return model_path


def decode_image(file_bytes: bytes) -> np.ndarray | None:
    """Decode raw image bytes into a BGR OpenCV array.

    Returns ``None`` when the bytes cannot be decoded.
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def run_detection(
    model: YOLO,
    image: np.ndarray,
    confidence: float,
) -> tuple[bytes, int, float]:
    """Run YOLO inference on *image* and return the annotated result.

    Returns:
        A tuple of ``(annotated_jpeg_bytes, detections_count, inference_time_ms)``.
    """
    start = time.perf_counter()
    results = model.predict(image, conf=confidence, verbose=False)
    inference_ms = (time.perf_counter() - start) * 1000

    annotated = results[0].plot()
    detections_count = len(results[0].boxes)

    success, jpeg_buf = cv2.imencode(".jpg", annotated)
    if not success:
        raise RuntimeError("Failed to encode annotated image as JPEG")

    return jpeg_buf.tobytes(), detections_count, inference_ms


# ──────────────────────────────────────────────────────────────
# Model manager (lazy-load + TTL)
# ──────────────────────────────────────────────────────────────
class ModelManager:
    """Thread-safe manager for a single YOLO model instance.

    The model is loaded lazily on the first request and automatically
    unloaded after *ttl_seconds* of inactivity.  Every request resets
    the TTL countdown.
    """

    def __init__(
        self,
        model_path: Path,
        model_url: str,
        ttl_seconds: int = 300,
    ) -> None:
        self._model: YOLO | None = None
        self._model_path = model_path
        self._model_url = model_url
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._is_loading = False

    # ── public properties ────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> Path:
        return self._model_path

    # ── model access ─────────────────────────────────────────

    def get_model(self) -> YOLO:
        """Return the loaded model, triggering a lazy load if needed.

        Raises ``ModelUnavailableError`` if another thread is already
        loading the model, and ``ModelLoadError`` if loading fails.
        """
        # Fast path — model already loaded
        if self._model is not None:
            self._reset_timer()
            return self._model

        # Another thread is loading — don't block, fail fast
        if self._is_loading:
            raise ModelUnavailableError(
                "Model is loading, please retry in a few seconds"
            )

        # Try to acquire the lock without blocking
        acquired = self._lock.acquire(timeout=0)
        if not acquired:
            raise ModelUnavailableError(
                "Model is loading, please retry in a few seconds"
            )

        try:
            # Double-check after acquiring lock
            if self._model is not None:
                self._reset_timer()
                return self._model

            self._is_loading = True
            try:
                ensure_model(self._model_path, self._model_url)
                self._model = YOLO(str(self._model_path))
                print(f"Model loaded from {self._model_path}")
            except ModelLoadError:
                raise
            except Exception as e:
                raise ModelLoadError(f"Failed to load model: {e}") from e
            finally:
                self._is_loading = False

            self._reset_timer()
            return self._model
        finally:
            self._lock.release()

    # ── hot-swap ─────────────────────────────────────────────

    def swap_model(self, new_weights_path: Path) -> None:
        """Replace the current model with weights from *new_weights_path*.

        Raises ``ModelLoadError`` if the new weights can't be loaded.
        """
        with self._lock:
            try:
                new_model = YOLO(str(new_weights_path))
            except Exception as e:
                raise ModelLoadError(
                    f"Failed to load new model weights: {e}"
                ) from e

            self._model = new_model
            self._model_path = new_weights_path
            self._reset_timer()
            print(f"Model hot-swapped to {new_weights_path}")

    # ── info ─────────────────────────────────────────────────

    def get_info(self) -> dict:
        """Return a JSON-serialisable dict with model metadata."""
        info: dict = {
            "status": "loaded" if self.is_loaded else "unloaded",
            "weights_file": self._model_path.name,
        }
        if self._model is not None:
            info["classes"] = list(self._model.names.values())
        return info

    # ── lifecycle ────────────────────────────────────────────

    def shutdown(self) -> None:
        """Release model and cancel any pending TTL timer."""
        if self._timer is not None:
            self._timer.cancel()
        self._model = None

    # ── private ──────────────────────────────────────────────

    def _reset_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._ttl_seconds, self._unload)
        self._timer.daemon = True
        self._timer.start()

    def _unload(self) -> None:
        with self._lock:
            self._model = None
            self._timer = None
            print("Model unloaded (TTL expired).")
