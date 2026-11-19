"""
Tests for core helper functions (core.py) and application settings (settings.py).

These unit tests verify individual functions in isolation without needing the HTTP server.
"""

import numpy as np
from core import decode_image
from settings import Settings


# ──────────────────────────────────────────────────────────────
# 1. Image Decoding Tests (core.py)
# ──────────────────────────────────────────────────────────────

def test_decode_image_with_valid_jpeg(dummy_jpeg_bytes):
    """
    Test decode_image() with valid JPEG bytes.
    Verifies that it returns a valid 3-channel OpenCV (BGR) NumPy array.
    """
    image = decode_image(dummy_jpeg_bytes)

    assert image is not None
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3  # (height, width, channels)
    assert image.shape == (50, 50, 3)  # Matches dummy_jpeg_bytes dimensions


def test_decode_image_with_corrupt_bytes():
    """
    Test decode_image() with corrupt or random bytes.
    Verifies that it gracefully returns None instead of raising an unhandled exception.
    """
    garbage_bytes = b"Not an image at all! 1234567890"
    image = decode_image(garbage_bytes)

    assert image is None


def test_decode_image_with_empty_bytes():
    """
    Test decode_image() with empty bytes b"".
    Verifies that empty input safely returns None.
    """
    image = decode_image(b"")

    assert image is None


# ──────────────────────────────────────────────────────────────
# 2. Application Settings Tests (settings.py)
# ──────────────────────────────────────────────────────────────

def test_settings_default_values():
    """
    Test that default Settings values match expected configuration.
    """
    settings = Settings()

    # Default confidence threshold
    assert settings.default_confidence == 0.25

    # Max upload limit
    assert settings.max_upload_size_mb == 30
    assert settings.max_upload_bytes == 30 * 1024 * 1024

    # Model path resolution
    assert settings.resolved_model_path.name == "best.pt"
