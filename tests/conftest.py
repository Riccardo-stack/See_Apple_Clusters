"""
Pytest configuration and shared fixtures.

In pytest, `conftest.py` is automatically discovered by pytest.
Any fixture defined here is available to all test files without needing an explicit import!
"""

from io import BytesIO
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import app


@pytest.fixture(scope="session")
def client():
    """
    Creates a FastAPI TestClient that simulates HTTP requests.
    This lets us test API endpoints (/health, /detect, etc.) without starting a real server.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def dummy_jpeg_bytes() -> bytes:
    """
    Generates a tiny 50x50 JPEG image in memory.
    This is extremely fast and perfect for testing image uploads without disk I/O.
    """
    img = Image.new("RGB", (50, 50), color=(255, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """
    Reads a real sample apple photo from the samples/ directory.
    Used for end-to-end detection tests with the trained YOLO model.
    """
    sample_path = Path(__file__).resolve().parent.parent / "samples" / "IMG_5822.jpeg"
    assert sample_path.exists(), f"Sample image not found at {sample_path}"
    return sample_path.read_bytes()
