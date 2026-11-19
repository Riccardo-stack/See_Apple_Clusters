from io import BytesIO
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def dummy_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (50, 50), color=(255, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def sample_image_bytes() -> bytes:
    sample_path = Path(__file__).resolve().parent.parent / "samples" / "IMG_5822.jpeg"
    assert sample_path.exists(), f"Sample image not found at {sample_path}"
    return sample_path.read_bytes()
