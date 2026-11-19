"""
Tests for FastAPI REST API endpoints (api.py).

These tests use FastAPI's TestClient to send simulated HTTP requests
and verify response status codes, headers, and JSON payloads.
"""

import pytest


# ──────────────────────────────────────────────────────────────
# 1. Basic Health & Info Endpoints
# ──────────────────────────────────────────────────────────────

def test_health_endpoint(client):
    """
    Test GET /health.
    Verifies that the server is alive and returns the expected status dictionary.
    """
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_model_info_endpoint(client):
    """
    Test GET /model.
    Verifies that the model metadata endpoint returns status and weights info.
    """
    response = client.get("/model")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] in ("loaded", "unloaded")
    assert "weights_file" in data


# ──────────────────────────────────────────────────────────────
# 2. POST /detect — Valid Detection Requests
# ──────────────────────────────────────────────────────────────

def test_detect_valid_sample_image(client, sample_image_bytes):
    """
    Test POST /detect with a real sample image from samples/.
    Verifies that inference runs successfully and returns:
      - 200 OK status
      - image/jpeg content type
      - Custom response headers with inference metadata
    """
    response = client.post(
        "/detect",
        files={"file": ("IMG_5822.jpeg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

    # Verify custom detection headers returned by our API
    assert "x-inference-time-ms" in response.headers
    assert "x-detections-count" in response.headers
    assert "x-confidence-threshold" in response.headers

    # Check that returned image bytes are non-empty
    assert len(response.content) > 0


def test_detect_with_custom_confidence(client, sample_image_bytes):
    """
    Test POST /detect with a custom ?confidence query parameter.
    Verifies that the API uses the requested confidence value.
    """
    custom_conf = 0.60
    response = client.post(
        f"/detect?confidence={custom_conf}",
        files={"file": ("IMG_5822.jpeg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.headers["x-confidence-threshold"] == str(custom_conf)


# ──────────────────────────────────────────────────────────────
# 3. POST /detect — Input Validation & Error Handling
# ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "invalid_filename",
    [
        "notes.txt",
        "document.pdf",
        "dataset.csv",
        "audio.mp3",
        "animation.gif",
    ],
)
def test_detect_rejects_unsupported_extensions(client, invalid_filename):
    """
    Test that uploading files with invalid extensions returns 400 Bad Request.
    @pytest.mark.parametrize runs this same test for each filename in the list!
    """
    response = client.post(
        "/detect",
        files={"file": (invalid_filename, b"fake file content", "application/octet-stream")},
    )
    assert response.status_code == 400

    error_data = response.json()
    assert "error" in error_data
    assert "Expected .jpg, .jpeg, or .png" in error_data["error"]


def test_detect_rejects_corrupted_image(client):
    """
    Test that a file with an image extension (.jpg) but corrupted bytes returns 400 Bad Request.
    """
    corrupt_bytes = b"This is not a real JPEG image file, just random text."
    response = client.post(
        "/detect",
        files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
    )
    assert response.status_code == 400

    error_data = response.json()
    assert error_data["error"] == "The uploaded file could not be read as an image"


@pytest.mark.parametrize("invalid_confidence", [-0.5, 1.5, 99.0])
def test_detect_rejects_out_of_range_confidence(client, dummy_jpeg_bytes, invalid_confidence):
    """
    Test that setting confidence outside the valid 0.0 - 1.0 range triggers
    FastAPI's built-in 422 Unprocessable Entity validation error.
    """
    response = client.post(
        f"/detect?confidence={invalid_confidence}",
        files={"file": ("photo.jpg", dummy_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 422


def test_detect_missing_file_payload(client):
    """
    Test that calling POST /detect without providing a file returns 422.
    """
    response = client.post("/detect")
    assert response.status_code == 422
