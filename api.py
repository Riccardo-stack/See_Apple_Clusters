"""
Apple Clusters Detector — FastAPI application.

Exposes the YOLO model as a REST API for image-based apple cluster detection.
Run with:  uv run uvicorn api:app --reload
Docs at:   http://localhost:8000/docs
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from core import (
    API_ACCEPTED_EXTENSIONS,
    MAX_IMAGE_DIMENSION,
    ModelLoadError,
    ModelManager,
    ModelUnavailableError,
    decode_image,
    run_detection,
)
from settings import Settings

# ──────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────
settings = Settings()

manager = ModelManager(
    model_path=settings.resolved_model_path,
    model_url=settings.model_url,
    ttl_seconds=settings.model_ttl_seconds,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — clean up model resources on shutdown."""
    yield
    manager.shutdown()


app = FastAPI(
    title="Apple Clusters Detector API",
    description=(
        "Detect apple clusters in images using a custom-trained YOLO model. "
        "Upload a photo and receive the annotated result with bounding boxes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://see-apple-clusters.vercel.app/"
        "http://localhost:8000",            # Per lo sviluppo locale
        "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=[
        "X-Inference-Time-Ms",
        "X-Detections-Count",
        "X-Confidence-Threshold",
    ],
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
_inference_semaphore = threading.Semaphore(2)


def _error(status_code: int, message: str) -> JSONResponse:
    """Return a JSON error response matching the spec envelope."""
    return JSONResponse(
        status_code=status_code,
        content={"error": message},
    )


# ──────────────────────────────────────────────────────────────
# POST /detect
# ──────────────────────────────────────────────────────────────
@app.post(
    "/detect",
    summary="Detect apple clusters in an image",
    response_class=Response,
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "Annotated image"},
        400: {"description": "Invalid file type or corrupt image"},
        413: {"description": "File too large"},
        422: {"description": "Invalid confidence value"},
        500: {"description": "Inference failed"},
        503: {"description": "Model unavailable"},
    },
)
def detect(
    file: UploadFile = File(
        ..., description="Image file (.jpg, .jpeg, .png) to run detection on"
    ),
    confidence: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold (0.0–1.0)",
    ),
):
    """Accept an image upload, run YOLO inference, and return the annotated
    image as a JPEG with detection metadata in custom response headers."""

    # Default confidence
    if confidence is None:
        confidence = settings.default_confidence

    # Validate file extension
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in API_ACCEPTED_EXTENSIONS:
        return _error(
            400,
            f"Expected .jpg, .jpeg, or .png, got {ext or 'no extension'}",
        )

    # Read file bytes and check size
    file_bytes = file.file.read()
    if len(file_bytes) > settings.max_upload_bytes:
        return _error(
            413, f"File exceeds the {settings.max_upload_size_mb} MB size limit"
        )

    # Decode image
    image = decode_image(file_bytes)
    if image is None:
        return _error(400, "The uploaded file could not be read as an image")

    # Guard against decompression bombs
    h, w = image.shape[:2]
    if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
        return _error(
            400,
            f"Image dimensions ({w}x{h}) exceed the "
            f"{MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} pixel limit",
        )

    # Limit concurrent inferences to protect the host machine
    acquired = _inference_semaphore.acquire(blocking=False)
    if not acquired:
        return _error(503, "Server is busy, please retry in a few seconds")

    try:
        # Load model (lazy)
        try:
            model = manager.get_model()
        except ModelUnavailableError as exc:
            return _error(503, str(exc))
        except ModelLoadError as exc:
            return _error(503, str(exc))

        # Run inference
        try:
            jpeg_bytes, detections_count, inference_ms = run_detection(
                model, image, confidence
            )
        except Exception:
            return _error(500, "Detection failed unexpectedly")
    finally:
        _inference_semaphore.release()

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "X-Inference-Time-Ms": f"{inference_ms:.1f}",
            "X-Detections-Count": str(detections_count),
            "X-Confidence-Threshold": str(confidence),
        },
    )


# ──────────────────────────────────────────────────────────────
# GET /model
# ──────────────────────────────────────────────────────────────
@app.get(
    "/model",
    summary="Get model metadata",
    responses={200: {"description": "Model information"}},
)
def get_model_info():
    """Return the current model's metadata (status, classes, weights path)."""
    return manager.get_info()


# ──────────────────────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────────────────────
@app.get(
    "/health",
    summary="Health check",
    responses={200: {"description": "Service is healthy"}},
)
def health():
    """Return the service health status and whether the model is loaded."""
    return {"status": "healthy", "model_loaded": manager.is_loaded}


# ──────────────────────────────────────────────────────────────
# Static frontend (must come AFTER all API routes)
# ──────────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
