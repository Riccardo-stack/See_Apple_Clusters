# Apple Clusters Detector — API Reference

The project ships a **FastAPI** REST service that exposes the YOLO model over HTTP.
This lets you integrate apple-cluster detection into any language or toolchain — no Python environment required on the client side.

---

## Table of Contents

- [Running the Server](#running-the-server)
- [Interactive Docs (Swagger UI)](#interactive-docs-swagger-ui)
- [Configuration](#configuration)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [POST /detect](#post-detect)
  - [GET /model](#get-model)
  - [GET /health](#get-health)
- [Response Headers](#response-headers)
- [Error Format](#error-format)
- [Code Examples](#code-examples)

---

## Running the Server

```bash
# Install dependencies (if not done already)
uv sync

# Start with auto-reload (development)
uv run uvicorn api:app --reload

# Start for production (no reload, bind to all interfaces)
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

The server is available at `http://localhost:8000` by default.

---

## Interactive Docs (Swagger UI)

FastAPI generates an interactive browser UI automatically:

| UI | URL |
|----|-----|
| Swagger UI | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |
| OpenAPI schema (JSON) | `http://localhost:8000/openapi.json` |

You can try every endpoint directly from the browser — no extra tools needed.

---

## Configuration

All settings have sensible defaults and work out of the box.  
Override any value via **environment variables** or a **`.env` file** in the project root.

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/best.pt` | Path to the YOLO `.pt` weights file |
| `MODEL_URL` | GitHub Releases URL | Remote URL to auto-download weights from |
| `DEFAULT_CONFIDENCE` | `0.25` | Confidence threshold used when none is supplied |
| `MAX_UPLOAD_SIZE_MB` | `30` | Maximum allowed image upload size (MB) |
| `MODEL_TTL_SECONDS` | `300` | Seconds of inactivity before the model is unloaded from memory |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |

**Example `.env` file:**

```env
DEFAULT_CONFIDENCE=0.4
MAX_UPLOAD_SIZE_MB=50
MODEL_TTL_SECONDS=600
```

---

## Authentication

The API has **no authentication** in its current form — it is intended for local or trusted-network use.  
If you expose it publicly, place it behind a reverse proxy (e.g. nginx) with your preferred auth layer.

---

## Endpoints

### POST /detect

Run apple-cluster detection on an uploaded image.

**Request**

| Field | In | Type | Required | Description |
|---|---|---|---|---|
| `file` | form-data | `image/jpeg`, `image/png` | ✅ | Image to analyse (`.jpg`, `.jpeg`, `.png`) |
| `confidence` | query | `float` [0.0 – 1.0] | ❌ | Detection confidence threshold. Defaults to `DEFAULT_CONFIDENCE` |

**Success response — `200 OK`**

- **Content-Type:** `image/jpeg`
- **Body:** The original image annotated with bounding boxes drawn around every detected apple cluster.
- Detection metadata is returned in [custom response headers](#response-headers).

**Error responses**

| Status | Meaning |
|---|---|
| `400` | Unsupported file extension, corrupt / unreadable image, or image dimensions too large |
| `413` | File exceeds the configured size limit |
| `422` | `confidence` value is outside the `0.0–1.0` range |
| `500` | Inference failed unexpectedly |
| `503` | Model is still loading (cold start) or server is busy — retry in a few seconds |

---

### GET /model

Return metadata about the currently loaded model.

**Request** — no parameters.

**Success response — `200 OK`**

```json
{
  "status": "loaded",
  "weights_file": "models/best.pt",
  "classes": ["apple_cluster"]
}
```

When the model has not yet been loaded (e.g. before the first `/detect` call), `"status"` will be `"unloaded"` and `"classes"` will be absent.

---

### GET /health

Lightweight health check — useful for load balancers and monitoring tools.

**Request** — no parameters.

**Success response — `200 OK`**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

`model_loaded` is `false` until the first `/detect` call triggers the lazy load.

---

## Response Headers

Successful `/detect` responses include three custom headers with inference metadata:

| Header | Type | Example | Description |
|---|---|---|---|
| `X-Inference-Time-Ms` | `float` | `"47.3"` | End-to-end YOLO inference time in milliseconds |
| `X-Detections-Count` | `integer` | `"5"` | Number of apple clusters found in the image |
| `X-Confidence-Threshold` | `float` | `"0.25"` | The confidence threshold that was applied |

---

## Error Format

All error responses share the same JSON envelope:

```json
{
  "error": "Human-readable description of what went wrong"
}
```

---

## Code Examples

### cURL

```bash
# Basic detection with default confidence
curl -X POST http://localhost:8000/detect \
  -F "file=@photo.jpg" \
  --output result.jpg \
  -D -   # print response headers to stdout

# Detection with custom confidence
curl -X POST "http://localhost:8000/detect?confidence=0.4" \
  -F "file=@photo.jpg" \
  --output result.jpg
```

### Python (`httpx`)

```python
import httpx

with open("photo.jpg", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/detect",
        files={"file": ("photo.jpg", f, "image/jpeg")},
        params={"confidence": 0.4},
    )

response.raise_for_status()

# Save annotated image
with open("result.jpg", "wb") as out:
    out.write(response.content)

# Read detection metadata from headers
print("Detections:", response.headers["X-Detections-Count"])
print("Inference time:", response.headers["X-Inference-Time-Ms"], "ms")
```

### JavaScript (`fetch`)

```js
const form = new FormData();
form.append("file", fileInput.files[0]);

const response = await fetch("http://localhost:8000/detect?confidence=0.35", {
  method: "POST",
  body: form,
});

if (!response.ok) {
  const { error } = await response.json();
  throw new Error(error);
}

const blob = await response.blob();
const imageUrl = URL.createObjectURL(blob);

console.log("Detections:", response.headers.get("X-Detections-Count"));
```

### Health check

```bash
curl http://localhost:8000/health
# → {"status":"healthy","model_loaded":true}
```
