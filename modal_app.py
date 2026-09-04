"""
Modal serverless deployment for Apple Clusters Detector API.

Deploy with:
    uv run modal deploy modal_app.py
"""

import modal

# Create Modal App
app = modal.App("apple-clusters-api")

# Define the container image with required system libraries and dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.34.0",
        "python-multipart>=0.0.18",
        "opencv-python-headless>=4.11.0",
        "ultralytics>=8.4.90",
        "pydantic-settings>=2.7.0",
        "pillow>=12.3.0",
        "numpy>=2.0.0",
    )
    .add_local_file("models/best.pt", remote_path="/root/models/best.pt")
    .add_local_file("core.py", remote_path="/root/core.py")
    .add_local_file("settings.py", remote_path="/root/settings.py")
    .add_local_file("api.py", remote_path="/root/api.py")
    .add_local_dir("frontend", remote_path="/root/frontend")
)


@app.function(
    image=image,
    scaledown_window=300,  # Keeps the container warm for 5 minutes after each request
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/root")
    from api import app as web_app

    return web_app
