FROM python:3.12-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reliable package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set up non-root user required by Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:/home/user/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Copy dependency specifications first for Docker layer caching
COPY --chown=user pyproject.toml uv.lock* ./

# Install project dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy project files and model weights
COPY --chown=user . .

# Hugging Face Spaces standard port
EXPOSE 7860

# Launch FastAPI server on port 7860
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
