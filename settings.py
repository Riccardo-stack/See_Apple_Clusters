"""
Configuration for the Apple Clusters Detector API.

All values have sensible defaults — the API works out of the box
without a .env file. Override any value via environment variables
or a .env file in the project root.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


# Resolve project root relative to this file (works regardless of cwd)
_PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_path: str = "models/best.pt"
    model_url: str = (
        "https://github.com/Riccardo-stack/See_Apple_Clusters/"
        "releases/download/v1.0/best.pt"
    )
    default_confidence: float = 0.25
    max_upload_size_mb: int = 30
    max_model_size_mb: int = 200
    model_ttl_seconds: int = 300
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def resolved_model_path(self) -> Path:
        """Return model_path as an absolute Path, resolved from project root."""
        p = Path(self.model_path)
        return p if p.is_absolute() else _PROJECT_ROOT / p

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def max_model_bytes(self) -> int:
        return self.max_model_size_mb * 1024 * 1024

    model_config = {
        "env_file": _PROJECT_ROOT / ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
