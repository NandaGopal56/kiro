from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Root directory of the whole project (the folder containing .models)."""
    return Path(__file__).resolve().parents[2]


def model_dir() -> Path:
    """Shared ``.models`` folder at the project root, created if missing."""
    path = project_root() / ".models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path(name: str) -> Path:
    """Absolute path to a model file inside the root ``.models`` folder."""
    return model_dir() / name
