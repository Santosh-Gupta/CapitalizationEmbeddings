"""Runtime path helpers for Colab, RunPod, and local development."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the current repository root from an installed package."""

    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    """Return the persistent workspace root for the active environment."""

    runpod_workspace = Path("/workspace")
    if runpod_workspace.exists():
        return runpod_workspace

    colab_drive = Path("/content/drive/MyDrive")
    if colab_drive.exists():
        return colab_drive

    return repo_root()


def project_root() -> Path:
    """Return the persistent project artifact root."""

    return workspace_root() / "capitalization_embeddings"


def checkpoint_dir(*parts: str) -> str:
    path = project_root() / "checkpoints" / Path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def cache_dir(*parts: str) -> str:
    path = workspace_root() / ".cache" / Path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def configure_huggingface_cache() -> str:
    """Set Hugging Face cache env vars to persistent storage."""

    path = cache_dir("huggingface")
    os.environ.setdefault("HF_HOME", path)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(path) / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(path) / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", path)
    return path
