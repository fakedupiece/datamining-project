"""Shared utilities for the Stage 0-2 research pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a configuration path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for an output path and return the path."""
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_directory(path: str | Path) -> Path:
    """Create an output directory and return it."""
    resolved = resolve_project_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 without loading the full file into memory."""
    digest = hashlib.sha256()
    with resolve_project_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_divide(numerator: Any, denominator: Any) -> np.ndarray:
    """Divide arrays while returning NaN for non-positive or invalid denominators."""
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    result = np.full(np.broadcast_shapes(numerator_array.shape, denominator_array.shape), np.nan)
    valid = np.isfinite(numerator_array) & np.isfinite(denominator_array) & (denominator_array > 0)
    np.divide(numerator_array, denominator_array, out=result, where=valid)
    return result


def write_json(payload: Any, path: str | Path) -> Path:
    """Write JSON with stable formatting and NumPy-aware scalar conversion."""
    output_path = ensure_parent(path)

    def default(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=default)
        handle.write("\n")
    return output_path
