"""Filesystem path helpers (no heavy dependencies)."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: Path) -> Path:
    """Resolve a path relative to the repository root."""
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
