"""Model backends: local and API."""
from __future__ import annotations

from typing import Any

from .base import LabelerBackend


def build_backend(cfg: dict) -> LabelerBackend:
    """Factory: build the appropriate backend from config."""
    kind = (cfg.get("backend") or "local").lower()
    if kind == "local":
        from .local_backend import LocalBackend
        return LocalBackend(cfg["local"])
    if kind == "api":
        from .api_backend import ApiBackend
        return ApiBackend(cfg["api"])
    raise ValueError(f"Unknown backend: {kind!r}. Expected 'local' or 'api'.")
