"""Localhost vision-model backend (Ollama, LM Studio, vLLM, custom).

Runs the model in a separate local server process. This app only sends HTTP
requests — no torch/CUDA needed in the labeler itself. Recommended on Windows
when PyTorch CUDA DLL errors occur.
"""
from __future__ import annotations

import logging
from typing import Optional

from .api_backend import ApiBackend

log = logging.getLogger(__name__)

# Default OpenAI-compatible base URLs for common local servers.
SERVER_PRESETS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "vllm": "http://localhost:8000/v1",
}


def resolve_localhost_api_cfg(cfg: dict) -> dict:
    """Convert localhost config block into ApiBackend-compatible settings."""
    server = (cfg.get("server") or "ollama").lower()
    base_url = cfg.get("base_url")

    if not base_url:
        if server in SERVER_PRESETS:
            base_url = SERVER_PRESETS[server]
        elif server == "custom":
            raise ValueError(
                "localhost.server is 'custom' but no base_url was set. "
                "Example: http://localhost:8080/v1"
            )
        else:
            raise ValueError(
                f"Unknown localhost server {server!r}. "
                f"Expected one of: {', '.join(SERVER_PRESETS)} or 'custom'."
            )

    return {
        "provider": "openai-compatible",
        "model": cfg["model"],
        "base_url": base_url,
        "max_tokens": int(cfg.get("max_tokens", 1024)),
        "temperature": float(cfg.get("temperature", 0.2)),
        "request_timeout": float(cfg.get("request_timeout", 300)),
        "max_retries": int(cfg.get("max_retries", 3)),
        # Not used for openai-compatible, but keeps ApiBackend happy.
        "api_key_env": cfg.get("api_key_env", "OPENAI_API_KEY"),
        "_localhost_server": server,
        "_localhost_base_url": base_url,
    }


class LocalhostBackend(ApiBackend):
    """Vision model served locally via an OpenAI-compatible HTTP endpoint."""

    name = "localhost"

    def __init__(self, cfg: dict):
        api_cfg = resolve_localhost_api_cfg(cfg)
        self.server = api_cfg.pop("_localhost_server")
        self.local_base_url = api_cfg.pop("_localhost_base_url")
        super().__init__(api_cfg)
        log.info(
            "Localhost backend: server=%s url=%s model=%s",
            self.server,
            self.local_base_url,
            self.model_id,
        )

    def info(self) -> dict:
        return {
            "backend": self.name,
            "server": self.server,
            "url": self.local_base_url,
            "model": self.model_id,
        }
