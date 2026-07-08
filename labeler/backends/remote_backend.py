"""Remote GPU worker backend.

Talks to a `labeler.worker.server` process running on a remote (cloud) GPU
box over HTTP. Storage (video files) and outputs stay entirely on the client
machine — only the prompt + base64-encoded frames travel over the wire, and
only the label text comes back.

Wire protocol (JSON):

    POST /label
      body: {"prompt": str, "frames": [b64_jpeg, ...]}
      returns: {"text": str}

    GET  /info
      returns: {"backend": "remote", "worker": {...}, "model": str}

    GET  /health
      returns: {"status": "ok"}

Optional bearer-token auth: set `remote.auth_token` in config (or pass
`--worker-token`) — the client sends it as `Authorization: Bearer <token>`.

This backend intentionally has zero torch/transformers dependencies; it is
usable with the lightweight `requirements-api.txt` install on the client.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import List, Optional

import requests
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import LabelerBackend

log = logging.getLogger(__name__)


def _pil_to_b64_jpeg(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class RemoteBackend(LabelerBackend):
    """HTTP client for a remote GPU worker running `labeler.worker.server`."""

    name = "remote"

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        url = self.cfg.get("url")
        if not url:
            raise ValueError(
                "remote.url is required (e.g. http://gpu-host:8000). "
                "Set it in config.yaml under `remote:` or pass --worker-url."
            )
        self.url = url.rstrip("/")
        self.auth_token: Optional[str] = self.cfg.get("auth_token") or None
        self.timeout = float(self.cfg.get("request_timeout", 600))
        self.max_retries = int(self.cfg.get("max_retries", 3))
        self.jpeg_quality = int(self.cfg.get("jpeg_quality", 90))

        # Populated lazily on first /info call so `info()` reflects the worker.
        self._remote_info: Optional[dict] = None
        self.model_id = self.cfg.get("model_hint") or "remote"

        log.info("Remote backend: url=%s", self.url)

    # ------------------------------------------------------------------
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.auth_token:
            h["Authorization"] = f"Bearer {self.auth_token}"
        return h

    def _fetch_info(self) -> dict:
        try:
            r = requests.get(
                self.url + "/info", headers=self._headers(), timeout=min(30, self.timeout)
            )
            r.raise_for_status()
            data = r.json() or {}
            self._remote_info = data
            if isinstance(data, dict) and data.get("model"):
                self.model_id = str(data["model"])
            return data
        except Exception as e:
            log.warning("Could not fetch remote worker info from %s/info: %s", self.url, e)
            return {}

    # ------------------------------------------------------------------
    def label(self, prompt: str, frames: List[Image.Image]) -> str:
        payload = {
            "prompt": prompt,
            "frames": [_pil_to_b64_jpeg(img, quality=self.jpeg_quality) for img in frames],
        }

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(
                (requests.ConnectionError, requests.Timeout, requests.HTTPError)
            ),
            reraise=True,
        )
        def _do_call() -> str:
            r = requests.post(
                self.url + "/label",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if r.status_code >= 500:
                r.raise_for_status()
            if r.status_code >= 400:
                # Non-retriable client error — surface the body for debuggability.
                raise RuntimeError(
                    f"Remote worker returned HTTP {r.status_code}: {r.text[:500]}"
                )
            data = r.json() or {}
            return (data.get("text") or "").strip()

        return _do_call()

    # ------------------------------------------------------------------
    def info(self) -> dict:
        if self._remote_info is None:
            self._fetch_info()
        return {
            "backend": self.name,
            "url": self.url,
            "model": self.model_id,
            "worker": self._remote_info or {},
        }

    def close(self) -> None:
        return None
