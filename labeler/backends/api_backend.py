"""API backend supporting cloud APIs and local vision-model servers.

Supported providers:
  - "openai"            OpenAI cloud (needs OPENAI_API_KEY)
  - "anthropic"         Anthropic cloud (needs ANTHROPIC_API_KEY)
  - "openai-compatible" any OpenAI-schema server (custom base_url)
  - "ollama"            local Ollama server (auto base_url http://localhost:11434/v1)
  - "lmstudio"          local LM Studio server (auto base_url http://localhost:1234/v1)
  - "vllm"              local vLLM server   (auto base_url http://localhost:8000/v1)

`ollama`, `lmstudio`, and `vllm` are convenience aliases for
`openai-compatible` with sensible localhost defaults — the app is otherwise
provider-agnostic.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from typing import List, Optional

from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import LabelerBackend

log = logging.getLogger(__name__)


def _pil_to_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def _pil_to_b64(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# Localhost providers that speak the OpenAI chat/completions schema.
# Users can still pass --api-base-url to override any of these.
LOCAL_SERVER_DEFAULTS = {
    "ollama":   "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "vllm":     "http://localhost:8000/v1",
}


class ApiBackend(LabelerBackend):
    name = "api"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.provider = (cfg.get("provider") or "openai").lower()
        self.model_id = cfg["model"]
        self.base_url = cfg.get("base_url") or None

        # Apply localhost defaults for convenience providers.
        if self.provider in LOCAL_SERVER_DEFAULTS and not self.base_url:
            self.base_url = LOCAL_SERVER_DEFAULTS[self.provider]

        self.max_tokens = int(cfg.get("max_tokens", 1024))
        self.temperature = float(cfg.get("temperature", 0.2))
        self.timeout = float(cfg.get("request_timeout", 120))
        self.max_retries = int(cfg.get("max_retries", 3))

        env_var = cfg.get("api_key_env") or (
            "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"
        )
        self.api_key = os.environ.get(env_var)
        # Local servers don't need a real key.
        needs_key = self.provider in ("openai", "anthropic")
        if not self.api_key and needs_key:
            log.warning(
                "Environment variable %s is not set. API calls will likely fail.", env_var
            )

        self._client = None

    # ------------------------------------------------------------------
    def _uses_openai_schema(self) -> bool:
        return self.provider in ("openai", "openai-compatible", *LOCAL_SERVER_DEFAULTS.keys())

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._uses_openai_schema():
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key or "not-required",
                base_url=self.base_url,
                timeout=self.timeout,
            )
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key, timeout=self.timeout)
        else:
            raise ValueError(f"Unknown provider: {self.provider!r}")
        return self._client

    # ------------------------------------------------------------------
    def label(self, prompt: str, frames: List[Image.Image]) -> str:
        if self.provider == "anthropic":
            return self._label_anthropic(prompt, frames)
        return self._label_openai(prompt, frames)

    # ------------------------------------------------------------------
    def _label_openai(self, prompt: str, frames: List[Image.Image]) -> str:
        client = self._get_client()

        content = []
        for i, img in enumerate(frames):
            content.append(
                {"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}}
            )
        content.append({"type": "text", "text": prompt})

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _do_call():
            resp = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return resp.choices[0].message.content or ""

        return _do_call().strip()

    # ------------------------------------------------------------------
    def _label_anthropic(self, prompt: str, frames: List[Image.Image]) -> str:
        client = self._get_client()

        content = []
        for img in frames:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": _pil_to_b64(img),
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _do_call():
            resp = client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": content}],
            )
            parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            return "\n".join(parts)

        return _do_call().strip()
