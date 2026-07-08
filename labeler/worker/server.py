"""GPU worker HTTP server.

Wraps the existing `LocalBackend` (unchanged) behind a small FastAPI app so a
remote client can send `(prompt, frames)` and receive the model's text output.

Run it on the cloud GPU box:

    # Install GPU deps once (torch + transformers + this repo).
    pip install -r requirements.txt
    pip install -r requirements-worker.txt

    # Then start the worker with the same knobs LocalBackend already accepts.
    python -m labeler.worker.server \\
        --host 0.0.0.0 --port 8000 \\
        --model Qwen/Qwen2.5-VL-3B-Instruct --load-in-4bit

    # (Optional) require a bearer token from clients.
    python -m labeler.worker.server --auth-token $(openssl rand -hex 16)

The client (labeler) then runs with:

    python main.py --input clip.mp4 \\
        --backend remote --worker-url http://<gpu-host>:8000 \\
        [--worker-token <token>]

Storage (videos) and outputs stay on the client. Only prompt text and
base64-JPEG frames traverse the network; only the label text comes back.
"""
import argparse
import base64
import io
import logging
import os
import sys
from typing import List, Optional

from PIL import Image
from pydantic import BaseModel

from labeler.backends.local_backend import LocalBackend

log = logging.getLogger("labeler.worker")

# Global backend instance (loaded once at startup, reused across requests).
_backend: Optional[LocalBackend] = None
_auth_token: Optional[str] = None


class LabelRequest(BaseModel):
    prompt: str
    frames: List[str] = []


class LabelResponse(BaseModel):
    text: str


def _decode_frame(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _require_auth(authorization_header: Optional[str]) -> None:
    if not _auth_token:
        return
    expected = f"Bearer {_auth_token}"
    if authorization_header != expected:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")


def _build_app():
    """Import FastAPI lazily so the module loads even without it installed."""
    try:
        from fastapi import Body, FastAPI, Header, HTTPException
    except ImportError as e:
        raise RuntimeError(
            "The worker requires fastapi + pydantic + uvicorn. "
            "Install them with `pip install -r requirements-worker.txt`."
        ) from e

    app = FastAPI(title="Auto-Labeler GPU Worker", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/info")
    def info(authorization: Optional[str] = Header(default=None)):
        _require_auth(authorization)
        if _backend is None:
            raise HTTPException(status_code=503, detail="Backend not initialised.")
        return _backend.info() | {"backend": "worker(local)"}

    @app.post("/label", response_model=LabelResponse)
    def label(
        req: LabelRequest = Body(...),
        authorization: Optional[str] = Header(default=None),
    ):
        _require_auth(authorization)
        if _backend is None:
            raise HTTPException(status_code=503, detail="Backend not initialised.")
        try:
            frames = [_decode_frame(b) for b in req.frames]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode frames: {e}")
        try:
            text = _backend.label(req.prompt, frames)
        except Exception as e:
            log.exception("Model call failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Model call failed: {e}")
        return LabelResponse(text=text)

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-labeler GPU worker (HTTP wrapper around LocalBackend).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host.")
    p.add_argument("--port", type=int, default=8000, help="Bind port.")
    p.add_argument("--auth-token", default=None,
                   help="Optional shared secret. If set, clients must send "
                        "'Authorization: Bearer <token>'. May also be provided "
                        "via WORKER_AUTH_TOKEN env var.")

    # These mirror LocalBackend config keys 1:1.
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct",
                   help="HuggingFace model id to load on the GPU.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--load-in-4bit", action="store_true", help="Load in 4-bit.")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-pixels", type=int, default=None,
                   help="Max pixels per image for the processor (VRAM control).")

    p.add_argument("--preload", action="store_true",
                   help="Load the model weights at startup instead of lazily on "
                        "the first request.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _local_cfg_from_args(args: argparse.Namespace) -> dict:
    cfg = {
        "model_id": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "load_in_4bit": bool(args.load_in_4bit),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    if args.max_pixels is not None:
        cfg["max_pixels"] = int(args.max_pixels)
    return cfg


def main() -> int:
    global _backend, _auth_token

    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _auth_token = args.auth_token or os.environ.get("WORKER_AUTH_TOKEN") or None
    if _auth_token:
        log.info("Bearer-token auth enabled.")
    else:
        log.warning("No auth token set — worker is open to any caller that can "
                    "reach the port. Bind to a private network or set --auth-token.")

    _backend = LocalBackend(_local_cfg_from_args(args))
    if args.preload:
        log.info("Preloading model %s ...", args.model)
        # LocalBackend loads lazily on the first .label() call; force it now by
        # doing a no-op internal load.
        _backend._load()  # pylint: disable=protected-access
        log.info("Model preloaded.")

    app = _build_app()

    try:
        import uvicorn
    except ImportError as e:
        log.error("uvicorn is required to run the worker. "
                  "Install with `pip install -r requirements-worker.txt`.")
        raise SystemExit(1) from e

    log.info("Starting worker on %s:%d (model=%s)", args.host, args.port, args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
