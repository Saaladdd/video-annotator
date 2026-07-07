"""Video frame extraction with multi-decoder fallback.

Decoder priority: decord -> pyav -> opencv -> imageio.
This maximises the range of container/codec combinations we can read.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int
    decoder: str = "unknown"


@dataclass
class ExtractedFrames:
    frames: List[Image.Image]
    indices: List[int]
    timestamps_sec: List[float]
    meta: VideoMeta


def _resize_pil(img: Image.Image, longest: Optional[int]) -> Image.Image:
    if not longest:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= longest:
        return img
    scale = longest / float(m)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)


def _pick_indices(
    frame_count: int,
    fps: float,
    duration_sec: float,
    num_frames: Optional[int],
    target_fps: Optional[float],
    start_sec: Optional[float],
    end_sec: Optional[float],
) -> List[int]:
    """Return sorted list of frame indices to sample."""
    if frame_count <= 0:
        return []

    lo = 0 if not start_sec else max(0, int(math.floor(start_sec * fps)))
    hi = frame_count - 1 if not end_sec else min(frame_count - 1, int(math.ceil(end_sec * fps)))
    if hi < lo:
        hi = lo

    total = hi - lo + 1

    if num_frames and num_frames > 0:
        n = min(num_frames, total)
        if n == 1:
            return [lo + total // 2]
        step = (total - 1) / (n - 1)
        return [int(round(lo + i * step)) for i in range(n)]

    if target_fps and target_fps > 0:
        stride = max(1, int(round(fps / target_fps)))
        return list(range(lo, hi + 1, stride))

    # Default: 8 frames.
    n = min(8, total)
    if n == 1:
        return [lo + total // 2]
    step = (total - 1) / (n - 1)
    return [int(round(lo + i * step)) for i in range(n)]


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

def _probe_and_extract_decord(path: Path, sampling: dict) -> Optional[ExtractedFrames]:
    try:
        import decord  # type: ignore
    except Exception:
        return None
    try:
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(path), num_threads=1)
        fps = float(vr.get_avg_fps()) or 30.0
        frame_count = len(vr)
        duration = frame_count / fps if fps else 0.0
        h, w, _ = vr[0].shape
        meta = VideoMeta(path, fps, frame_count, duration, int(w), int(h), decoder="decord")

        idxs = _pick_indices(
            frame_count, fps, duration,
            sampling.get("num_frames"), sampling.get("target_fps"),
            sampling.get("start_sec"), sampling.get("end_sec"),
        )
        if not idxs:
            return None
        batch = vr.get_batch(idxs).asnumpy()  # (N, H, W, 3) uint8
        frames = [Image.fromarray(f) for f in batch]
        frames = [_resize_pil(f, sampling.get("resize_longest")) for f in frames]
        timestamps = [i / fps for i in idxs]
        return ExtractedFrames(frames, idxs, timestamps, meta)
    except Exception as e:
        log.debug("decord failed on %s: %s", path, e)
        return None


def _probe_and_extract_pyav(path: Path, sampling: dict) -> Optional[ExtractedFrames]:
    try:
        import av  # type: ignore
    except Exception:
        return None
    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        fps = float(stream.average_rate) if stream.average_rate else 30.0
        frame_count = stream.frames or 0
        duration = float(container.duration or 0) / 1_000_000.0
        if not frame_count and duration and fps:
            frame_count = int(duration * fps)
        if not duration and frame_count and fps:
            duration = frame_count / fps

        w = int(stream.codec_context.width)
        h = int(stream.codec_context.height)
        meta = VideoMeta(path, fps, frame_count, duration, w, h, decoder="pyav")

        idxs = _pick_indices(
            frame_count or 1, fps, duration,
            sampling.get("num_frames"), sampling.get("target_fps"),
            sampling.get("start_sec"), sampling.get("end_sec"),
        )
        if not idxs:
            container.close()
            return None

        wanted = set(idxs)
        picked: dict[int, Image.Image] = {}
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                img = frame.to_image()
                picked[i] = _resize_pil(img, sampling.get("resize_longest"))
                if len(picked) == len(wanted):
                    break
        container.close()

        ordered = sorted(picked.keys())
        frames = [picked[i] for i in ordered]
        timestamps = [i / fps for i in ordered]
        return ExtractedFrames(frames, ordered, timestamps, meta)
    except Exception as e:
        log.debug("pyav failed on %s: %s", path, e)
        return None


def _probe_and_extract_opencv(path: Path, sampling: dict) -> Optional[ExtractedFrames]:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps else 0.0
        meta = VideoMeta(path, fps, frame_count, duration, w, h, decoder="opencv")

        idxs = _pick_indices(
            frame_count, fps, duration,
            sampling.get("num_frames"), sampling.get("target_fps"),
            sampling.get("start_sec"), sampling.get("end_sec"),
        )
        if not idxs:
            return None

        frames: list[Image.Image] = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            frames.append(_resize_pil(img, sampling.get("resize_longest")))
        if not frames:
            return None
        timestamps = [i / fps for i in idxs[: len(frames)]]
        return ExtractedFrames(frames, idxs[: len(frames)], timestamps, meta)
    except Exception as e:
        log.debug("opencv failed on %s: %s", path, e)
        return None
    finally:
        cap.release()


def _probe_and_extract_imageio(path: Path, sampling: dict) -> Optional[ExtractedFrames]:
    try:
        import imageio.v3 as iio  # type: ignore
    except Exception:
        return None
    try:
        meta_dict = iio.immeta(str(path), plugin="pyav")
        fps = float(meta_dict.get("fps") or 30.0)
        duration = float(meta_dict.get("duration") or 0.0)
        # imageio cannot cheaply give frame_count; approximate.
        frame_count = int(duration * fps) if (duration and fps) else 0

        all_frames = []
        for i, frame in enumerate(iio.imiter(str(path), plugin="pyav")):
            all_frames.append(frame)
        if not all_frames:
            return None
        frame_count = len(all_frames)
        h, w, _ = all_frames[0].shape
        duration = frame_count / fps if fps else 0.0
        meta = VideoMeta(path, fps, frame_count, duration, int(w), int(h), decoder="imageio")

        idxs = _pick_indices(
            frame_count, fps, duration,
            sampling.get("num_frames"), sampling.get("target_fps"),
            sampling.get("start_sec"), sampling.get("end_sec"),
        )
        frames = [_resize_pil(Image.fromarray(all_frames[i]), sampling.get("resize_longest")) for i in idxs]
        timestamps = [i / fps for i in idxs]
        return ExtractedFrames(frames, idxs, timestamps, meta)
    except Exception as e:
        log.debug("imageio failed on %s: %s", path, e)
        return None


def extract_frames(path: str | Path, sampling: dict) -> ExtractedFrames:
    """Extract frames from a video, trying multiple decoders in order."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {p}")

    for decoder in (
        _probe_and_extract_decord,
        _probe_and_extract_pyav,
        _probe_and_extract_opencv,
        _probe_and_extract_imageio,
    ):
        result = decoder(p, sampling)
        if result and result.frames:
            log.info(
                "Decoded %s with %s (%d frames sampled, %.2fs, %.2f fps)",
                p.name, result.meta.decoder, len(result.frames),
                result.meta.duration_sec, result.meta.fps,
            )
            return result

    raise RuntimeError(
        f"Could not decode {p}. Tried decord, pyav, opencv, imageio. "
        "Install ffmpeg or check that the file is a valid video."
    )


def discover_videos(input_path: str | Path, extensions: List[str]) -> List[Path]:
    """Return a sorted list of video files. Accepts a file or directory."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"Input path not found: {p}")

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    videos = [q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in exts]
    return sorted(videos)
