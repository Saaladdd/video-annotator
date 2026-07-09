"""Egocentric video auto-labeler — CLI entry point.

Default pipeline: sample frames -> (optional) whole-video global context ->
chunked subtask segmentation -> one continuous, de-duplicated subtask timeline
written to a single .txt per video.

Examples:
    # Segment a single video into a subtask timeline (default mode).
    python main.py --input path/to/clip.mp4

    # Sample a 3-min 24fps clip at ~2 fps, 12 frames per chunk, 2 overlap.
    python main.py --input clip.mov --target-fps 2 --chunk-frames 12 --chunk-overlap 2

    # Also export the timeline as CSV/JSON for downstream tooling.
    python main.py --input clip.mp4 --emit-csv --emit-json

    # Use an OpenAI API instead of the local model.
    export OPENAI_API_KEY=sk-...
    python main.py --input clip.mp4 --backend api --api-model gpt-4o

    # Use a local Ollama server via the OpenAI-compatible API.
    python main.py --input clip.mp4 --backend api \\
        --api-provider openai-compatible \\
        --api-base-url http://localhost:11434/v1 \\
        --api-model qwen2.5vl:7b

    # Use a different prompt (edit the file or point to another).
    python main.py --input clip.mp4 --prompt prompts/my_sop.txt
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from labeler.backends import build_backend
from labeler.config import DEFAULT_CONFIG_PATH, load_config, load_prompt_template
from labeler.output import write_label_txt
from labeler.overlays import annotate_frame
from labeler.segments import (
    chunk_ranges,
    format_timeline,
    median_dt,
    merge_segments,
    parse_subtask_lines,
)
from labeler.video import discover_videos, extract_frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Egocentric video auto-labeler (local model or API).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True, help="Video file OR directory of videos.")
    p.add_argument("--config", "-c", default=None, help="Path to config.yaml.")
    p.add_argument("--output-dir", "-o", default=None, help="Directory for .txt outputs.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    p.add_argument("--emit-json", action="store_true", help="Also write a .json sidecar per video.")
    p.add_argument("--emit-csv", action="store_true",
                   help="Also write a .csv of the subtask timeline (subtask-segments mode).")

    # Backend
    p.add_argument(
        "--backend",
        choices=["local", "api", "remote"],
        default=None,
        help="Backend to use. 'remote' talks to a GPU worker we run ourselves "
             "(labeler.worker.server) — videos and outputs stay local.",
    )

    # Local
    p.add_argument("--model", default=None, help="Local HuggingFace model id.")
    p.add_argument("--device", default=None, choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    p.add_argument("--load-in-4bit", action="store_true", help="Load local model in 4-bit.")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)

    # API
    p.add_argument(
        "--api-provider",
        choices=[
            "openai", "anthropic", "openai-compatible",
            "ollama", "lmstudio", "vllm",
        ],
        default=None,
        help="Cloud provider or local server type. ollama/lmstudio/vllm "
             "auto-fill base_url to their default localhost port.",
    )
    p.add_argument("--api-model", default=None, help="Model name at the API provider.")
    p.add_argument("--api-base-url", default=None, help="Override API base URL.")

    # Remote GPU worker
    p.add_argument(
        "--worker-url",
        default=None,
        help="Base URL of the remote GPU worker (used with --backend remote). "
             "e.g. http://gpu-host:8000",
    )
    p.add_argument(
        "--worker-token",
        default=None,
        help="Bearer token for the remote worker (must match its --auth-token). "
             "May also be read from WORKER_AUTH_TOKEN env var.",
    )
    p.add_argument(
        "--worker-timeout",
        type=float,
        default=None,
        help="Per-request timeout in seconds for the remote worker.",
    )

    # Sampling
    p.add_argument("--num-frames", "-n", type=int, default=None,
                   help="Number of frames uniformly sampled. Overrides --target-fps.")
    p.add_argument("--target-fps", type=float, default=None, help="Sample at this rate (fps).")
    p.add_argument("--resize-longest", type=int, default=None, help="Resize longest side (px).")
    p.add_argument("--start-sec", type=float, default=None)
    p.add_argument("--end-sec", type=float, default=None)
    p.add_argument(
        "--label-mode",
        choices=["subtask-segments", "context-window", "per-frame", "multi-frame"],
        default=None,
        help="subtask-segments (default): chunk frames and emit one continuous "
             "subtask timeline. Others: per-frame/context-window annotation, or "
             "all frames in a single call.",
    )
    p.add_argument(
        "--chunk-frames",
        type=int,
        default=None,
        help="subtask-segments: number of sampled frames per chunk sent to the model.",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="subtask-segments: sampled frames shared between consecutive chunks.",
    )
    p.add_argument(
        "--prior-summary-mode",
        choices=["timeline", "model", "off"],
        default=None,
        help="How to summarize previous chunks for the next chunk prompt. "
             "timeline (default): reuse the merged subtask lines. "
             "model: run a small text-only backend call to compress them. "
             "off: do not inject any prior-chunks summary.",
    )
    p.add_argument(
        "--prior-summary-max-lines",
        type=int,
        default=None,
        help="Cap how many prior subtask lines are carried forward into the next chunk.",
    )
    p.add_argument(
        "--prior-summary-prompt",
        default=None,
        help="Path to the prompt used when --prior-summary-mode=model.",
    )
    p.add_argument(
        "--context-before",
        type=int,
        default=None,
        help="When using context-window mode, include this many earlier sampled frames.",
    )
    p.add_argument(
        "--context-after",
        type=int,
        default=None,
        help="When using context-window mode, include this many later sampled frames.",
    )

    # Global context (whole-video understanding)
    p.add_argument(
        "--global-context",
        dest="global_context",
        action="store_true",
        default=None,
        help="Enable the global-context stage (default: on via config).",
    )
    p.add_argument(
        "--no-global-context",
        dest="global_context",
        action="store_false",
        help="Disable the global-context stage.",
    )
    p.add_argument(
        "--global-num-frames",
        type=int,
        default=None,
        help="How many coarse frames to sample across the whole video for global context.",
    )
    p.add_argument(
        "--global-prompt",
        default=None,
        help="Path to the global-snippet prompt used per coarse frame.",
    )

    # Timestamps
    p.add_argument(
        "--overlay-timestamp",
        dest="overlay_timestamp",
        action="store_true",
        default=None,
        help="Burn a timestamp badge onto every frame image sent to the model "
             "(default: on via config). Improves timestamp reasoning.",
    )
    p.add_argument(
        "--no-overlay-timestamp",
        dest="overlay_timestamp",
        action="store_false",
        help="Do not overlay timestamps on frame images.",
    )
    p.add_argument(
        "--overlay-position",
        choices=[
            "extend-bottom", "extend-top",
            "overlay-topleft", "overlay-topright",
            "overlay-bottomleft", "overlay-bottomright",
        ],
        default=None,
        help="Timestamp label placement. extend-* adds a strip outside the frame "
             "so no original pixel is occluded (default: extend-bottom).",
    )

    # Prompt
    p.add_argument("--prompt", default=None, help="Path to a prompt template (overrides config).")
    p.add_argument(
        "--context",
        "--video-context",
        dest="video_context",
        default=None,
        help="One-sentence description of what the video shows (e.g. 'A person "
             "restocks bottles from a shopping basket onto a store shelf'). "
             "Injected into every prompt as authoritative material-flow context. "
             "Massively reduces hallucinations for repetitive tasks.",
    )

    # Post-process reviewer (text-only correction pass on the stitched timeline)
    p.add_argument(
        "--review",
        dest="review",
        action="store_true",
        default=None,
        help="After chunking, run a text-only reviewer pass on the stitched "
             "timeline to fix pick/place direction errors, merge duplicates, "
             "and remove hallucinations (default: on via config for "
             "subtask-segments mode).",
    )
    p.add_argument(
        "--no-review",
        dest="review",
        action="store_false",
        help="Disable the post-process reviewer pass.",
    )
    p.add_argument(
        "--review-prompt",
        default=None,
        help="Path to the reviewer prompt template. Default: prompts/review_timeline.txt.",
    )

    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    ov: dict = {}
    if args.backend:
        ov["backend"] = args.backend

    local: dict = {}
    if args.model: local["model_id"] = args.model
    if args.device: local["device"] = args.device
    if args.dtype: local["dtype"] = args.dtype
    if args.load_in_4bit: local["load_in_4bit"] = True
    if args.max_new_tokens is not None: local["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None: local["temperature"] = args.temperature
    if local: ov["local"] = local

    api: dict = {}
    if args.api_provider: api["provider"] = args.api_provider
    if args.api_model: api["model"] = args.api_model
    if args.api_base_url is not None: api["base_url"] = args.api_base_url
    if args.max_new_tokens is not None: api["max_tokens"] = args.max_new_tokens
    if args.temperature is not None: api["temperature"] = args.temperature
    if api: ov["api"] = api

    remote: dict = {}
    if args.worker_url is not None: remote["url"] = args.worker_url
    if args.worker_token is not None: remote["auth_token"] = args.worker_token
    else:
        env_token = os.environ.get("WORKER_AUTH_TOKEN")
        if env_token:
            remote["auth_token"] = env_token
    if args.worker_timeout is not None: remote["request_timeout"] = args.worker_timeout
    if remote: ov["remote"] = remote

    sampling: dict = {}
    if args.num_frames is not None: sampling["num_frames"] = args.num_frames
    if args.target_fps is not None:
        sampling["target_fps"] = args.target_fps
        if args.num_frames is None:
            sampling["num_frames"] = None
    if args.resize_longest is not None: sampling["resize_longest"] = args.resize_longest
    if args.start_sec is not None: sampling["start_sec"] = args.start_sec
    if args.end_sec is not None: sampling["end_sec"] = args.end_sec
    if sampling: ov["sampling"] = sampling

    if args.label_mode:
        ov["labeling"] = {"mode": args.label_mode.replace("-", "_")}
    if args.context_before is not None or args.context_after is not None:
        ov.setdefault("labeling", {})
        if args.context_before is not None:
            ov["labeling"]["context_frames_before"] = args.context_before
        if args.context_after is not None:
            ov["labeling"]["context_frames_after"] = args.context_after
    if args.chunk_frames is not None or args.chunk_overlap is not None:
        ov.setdefault("labeling", {})
        if args.chunk_frames is not None:
            ov["labeling"]["chunk_frames"] = args.chunk_frames
        if args.chunk_overlap is not None:
            ov["labeling"]["chunk_overlap"] = args.chunk_overlap
    if (
        args.prior_summary_mode is not None
        or args.prior_summary_max_lines is not None
        or args.prior_summary_prompt is not None
    ):
        ov.setdefault("labeling", {})
        ps: dict = ov["labeling"].setdefault("prior_chunks_summary", {})
        if args.prior_summary_mode is not None:
            if args.prior_summary_mode == "off":
                ps["enabled"] = False
            else:
                ps["enabled"] = True
                ps["mode"] = args.prior_summary_mode
        if args.prior_summary_max_lines is not None:
            ps["max_lines"] = args.prior_summary_max_lines
        if args.prior_summary_prompt is not None:
            ps["prompt_path"] = args.prior_summary_prompt

    if args.review is not None or args.review_prompt is not None:
        ov.setdefault("labeling", {})
        rv: dict = ov["labeling"].setdefault("review", {})
        if args.review is not None:
            rv["enabled"] = bool(args.review)
        if args.review_prompt is not None:
            rv["prompt_path"] = args.review_prompt

    gctx: dict = {}
    if args.global_context is not None:
        gctx["enabled"] = bool(args.global_context)
    if args.global_num_frames is not None:
        gctx["num_frames"] = args.global_num_frames
    if args.global_prompt is not None:
        gctx["snippet_prompt_path"] = args.global_prompt
    if gctx: ov["global_context"] = gctx

    if args.overlay_timestamp is not None:
        ov.setdefault("sampling", {})["overlay_timestamp"] = bool(args.overlay_timestamp)
    if args.overlay_position is not None:
        ov.setdefault("sampling", {})["overlay_position"] = args.overlay_position

    if args.prompt: ov["prompt"] = {"path": args.prompt}
    if args.video_context is not None:
        ov.setdefault("prompt", {})["video_context"] = args.video_context
    output: dict = {}
    if args.output_dir: output["dir"] = args.output_dir
    if args.overwrite: output["overwrite"] = True
    if args.emit_json: output["emit_json_sidecar"] = True
    if args.emit_csv: output["emit_csv"] = True
    if output: ov["output"] = output

    return ov


def render_prompt(
    template: str,
    video_path: Path,
    extracted,
    sampled_position: int | None = None,
    frame_index: int | None = None,
    timestamp_sec: float | None = None,
    context_start_position: int | None = None,
    context_end_position: int | None = None,
    context_frame_count: int | None = None,
    context_start_timestamp_sec: float | None = None,
    context_end_timestamp_sec: float | None = None,
    global_summary: str | None = None,
    frames_manifest: str | None = None,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
    chunk_frame_count: int | None = None,
    chunk_start_timestamp_sec: float | None = None,
    chunk_end_timestamp_sec: float | None = None,
    prior_chunks_summary: str | None = None,
    user_context: str | None = None,
) -> str:
    """Fill in prompt placeholders using video-level and optional frame-level metadata."""
    replacements = {
        "video_name": video_path.name,
        "num_frames": len(extracted.frames),
        "duration_sec": f"{extracted.meta.duration_sec:.2f}",
        "fps": f"{extracted.meta.fps:.2f}",
        "sampled_position": "" if sampled_position is None else sampled_position,
        "frame_index": "" if frame_index is None else frame_index,
        "timestamp_sec": "" if timestamp_sec is None else f"{timestamp_sec:.3f}",
        "context_start_position": "" if context_start_position is None else context_start_position,
        "context_end_position": "" if context_end_position is None else context_end_position,
        "context_frame_count": "" if context_frame_count is None else context_frame_count,
        "context_start_timestamp_sec": (
            "" if context_start_timestamp_sec is None else f"{context_start_timestamp_sec:.3f}"
        ),
        "context_end_timestamp_sec": (
            "" if context_end_timestamp_sec is None else f"{context_end_timestamp_sec:.3f}"
        ),
        "global_summary": global_summary or "(global summary not available)",
        "frames_manifest": frames_manifest or "(no frame manifest available)",
        "chunk_index": "" if chunk_index is None else chunk_index,
        "chunk_count": "" if chunk_count is None else chunk_count,
        "chunk_frame_count": "" if chunk_frame_count is None else chunk_frame_count,
        "chunk_start_timestamp_sec": (
            "" if chunk_start_timestamp_sec is None else f"{chunk_start_timestamp_sec:.3f}"
        ),
        "chunk_end_timestamp_sec": (
            "" if chunk_end_timestamp_sec is None else f"{chunk_end_timestamp_sec:.3f}"
        ),
        "prior_chunks_summary": prior_chunks_summary or "(none — this is the first chunk)",
        "user_context": (user_context or "").strip() or "(none provided)",
    }
    out = template
    for k, v in replacements.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _stamp_frame(frame, timestamp_sec, sampled_position, num_total, overlay_enabled, position):
    """Return a possibly-overlayed frame. Never mutate the original.

    With `extend-*` positions, the original video pixels are preserved
    completely — the timestamp is written in a small strip added to the canvas.
    """
    if not overlay_enabled:
        return frame
    text = f"t={timestamp_sec:.3f}s  frame {sampled_position}/{num_total}"
    return annotate_frame(frame, text, position=position)


def _build_manifest(
    positions: list[int],
    frame_indices: list[int],
    timestamps_sec: list[float],
    target_pos: int | None,
) -> str:
    """Build a human/model readable list mapping each image sent to its timestamp."""
    lines = []
    for slot, (pos, idx, ts) in enumerate(zip(positions, frame_indices, timestamps_sec), start=1):
        tag = "  (TARGET)" if target_pos is not None and pos == target_pos else ""
        lines.append(
            f"- image {slot}: sampled_position={pos}  source_frame_index={idx}  "
            f"timestamp_sec={ts:.3f}{tag}"
        )
    return "\n".join(lines)


def _timeline_summary(segments: list[dict], max_lines: int) -> str:
    """Render at most `max_lines` most-recent merged subtask lines as text."""
    if not segments:
        return "(no subtasks annotated yet)"
    tail = segments if max_lines <= 0 else segments[-max_lines:]
    header = ""
    if max_lines > 0 and len(segments) > len(tail):
        header = f"(showing last {len(tail)} of {len(segments)} prior subtasks)\n"
    return header + format_timeline(tail)


def _model_summary(
    backend,
    template: str,
    video_path: Path,
    extracted,
    prior_segments: list[dict],
    max_lines: int,
    chunk_index: int,
    chunk_count: int,
    global_summary: str | None,
    log: logging.Logger,
    user_context: str | None = None,
) -> str:
    """Compress prior subtask lines into a prose paragraph via a text-only call.

    Falls back to the plain timeline text if the backend call fails or returns
    nothing. Passes zero image frames so it's cheap and non-visual.
    """
    timeline_text = _timeline_summary(prior_segments, max_lines)
    prompt = render_prompt(
        template,
        video_path,
        extracted,
        global_summary=global_summary,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        prior_chunks_summary=timeline_text,
        user_context=user_context,
    )
    try:
        text = (backend.label(prompt, []) or "").strip()
        return text or timeline_text
    except Exception as e:
        log.warning("Prior-chunks model summary failed: %s — using timeline text.", e)
        return timeline_text


def review_timeline(
    backend,
    review_template: str,
    raw_segments: list[dict],
    global_summary: str | None,
    duration_sec: float,
    log: logging.Logger,
    user_context: str | None = None,
) -> tuple[list[dict], str] | tuple[None, None]:
    """Run a text-only correction pass on the stitched timeline.

    Sends zero image frames to the backend — the reviewer sees only the
    global summary and the noisy timeline text. Returns
    (reviewed_segments, raw_reviewer_response) on success, or (None, None) if
    the reviewer returned garbage / failed.
    """
    if not raw_segments:
        return None, None
    raw_text = format_timeline(raw_segments)
    prompt = (
        review_template
        .replace("{global_summary}", global_summary or "(global summary not available)")
        .replace("{raw_timeline}", raw_text)
        .replace("{num_entries}", str(len(raw_segments)))
        .replace("{duration_sec:.2f}", f"{duration_sec:.2f}")
        .replace("{duration_sec}", f"{duration_sec:.2f}")
        .replace("{user_context}", (user_context or "").strip() or "(none provided)")
    )
    try:
        raw_response = (backend.label(prompt, []) or "").strip()
    except Exception as e:
        log.warning("Reviewer call failed: %s — keeping raw timeline.", e)
        return None, None
    if not raw_response:
        log.warning("Reviewer returned empty response — keeping raw timeline.")
        return None, None
    reviewed = parse_subtask_lines(raw_response, duration_sec=duration_sec)
    if not reviewed:
        log.warning(
            "Reviewer response had no parseable subtask lines — keeping raw timeline. "
            "Raw reviewer output (first 300 chars): %r",
            raw_response[:300],
        )
        return None, raw_response
    log.info(
        "Reviewer produced %d line(s) (raw had %d).",
        len(reviewed), len(raw_segments),
    )
    return reviewed, raw_response


def _pick_coarse_indices(total: int, k: int) -> list[int]:
    """Return up to k indices uniformly spread across range(total)."""
    if total <= 0 or k <= 0:
        return []
    if k >= total:
        return list(range(total))
    if k == 1:
        return [total // 2]
    step = (total - 1) / (k - 1)
    return [int(round(i * step)) for i in range(k)]


def build_global_summary(
    backend,
    snippet_template: str,
    video_path: Path,
    extracted,
    num_coarse: int,
    overlay_enabled: bool,
    overlay_position: str,
    log: logging.Logger,
    user_context: str | None = None,
) -> tuple[str, list[dict]]:
    """Generate a whole-video timeline by describing coarse frames one at a time.

    Compatible with single-image local vision servers (e.g. Ollama) because each
    coarse frame is sent as its own single-image request.
    Returns (formatted_summary_text, snippets_list).
    """
    n_available = len(extracted.frames)
    coarse_zero_idxs = _pick_coarse_indices(n_available, num_coarse)
    if not coarse_zero_idxs:
        return "(no frames available for global context)", []

    snippets: list[dict] = []
    total = len(coarse_zero_idxs)
    for pos, zero_idx in enumerate(coarse_zero_idxs, start=1):
        frame = extracted.frames[zero_idx]
        frame_index = extracted.indices[zero_idx]
        timestamp_sec = extracted.timestamps_sec[zero_idx]
        stamped = _stamp_frame(frame, timestamp_sec, pos, total, overlay_enabled, overlay_position)
        manifest = _build_manifest([pos], [frame_index], [timestamp_sec], target_pos=pos)
        prompt = render_prompt(
            snippet_template,
            video_path,
            extracted,
            sampled_position=pos,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            frames_manifest=manifest,
            user_context=user_context,
        )
        try:
            text = backend.label(prompt, [stamped]).strip()
        except Exception as e:
            log.warning("Global-context snippet failed at t=%.3fs: %s", timestamp_sec, e)
            text = "(snippet unavailable)"
        snippets.append(
            {
                "position": pos,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "text": text,
            }
        )

    header = (
        f"Coarse timeline of the whole clip ({len(snippets)} sampled points across "
        f"{extracted.meta.duration_sec:.2f}s):"
    )
    body_lines = []
    for s in snippets:
        indented = s["text"].replace("\n", "\n    ")
        body_lines.append(f"- t={s['timestamp_sec']:.3f}s (point {s['position']}):\n    {indented}")
    return header + "\n" + "\n".join(body_lines), snippets


def main() -> int:
    load_dotenv()
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("labeler")

    cfg = load_config(args.config, overrides=build_overrides(args))
    prompt_template = load_prompt_template(cfg)
    user_context = str(cfg.get("prompt", {}).get("video_context") or "").strip() or None
    if user_context:
        log.info("User-provided video context: %s", user_context)

    videos = discover_videos(args.input, cfg["video"]["extensions"])
    if not videos:
        log.error("No videos found at %s", args.input)
        return 2
    log.info("Found %d video(s) to label.", len(videos))

    backend = build_backend(cfg)
    output_dir = cfg["output"]["dir"]
    overwrite = bool(cfg["output"]["overwrite"])
    emit_json = bool(cfg["output"]["emit_json_sidecar"])
    emit_csv = bool(cfg["output"].get("emit_csv", False))
    labeling_cfg = cfg.get("labeling", {})
    label_mode = (labeling_cfg.get("mode") or "subtask_segments").lower()
    context_before = int(labeling_cfg.get("context_frames_before", 1))
    context_after = int(labeling_cfg.get("context_frames_after", 1))
    chunk_frames = int(labeling_cfg.get("chunk_frames", 12))
    chunk_overlap = int(labeling_cfg.get("chunk_overlap", 2))

    prior_cfg = labeling_cfg.get("prior_chunks_summary", {}) or {}
    prior_enabled = bool(prior_cfg.get("enabled", True))
    prior_mode = str(prior_cfg.get("mode", "timeline")).lower()
    prior_max_lines = int(prior_cfg.get("max_lines", 40))
    prior_summary_template: str | None = None
    if prior_enabled and prior_mode == "model":
        p_path = Path(prior_cfg.get("prompt_path") or "prompts/chunk_summary.txt")
        if not p_path.is_absolute():
            p_path = DEFAULT_CONFIG_PATH.parent / p_path
        if p_path.exists():
            prior_summary_template = p_path.read_text(encoding="utf-8")
        else:
            log.warning(
                "Prior-summary prompt not found at %s — falling back to timeline mode.",
                p_path,
            )
            prior_mode = "timeline"

    review_cfg = labeling_cfg.get("review", {}) or {}
    review_enabled = bool(review_cfg.get("enabled", True))
    review_template: str | None = None
    if review_enabled:
        r_path = Path(review_cfg.get("prompt_path") or "prompts/review_timeline.txt")
        if not r_path.is_absolute():
            r_path = DEFAULT_CONFIG_PATH.parent / r_path
        if r_path.exists():
            review_template = r_path.read_text(encoding="utf-8")
        else:
            log.warning(
                "Reviewer prompt not found at %s — disabling review pass.", r_path,
            )
            review_enabled = False

    sampling_cfg = cfg.get("sampling", {}) or {}
    overlay_enabled = bool(sampling_cfg.get("overlay_timestamp", True))
    overlay_position = str(sampling_cfg.get("overlay_position", "extend-bottom"))

    global_cfg = cfg.get("global_context", {}) or {}
    global_enabled = bool(global_cfg.get("enabled", True))
    global_num_frames = int(global_cfg.get("num_frames", 8))
    snippet_template: str | None = None
    if global_enabled:
        snippet_path = Path(global_cfg.get("snippet_prompt_path") or "prompts/global_snippet.txt")
        if not snippet_path.is_absolute():
            snippet_path = DEFAULT_CONFIG_PATH.parent / snippet_path
        if snippet_path.exists():
            snippet_template = snippet_path.read_text(encoding="utf-8")
        else:
            log.warning("Global snippet prompt not found at %s — disabling global context.", snippet_path)
            global_enabled = False

    n_ok, n_skip, n_fail = 0, 0, 0
    try:
        for video_path in tqdm(videos, desc="Labeling", unit="vid"):
            target = Path(output_dir) / f"{video_path.stem}.label.txt"
            if target.exists() and not overwrite:
                log.info("Skipping (exists): %s", target)
                n_skip += 1
                continue
            try:
                extracted = extract_frames(video_path, cfg["sampling"])

                global_summary_text: str | None = None
                global_snippets: list[dict] | None = None
                if global_enabled and snippet_template:
                    log.info("Building global-context timeline for %s ...", video_path.name)
                    global_summary_text, global_snippets = build_global_summary(
                        backend=backend,
                        snippet_template=snippet_template,
                        video_path=video_path,
                        extracted=extracted,
                        num_coarse=global_num_frames,
                        overlay_enabled=overlay_enabled,
                        overlay_position=overlay_position,
                        log=log,
                        user_context=user_context,
                    )

                subtask_timeline: list[dict] | None = None
                raw_subtask_timeline: list[dict] | None = None
                reviewer_info: dict | None = None
                chunk_outputs: list[dict] | None = None
                prompt = None
                label_text = None
                frame_outputs = None

                if label_mode == "subtask_segments":
                    total_sampled = len(extracted.frames)
                    ranges = chunk_ranges(total_sampled, chunk_frames, chunk_overlap)
                    chunk_outputs = []
                    all_segments: list[dict] = []
                    gap_tol = median_dt(extracted.timestamps_sec) * 1.5
                    merged_so_far: list[dict] = []
                    for ci, (lo, hi) in enumerate(ranges, start=1):
                        positions = list(range(lo + 1, hi + 1))
                        frame_indices = extracted.indices[lo:hi]
                        timestamps = extracted.timestamps_sec[lo:hi]
                        chunk_frames_raw = extracted.frames[lo:hi]
                        chunk_start_ts = timestamps[0]
                        chunk_end_ts = timestamps[-1]
                        stamped = [
                            _stamp_frame(f, ts, pos, total_sampled, overlay_enabled, overlay_position)
                            for f, ts, pos in zip(chunk_frames_raw, timestamps, positions)
                        ]
                        manifest = _build_manifest(
                            positions, frame_indices, timestamps, target_pos=None,
                        )

                        prior_summary_text: str | None = None
                        if prior_enabled:
                            if ci == 1:
                                prior_summary_text = "(none — this is the first chunk)"
                            elif prior_mode == "model" and prior_summary_template:
                                prior_summary_text = _model_summary(
                                    backend=backend,
                                    template=prior_summary_template,
                                    video_path=video_path,
                                    extracted=extracted,
                                    prior_segments=merged_so_far,
                                    max_lines=prior_max_lines,
                                    chunk_index=ci,
                                    chunk_count=len(ranges),
                                    global_summary=global_summary_text,
                                    log=log,
                                    user_context=user_context,
                                )
                            else:
                                prior_summary_text = _timeline_summary(
                                    merged_so_far, prior_max_lines
                                )

                        chunk_prompt = render_prompt(
                            prompt_template,
                            video_path,
                            extracted,
                            global_summary=global_summary_text,
                            frames_manifest=manifest,
                            chunk_index=ci,
                            chunk_count=len(ranges),
                            chunk_frame_count=len(chunk_frames_raw),
                            chunk_start_timestamp_sec=chunk_start_ts,
                            chunk_end_timestamp_sec=chunk_end_ts,
                            prior_chunks_summary=prior_summary_text,
                            user_context=user_context,
                        )
                        raw = backend.label(chunk_prompt, stamped)
                        segs = parse_subtask_lines(raw, duration_sec=extracted.meta.duration_sec)
                        chunk_outputs.append(
                            {
                                "chunk_index": ci,
                                "chunk_count": len(ranges),
                                "start_timestamp_sec": chunk_start_ts,
                                "end_timestamp_sec": chunk_end_ts,
                                "frame_count": len(chunk_frames_raw),
                                "prompt": chunk_prompt,
                                "raw": raw,
                                "segments": segs,
                                "prior_summary": prior_summary_text,
                            }
                        )
                        all_segments.extend(segs)
                        merged_so_far = merge_segments(all_segments, gap_tol=gap_tol)
                        log.info(
                            "Chunk %d/%d (t=%.3f-%.3fs): %d subtask line(s), "
                            "rolling timeline=%d entries",
                            ci, len(ranges), chunk_start_ts, chunk_end_ts,
                            len(segs), len(merged_so_far),
                        )
                    subtask_timeline = merged_so_far
                    log.info(
                        "Merged %d raw subtask line(s) into %d timeline entries.",
                        len(all_segments), len(subtask_timeline),
                    )

                    if review_enabled and review_template and subtask_timeline:
                        log.info(
                            "Running reviewer pass over %d stitched entries ...",
                            len(subtask_timeline),
                        )
                        reviewed, reviewer_raw = review_timeline(
                            backend=backend,
                            review_template=review_template,
                            raw_segments=subtask_timeline,
                            global_summary=global_summary_text,
                            duration_sec=extracted.meta.duration_sec,
                            log=log,
                            user_context=user_context,
                        )
                        if reviewed:
                            raw_subtask_timeline = subtask_timeline
                            subtask_timeline = reviewed
                            reviewer_info = {
                                "applied": True,
                                "raw_count": len(raw_subtask_timeline),
                                "reviewed_count": len(subtask_timeline),
                                "reviewer_raw": reviewer_raw or "",
                            }
                        else:
                            raw_subtask_timeline = None
                            reviewer_info = {
                                "applied": False,
                                "reviewer_raw": reviewer_raw or "",
                            }
                    else:
                        raw_subtask_timeline = None
                        reviewer_info = None

                elif label_mode in ("per_frame", "context_window"):
                    frame_outputs = []
                    total_sampled = len(extracted.frames)
                    for zero_idx, (frame, frame_index, timestamp_sec) in enumerate(
                        zip(extracted.frames, extracted.indices, extracted.timestamps_sec),
                    ):
                        sampled_position = zero_idx + 1
                        if label_mode == "context_window":
                            ctx_lo = max(0, zero_idx - context_before)
                            ctx_hi = min(len(extracted.frames), zero_idx + context_after + 1)
                            ctx_positions = list(range(ctx_lo + 1, ctx_hi + 1))
                            ctx_frame_indices = extracted.indices[ctx_lo:ctx_hi]
                            ctx_timestamps = extracted.timestamps_sec[ctx_lo:ctx_hi]
                            context_frames_raw = extracted.frames[ctx_lo:ctx_hi]
                            context_start_position = ctx_lo + 1
                            context_end_position = ctx_hi
                            context_start_timestamp_sec = extracted.timestamps_sec[ctx_lo]
                            context_end_timestamp_sec = extracted.timestamps_sec[ctx_hi - 1]
                        else:
                            ctx_positions = [sampled_position]
                            ctx_frame_indices = [frame_index]
                            ctx_timestamps = [timestamp_sec]
                            context_frames_raw = [frame]
                            context_start_position = sampled_position
                            context_end_position = sampled_position
                            context_start_timestamp_sec = timestamp_sec
                            context_end_timestamp_sec = timestamp_sec

                        context_frames = [
                            _stamp_frame(f, ts, pos, total_sampled, overlay_enabled, overlay_position)
                            for f, ts, pos in zip(context_frames_raw, ctx_timestamps, ctx_positions)
                        ]
                        frames_manifest = _build_manifest(
                            ctx_positions, ctx_frame_indices, ctx_timestamps,
                            target_pos=sampled_position,
                        )

                        prompt = render_prompt(
                            prompt_template,
                            video_path,
                            extracted,
                            sampled_position=sampled_position,
                            frame_index=frame_index,
                            timestamp_sec=timestamp_sec,
                            context_start_position=context_start_position,
                            context_end_position=context_end_position,
                            context_frame_count=len(context_frames),
                            context_start_timestamp_sec=context_start_timestamp_sec,
                            context_end_timestamp_sec=context_end_timestamp_sec,
                            global_summary=global_summary_text,
                            frames_manifest=frames_manifest,
                            user_context=user_context,
                        )
                        label_text = backend.label(prompt, context_frames)
                        frame_outputs.append(
                            {
                                "sampled_position": sampled_position,
                                "frame_index": frame_index,
                                "timestamp_sec": timestamp_sec,
                                "context_start_position": context_start_position,
                                "context_end_position": context_end_position,
                                "context_frame_count": len(context_frames),
                                "context_start_timestamp_sec": context_start_timestamp_sec,
                                "context_end_timestamp_sec": context_end_timestamp_sec,
                                "prompt": prompt,
                                "label": label_text,
                            }
                        )
                    prompt = None
                    label_text = None
                else:
                    total_sampled = len(extracted.frames)
                    all_positions = list(range(1, total_sampled + 1))
                    frames_manifest = _build_manifest(
                        all_positions, extracted.indices, extracted.timestamps_sec,
                        target_pos=None,
                    )
                    stamped_all = [
                        _stamp_frame(f, ts, pos, total_sampled, overlay_enabled, overlay_position)
                        for f, ts, pos in zip(
                            extracted.frames, extracted.timestamps_sec, all_positions,
                        )
                    ]
                    prompt = render_prompt(
                        prompt_template,
                        video_path,
                        extracted,
                        global_summary=global_summary_text,
                        frames_manifest=frames_manifest,
                        user_context=user_context,
                    )
                    label_text = backend.label(prompt, stamped_all)
                    frame_outputs = None
                out_path = write_label_txt(
                    output_dir=output_dir,
                    video_path=video_path,
                    extracted=extracted,
                    prompt_used=prompt,
                    label_text=label_text,
                    frame_outputs=frame_outputs,
                    backend_info=backend.info(),
                    label_mode=label_mode,
                    global_summary_text=global_summary_text,
                    global_snippets=global_snippets,
                    user_context=user_context,
                    subtask_timeline=subtask_timeline,
                    raw_subtask_timeline=raw_subtask_timeline,
                    reviewer_info=reviewer_info,
                    chunk_outputs=chunk_outputs,
                    overwrite=overwrite,
                    emit_json_sidecar=emit_json,
                    emit_csv=emit_csv,
                )
                log.info("Wrote %s", out_path)
                n_ok += 1
            except Exception as e:
                log.exception("Failed to label %s: %s", video_path, e)
                n_fail += 1
    finally:
        backend.close()

    log.info("Done. ok=%d skipped=%d failed=%d", n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
