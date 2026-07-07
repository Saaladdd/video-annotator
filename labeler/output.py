"""Text output writers."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from .segments import format_timeline
from .video import ExtractedFrames


def _fmt_timestamps(extracted: ExtractedFrames) -> str:
    lines = []
    for idx, ts in zip(extracted.indices, extracted.timestamps_sec):
        lines.append(f"  - frame_idx={idx}  t={ts:.3f}s")
    return "\n".join(lines)


def _fmt_frame_outputs(frame_outputs: list[dict]) -> str:
    lines = []
    for item in frame_outputs:
        lines.extend(
            [
                (
                    f"=== FRAME {item['sampled_position']} / "
                    f"{len(frame_outputs)} | source_idx={item['frame_index']} "
                    f"| t={item['timestamp_sec']:.3f}s ==="
                ),
                (
                    f"context_window: sampled {item['context_start_position']}-"
                    f"{item['context_end_position']} | frames={item['context_frame_count']} "
                    f"| t={item['context_start_timestamp_sec']:.3f}s-"
                    f"{item['context_end_timestamp_sec']:.3f}s"
                ),
                "",
                "----- PROMPT -----",
                item["prompt"].rstrip(),
                "",
                "----- LABEL OUTPUT -----",
                item["label"].rstrip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def _fmt_global_snippets(snippets: list[dict]) -> str:
    lines = []
    for s in snippets:
        indented = s["text"].replace("\n", "\n    ")
        lines.append(
            f"- point {s['position']} @ t={s['timestamp_sec']:.3f}s "
            f"(source_idx={s['frame_index']}):\n    {indented}"
        )
    return "\n".join(lines)


def _fmt_chunk_outputs(chunk_outputs: list[dict]) -> str:
    lines = []
    for c in chunk_outputs:
        lines.extend(
            [
                (
                    f"=== CHUNK {c['chunk_index']} / {c['chunk_count']} "
                    f"| frames={c['frame_count']} "
                    f"| t={c['start_timestamp_sec']:.3f}s-{c['end_timestamp_sec']:.3f}s "
                    f"| parsed_lines={len(c['segments'])} ==="
                ),
                "",
                "----- RAW MODEL OUTPUT -----",
                (c["raw"] or "").rstrip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def _segments_csv(segments: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["start_sec", "end_sec", "subtask"])
    for s in segments:
        writer.writerow([f"{s['start']:.3f}", f"{s['end']:.3f}", s["sentence"]])
    return buf.getvalue()


def write_label_txt(
    output_dir: str | Path,
    video_path: Path,
    extracted: ExtractedFrames,
    prompt_used: str | None,
    label_text: str | None,
    frame_outputs: list[dict] | None,
    backend_info: dict,
    label_mode: str,
    global_summary_text: str | None = None,
    global_snippets: list[dict] | None = None,
    subtask_timeline: list[dict] | None = None,
    chunk_outputs: list[dict] | None = None,
    overwrite: bool = False,
    emit_json_sidecar: bool = False,
    emit_csv: bool = False,
) -> Path:
    """Write the labeler output to a .txt file. Returns the path written."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    txt_path = out_dir / f"{stem}.label.txt"
    if txt_path.exists() and not overwrite:
        return txt_path

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header = [
        "===== EGOCENTRIC AUTO-LABEL =====",
        f"video:          {video_path}",
        f"generated_at:   {now}",
        f"decoder:        {extracted.meta.decoder}",
        f"video_fps:      {extracted.meta.fps:.3f}",
        f"video_frames:   {extracted.meta.frame_count}",
        f"video_duration: {extracted.meta.duration_sec:.3f}s",
        f"video_size:     {extracted.meta.width}x{extracted.meta.height}",
        f"sampled_frames: {len(extracted.frames)}",
        _fmt_timestamps(extracted),
        f"backend:        {backend_info.get('backend')}",
        f"model:          {backend_info.get('model')}",
        f"label_mode:     {label_mode}",
        f"global_context: {'yes' if global_snippets else 'no'}"
        + (f" ({len(global_snippets)} points)" if global_snippets else ""),
        "",
    ]

    body: list[str] = []

    if global_snippets:
        body.extend(
            [
                "----- GLOBAL VIDEO CONTEXT (whole-clip timeline) -----",
                _fmt_global_snippets(global_snippets),
                "",
            ]
        )

    if subtask_timeline is not None:
        # Primary deliverable: the continuous subtask timeline for the whole video.
        body.extend(
            [
                "===== SUBTASK TIMELINE (whole video) =====",
                format_timeline(subtask_timeline),
                "",
            ]
        )
        if chunk_outputs:
            body.extend(
                [
                    f"----- RAW CHUNK OUTPUTS ({len(chunk_outputs)} chunk(s), for audit) -----",
                    _fmt_chunk_outputs(chunk_outputs),
                    "",
                ]
            )
    elif frame_outputs:
        body.extend(
            [
                "----- FRAME OUTPUTS -----",
                _fmt_frame_outputs(frame_outputs),
                "",
            ]
        )
    else:
        body.extend(
            [
                "----- PROMPT -----",
                (prompt_used or "").rstrip(),
                "",
                "----- LABEL OUTPUT -----",
                (label_text or "").rstrip(),
                "",
            ]
        )

    txt_path.write_text("\n".join(header + body), encoding="utf-8")

    if emit_csv and subtask_timeline is not None:
        (out_dir / f"{stem}.timeline.csv").write_text(
            _segments_csv(subtask_timeline), encoding="utf-8"
        )

    if emit_json_sidecar:
        side = {
            "video": str(video_path),
            "generated_at": now,
            "decoder": extracted.meta.decoder,
            "video_fps": extracted.meta.fps,
            "video_frame_count": extracted.meta.frame_count,
            "video_duration_sec": extracted.meta.duration_sec,
            "video_width": extracted.meta.width,
            "video_height": extracted.meta.height,
            "sampled_indices": extracted.indices,
            "sampled_timestamps_sec": extracted.timestamps_sec,
            "backend": backend_info,
            "label_mode": label_mode,
            "global_summary_text": global_summary_text,
            "global_snippets": global_snippets,
            "subtask_timeline": subtask_timeline,
            "chunk_outputs": chunk_outputs,
            "label": label_text,
            "frame_outputs": frame_outputs,
        }
        (out_dir / f"{stem}.label.json").write_text(
            json.dumps(side, indent=2), encoding="utf-8"
        )

    return txt_path
