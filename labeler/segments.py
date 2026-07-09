"""Subtask segmentation helpers: chunking, parsing, and merging.

The subtask SOP produces a continuous timeline of instructional sentences of the
form:

    [<start_sec>-<end_sec>] <Action + Object + Location/Context + Hand>.

To keep requests small (and compatible with local vision servers), the video's
sampled frames are processed in overlapping chunks. Each chunk yields subtask
lines that are parsed and then merged into one continuous, de-duplicated
timeline for the whole video.
"""
from __future__ import annotations

import re
from statistics import median

# Accepts hyphen, en-dash, or em-dash between the two timestamps.
_SEG_RE = re.compile(
    r"^\s*\[\s*(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*\]\s*(.+?)\s*$"
)


def chunk_ranges(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Split range(n) into [lo, hi) windows of `size` frames sharing `overlap`."""
    if n <= 0:
        return []
    size = max(1, size)
    overlap = max(0, min(overlap, size - 1))
    step = size - overlap
    ranges: list[tuple[int, int]] = []
    i = 0
    while i < n:
        hi = min(n, i + size)
        ranges.append((i, hi))
        if hi >= n:
            break
        i += step
    return ranges


def median_dt(timestamps_sec: list[float]) -> float:
    """Median spacing between consecutive sampled timestamps (0 if < 2 points)."""
    if not timestamps_sec or len(timestamps_sec) < 2:
        return 0.0
    diffs = [
        b - a
        for a, b in zip(timestamps_sec, timestamps_sec[1:])
        if b - a > 0
    ]
    return float(median(diffs)) if diffs else 0.0


def parse_subtask_lines(text: str, duration_sec: float | None = None) -> list[dict]:
    """Parse '[start-end] sentence.' lines into structured segments."""
    segs: list[dict] = []
    for line in (text or "").splitlines():
        m = _SEG_RE.match(line)
        if not m:
            continue
        start = float(m.group(1))
        end = float(m.group(2))
        sentence = m.group(3).strip()
        if not sentence:
            continue
        if end < start:
            start, end = end, start
        if duration_sec is not None and duration_sec > 0:
            start = max(0.0, min(start, duration_sec))
            end = max(0.0, min(end, duration_sec))
        segs.append({"start": start, "end": end, "sentence": sentence})
    return segs


def _normalize(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence.strip().rstrip(".").lower())


def merge_segments(segments: list[dict], gap_tol: float = 0.0) -> list[dict]:
    """Sort by time and merge consecutive duplicates (same sentence, contiguous).

    `gap_tol` allows two identical-sentence segments to merge even if there is a
    small gap between them (typically one sampling interval), which removes the
    duplication introduced by overlapping chunks.
    """
    ordered = sorted(segments, key=lambda s: (s["start"], s["end"]))
    merged: list[dict] = []
    for seg in ordered:
        if merged:
            prev = merged[-1]
            same = _normalize(seg["sentence"]) == _normalize(prev["sentence"])
            contiguous = seg["start"] <= prev["end"] + gap_tol
            if same and contiguous:
                prev["end"] = max(prev["end"], seg["end"])
                continue
        merged.append(dict(seg))
    return merged


def segment_line_chunks(
    segments: list[dict],
    max_lines: int,
    overlap: int,
) -> list[list[dict]]:
    """Split a timeline into overlapping line windows for chunked text review."""
    if not segments or max_lines <= 0 or len(segments) <= max_lines:
        return [segments] if segments else []
    overlap = max(0, min(overlap, max_lines - 1))
    step = max(1, max_lines - overlap)
    chunks: list[list[dict]] = []
    i = 0
    while i < len(segments):
        chunk = segments[i : i + max_lines]
        if chunk:
            chunks.append(chunk)
        if i + max_lines >= len(segments):
            break
        i += step
    return chunks


def format_timeline(segments: list[dict]) -> str:
    """Render segments as the canonical '[start-end] sentence' timeline text."""
    if not segments:
        return "(none)"
    return "\n".join(
        f"[{s['start']:.3f}-{s['end']:.3f}] {s['sentence']}" for s in segments
    )
