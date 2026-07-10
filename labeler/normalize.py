"""Task-agnostic post-processing to align model output with the SOP.

Conservative cleanup: strip appearance adjectives, fix place wording, resolve
timestamp overlaps, refine action spans (anti frame-grid), snap starts to
sampled frame times. Never invents hands or new actions.
"""
from __future__ import annotations

import re

from .segments import median_dt, merge_segments

_APPEARANCE_WORDS = frozenset({
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "grey", "gray", "beige", "tan", "gold", "silver",
    "cyan", "magenta", "violet", "crimson", "maroon", "navy", "teal",
    "turquoise", "ivory", "cream", "amber",
    "large", "small", "big", "tiny", "huge", "little", "medium", "mini",
    "massive", "compact",
    "round", "square", "rectangular", "cylindrical", "flat", "long", "short",
    "wide", "narrow", "thick", "thin",
    "wooden", "metal", "plastic", "glass", "ceramic", "stainless", "chrome",
    "rubber", "leather", "fabric", "cardboard", "paper", "foil", "shiny",
    "matte", "transparent", "opaque",
})

_NONE_RE = re.compile(r"^\(?\s*none\s*\)?\.?$", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-zA-Z]+")
# "Place a bottle from the shopping basket onto the store shelf with ..."
_PLACE_FROM_RE = re.compile(
    r"^Place (?:a|the) (.+?) from .+? (?:onto|on|in) (?:the |a )?(.+?)"
    r"(\s+with\b[^.]+)?\.?$",
    re.IGNORECASE,
)
_PICK_RE = re.compile(r"^Pick\b", re.IGNORECASE)
_PLACE_RE = re.compile(r"^Place\b", re.IGNORECASE)


def strip_appearance_adjectives(sentence: str) -> str:
    """Remove SOP-forbidden appearance adjectives from a subtask sentence."""
    if not sentence:
        return sentence

    def _replace_word(m: re.Match) -> str:
        word = m.group(0)
        if word.lower() in _APPEARANCE_WORDS:
            return ""
        return word

    cleaned = _WORD_RE.sub(_replace_word, sentence)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.])", r"\1", cleaned)
    cleaned = re.sub(r"\bthe the\b", "the", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def fix_place_wording(sentence: str) -> str:
    """Rewrite invalid 'Place X from ORIGIN onto DEST' -> 'Place the X on DEST'.

    Per SOP §9, place moves object from hand to destination — origin must not
    appear on place lines.
    """
    s = sentence.strip()
    if not s.endswith("."):
        s = s + "."
    m = _PLACE_FROM_RE.match(s)
    if not m:
        return s
    obj, dest, hand = m.group(1).strip(), m.group(2).strip(), m.group(3) or ""
    if not dest.lower().startswith("the "):
        dest = f"the {dest}"
    return f"Place the {obj} on {dest}{hand}."


def _is_none_segment(sentence: str) -> bool:
    return bool(_NONE_RE.match(sentence.strip()))


def _nearest_frame(t: float, frame_times: list[float]) -> float:
    return min(frame_times, key=lambda f: abs(f - t))


def _segment_duration(seg: dict) -> float:
    return max(0.0, seg["end"] - seg["start"])


def _is_micro_segment(seg: dict, dt: float, eps: float = 0.05) -> bool:
    if dt <= 0:
        return False
    return _segment_duration(seg) <= dt * 1.25 + eps


def _is_pick(sentence: str) -> bool:
    return bool(_PICK_RE.match(sentence.strip()))


def _is_place(sentence: str) -> bool:
    return bool(_PLACE_RE.match(sentence.strip()))


def _is_pick_place_pair(a: dict, b: dict) -> bool:
    return _is_pick(a["sentence"]) and _is_place(b["sentence"])


def snap_segment_starts_to_frames(
    segments: list[dict],
    frame_times: list[float],
    tolerance: float | None = None,
) -> list[dict]:
    """Snap only action START times to the nearest sampled frame.

  Per SOP §8, starts should align to frames where an action begins; ends may
  span multiple frames and should not be forced onto the sampling grid.
    """
    if not segments or not frame_times:
        return segments
    if tolerance is None:
        dt = median_dt(frame_times)
        tolerance = max(0.26, dt * 0.6) if dt > 0 else 0.26

    snapped: list[dict] = []
    for seg in segments:
        start, end = seg["start"], seg["end"]
        ns = _nearest_frame(start, frame_times)
        if abs(ns - start) <= tolerance:
            start = ns
        if end <= start:
            end = min(frame_times[-1], start + tolerance)
        snapped.append({"start": start, "end": end, "sentence": seg["sentence"]})
    return snapped


def snap_to_frame_times(
    segments: list[dict],
    frame_times: list[float],
    tolerance: float | None = None,
) -> list[dict]:
    """Backward-compatible alias: snap starts only (see snap_segment_starts_to_frames)."""
    return snap_segment_starts_to_frames(segments, frame_times, tolerance=tolerance)


def thin_dense_frame_grid(
    segments: list[dict],
    frame_times: list[float],
    *,
    duration_sec: float | None = None,
    max_cycles_per_sec: float = 0.55,
) -> list[dict]:
    """Drop excess 1-frame pick/place pairs when the model grids one cycle per frame.

  SOP §8 forbids emitting one subtask per sampled frame unless each frame shows
  a genuinely new action. When bursts exceed a plausible physical rate, keep
  evenly-spaced pairs and discard the rest (never merges pick+place into one line).
    """
    if not segments or not frame_times:
        return segments
    dt = median_dt(frame_times)
    if dt <= 0:
        return segments

    ordered = sorted(segments, key=lambda s: (s["start"], s["end"]))
    keep = [True] * len(ordered)
    i = 0
    while i < len(ordered) - 1:
        if not (_is_pick_place_pair(ordered[i], ordered[i + 1])
                and _is_micro_segment(ordered[i], dt)
                and _is_micro_segment(ordered[i + 1], dt)):
            i += 1
            continue

        j = i
        while j < len(ordered) - 1:
            if not (_is_pick_place_pair(ordered[j], ordered[j + 1])
                    and _is_micro_segment(ordered[j], dt)
                    and _is_micro_segment(ordered[j + 1], dt)):
                break
            j += 2
        burst_end = ordered[j - 1]["end"] if j > i else ordered[i + 1]["end"]
        burst_start = ordered[i]["start"]
        span = max(dt, burst_end - burst_start)
        n_pairs = (j - i) // 2
        max_pairs = max(1, int(span * max_cycles_per_sec + 0.999))
        if n_pairs > max_pairs:
            keep_pairs = set()
            if max_pairs == 1:
                keep_pairs.add(0)
            else:
                for k in range(max_pairs):
                    keep_pairs.add(int(round(k * (n_pairs - 1) / (max_pairs - 1))))
            for p in range(n_pairs):
                if p not in keep_pairs:
                    keep[i + 2 * p] = False
                    keep[i + 2 * p + 1] = False
        i = j if j > i else i + 2

    thinned = [dict(seg) for seg, ok in zip(ordered, keep) if ok]
    if duration_sec and thinned:
        thinned[-1]["end"] = min(thinned[-1]["end"], duration_sec)
    return thinned


def refine_action_spans(
    segments: list[dict],
    frame_times: list[float],
    *,
    duration_sec: float | None = None,
    min_action_sec: float = 1.0,
    pick_weight: float = 0.45,
    gap_absorb_ratio: float = 0.65,
) -> list[dict]:
    """Expand single-frame segments using idle gaps and SOP-like pick/place splits.

  When pick and place are each only one sampling interval wide, borrow time from
  the following idle gap (if any) so actions span realistic durations instead
  of a rigid 0.5s grid.
    """
    if not segments or not frame_times:
        return segments
    dt = median_dt(frame_times)
    if dt <= 0:
        return segments

    clip_end = duration_sec if duration_sec and duration_sec > 0 else frame_times[-1]
    ordered = [dict(s) for s in sorted(segments, key=lambda x: (x["start"], x["end"]))]
    i = 0
    while i < len(ordered):
        seg = ordered[i]
        dur = _segment_duration(seg)
        if dur >= min_action_sec or not _is_micro_segment(seg, dt):
            i += 1
            continue

        # pick/place pair: redistribute across pair window + trailing idle gap
        if i + 1 < len(ordered) and _is_pick_place_pair(seg, ordered[i + 1]):
            pick, place = seg, ordered[i + 1]
            if _is_micro_segment(place, dt):
                pair_start = pick["start"]
                pair_end = place["end"]
                next_start = (
                    ordered[i + 2]["start"] if i + 2 < len(ordered) else clip_end
                )
                idle = max(0.0, next_start - pair_end)
                absorb = idle * gap_absorb_ratio
                target = max(min_action_sec * 2, pair_end - pair_start + absorb)
                target = min(target, max(pair_end - pair_start + idle, min_action_sec * 2))
                pick_dur = max(min_action_sec, target * pick_weight)
                place_dur = max(min_action_sec, target - pick_dur)
                # Do not collide with the next event.
                if i + 2 < len(ordered):
                    max_end = next_start
                    if pick_dur + place_dur > max_end - pair_start:
                        scale = (max_end - pair_start) / max(pick_dur + place_dur, 1e-6)
                        pick_dur *= scale
                        place_dur *= scale
                pick["end"] = pick["start"] + pick_dur
                place["start"] = pick["end"]
                place["end"] = min(place["start"] + place_dur, clip_end)
                i += 2
                continue

        # lone micro segment: extend into following gap up to min_action_sec
        next_start = ordered[i + 1]["start"] if i + 1 < len(ordered) else clip_end
        idle = max(0.0, next_start - seg["end"])
        extend = min(idle * gap_absorb_ratio, max(0.0, min_action_sec - dur))
        seg["end"] = min(seg["start"] + dur + extend, clip_end)
        i += 1

    return ordered


def resolve_overlapping_segments(segments: list[dict]) -> list[dict]:
    """Trim overlaps so each segment ends when the next begins."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: (s["start"], s["end"]))
    out: list[dict] = []
    for seg in ordered:
        if seg["end"] <= seg["start"]:
            continue
        if not out:
            out.append(dict(seg))
            continue
        prev = out[-1]
        if seg["start"] < prev["end"]:
            if seg["start"] > prev["start"]:
                prev["end"] = seg["start"]
            else:
                # Same start window: keep the longer span.
                if (seg["end"] - seg["start"]) > (prev["end"] - prev["start"]):
                    out[-1] = dict(seg)
                continue
        if seg["end"] <= seg["start"]:
            continue
        out.append(dict(seg))
    return out


def dedupe_same_start(segments: list[dict], eps: float = 0.01) -> list[dict]:
    """Collapse duplicate lines that share the same start instant."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: (s["start"], s["end"]))
    out: list[dict] = []
    for seg in ordered:
        if not out:
            out.append(dict(seg))
            continue
        prev = out[-1]
        if abs(seg["start"] - prev["start"]) <= eps:
            if seg["sentence"].strip().lower() == prev["sentence"].strip().lower():
                prev["end"] = max(prev["end"], seg["end"])
                continue
            if (seg["end"] - seg["start"]) > (prev["end"] - prev["start"]):
                out[-1] = dict(seg)
            continue
        out.append(dict(seg))
    return out


def normalize_timeline(
    segments: list[dict],
    *,
    gap_tol: float = 0.0,
    frame_times: list[float] | None = None,
    duration_sec: float | None = None,
    drop_none: bool = True,
    strip_appearance: bool = True,
    fix_place: bool = True,
    resolve_overlaps: bool = True,
    thin_frame_grid: bool = True,
    refine_spans: bool = True,
    snap_frames: bool = True,
    min_action_sec: float = 1.0,
    max_cycles_per_sec: float = 0.55,
) -> list[dict]:
    """Apply SOP-aligned cleanup without inventing hands or actions."""
    if not segments:
        return []

    out: list[dict] = []
    for seg in segments:
        sentence = seg["sentence"]
        if drop_none and _is_none_segment(sentence):
            continue
        if strip_appearance:
            sentence = strip_appearance_adjectives(sentence)
        if fix_place:
            sentence = fix_place_wording(sentence)
        if not sentence:
            continue
        out.append({"start": seg["start"], "end": seg["end"], "sentence": sentence})

    out = merge_segments(out, gap_tol=gap_tol)

    if resolve_overlaps:
        out = resolve_overlapping_segments(out)
        out = dedupe_same_start(out)

    if thin_frame_grid and frame_times:
        out = thin_dense_frame_grid(
            out,
            frame_times,
            duration_sec=duration_sec,
            max_cycles_per_sec=max_cycles_per_sec,
        )

    if refine_spans and frame_times:
        out = refine_action_spans(
            out,
            frame_times,
            duration_sec=duration_sec,
            min_action_sec=min_action_sec,
        )

    if snap_frames and frame_times:
        out = snap_segment_starts_to_frames(out, frame_times)

    if resolve_overlaps:
        out = resolve_overlapping_segments(out)

    return merge_segments(out, gap_tol=gap_tol)
