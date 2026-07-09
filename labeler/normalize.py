"""Task-agnostic post-processing to align model output with the SOP.

Strips appearance adjectives (color, size, shape, material), removes junk
segments, fixes zero-duration lines, and re-merges consecutive duplicates.
Works for any egocentric task — not tied to a specific scene or object set.
"""
from __future__ import annotations

import re

from .segments import merge_segments

# Appearance words the SOP says to omit unless disambiguating.
# Kept as a set for O(1) lookup; matched as whole words only.
_APPEARANCE_WORDS = frozenset({
    # colors
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "grey", "gray", "beige", "tan", "gold", "silver",
    "cyan", "magenta", "violet", "crimson", "maroon", "navy", "teal",
    "turquoise", "ivory", "cream", "amber",
    # sizes
    "large", "small", "big", "tiny", "huge", "little", "medium", "mini",
    "massive", "compact",
    # shapes / descriptors
    "round", "square", "rectangular", "cylindrical", "flat", "long", "short",
    "wide", "narrow", "thick", "thin",
    # materials / finishes (visual, not functional)
    "wooden", "metal", "plastic", "glass", "ceramic", "stainless", "chrome",
    "rubber", "leather", "fabric", "cardboard", "paper", "foil", "shiny",
    "matte", "transparent", "opaque",
})

_NONE_RE = re.compile(r"^\(?\s*none\s*\)?\.?$", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-zA-Z]+")
_HAND_RE = re.compile(
    r"\bwith (?:the )?(?:left|right) hand\b|\bwith both hands\b",
    re.IGNORECASE,
)
_HAND_EXTRACT_RE = re.compile(
    r"\bwith (?:the )?(left|right) hand\b",
    re.IGNORECASE,
)
# Manipulation / reach verbs that require a hand per SOP §5.
_HAND_REQUIRED_RE = re.compile(
    r"\b(?:extend|reach|grasp|pick|place|hold|carry|push|pull|rotate|"
    r"open|close|insert|remove|adjust|align|fold|wipe|transfer)\b",
    re.IGNORECASE,
)


def _extract_hand(sentence: str) -> str | None:
    if re.search(r"\bwith both hands\b", sentence, re.IGNORECASE):
        return "both"
    m = _HAND_EXTRACT_RE.search(sentence)
    if m:
        return m.group(1).lower()
    return None


def _append_hand(sentence: str, hand: str) -> str:
    base = sentence.strip().rstrip(".")
    if hand == "both":
        return f"{base} with both hands."
    return f"{base} with the {hand} hand."


def ensure_hand_specified(segments: list[dict]) -> list[dict]:
    """Add missing hand clauses by propagating from adjacent subtasks."""
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: (s["start"], s["end"]))
    hands: list[str | None] = [_extract_hand(s["sentence"]) for s in ordered]

    # Forward pass: carry last known hand to lines that need one.
    last: str | None = None
    for i, seg in enumerate(ordered):
        h = hands[i]
        if h:
            last = h
        elif last and _HAND_REQUIRED_RE.search(seg["sentence"]):
            hands[i] = last

    # Backward pass for leading lines before any hand was seen.
    last = None
    for i in range(len(ordered) - 1, -1, -1):
        h = hands[i]
        if h:
            last = h
        elif last and _HAND_REQUIRED_RE.search(ordered[i]["sentence"]):
            hands[i] = last

    out: list[dict] = []
    for seg, hand in zip(ordered, hands):
        sentence = seg["sentence"]
        if hand and not _HAND_RE.search(sentence):
            sentence = _append_hand(sentence, hand)
        out.append({"start": seg["start"], "end": seg["end"], "sentence": sentence})
    return out


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
    # Collapse whitespace and fix stray spaces before punctuation.
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.])", r"\1", cleaned)
    cleaned = re.sub(r"\bthe the\b", "the", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _is_none_segment(sentence: str) -> bool:
    return bool(_NONE_RE.match(sentence.strip()))


def _action_triplet(sentence: str) -> str:
    """Normalize a sentence to a coarse action-object-location key for dedup."""
    s = strip_appearance_adjectives(sentence).lower().rstrip(".")
    s = re.sub(r"\s+", " ", s)
    # Drop hand suffix so pick/place pairs with different hands still dedup
    # when the rest is identical (rare duplicate from chunk overlap).
    s = re.sub(
        r"\s+with (?:the )?(?:left|right) hand$",
        "",
        s,
    )
    s = re.sub(
        r"\s+with both hands$",
        "",
        s,
    )
    return s


def fix_zero_duration_segments(
    segments: list[dict],
    min_span: float = 0.001,
) -> list[dict]:
    """Extend or drop segments where start == end."""
    fixed: list[dict] = []
    for seg in segments:
        start, end = seg["start"], seg["end"]
        if end <= start:
            if fixed:
                # Attach instantaneous blip to the previous segment's end.
                fixed[-1]["end"] = max(fixed[-1]["end"], start + min_span)
            continue
        fixed.append(dict(seg))
    return fixed


def collapse_consecutive_triplets(segments: list[dict]) -> list[dict]:
    """Merge consecutive lines with the same action-object-location triple."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: (s["start"], s["end"]))
    merged: list[dict] = []
    for seg in ordered:
        triplet = _action_triplet(seg["sentence"])
        if merged:
            prev = merged[-1]
            if _action_triplet(prev["sentence"]) == triplet:
                prev["end"] = max(prev["end"], seg["end"])
                continue
        merged.append(dict(seg))
    return merged


def normalize_timeline(
    segments: list[dict],
    *,
    gap_tol: float = 0.0,
    drop_none: bool = True,
    strip_appearance: bool = True,
    ensure_hand: bool = True,
    collapse_triplets: bool = True,
    fix_zero_duration: bool = True,
) -> list[dict]:
    """Apply SOP-aligned cleanup to a parsed subtask timeline."""
    if not segments:
        return []

    out: list[dict] = []
    for seg in segments:
        sentence = seg["sentence"]
        if drop_none and _is_none_segment(sentence):
            continue
        if strip_appearance:
            sentence = strip_appearance_adjectives(sentence)
        if not sentence:
            continue
        out.append({"start": seg["start"], "end": seg["end"], "sentence": sentence})

    if ensure_hand:
        out = ensure_hand_specified(out)

    if fix_zero_duration:
        out = fix_zero_duration_segments(out)

    if collapse_triplets:
        out = collapse_consecutive_triplets(out)

    return merge_segments(out, gap_tol=gap_tol)
