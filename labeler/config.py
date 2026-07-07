"""Configuration loading and merging."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` and return a new dict."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str | Path] = None, overrides: Optional[dict] = None) -> dict:
    """Load YAML config from `path` (or the default) and apply CLI overrides."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    return cfg


def load_prompt_template(cfg: dict) -> str:
    """Load the raw SOP prompt template referenced by the config."""
    prompt_path = Path(cfg["prompt"]["path"])
    if not prompt_path.is_absolute():
        prompt_path = DEFAULT_CONFIG_PATH.parent / prompt_path
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")
