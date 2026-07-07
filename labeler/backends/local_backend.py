"""Local HuggingFace VLM backend.

Default model: Qwen/Qwen2.5-VL-7B-Instruct — strong open VLM with native
multi-image (multi-frame) support, good performance on egocentric video.

Other tested models:
  - Qwen/Qwen2.5-VL-3B-Instruct  (smaller, ~6GB VRAM)
  - openbmb/MiniCPM-V-2_6        (lightweight fallback)
Any HF causal VLM using the chat template + processor pattern should work.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from PIL import Image

from .base import LabelerBackend

log = logging.getLogger(__name__)


def _resolve_device(pref: str) -> str:
    import torch
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return "cuda"
    if pref == "mps" or (pref == "auto" and torch.backends.mps.is_available()):
        return "mps"
    return "cpu"


def _resolve_dtype(name: str):
    import torch
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


class LocalBackend(LabelerBackend):
    name = "local"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_id = cfg["model_id"]
        self.device = _resolve_device(cfg.get("device", "auto"))
        self.dtype = _resolve_dtype(cfg.get("dtype", "bfloat16"))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 1024))
        self.temperature = float(cfg.get("temperature", 0.2))
        self.max_pixels = cfg.get("max_pixels")
        self.load_in_4bit = bool(cfg.get("load_in_4bit", False))

        self._model = None
        self._processor = None
        self._is_qwen_vl = "qwen" in self.model_id.lower() and "vl" in self.model_id.lower()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor

        log.info("Loading local model %s on %s (%s)", self.model_id, self.device, self.dtype)

        quant_kwargs = {}
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as e:
                log.warning("4-bit requested but bitsandbytes unavailable: %s", e)

        proc_kwargs = {}
        if self.max_pixels:
            proc_kwargs["max_pixels"] = int(self.max_pixels)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, **proc_kwargs)

        if self._is_qwen_vl:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as _Cls
            except Exception:
                from transformers import Qwen2VLForConditionalGeneration as _Cls  # type: ignore
            self._model = _Cls.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
                **quant_kwargs,
            )
        else:
            from transformers import AutoModelForCausalLM
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
                **quant_kwargs,
            )
        if self.device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()

    # ------------------------------------------------------------------
    def label(self, prompt: str, frames: List[Image.Image]) -> str:
        self._load()
        import torch

        if self._is_qwen_vl:
            return self._label_qwen_vl(prompt, frames)
        return self._label_generic(prompt, frames)

    # ---- Qwen2.5-VL specific path ---------------------------------------
    def _label_qwen_vl(self, prompt: str, frames: List[Image.Image]) -> str:
        import torch
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "qwen-vl-utils is required for Qwen2.5-VL. Install with "
                "`pip install qwen-vl-utils`."
            ) from e

        content = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: (v.to(self._model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.inference_mode():
            gen_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
            )
        input_len = inputs["input_ids"].shape[1]
        trimmed = gen_ids[:, input_len:]
        out = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return out.strip()

    # ---- Generic HF VLM fallback (best-effort) ---------------------------
    def _label_generic(self, prompt: str, frames: List[Image.Image]) -> str:
        import torch
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in frames]
                + [{"type": "text", "text": prompt}],
            }
        ]
        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt

        inputs = self._processor(text=text, images=frames, return_tensors="pt")
        inputs = {k: (v.to(self._model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.inference_mode():
            gen_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
            )
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        trimmed = gen_ids[:, input_len:] if input_len else gen_ids
        if hasattr(self._processor, "batch_decode"):
            out = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        else:
            out = self._processor.tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]
        return out.strip()

    def close(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
