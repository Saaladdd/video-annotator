"""Backend abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from PIL import Image


class LabelerBackend(ABC):
    """A backend takes a prompt + list of frames and returns a text label."""

    name: str = "base"
    model_id: str = "unknown"

    @abstractmethod
    def label(self, prompt: str, frames: List[Image.Image]) -> str:
        """Run the model and return the label text."""
        raise NotImplementedError

    def info(self) -> dict:
        return {"backend": self.name, "model": self.model_id}

    def close(self) -> None:
        """Release resources. Default no-op."""
        return None
