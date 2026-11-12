"""Abstract vision provider."""
from __future__ import annotations

from abc import ABC, abstractmethod

from snap2cook.models import DetectedIngredient


class VisionProvider(ABC):
    """Turns image bytes into detected ingredients."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider name for the UI/logs."""

    @abstractmethod
    def detect(self, image_bytes: bytes) -> list[DetectedIngredient]:
        """Return ingredients seen in the (already validated) JPEG bytes."""
