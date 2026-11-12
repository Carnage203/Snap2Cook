"""Offline vision stand-in used when no Gemini key is configured."""
from __future__ import annotations

import hashlib

from snap2cook.models import DetectedIngredient
from snap2cook.vision.base import VisionProvider

_PRESETS: tuple[tuple[tuple[str, float], ...], ...] = (
    (("oats", 0.93), ("banana", 0.9), ("peanut butter", 0.8)),
    (("egg", 0.95), ("bread", 0.9), ("onion", 0.8), ("tomato", 0.7)),
    (("instant noodles", 0.96), ("onion", 0.7), ("tomato", 0.6)),
    (("potato", 0.92), ("onion", 0.85), ("tomato", 0.8)),
    (("poha", 0.9), ("onion", 0.8), ("curd", 0.6)),
)


class MockVisionProvider(VisionProvider):
    """Deterministically picks a preset ingredient set from the image hash."""

    @property
    def name(self) -> str:
        return "mock"

    def detect(self, image_bytes: bytes) -> list[DetectedIngredient]:
        index = int(hashlib.sha256(image_bytes).hexdigest(), 16) % len(_PRESETS)
        return [DetectedIngredient(name=n, confidence=c) for n, c in _PRESETS[index]]
