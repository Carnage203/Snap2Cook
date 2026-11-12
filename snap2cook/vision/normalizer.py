"""Maps free-form ingredient names to the corpus vocabulary."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from snap2cook.fileio.safe_io import SafeFile
from snap2cook.models import DetectedIngredient


class IngredientNormalizer:
    """Lower-cases, applies the synonym table, then un-pluralises when the singular is known."""

    def __init__(self, synonyms: dict[str, str], vocabulary: Iterable[str] = ()) -> None:
        self._synonyms = {k.lower(): v.lower() for k, v in synonyms.items()}
        self._vocab = {v.lower() for v in vocabulary}

    @classmethod
    def from_file(cls, path: Path, vocabulary: Iterable[str] = ()) -> "IngredientNormalizer":
        """Load synonyms.json via SafeFile."""
        return cls(SafeFile(path).read_json(), vocabulary)

    def normalize(self, name: str) -> str:
        """Return the canonical name ('' if nothing is left)."""
        cleaned = re.sub(r"[^a-z\s-]", "", name.lower()).replace("-", " ")
        cleaned = " ".join(cleaned.split())
        if cleaned in self._synonyms:
            return self._synonyms[cleaned]
        if cleaned in self._vocab:
            return cleaned
        for singular in (cleaned[:-3] + "y" if cleaned.endswith("ies") else "",
                         cleaned[:-2] if cleaned.endswith("es") else "",
                         cleaned[:-1] if cleaned.endswith("s") else ""):
            if singular and (singular in self._vocab or singular in self._synonyms):
                return self._synonyms.get(singular, singular)
        return cleaned

    def normalize_all(self, items: Iterable[DetectedIngredient]) -> list[DetectedIngredient]:
        """Normalise names, dropping empties."""
        out = [DetectedIngredient(name=self.normalize(i.name), confidence=i.confidence) for i in items]
        return [i for i in out if i.name]
