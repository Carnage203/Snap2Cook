"""Reranking of retrieved candidates by ingredient coverage."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from snap2cook.config import Settings
from snap2cook.models import Recipe


@dataclass(frozen=True)
class ScoredRecipe:
    """A candidate with its similarity, coverage, missing items and final score."""

    recipe: Recipe
    similarity: float
    coverage: float
    missing: tuple[str, ...]
    overlap: int
    score: float


class Reranker(ABC):
    """Orders (recipe, similarity) candidates."""

    @abstractmethod
    def rerank(self, candidates: list[tuple[Recipe, float]], available: set[str],
               detected: set[str]) -> list[ScoredRecipe]:
        """Return all candidates scored and sorted best first."""


class CoverageReranker(Reranker):
    """final = w_sim*similarity + w_cov*coverage - w_miss*missing_count."""

    def __init__(self, w_similarity: float = 0.6, w_coverage: float = 0.4, w_missing: float = 0.1) -> None:
        self._ws, self._wc, self._wm = w_similarity, w_coverage, w_missing

    @classmethod
    def from_settings(cls, settings: Settings) -> "CoverageReranker":
        """Build with the weights from Settings."""
        return cls(settings.w_similarity, settings.w_coverage, settings.w_missing)

    def rerank(self, candidates: list[tuple[Recipe, float]], available: set[str],
               detected: set[str]) -> list[ScoredRecipe]:
        scored = []
        for recipe, similarity in candidates:
            required = recipe.required_names
            missing = tuple(n for n in required if n not in available)
            coverage = (len(required) - len(missing)) / len(required) if required else 1.0
            overlap = len({i.name for i in recipe.ingredients} & detected)
            score = self._ws * similarity + self._wc * coverage - self._wm * len(missing)
            scored.append(ScoredRecipe(recipe, similarity, coverage, missing, overlap, score))
        return sorted(scored, key=lambda s: s.score, reverse=True)
