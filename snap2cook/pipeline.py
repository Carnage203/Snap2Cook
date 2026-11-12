"""Pipeline: orchestrates vision -> retrieval -> generation. All collaborators are injected."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Iterable

import pandas as pd

from snap2cook.analytics.series import ingredient_series, score_series
from snap2cook.cache import Cache
from snap2cook.config import Settings
from snap2cook.exceptions import GenerationError, NoIngredientsDetectedError
from snap2cook.limiter import RateLimiter
from snap2cook.models import Conditions, GeneratedRecipe, PipelineResult
from snap2cook.rag.generator import GenerationContext, Generator, VerbatimGenerator
from snap2cook.rag.retriever import Retriever
from snap2cook.tiers import Tier
from snap2cook.vision.base import VisionProvider
from snap2cook.vision.image_utils import PreparedImage
from snap2cook.vision.normalizer import IngredientNormalizer

log = logging.getLogger("snap2cook.pipeline")


class Pipeline:
    """Orchestrator; every dependency arrives through the constructor (easy to mock)."""

    def __init__(self, vision: VisionProvider, retriever: Retriever, generator: Generator,
                 normalizer: IngredientNormalizer, cache: Cache, limiter: RateLimiter,
                 settings: Settings, fallback_generator: Generator | None = None) -> None:
        self._vision, self._retriever, self._generator = vision, retriever, generator
        self._normalizer, self._cache, self._limiter, self._settings = normalizer, cache, limiter, settings
        self._fallback = fallback_generator or VerbatimGenerator()

    @property
    def mode(self) -> str:
        """'live' when the vision provider is not the mock."""
        return "mock" if self._vision.name == "mock" else "live"

    def detect(self, image: PreparedImage) -> tuple[pd.Series, bool]:
        """Detect ingredients (cached by image hash). Returns (Series, cache_hit)."""
        key = f"vision:{self._vision.name}:{image.sha256}"
        cached = self._cache.get(key)
        if cached is not None:
            return pd.Series(cached, name="confidence", dtype=float), True
        items = self._normalizer.normalize_all(self._vision.detect(image.data))
        series = ingredient_series(items)
        if series.empty:
            raise NoIngredientsDetectedError("No ingredients left after normalisation")
        self._cache.set(key, series.to_dict(), self._settings.vision_ttl_s)
        return series, False

    @staticmethod
    def _recipe_key(tier: Tier, ingredients: list[str], extras: list[str], conditions: Conditions) -> str:
        payload = json.dumps([tier.name, sorted(tier.allowed_equipment), ingredients, extras,
                              conditions.diet, conditions.max_time])
        return "recipe:" + hashlib.sha256(payload.encode()).hexdigest()

    def recommend(self, ingredients: Iterable[str], tier: Tier, conditions: Conditions,
                  extras: Iterable[str], session_id: str) -> PipelineResult:
        """Retrieve, rerank and generate a recipe for the (user-edited) ingredient list."""
        timings: dict[str, float] = {}
        names = sorted({n for n in (self._normalizer.normalize(i) for i in ingredients) if n})
        if not names:
            raise NoIngredientsDetectedError("The ingredient list is empty")
        extra_names = sorted(self._normalizer.normalize(e) for e in extras) if tier.allow_extra_supplies else []
        series = pd.Series(1.0, index=names, name="confidence")

        key = self._recipe_key(tier, names, extra_names, conditions)
        cached = self._cache.get(key)
        if cached is not None:
            return self._from_cache(cached, series)

        start = time.perf_counter()
        result = self._retriever.search(set(names), tier, conditions, set(extra_names))
        timings["retrieve_s"] = time.perf_counter() - start
        available = frozenset(tier.available(names, extra_names))
        fallback_used = False
        recipe: GeneratedRecipe | None = None
        if result.recipes:
            ctx = GenerationContext(tier, conditions, available, tuple(result.recipes))
            self._limiter.check(session_id)
            start = time.perf_counter()
            try:
                recipe = self._generator.generate(ctx)
            except GenerationError as exc:
                log.warning("Generator failed (%s); using verbatim fallback", exc.message)
                recipe = self._fallback.generate(ctx)
                fallback_used = True
            timings["generate_s"] = time.perf_counter() - start

        candidates = [{"id": s.recipe.id, "title": s.recipe.title, "similarity": round(s.similarity, 3),
                       "coverage": round(s.coverage, 2), "missing": list(s.missing), "score": round(s.score, 3)}
                      for s in result.recipes]
        out = PipelineResult(recipe, series, result.scores, timings,
                             {"query_embedding": result.cache_hit, "recipe": False},
                             fallback_used, result.suggestions, candidates, f"{self.mode}/{result.embedder}")
        self._cache.set(key, out.to_cache(), self._settings.recipe_ttl_s)
        return out

    def _from_cache(self, cached: dict, series: pd.Series) -> PipelineResult:
        recipe = GeneratedRecipe(**cached["recipe"]) if cached["recipe"] else None
        scores = score_series(cached["scores"].items())
        return PipelineResult(recipe, series, scores, {}, {"query_embedding": False, "recipe": True},
                              cached["fallback_used"], cached["suggestions"], cached["candidates"], cached["mode"])
