"""Retriever: filter -> embed query -> cosine top-k -> rerank -> tier rules."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from snap2cook.analytics.series import score_series
from snap2cook.cache import Cache
from snap2cook.exceptions import EmbeddingError, RetrievalError
from snap2cook.models import Conditions
from snap2cook.rag.corpus import Corpus
from snap2cook.rag.embedder import Embedder, LexicalEmbedder
from snap2cook.rag.reranker import Reranker, ScoredRecipe
from snap2cook.tiers import Tier

log = logging.getLogger("snap2cook.retriever")


@dataclass
class RetrievalResult:
    """Top recipes, their labelled scores and, if none fit, what to add."""

    recipes: list[ScoredRecipe]
    scores: pd.Series
    suggestions: list[str] = field(default_factory=list)
    embedder: str = ""
    cache_hit: bool = False


class Retriever:
    """Composes Corpus + Embedder + Reranker (+ lexical fallback and a query cache)."""

    def __init__(self, corpus: Corpus, embedder: Embedder, reranker: Reranker,
                 fallback: LexicalEmbedder, cache: Cache | None = None, k: int = 20, top: int = 3) -> None:
        self._corpus, self._embedder, self._reranker = corpus, embedder, reranker
        self._fallback, self._cache, self._k, self._top = fallback, cache, k, top
        self._fallback_matrix = fallback.embed_documents([r.ingredient_text for r in corpus])

    def _attempts(self) -> list[tuple[Embedder, np.ndarray]]:
        """Embedders to try in order, each with its matching document matrix."""
        attempts: list[tuple[Embedder, np.ndarray]] = []
        if self._embedder.uses_corpus_vectors and self._corpus.vectors is not None:
            attempts.append((self._embedder, self._corpus.vectors))
        attempts.append((self._fallback, self._fallback_matrix))
        return attempts

    def _embed_query(self, embedder: Embedder, text: str) -> tuple[np.ndarray, bool]:
        key = f"q:{embedder.name}:{text.lower()}"
        cached = self._cache.get(key) if self._cache else None
        if cached is not None:
            return np.array(cached, dtype=np.float32), True
        vector = embedder.embed_query(text)
        if self._cache:
            self._cache.set(key, vector.tolist())
        return vector, False

    def search(self, detected: set[str], tier: Tier, conditions: Conditions, extras: set[str]) -> RetrievalResult:
        """Return the best recipes for the ingredients under the tier rules."""
        if len(self._corpus) == 0:
            raise RetrievalError("The corpus is empty")
        rows = self._corpus.filter(tier, conditions)
        if not rows:
            return RetrievalResult([], score_series([]))
        query = f"Recipe using: {', '.join(sorted(detected))} ({tier.name} kitchen)"
        for embedder, matrix in self._attempts():
            try:
                vector, hit = self._embed_query(embedder, query)
                break
            except EmbeddingError as exc:
                log.warning("Embedding failed (%s); trying lexical fallback", exc.message)
        else:
            raise RetrievalError("No embedder could embed the query")
        sims = matrix[rows] @ vector
        order = np.argsort(-sims)[: self._k]
        candidates = [(self._corpus[rows[i]], float(sims[i])) for i in order]
        available = tier.available(detected, extras)
        scored = self._reranker.rerank(candidates, available, detected)
        fits = [s for s in scored if s.overlap > 0 and len(s.missing) <= tier.max_missing][: self._top]
        scores = score_series((s.recipe.id, s.score) for s in fits)
        suggestions = [] if fits else self._suggest(scored)
        return RetrievalResult(fits, scores, suggestions, embedder.name, hit)

    @staticmethod
    def _suggest(scored: list[ScoredRecipe]) -> list[str]:
        """Most common missing ingredients among the nearest almost-possible recipes."""
        near = sorted((s for s in scored if s.overlap > 0), key=lambda s: (len(s.missing), -s.score))[:5]
        counts = Counter(name for s in near for name in s.missing)
        return [name for name, _ in counts.most_common(3)]
