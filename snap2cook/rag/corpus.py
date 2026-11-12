"""The validated recipe corpus plus its embedding matrix."""
from __future__ import annotations

from typing import Iterator

import numpy as np

from snap2cook.exceptions import CorpusLoadError
from snap2cook.models import Conditions, Recipe
from snap2cook.tiers import Tier


class Corpus:
    """Recipes (row i) and optional document vectors (row i of ``vectors``)."""

    def __init__(self, recipes: list[Recipe], vectors: np.ndarray | None = None, source: str = "memory") -> None:
        if vectors is not None and vectors.shape[0] != len(recipes):
            raise CorpusLoadError("Embedding rows do not match recipe count",
                                  rows=vectors.shape[0], recipes=len(recipes))
        ids = [r.id for r in recipes]
        if len(set(ids)) != len(ids):
            raise CorpusLoadError("Duplicate recipe ids in corpus")
        self._recipes = recipes
        self._vectors = vectors
        self._source = source
        self._by_id = {r.id: i for i, r in enumerate(recipes)}

    @property
    def vectors(self) -> np.ndarray | None:
        """Document vectors, or None when the corpus has no embeddings."""
        return self._vectors

    @property
    def source(self) -> str:
        """Where the corpus was loaded from (cloud or local)."""
        return self._source

    @property
    def vocabulary(self) -> set[str]:
        """Every ingredient name in the corpus."""
        return {i.name for r in self._recipes for i in r.ingredients}

    def __len__(self) -> int:
        return len(self._recipes)

    def __iter__(self) -> Iterator[Recipe]:
        return iter(self._recipes)

    def __getitem__(self, key: int | str) -> Recipe:
        """Look a recipe up by row number or by id."""
        return self._recipes[self._by_id[key] if isinstance(key, str) else key]

    def filter(self, tier: Tier, conditions: Conditions) -> list[int]:
        """Row numbers of recipes the tier accepts under the conditions."""
        return [i for i, r in enumerate(self._recipes) if tier.accepts(r, conditions)]
