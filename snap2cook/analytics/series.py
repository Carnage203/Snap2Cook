# SYLLABUS Q10 - pandas Series with labels
"""Builders for the two labelled Series used by the app."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from snap2cook.models import DetectedIngredient


def ingredient_series(items: Iterable[DetectedIngredient]) -> pd.Series:
    """Ingredient name -> confidence (best confidence kept for duplicates, sorted high to low)."""
    pairs = [(item.name, item.confidence) for item in items]
    if not pairs:
        return pd.Series(dtype=float, name="confidence")
    names, confs = zip(*pairs)
    series = pd.Series(confs, index=pd.Index(names), name="confidence", dtype=float)
    return series.groupby(level=0).max().sort_values(ascending=False)


def score_series(pairs: Iterable[tuple[str, float]]) -> pd.Series:
    """Recipe id -> final score, sorted high to low."""
    pairs = list(pairs)
    if not pairs:
        return pd.Series(dtype=float, name="score")
    ids, scores = zip(*pairs)
    return pd.Series(scores, index=pd.Index(ids), name="score", dtype=float).sort_values(ascending=False)


def demo() -> None:
    """Label-based access on an ingredient Series (viva demo)."""
    conf = ingredient_series(
        [DetectedIngredient(name=n, confidence=c) for n, c in
         [("egg", 0.95), ("bread", 0.9), ("onion", 0.62), ("tomato", 0.4), ("egg", 0.7)]]
    )
    print(conf, end="\n\n")
    print("conf.loc['egg']            :", conf.loc["egg"])
    print("label slice 'bread':'onion':", conf.loc["bread":"onion"].to_dict())
    print("sorted ascending           :", conf.sort_values().index.tolist())
    print("boolean mask conf >= 0.6   :", conf[conf >= 0.6].index.tolist())


if __name__ == "__main__":
    demo()
