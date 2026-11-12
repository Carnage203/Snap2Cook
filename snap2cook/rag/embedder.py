"""Embedders: Gemini (cloud) and a lexical fallback that needs no network."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable, Iterable

import numpy as np
from google import genai
from google.genai import types

from snap2cook.exceptions import EmbeddingError
from snap2cook.gemini import call_with_retries


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length (zero rows stay zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class Embedder(ABC):
    """Turns text into unit-length vectors."""

    #: True when vectors match the precomputed corpus matrix.
    uses_corpus_vectors = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier used in cache keys."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed one search query (1-D vector)."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed documents (2-D matrix, one row each)."""


class GeminiEmbedder(Embedder):
    """Gemini embeddings truncated to ``dim`` and L2-normalised by us."""

    uses_corpus_vectors = True

    def __init__(self, client: genai.Client, model: str, dim: int, retries: int = 2,
                 sleep: Callable[[float], None] | None = None) -> None:
        self._client, self._model, self._dim, self._retries = client, model, dim, retries
        self._sleep = sleep

    @property
    def name(self) -> str:
        return f"gemini:{self._model}:{self._dim}"

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        def call() -> np.ndarray:
            response = self._client.models.embed_content(
                model=self._model, contents=texts,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self._dim))
            return np.array([e.values for e in response.embeddings], dtype=np.float32)

        kwargs = {"sleep": self._sleep} if self._sleep else {}
        matrix = call_with_retries(call, EmbeddingError, "Embedding request", self._retries, **kwargs)
        if matrix.ndim != 2 or matrix.shape != (len(texts), self._dim):
            raise EmbeddingError("Unexpected embedding shape", shape=matrix.shape)
        return l2_normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")


class LexicalEmbedder(Embedder):
    """Bag-of-words over the ingredient vocabulary; cosine equals normalised word overlap."""

    def __init__(self, vocabulary: Iterable[str]) -> None:
        words = sorted({w for phrase in vocabulary for w in self._tokens(phrase)})
        self._index = {w: i for i, w in enumerate(words)}

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    @property
    def name(self) -> str:
        return "lexical"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self._index), dtype=np.float32)
        for word in self._tokens(text):
            if word in self._index:
                vec[self._index[word]] = 1.0
        return vec

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._vector(text)[None, :])[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(np.vstack([self._vector(t) for t in texts]))
