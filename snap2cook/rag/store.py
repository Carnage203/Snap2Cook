"""Corpus storage backends: Hugging Face dataset (cloud) and a bundled local copy."""
from __future__ import annotations

import io
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import requests
from pydantic import ValidationError

from snap2cook.config import Settings
from snap2cook.exceptions import CorpusLoadError, FileOperationError
from snap2cook.fileio.safe_io import SafeFile
from snap2cook.models import Recipe
from snap2cook.rag.corpus import Corpus

log = logging.getLogger("snap2cook.store")
FILES = ("recipes.jsonl", "embeddings.npy", "meta.json")


class CorpusStore(ABC):
    """Template: subclasses fetch raw bytes; ``load`` parses and validates them."""

    #: Local copies may lack embeddings (then retrieval is lexical); cloud copies may not.
    vectors_optional = False

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    @abstractmethod
    def name(self) -> str:
        """Human readable source name."""

    @abstractmethod
    def fetch(self, filename: str) -> bytes:
        """Return the raw bytes of one corpus file or raise CorpusLoadError."""

    def load(self) -> Corpus:
        """Fetch, validate against current config and build the Corpus."""
        recipes = self.parse_recipes(self.fetch("recipes.jsonl"))
        try:
            raw_vectors = self.fetch("embeddings.npy")
        except CorpusLoadError:
            if not self.vectors_optional:
                raise
            log.info("%s has no embeddings; lexical retrieval will be used", self.name)
            return Corpus(recipes, None, self.name)
        vectors = self._parse_vectors(raw_vectors, len(recipes))
        self._check_meta(self.fetch("meta.json"), len(recipes))
        return Corpus(recipes, vectors, self.name)

    @staticmethod
    def parse_recipes(raw: bytes) -> list[Recipe]:
        """Validate every JSONL row with pydantic."""
        try:
            lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
            return [Recipe.model_validate_json(ln) for ln in lines]
        except (UnicodeDecodeError, ValidationError) as exc:
            raise CorpusLoadError(f"recipes.jsonl failed validation: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _parse_vectors(raw: bytes, count: int) -> np.ndarray:
        try:
            vectors = np.load(io.BytesIO(raw), allow_pickle=False)
        except (ValueError, OSError) as exc:
            raise CorpusLoadError("embeddings.npy is not a valid array") from exc
        if vectors.ndim != 2 or vectors.shape[0] != count:
            raise CorpusLoadError("Embedding matrix shape does not match recipes", shape=vectors.shape, recipes=count)
        return vectors.astype(np.float32)

    def _check_meta(self, raw: bytes, count: int) -> None:
        try:
            meta = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorpusLoadError("meta.json is not valid JSON") from exc
        expected = {"embed_model": self._settings.embed_model, "dim": self._settings.embed_dim, "count": count}
        actual = {k: meta.get(k) for k in expected}
        if actual != expected:
            raise CorpusLoadError("Corpus metadata does not match current config "
                                  "(different embedding model or dimension)", expected=expected, actual=actual)


class HFDatasetStore(CorpusStore):
    """Public Hugging Face dataset read over plain HTTPS (timeout + retry)."""

    def __init__(self, settings: Settings, session: requests.Session | None = None,
                 retries: int = 2, sleep=time.sleep) -> None:
        super().__init__(settings)
        self._session = session or requests.Session()
        self._retries = retries
        self._sleep = sleep

    @property
    def name(self) -> str:
        return f"cloud:hf:{self._settings.hf_dataset_repo}"

    def fetch(self, filename: str) -> bytes:
        if not self._settings.hf_dataset_repo:
            raise CorpusLoadError("HF_DATASET_REPO is not configured")
        url = f"https://huggingface.co/datasets/{self._settings.hf_dataset_repo}/resolve/main/{filename}"
        for attempt in range(self._retries + 1):
            try:
                response = self._session.get(url, timeout=self._settings.api_timeout_s)
                response.raise_for_status()
                return response.content
            except requests.RequestException as exc:
                if attempt == self._retries:
                    raise CorpusLoadError(f"Cloud fetch of {filename} failed: {type(exc).__name__}") from exc
                self._sleep(2 ** attempt + random.random())
        raise CorpusLoadError(f"Cloud fetch of {filename} failed")  # unreachable


class LocalStore(CorpusStore):
    """The copy bundled in ``data/`` (offline fallback)."""

    vectors_optional = True

    def __init__(self, settings: Settings, directory: Path | None = None) -> None:
        super().__init__(settings)
        self._dir = directory or settings.data_dir

    @property
    def name(self) -> str:
        return "local"

    def fetch(self, filename: str) -> bytes:
        try:
            return SafeFile(self._dir / filename).read_bytes()
        except FileOperationError as exc:
            raise CorpusLoadError(f"Local file {filename} unavailable: {exc.message}") from exc


def load_with_fallback(primary: CorpusStore, fallback: CorpusStore) -> Corpus:
    """Try the cloud store; on any CorpusLoadError use the local copy."""
    try:
        corpus = primary.load()
        log.info("Corpus loaded from %s (%d recipes)", primary.name, len(corpus))
        return corpus
    except CorpusLoadError as exc:
        log.warning("Cloud corpus unavailable (%s); falling back to %s", exc.message, fallback.name)
        corpus = fallback.load()
        log.info("Corpus loaded from %s (%d recipes)", fallback.name, len(corpus))
        return corpus
