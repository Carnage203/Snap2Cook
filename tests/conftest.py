"""Shared fixtures. Tests never touch the network and never use a real API key."""
from __future__ import annotations

import pytest

from snap2cook.cache import LRUCache, TieredCache
from snap2cook.config import Settings
from snap2cook.limiter import RateLimiter
from snap2cook.pipeline import Pipeline
from snap2cook.rag.embedder import LexicalEmbedder
from snap2cook.rag.generator import Generator, VerbatimGenerator
from snap2cook.rag.reranker import CoverageReranker
from snap2cook.rag.retriever import Retriever
from snap2cook.rag.store import LocalStore
from snap2cook.tiers import Pantry, TierFactory
from snap2cook.vision.mock_provider import MockVisionProvider
from snap2cook.vision.normalizer import IngredientNormalizer


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(gemini_api_key="")


@pytest.fixture(scope="session")
def pantry(settings: Settings) -> Pantry:
    return Pantry.load(settings.data_dir / "pantry.json")


@pytest.fixture(scope="session")
def corpus(settings: Settings):
    return LocalStore(settings).load()


@pytest.fixture()
def hostel(pantry):
    return TierFactory.create("hostel", pantry)


@pytest.fixture()
def pg(pantry):
    return TierFactory.create("pg", pantry, ["gas_stove"])


@pytest.fixture()
def home(pantry):
    return TierFactory.create("home", pantry, ["gas_stove", "microwave"])


@pytest.fixture()
def retriever(corpus):
    lexical = LexicalEmbedder(corpus.vocabulary)
    return Retriever(corpus, lexical, CoverageReranker(), lexical)


class CountingGenerator(Generator):
    """Fake generator that counts calls (stands in for the LLM)."""

    def __init__(self) -> None:
        self.calls = 0
        self._inner = VerbatimGenerator()

    @property
    def name(self) -> str:
        return "counting"

    def generate(self, ctx):
        self.calls += 1
        return self._inner.generate(ctx)


@pytest.fixture()
def make_pipeline(settings, corpus, retriever):
    def build(generator: Generator, limiter: RateLimiter | None = None) -> Pipeline:
        normalizer = IngredientNormalizer.from_file(settings.data_dir / "synonyms.json", corpus.vocabulary)
        return Pipeline(MockVisionProvider(), retriever, generator, normalizer, TieredCache(LRUCache()),
                        limiter or RateLimiter(), settings)
    return build
