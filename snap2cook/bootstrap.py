"""Composition root: wires concrete classes into a Pipeline (used by app.py and scripts)."""
from __future__ import annotations

from dataclasses import dataclass

from snap2cook.cache import FileCache, LRUCache, TieredCache
from snap2cook.config import Settings
from snap2cook.gemini import build_client, model_chain
from snap2cook.groq import GroqClient
from snap2cook.limiter import RateLimiter
from snap2cook.pipeline import Pipeline
from snap2cook.rag.embedder import GeminiEmbedder, LexicalEmbedder
from snap2cook.rag.generator import GeminiGenerator, VerbatimGenerator
from snap2cook.rag.reranker import CoverageReranker
from snap2cook.rag.retriever import Retriever
from snap2cook.rag.store import HFDatasetStore, LocalStore, load_with_fallback
from snap2cook.tiers import Pantry
from snap2cook.vision.gemini_provider import GeminiVisionProvider
from snap2cook.vision.mock_provider import MockVisionProvider
from snap2cook.vision.normalizer import IngredientNormalizer


@dataclass(frozen=True)
class App:
    """Long-lived objects shared by all Streamlit sessions."""

    settings: Settings
    pipeline: Pipeline
    pantry: Pantry
    cache: TieredCache
    limiter: RateLimiter
    vocabulary: list[str]
    corpus_source: str


def build_app(settings: Settings) -> App:
    """Create everything. Live mode needs a key; otherwise mock providers are used."""
    corpus = load_with_fallback(HFDatasetStore(settings), LocalStore(settings))
    vocabulary = sorted(corpus.vocabulary)
    pantry = Pantry.load(settings.data_dir / "pantry.json")
    normalizer = IngredientNormalizer.from_file(settings.data_dir / "synonyms.json", vocabulary)
    cache = TieredCache(LRUCache(512), FileCache(settings.cache_dir))
    limiter = RateLimiter(settings.rate_limit_calls, settings.rate_limit_window_s, settings.daily_limit)
    lexical = LexicalEmbedder(vocabulary)
    reranker = CoverageReranker.from_settings(settings)
    if settings.live_mode:
        client = build_client(settings)
        groq = GroqClient(settings.groq_api_key) if settings.groq_api_key else None
        vision_models = model_chain(settings.vision_model, settings.fallback_models)
        gen_models = model_chain(settings.gen_model, settings.fallback_models,
                                 settings.groq_gen_models if groq else ())
        vision = GeminiVisionProvider(client, vision_models, settings.api_retries)
        embedder = GeminiEmbedder(client, settings.embed_model, settings.embed_dim, settings.api_retries)
        generator = GeminiGenerator(client, gen_models, settings.api_retries, groq)
    else:
        vision, embedder, generator = MockVisionProvider(), lexical, VerbatimGenerator()
    retriever = Retriever(corpus, embedder, reranker, lexical, cache)
    pipeline = Pipeline(vision, retriever, generator, normalizer, cache, limiter, settings)
    return App(settings, pipeline, pantry, cache, limiter, vocabulary, corpus.source)
