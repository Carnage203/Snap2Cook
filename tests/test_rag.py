"""Corpus stores, retrieval, grounding and the end-to-end pipeline (no network)."""
from __future__ import annotations

import io

import numpy as np
import pytest
import requests

from snap2cook.config import Settings
from snap2cook.exceptions import CorpusLoadError, GenerationError, NoIngredientsDetectedError, RateLimitExceededError
from snap2cook.limiter import RateLimiter
from snap2cook.models import Conditions, GeneratedRecipe
from snap2cook.rag.generator import GenerationContext, Generator, validate_grounded
from snap2cook.rag.store import CorpusStore, HFDatasetStore, LocalStore, load_with_fallback
from snap2cook.vision.image_utils import ImageValidator
from tests.conftest import CountingGenerator


class DictStore(CorpusStore):
    """In-memory store for testing validation."""

    def __init__(self, settings: Settings, files: dict[str, bytes]) -> None:
        super().__init__(settings)
        self._files = files

    @property
    def name(self) -> str:
        return "dict"

    def fetch(self, filename: str) -> bytes:
        try:
            return self._files[filename]
        except KeyError as exc:
            raise CorpusLoadError(f"{filename} missing") from exc


def _npy(rows: int, dim: int) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ones((rows, dim), dtype=np.float32))
    return buffer.getvalue()


def test_meta_mismatch_raises_corpus_load_error(settings) -> None:
    recipes = (settings.data_dir / "recipes.jsonl").read_bytes()
    count = len(recipes.decode().splitlines())
    meta = b'{"embed_model": "some-other-model", "dim": 768, "count": %d}' % count
    store = DictStore(settings, {"recipes.jsonl": recipes, "embeddings.npy": _npy(count, 768), "meta.json": meta})
    with pytest.raises(CorpusLoadError, match="metadata"):
        store.load()


def test_matching_meta_loads(settings) -> None:
    recipes = (settings.data_dir / "recipes.jsonl").read_bytes()
    count = len(recipes.decode().splitlines())
    meta = ('{"embed_model": "%s", "dim": %d, "count": %d}' % (settings.embed_model, settings.embed_dim, count)).encode()
    store = DictStore(settings, {"recipes.jsonl": recipes, "embeddings.npy": _npy(count, settings.embed_dim), "meta.json": meta})
    assert store.load().vectors.shape == (count, settings.embed_dim)


def test_invalid_recipe_row_is_rejected(settings) -> None:
    with pytest.raises(CorpusLoadError):
        DictStore(settings, {"recipes.jsonl": b'{"id": "x"}\n'}).load()


def test_corpus_has_90_recipes_30_per_tier(corpus) -> None:
    firsts = [r.tiers[0] for r in corpus]
    assert len(corpus) == 90 and {t: firsts.count(t) for t in set(firsts)} == {"hostel": 30, "pg": 30, "home": 30}


def test_cloud_failure_falls_back_to_local(settings) -> None:
    class Down(requests.Session):
        def get(self, *args, **kwargs):
            raise requests.ConnectionError("offline")

    cloud_settings = Settings(hf_dataset_repo="someone/snap2cook")
    cloud = HFDatasetStore(cloud_settings, session=Down(), retries=1, sleep=lambda _: None)
    corpus = load_with_fallback(cloud, LocalStore(settings))
    assert corpus.source == "local" and len(corpus) == 90


def test_retriever_returns_at_most_top_and_labelled_scores(retriever, pg) -> None:
    result = retriever.search({"egg", "onion", "tomato"}, pg, Conditions(), set())
    assert 0 < len(result.recipes) <= 3
    assert result.scores.name == "score" and list(result.scores.index) == [s.recipe.id for s in result.recipes]


def test_pg_respects_max_missing(retriever, pg) -> None:
    result = retriever.search({"potato", "carrot", "onion"}, pg, Conditions(), set())
    assert all(len(s.missing) <= 1 for s in result.recipes)


def test_hostel_with_unusable_items_returns_no_recipe(retriever, hostel) -> None:
    result = retriever.search({"egg", "paneer"}, hostel, Conditions(), set())
    assert result.recipes == [] and result.scores.empty


def test_hostel_near_miss_suggests_an_ingredient_to_add(retriever, hostel) -> None:
    result = retriever.search({"egg", "bread"}, hostel, Conditions(), set())
    assert result.recipes == [] and 1 <= len(result.suggestions) <= 3
    assert "banana" in result.suggestions or "jam" in result.suggestions


def test_pipeline_end_to_end_mock(make_pipeline, hostel) -> None:
    pipeline = make_pipeline(CountingGenerator())
    image = ImageValidator().prepare(_png_bytes())
    series, hit = pipeline.detect(image)
    assert not hit and series.name == "confidence" and len(series) >= 2
    result = pipeline.recommend(list(series.index), hostel, Conditions(), [], "s1")
    assert result.mode.startswith("mock") and (result.recipe is not None or result.suggestions)
    assert pipeline.detect(image)[1] is True


def _png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "green").save(buffer, format="PNG")
    return buffer.getvalue()


def test_cache_hit_skips_the_llm_call(make_pipeline, pg) -> None:
    generator = CountingGenerator()
    pipeline = make_pipeline(generator)
    first = pipeline.recommend(["egg", "onion", "tomato"], pg, Conditions(), [], "s1")
    second = pipeline.recommend(["Tomato", "EGG", "onion"], pg, Conditions(), [], "s1")
    assert generator.calls == 1 and first.cache_hits["recipe"] is False and second.cache_hits["recipe"] is True
    assert second.recipe == first.recipe


def test_generator_failure_uses_verbatim_fallback(make_pipeline, pg) -> None:
    class Broken(Generator):
        name = "broken"  # type: ignore[assignment]

        def generate(self, ctx):
            raise GenerationError("LLM down")

    result = make_pipeline(Broken()).recommend(["egg", "onion", "tomato"], pg, Conditions(), [], "s1")
    assert result.fallback_used and result.recipe is not None


def test_rate_limit_blocks_after_limit(make_pipeline, pg) -> None:
    pipeline = make_pipeline(CountingGenerator(), RateLimiter(1, 600, 100))
    pipeline.recommend(["egg", "onion", "tomato"], pg, Conditions(), [], "s1")
    with pytest.raises(RateLimitExceededError):
        pipeline.recommend(["egg", "onion"], pg, Conditions(), [], "s1")


def test_empty_ingredient_list_raises(make_pipeline, hostel) -> None:
    with pytest.raises(NoIngredientsDetectedError):
        make_pipeline(CountingGenerator()).recommend(["  ", ""], hostel, Conditions(), [], "s1")


def test_grounding_rejects_invented_ingredient_and_technique(retriever, hostel) -> None:
    result = retriever.search({"oats", "banana"}, hostel, Conditions(), set())
    ctx = GenerationContext(hostel, Conditions(), frozenset({"oats", "banana", "water", "salt"}), tuple(result.recipes))
    bad = GeneratedRecipe(title="x", time_min=5, ingredients_used=["oats", "saffron"], extra_needed=[],
                          techniques_used=["fry"], steps=["do"], substitutions=[], safety_note="careful",
                          source_recipe_ids=[result.recipes[0].recipe.id])
    problems = validate_grounded(bad, ctx)
    assert any("saffron" in p for p in problems) and any("fry" in p for p in problems)


def test_detect_many_merges_photos_and_limits_to_max(make_pipeline) -> None:
    from PIL import Image

    def png(color: str) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
        return buffer.getvalue()

    pipeline = make_pipeline(CountingGenerator())
    images = [ImageValidator().prepare(png(c)) for c in ("red", "green", "blue", "white")]
    single = pipeline.detect(images[0])[0]
    merged, empty = pipeline.detect_many(images)
    assert empty == 0 and merged.name == "confidence" and set(single.index) <= set(merged.index)
    assert merged.index.is_unique
