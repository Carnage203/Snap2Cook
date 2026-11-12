"""Exceptions (Q9), Series (Q10), tiers, caches, limiter, normalizer, images."""
from __future__ import annotations

import io

import pytest
from PIL import Image

from snap2cook.analytics.series import ingredient_series, score_series
from snap2cook.cache import FileCache, LRUCache, TieredCache
from snap2cook.exceptions import (
    FileOperationError, InvalidImageError, RateLimitExceededError, Snap2CookError, UnsupportedTierError)
from snap2cook.limiter import RateLimiter
from snap2cook.models import HOSTEL_TECHNIQUES, Conditions, DetectedIngredient
from snap2cook.tiers import TierFactory
from snap2cook.vision.image_utils import ImageValidator
from snap2cook.vision.normalizer import IngredientNormalizer


def test_q9_hierarchy_and_user_message() -> None:
    err = FileOperationError("boom", "x.txt", "read")
    assert isinstance(err, Snap2CookError) and err.context["op"] == "read" and err.user_message


def test_q9_rate_limit_error_carries_retry_after() -> None:
    err = RateLimitExceededError("slow down", retry_after=120)
    assert err.retry_after == 120 and "2 minute" in err.user_message


def test_q10_ingredient_series_labels() -> None:
    s = ingredient_series([DetectedIngredient(name="egg", confidence=0.9),
                           DetectedIngredient(name="egg", confidence=0.5),
                           DetectedIngredient(name="bread", confidence=0.7)])
    assert s.name == "confidence" and s.loc["egg"] == 0.9 and list(s.index) == ["egg", "bread"]
    assert list(s[s >= 0.8].index) == ["egg"]
    assert score_series([("r1", 0.2), ("r2", 0.9)]).index[0] == "r2"


def test_unknown_tier_and_bad_equipment(pantry) -> None:
    with pytest.raises(UnsupportedTierError):
        TierFactory.create("castle", pantry)
    with pytest.raises(UnsupportedTierError):
        TierFactory.create("pg", pantry, ["oven", "gas_stove"])
    with pytest.raises(UnsupportedTierError):
        TierFactory.create("hostel", pantry, ["gas_stove"])


def test_hostel_never_gets_heat_recipes(corpus, hostel) -> None:
    rows = corpus.filter(hostel, Conditions())
    assert len(rows) >= 25
    assert all(set(corpus[i].techniques) <= HOSTEL_TECHNIQUES for i in rows)
    assert not any("fry" in corpus[i].techniques for i in rows)


def test_diet_and_time_conditions(corpus, home) -> None:
    rows = corpus.filter(home, Conditions(diet="vegan", max_time=15))
    assert rows and all(corpus[i].diet == "vegan" and corpus[i].time_min <= 15 for i in rows)


def test_tier_available_ingredients(hostel, home) -> None:
    assert hostel.available(["egg"], ["oil"]) == {"egg", "water", "salt"}
    assert "oil" in home.available(["egg"], ["oil"])


def test_normalizer_synonyms_and_plurals() -> None:
    norm = IngredientNormalizer({"aloo": "potato"}, ["tomato", "onion"])
    assert norm.normalize("Aloo") == "potato"
    assert norm.normalize("Tomatoes") == "tomato"
    assert norm.normalize("onions!") == "onion"


def test_lru_eviction_and_ttl() -> None:
    now = [0.0]
    cache = LRUCache(max_size=2, clock=lambda: now[0])
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")
    cache.set("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1
    cache.set("t", 9, ttl=10)
    now[0] = 11
    assert cache.get("t") is None


def test_tiered_cache_promotes_and_tracks_hit_rate(tmp_path) -> None:
    l2 = FileCache(tmp_path)
    l2.set("k", {"v": 1})
    tiered = TieredCache(LRUCache(), l2)
    assert tiered.get("k") == {"v": 1}
    assert tiered.get("missing") is None
    assert tiered.hit_rate == 0.5


def test_file_cache_corrupt_file_is_a_miss(tmp_path) -> None:
    cache = FileCache(tmp_path)
    cache.set("k", 1)
    for path in tmp_path.iterdir():
        path.write_text("{broken")
    assert cache.get("k") is None


def test_rate_limiter_blocks_seventh_call_then_recovers() -> None:
    now = [1000.0]
    limiter = RateLimiter(6, 600, 300, clock=lambda: now[0])
    for _ in range(6):
        limiter.check("s")
    with pytest.raises(RateLimitExceededError) as info:
        limiter.check("s")
    assert info.value.retry_after > 0
    limiter.check("other-session")
    now[0] += 601
    limiter.check("s")


def test_global_daily_limit() -> None:
    limiter = RateLimiter(10, 600, 2, clock=lambda: 5.0)
    limiter.check("a")
    limiter.check("b")
    with pytest.raises(RateLimitExceededError):
        limiter.check("c")


def _png(size=(50, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_validator_accepts_png_and_resizes() -> None:
    prepared = ImageValidator().prepare(_png((3000, 1500)))
    assert max(prepared.width, prepared.height) == 1024 and len(prepared.sha256) == 64
    assert Image.open(io.BytesIO(prepared.data)).format == "JPEG"


@pytest.mark.parametrize("raw", [b"", b"not an image", b"x" * (5 * 1024 * 1024 + 1)])
def test_image_validator_rejects_bad_input(raw: bytes) -> None:
    with pytest.raises(InvalidImageError):
        ImageValidator().prepare(raw)
