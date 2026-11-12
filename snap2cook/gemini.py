"""Shared helpers for calling the Gemini API: client factory and retry policy."""
from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

from google import genai
from google.genai import types

from snap2cook.config import Settings
from snap2cook.exceptions import Snap2CookError

log = logging.getLogger("snap2cook.gemini")
T = TypeVar("T")

_FATAL_CODES = {400, 401, 403, 404}


def build_client(settings: Settings) -> genai.Client:
    """Create a Gemini client with the configured timeout (milliseconds in the SDK)."""
    return genai.Client(api_key=settings.gemini_api_key,
                        http_options=types.HttpOptions(timeout=settings.api_timeout_s * 1000))


def call_with_retries(call: Callable[[], T], error_cls: type[Snap2CookError], what: str,
                      retries: int = 2, sleep: Callable[[float], None] = time.sleep) -> T:
    """Run ``call`` with exponential backoff + jitter, mapping failures to ``error_cls``.

    Quota (429) and permanent client errors are not retried.
    """
    for attempt in range(retries + 1):
        try:
            return call()
        except Snap2CookError:
            raise
        except Exception as exc:  # SDK raises many types; all are mapped here
            code = getattr(exc, "code", None)
            if code == 429 or "RESOURCE_EXHAUSTED" in str(exc):
                raise error_cls(f"{what}: quota exceeded",
                                "The AI service is busy (free quota reached). Please try again in a minute.") from exc
            if code in _FATAL_CODES:
                raise error_cls(f"{what}: rejected with HTTP {code}",
                                "The AI service rejected the request. Check the API key and model names.") from exc
            if attempt == retries:
                raise error_cls(f"{what} failed after {retries + 1} attempts: {type(exc).__name__}") from exc
            delay = 2 ** attempt + random.random()
            log.warning("%s failed (%s), retrying in %.1fs", what, type(exc).__name__, delay)
            sleep(delay)
    raise error_cls(f"{what} failed")  # unreachable; satisfies type checkers


def model_chain(primary: str, fallbacks: tuple[str, ...]) -> tuple[str, ...]:
    """Primary model first, then the fallbacks, without duplicates."""
    return tuple(dict.fromkeys(m for m in (primary, *fallbacks) if m))


def call_with_model_fallback(models: tuple[str, ...], call: Callable[[str], T], error_cls: type[Snap2CookError],
                             what: str, retries: int = 2) -> T:
    """Try each model in order; move to the next one when a model fails (overload, quota, bad name).

    Models before the last get no retries (switching model is the retry); the last gets the full policy.
    """
    last_error: Snap2CookError | None = None
    for index, model in enumerate(models):
        last = index == len(models) - 1
        try:
            return call_with_retries(lambda: call(model), error_cls, f"{what} [{model}]", retries if last else 0)
        except error_cls as exc:
            last_error = exc
            if not last:
                log.warning("%s failed on %s, trying %s", what, model, models[index + 1])
    assert last_error is not None
    raise last_error
