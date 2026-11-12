# SYLLABUS Q9 - user-defined exceptions
"""Custom exception hierarchy for Snap2Cook.

Every domain failure derives from :class:`Snap2CookError`, which carries a
technical ``message`` (for logs), a friendly ``user_message`` (for the UI) and a
``context`` dict. The UI catches the base class once and shows ``user_message``.
"""
from __future__ import annotations

from typing import Any


class Snap2CookError(Exception):
    """Base class of all application errors."""

    default_user_message = "Something went wrong. Please try again."

    def __init__(self, message: str, user_message: str | None = None, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.user_message = user_message or self.default_user_message
        self.context: dict[str, Any] = context

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} [{details}]"


class FileOperationError(Snap2CookError):
    """A disk operation failed; ``path`` and ``op`` are kept in the context."""

    default_user_message = "We could not read or write a file."

    def __init__(self, message: str, path: object, op: str, user_message: str | None = None) -> None:
        super().__init__(message, user_message, path=str(path), op=op)
        self.path = str(path)
        self.op = op


class InvalidImageError(Snap2CookError):
    """The uploaded image is the wrong type, too large or corrupt."""

    default_user_message = "That image could not be used. Please upload a JPEG, PNG or WEBP under 5 MB."


class NoIngredientsDetectedError(Snap2CookError):
    """Vision found nothing edible in the photo."""

    default_user_message = "We could not spot any ingredients. Try a clearer photo or add them manually."


class VisionProviderError(Snap2CookError):
    """The vision API failed (down, auth, quota, timeout or bad JSON)."""

    default_user_message = "Photo analysis is unavailable right now. Please add your ingredients manually."


class EmbeddingError(Snap2CookError):
    """The embedding API failed."""

    default_user_message = "Smart search is unavailable, using simpler matching instead."


class CorpusLoadError(Snap2CookError):
    """The recipe corpus could not be fetched or failed validation."""

    default_user_message = "The recipe library could not be loaded."


class RetrievalError(Snap2CookError):
    """Recipe retrieval failed."""

    default_user_message = "We could not search the recipe library."


class GenerationError(Snap2CookError):
    """Recipe generation failed or produced an ungrounded recipe."""

    default_user_message = "We could not write a recipe this time."


class UnsupportedTierError(Snap2CookError):
    """Unknown kitchen tier or an invalid equipment choice for a tier."""

    default_user_message = "That kitchen type or equipment choice is not supported."


class RateLimitExceededError(Snap2CookError):
    """Too many generations; ``retry_after`` is the wait in seconds."""

    def __init__(self, message: str, retry_after: int) -> None:
        minutes = max(1, round(retry_after / 60))
        super().__init__(
            message,
            f"You are going a bit fast! Please wait about {minutes} minute(s) and try again.",
            retry_after=retry_after,
        )
        self.retry_after = retry_after


def demo() -> None:
    """Raise and catch each exception type (viva demo)."""
    samples: list[Snap2CookError] = [
        FileOperationError("cannot open", "missing.txt", "read"),
        InvalidImageError("bad bytes"),
        NoIngredientsDetectedError("empty list"),
        VisionProviderError("429"),
        EmbeddingError("timeout"),
        CorpusLoadError("dim mismatch"),
        RetrievalError("empty corpus"),
        GenerationError("bad JSON twice"),
        UnsupportedTierError("tier=castle"),
        RateLimitExceededError("7th call", retry_after=120),
    ]
    for err in samples:
        try:
            raise err
        except Snap2CookError as caught:
            print(f"{type(caught).__name__:28} -> {caught.user_message}")


if __name__ == "__main__":
    demo()
