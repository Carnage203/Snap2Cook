"""Gemini-backed ingredient detection."""
from __future__ import annotations

import logging

from google import genai
from google.genai import types
from pydantic import ValidationError

from snap2cook.exceptions import NoIngredientsDetectedError, VisionProviderError
from snap2cook.gemini import call_with_retries
from snap2cook.models import DetectedIngredient, DetectionResult
from snap2cook.vision.base import VisionProvider

log = logging.getLogger("snap2cook.vision")

PROMPT = (
    "List the raw food ingredients and ready-to-eat food items clearly visible in this photo. "
    "Use short, common, singular English names (e.g. 'egg', 'tomato', 'bread'). "
    "Give each a confidence between 0 and 1. Do not guess items that are not visible. "
    "Ignore packaging text unless it clearly names the food."
)


class GeminiVisionProvider(VisionProvider):
    """Calls Gemini in JSON mode with a pydantic response schema."""

    def __init__(self, client: genai.Client, model: str, retries: int = 2) -> None:
        self._client = client
        self._model = model
        self._retries = retries

    @property
    def name(self) -> str:
        return f"gemini:{self._model}"

    def _call(self, image_bytes: bytes) -> str | None:
        response = self._client.models.generate_content(
            model=self._model,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=DetectionResult, temperature=0.0),
        )
        return response.text

    def detect(self, image_bytes: bytes) -> list[DetectedIngredient]:
        """Detect ingredients; one repair retry if the JSON is invalid."""
        for attempt in (1, 2):
            text = call_with_retries(lambda: self._call(image_bytes), VisionProviderError,
                                     "Vision request", self._retries)
            try:
                result = DetectionResult.model_validate_json(text or "")
                break
            except ValidationError:
                log.warning("vision returned invalid JSON (attempt %d)", attempt)
        else:
            raise VisionProviderError("Vision returned invalid JSON twice")
        if not result.ingredients:
            raise NoIngredientsDetectedError("Vision returned an empty ingredient list")
        return result.ingredients
