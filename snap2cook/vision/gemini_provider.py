"""Gemini-backed ingredient detection."""
from __future__ import annotations

import logging

from google import genai
from google.genai import types
from pydantic import ValidationError

from snap2cook.exceptions import NoIngredientsDetectedError, VisionProviderError
from snap2cook.gemini import call_with_model_fallback
from snap2cook.llm import GROQ_PREFIX, OPENROUTER_PREFIX, ChatClient
from snap2cook.models import DetectedIngredient, DetectionResult
from snap2cook.vision.base import VisionProvider

log = logging.getLogger("snap2cook.vision")
JSON_INSTRUCTION = ('Reply with ONLY one JSON object like {"ingredients": [{"name": "egg", "confidence": 0.9}]} '
                    "using exactly those keys.")
FAST_THINKING = types.ThinkingConfig(thinking_level="low")  # naming foods needs no deep reasoning

PROMPT = (
    "List the raw food ingredients and ready-to-eat food items clearly visible in this photo. "
    "Use short, common, singular English names (e.g. 'egg', 'tomato', 'bread'). "
    "Give each a confidence between 0 and 1. Do not guess items that are not visible. "
    "Ignore packaging text unless it clearly names the food."
)


class GeminiVisionProvider(VisionProvider):
    """Detects ingredients along a model chain: Gemini directly, Groq/OpenRouter via ``chat_clients``."""

    def __init__(self, client: genai.Client, models: tuple[str, ...], retries: int = 2,
                 chat_clients: dict[str, ChatClient] | None = None) -> None:
        self._client = client
        self._chat = chat_clients or {}
        self._models = models
        self._retries = retries

    @property
    def name(self) -> str:
        return f"vision:{self._models[0]}"

    def _call(self, model: str, image_bytes: bytes) -> str | None:
        for prefix, chat in self._chat.items():
            if model.startswith(prefix):
                return chat.chat_json(model, PROMPT + " " + JSON_INSTRUCTION, image_jpeg=image_bytes,
                                      temperature=0.0, max_tokens=400)
        if model.startswith((GROQ_PREFIX, OPENROUTER_PREFIX)):
            raise VisionProviderError(f"{model} is in the chain but its provider has no API key")
        response = self._client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=DetectionResult, temperature=0.0,
                thinking_config=FAST_THINKING if model.startswith("gemini-4") else None),
        )
        return response.text

    def detect(self, image_bytes: bytes) -> list[DetectedIngredient]:
        """Detect ingredients; one repair retry if the JSON is invalid."""
        for attempt in (1, 2):
            text = call_with_model_fallback(
                self._models, lambda model: self._call(model, image_bytes), VisionProviderError,
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
