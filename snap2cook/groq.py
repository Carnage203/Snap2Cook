"""Groq (OpenAI-compatible REST) client used as a low-latency fallback after Gemini.

Models are addressed in a shared chain as ``groq:<model id>`` so the same fallback loop
that handles Gemini models can switch providers.
"""
from __future__ import annotations

import base64
import re
from typing import Any

import requests

GROQ_PREFIX = "groq:"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class GroqHTTPError(Exception):
    """HTTP failure from Groq; ``code`` lets the shared retry policy classify it."""

    def __init__(self, code: int, detail: str) -> None:
        super().__init__(f"Groq HTTP {code}: {detail}")
        self.code = code


def is_groq(model: str) -> bool:
    """True when a chain entry refers to a Groq model."""
    return model.startswith(GROQ_PREFIX)


def extract_json(text: str) -> str:
    """Strip reasoning tags and surrounding prose, returning the outermost JSON object."""
    text = _THINK.sub("", text)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text


class GroqClient:
    """Minimal chat-completions client in JSON mode."""

    def __init__(self, api_key: str, timeout_s: int = 20) -> None:
        self._api_key = api_key
        self._timeout = timeout_s

    def chat_json(self, model: str, prompt: str, image_jpeg: bytes | None = None,
                  temperature: float = 0.3) -> str:
        """Send one prompt (optionally with a JPEG) and return the JSON text of the reply."""
        model_id = model.removeprefix(GROQ_PREFIX)
        content: Any = prompt
        if image_jpeg is not None:
            data_url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode()
            content = [{"type": "text", "text": prompt},
                       {"type": "image_url", "image_url": {"url": data_url}}]
        body: dict[str, Any] = {
            "model": model_id, "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
        }
        if "gpt-oss" in model_id:
            body["reasoning_effort"] = "low"
        elif "qwen" in model_id:
            body["reasoning_format"] = "hidden"
        response = requests.post(GROQ_URL, json=body, timeout=self._timeout,
                                 headers={"Authorization": f"Bearer {self._api_key}"})
        if response.status_code != 200:
            raise GroqHTTPError(response.status_code, response.text[:200])
        return extract_json(response.json()["choices"][0]["message"]["content"] or "")
