"""OpenAI-compatible chat clients (Groq, OpenRouter) used as fast fallbacks next to Gemini.

Models are addressed in a shared chain as ``<prefix><model id>`` (``groq:`` / ``or:``) so the same
fallback loop that handles Gemini models can switch providers.
"""
from __future__ import annotations

import base64
import re
from typing import Any

import requests

GROQ_PREFIX = "groq:"
OPENROUTER_PREFIX = "or:"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class ChatHTTPError(Exception):
    """HTTP failure from a chat provider; ``code`` lets the shared retry policy classify it."""

    def __init__(self, code: int, detail: str) -> None:
        super().__init__(f"HTTP {code}: {detail}")
        self.code = code


def is_groq(model: str) -> bool:
    """True when a chain entry refers to a Groq model."""
    return model.startswith(GROQ_PREFIX)


def is_openrouter(model: str) -> bool:
    """True when a chain entry refers to an OpenRouter model."""
    return model.startswith(OPENROUTER_PREFIX)


def extract_json(text: str) -> str:
    """Strip reasoning tags and surrounding prose, returning the outermost JSON object."""
    text = _THINK.sub("", text)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text


class ChatClient:
    """Minimal chat-completions client in JSON mode; subclasses add provider-specific options."""

    url = ""
    prefix = ""

    def __init__(self, api_key: str, timeout_s: int = 20) -> None:
        self._api_key = api_key
        self._timeout = timeout_s

    def _extra_options(self, model_id: str) -> dict[str, Any]:
        """Provider/model specific request fields."""
        return {}

    def chat_json(self, model: str, prompt: str, image_jpeg: bytes | None = None,
                  temperature: float = 0.3, max_tokens: int | None = None) -> str:
        """Send one prompt (optionally with a JPEG) and return the JSON text of the reply."""
        model_id = model.removeprefix(self.prefix)
        content: Any = prompt
        if image_jpeg is not None:
            data_url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode()
            content = [{"type": "text", "text": prompt},
                       {"type": "image_url", "image_url": {"url": data_url}}]
        body: dict[str, Any] = {
            "model": model_id, "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"}, **self._extra_options(model_id),
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        response = requests.post(self.url, json=body, timeout=self._timeout,
                                 headers={"Authorization": f"Bearer {self._api_key}"})
        if response.status_code != 200:
            raise ChatHTTPError(response.status_code, response.text[:200])
        return extract_json(response.json()["choices"][0]["message"]["content"] or "")


class GroqClient(ChatClient):
    """Groq: very low latency; reasoning kept short."""

    url, prefix = GROQ_URL, GROQ_PREFIX

    def _extra_options(self, model_id: str) -> dict[str, Any]:
        if "gpt-oss" in model_id:
            return {"reasoning_effort": "low"}
        if "qwen" in model_id:
            return {"reasoning_format": "hidden"}
        return {}


class OpenRouterClient(ChatClient):
    """OpenRouter: access to many providers through one key."""

    url, prefix = OPENROUTER_URL, OPENROUTER_PREFIX

    def _extra_options(self, model_id: str) -> dict[str, Any]:
        return {"reasoning": {"effort": "low"}}
