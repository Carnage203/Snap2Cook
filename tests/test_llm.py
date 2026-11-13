"""Groq fallback: chain ordering, JSON extraction, and dispatch inside the model loop."""
from snap2cook.gemini import call_with_model_fallback, model_chain
from snap2cook.exceptions import GenerationError
from snap2cook.llm import ChatHTTPError, extract_json, is_groq


def test_chain_puts_groq_right_after_primary() -> None:
    chain = model_chain("g-main", ("g-b", "g-c"), ("openai/gpt-oss-20b", "qwen/qwen3.8-27b"))
    assert chain == ("g-main", "groq:openai/gpt-oss-20b", "groq:qwen/qwen3.8-27b", "g-b", "g-c")
    assert [is_groq(m) for m in chain] == [False, True, True, False, False]


def test_extract_json_strips_think_and_prose() -> None:
    assert extract_json('<think>hmm {x}</think> Sure: {"a": 1} done') == '{"a": 1}'


def test_failing_gemini_falls_through_to_groq() -> None:
    def call(model: str) -> str:
        if not is_groq(model):
            raise ChatHTTPError(503, "overloaded")
        return "ok:" + model
    chain = model_chain("g-main", (), ("m1",))
    assert call_with_model_fallback(chain, call, GenerationError, "test") == "ok:groq:m1"


def test_failed_model_is_skipped_during_cooldown() -> None:
    calls: list[str] = []

    def call(model: str) -> str:
        calls.append(model)
        if model == "flaky":
            raise ChatHTTPError(503, "overloaded")
        return "ok"
    chain = ("flaky", "groq:m1")
    call_with_model_fallback(chain, call, GenerationError, "t")
    call_with_model_fallback(chain, call, GenerationError, "t")
    assert calls == ["flaky", "groq:m1", "groq:m1"]


def test_settings_accept_both_openrouter_key_spellings() -> None:
    from snap2cook.config import Settings
    assert Settings.load({"Open_Router_API_KEY": "k1"}).openrouter_api_key == "k1"
    assert Settings.load({"OPENROUTER_API_KEY": "k2"}).openrouter_api_key == "k2"


def test_vision_chain_routes_by_prefix_and_falls_back() -> None:
    from snap2cook.llm import ChatClient, GROQ_PREFIX, OPENROUTER_PREFIX
    from snap2cook.vision.gemini_provider import GeminiVisionProvider

    class Fake(ChatClient):
        def __init__(self, prefix: str, fail: bool) -> None:
            self.prefix, self._fail = prefix, fail

        def chat_json(self, model: str, prompt: str, image_jpeg=None, temperature=0.3, max_tokens=None) -> str:
            if self._fail:
                raise ChatHTTPError(503, "down")
            return '{"ingredients": [{"name": "egg", "confidence": 0.9}]}'

    clients = {GROQ_PREFIX: Fake(GROQ_PREFIX, True), OPENROUTER_PREFIX: Fake(OPENROUTER_PREFIX, False)}
    provider = GeminiVisionProvider(None, ("groq:fast", "or:backup"), 0, clients)  # type: ignore[arg-type]
    assert [i.name for i in provider.detect(b"img")] == ["egg"]
