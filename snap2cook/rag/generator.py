"""Recipe generation grounded on retrieved recipes, with validation and fallback."""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from google import genai
from google.genai import types
from pydantic import ValidationError

from snap2cook.exceptions import GenerationError
from snap2cook.gemini import call_with_model_fallback
from snap2cook.llm import GroqClient, is_groq
from snap2cook.models import Conditions, GeneratedRecipe, Recipe
from snap2cook.rag.reranker import ScoredRecipe
from snap2cook.tiers import Tier

log = logging.getLogger("snap2cook.generator")
JSON_EXAMPLE = json.dumps({
    "title": "Recipe name", "time_min": 15, "ingredients_used": ["egg"], "extra_needed": [],
    "techniques_used": ["mix"], "steps": ["First step.", "Second step."],
    "substitutions": ["Optional swap or tip."], "safety_note": "Short safety advice.",
    "source_recipe_ids": ["r001"]})
JSON_INSTRUCTION = ("Reply with ONLY one JSON object using exactly these keys and value types "
                    "(replace the example values): " + JSON_EXAMPLE)

SAFETY_DEFAULT = ("Be careful with hot water and hot surfaces. Cook eggs and meat all the way through, "
                  "and check ingredients for allergens such as nuts and dairy.")
HOSTEL_SAFETY = ("The kettle only boils plain water. Pour the hot water over the food and never heat food "
                 "inside the kettle. Handle hot water carefully.")


@dataclass(frozen=True)
class GenerationContext:
    """Everything a generator may use; nothing else is allowed into the prompt."""

    tier: Tier
    conditions: Conditions
    available: frozenset[str]
    candidates: tuple[ScoredRecipe, ...]


def validate_grounded(recipe: GeneratedRecipe, ctx: GenerationContext) -> list[str]:
    """Return a list of rule violations (empty means the recipe is grounded)."""
    problems: list[str] = []
    known_ids = {c.recipe.id for c in ctx.candidates}
    corpus_names = {i.name for c in ctx.candidates for i in c.recipe.ingredients}
    if not recipe.source_recipe_ids or not set(recipe.source_recipe_ids) <= known_ids:
        problems.append(f"source_recipe_ids must be a non-empty subset of {sorted(known_ids)}")
    if len(recipe.extra_needed) > ctx.tier.max_missing:
        problems.append(f"extra_needed may have at most {ctx.tier.max_missing} item(s)")
    if not set(recipe.extra_needed) <= corpus_names - ctx.available:
        problems.append("extra_needed may only contain ingredients from the source recipes that are not available")
    invented = set(recipe.ingredients_used) - ctx.available - set(recipe.extra_needed)
    if invented:
        problems.append(f"ingredients not allowed: {sorted(invented)}")
    bad_tech = set(recipe.techniques_used) - ctx.tier.allowed_techniques
    if bad_tech:
        problems.append(f"techniques not allowed in a {ctx.tier.name} kitchen: {sorted(bad_tech)}")
    if not recipe.steps or not recipe.safety_note.strip():
        problems.append("steps and safety_note are required")
    return problems


class Generator(ABC):
    """Produces a GeneratedRecipe from a GenerationContext."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name shown in the UI."""

    @abstractmethod
    def generate(self, ctx: GenerationContext) -> GeneratedRecipe:
        """Return a grounded recipe or raise GenerationError."""


def recipe_to_generated(scored: ScoredRecipe, ctx: GenerationContext) -> GeneratedRecipe:
    """Copy a corpus recipe verbatim into the display model."""
    recipe: Recipe = scored.recipe
    names = [i.name for i in recipe.ingredients]
    safety = HOSTEL_SAFETY if ctx.tier.name == "hostel" else SAFETY_DEFAULT
    return GeneratedRecipe(
        title=recipe.title, time_min=recipe.time_min,
        ingredients_used=[n for n in names if n in ctx.available],
        extra_needed=list(scored.missing), techniques_used=list(recipe.techniques),
        steps=list(recipe.steps), substitutions=[recipe.tips] if recipe.tips else [],
        safety_note=safety, source_recipe_ids=[recipe.id])


class VerbatimGenerator(Generator):
    """No-LLM generator: returns the best retrieved recipe unchanged."""

    @property
    def name(self) -> str:
        return "verbatim"

    def generate(self, ctx: GenerationContext) -> GeneratedRecipe:
        if not ctx.candidates:
            raise GenerationError("No candidate recipe to copy")
        return recipe_to_generated(ctx.candidates[0], ctx)


class GeminiGenerator(Generator):
    """Gemini in JSON mode, grounded on the retrieved recipes, with one repair retry."""

    def __init__(self, client: genai.Client, models: tuple[str, ...], retries: int = 2,
                 groq: GroqClient | None = None) -> None:
        self._client, self._models, self._retries, self._groq = client, models, retries, groq

    @property
    def name(self) -> str:
        return f"gemini:{self._models[0]}"

    @staticmethod
    def build_prompt(ctx: GenerationContext, problems: list[str] | None = None) -> str:
        """Prompt containing ONLY retrieved recipes, tier rules and the allowed ingredients."""
        sources = "\n\n".join(f"[{c.recipe.id}]\n{c.recipe}" for c in ctx.candidates)
        prompt = (
            "You write one beginner-friendly recipe for a student who cannot cook.\n"
            f"RULES: {ctx.tier.prompt_rules()}\n"
            f"ALLOWED INGREDIENTS (use nothing else): {', '.join(sorted(ctx.available))}\n"
            f"ALLOWED TECHNIQUES: {', '.join(sorted(ctx.tier.allowed_techniques))}\n"
            f"Diet: {ctx.conditions.diet}. Max time: {ctx.conditions.max_time or 'no limit'} minutes.\n"
            "Base the recipe on the SOURCE RECIPES below and list the ids you used in source_recipe_ids. "
            "Put any needed-but-missing ingredient in extra_needed (never invent ingredients). "
            "Always include a short safety_note.\n\nSOURCE RECIPES:\n" + sources)
        if problems:
            prompt += "\n\nYour previous answer broke these rules, fix them:\n- " + "\n- ".join(problems)
        return prompt

    def _call(self, model: str, prompt: str) -> str | None:
        if is_groq(model):
            if self._groq is None:
                raise GenerationError("Groq model in chain but no Groq key configured")
            return self._groq.chat_json(model, prompt + "\n\n" + JSON_INSTRUCTION)
        response = self._client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                               response_schema=GeneratedRecipe, temperature=0.4))
        return response.text

    def generate(self, ctx: GenerationContext) -> GeneratedRecipe:
        problems: list[str] = []
        for attempt in (1, 2):
            text = call_with_model_fallback(
                self._models, lambda model: self._call(model, self.build_prompt(ctx, problems)),
                GenerationError, "Generation request", self._retries)
            try:
                recipe = GeneratedRecipe.model_validate_json(text or "")
            except ValidationError:
                problems = ["Return valid JSON matching the schema."]
                log.warning("generation JSON invalid (attempt %d)", attempt)
                continue
            problems = validate_grounded(recipe, ctx)
            if not problems:
                return recipe
            log.warning("generation rejected (attempt %d): %s", attempt, problems)
        raise GenerationError("Generation failed validation twice", problems=problems)
