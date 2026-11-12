"""Pydantic domain models shared across the app."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, field_validator, model_validator

HOSTEL_TECHNIQUES = frozenset({"boil", "steep", "soak", "mix", "pour", "mash", "chop"})
ALL_TECHNIQUES = HOSTEL_TECHNIQUES | {"fry", "saute", "simmer", "bake", "roast", "toast", "microwave", "steam"}
DIETS = ("any", "veg", "vegan", "egg", "non-veg")

TierName = Literal["hostel", "pg", "home"]
Equipment = Literal["kettle", "gas_stove", "microwave", "oven"]


class RecipeIngredient(BaseModel):
    """One ingredient line of a corpus recipe."""

    name: str
    required: bool = True


class Recipe(BaseModel):
    """A validated corpus recipe (one row of recipes.jsonl)."""

    id: str
    title: str
    tiers: list[TierName]
    equipment_any: list[Equipment]
    ingredients: list[RecipeIngredient]
    techniques: list[str]
    time_min: int
    calories: int
    cuisine: str
    diet: Literal["veg", "vegan", "egg", "non-veg"]
    difficulty: Literal["easy", "medium"]
    steps: list[str]
    tips: str
    doc_text: str

    @model_validator(mode="after")
    def _check_techniques(self) -> "Recipe":
        unknown = set(self.techniques) - ALL_TECHNIQUES
        if unknown:
            raise ValueError(f"unknown techniques {sorted(unknown)} in {self.id}")
        if "hostel" in self.tiers and not set(self.techniques) <= HOSTEL_TECHNIQUES:
            raise ValueError(f"hostel recipe {self.id} uses non-kettle techniques")
        if not self.steps:
            raise ValueError(f"recipe {self.id} has no steps")
        return self

    @property
    def required_names(self) -> list[str]:
        """Names of required ingredients."""
        return [i.name for i in self.ingredients if i.required]

    @property
    def ingredient_text(self) -> str:
        """Comma separated ingredient names (used for lexical matching)."""
        return ", ".join(i.name for i in self.ingredients)

    def __str__(self) -> str:
        lines = [f"{self.title} ({self.time_min} min)", "Ingredients: " + self.ingredient_text]
        lines += [f"{n}. {s}" for n, s in enumerate(self.steps, 1)]
        return "\n".join(lines)


class DetectedIngredient(BaseModel):
    """An ingredient spotted in a photo with a confidence in [0, 1]."""

    name: str
    confidence: float = 0.5

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return min(1.0, max(0.0, value))

    def __str__(self) -> str:
        return f"{self.name} ({self.confidence:.0%})"


class DetectionResult(BaseModel):
    """JSON schema wrapper for the vision response."""

    ingredients: list[DetectedIngredient]


class GeneratedRecipe(BaseModel):
    """The recipe shown to the user (generated or copied from the corpus)."""

    title: str
    time_min: int
    ingredients_used: list[str]
    extra_needed: list[str]
    techniques_used: list[str]
    steps: list[str]
    substitutions: list[str]
    safety_note: str
    source_recipe_ids: list[str]

    def __str__(self) -> str:
        lines = [self.title.upper(), f"Time: {self.time_min} minutes", "",
                 "You will use: " + ", ".join(self.ingredients_used)]
        if self.extra_needed:
            lines.append("You also need: " + ", ".join(self.extra_needed))
        lines += ["", "Steps:"] + [f"{n}. {s}" for n, s in enumerate(self.steps, 1)]
        if self.substitutions:
            lines += ["", "Substitutions:"] + [f"- {s}" for s in self.substitutions]
        lines += ["", f"Safety: {self.safety_note}", "Based on: " + ", ".join(self.source_recipe_ids)]
        return "\n".join(lines)


@dataclass(frozen=True)
class Conditions:
    """Optional user conditions applied as hard filters."""

    diet: str = "any"
    max_time: int | None = None


@dataclass
class PipelineResult:
    """Everything the UI needs after a run."""

    recipe: GeneratedRecipe | None
    ingredients: pd.Series
    scores: pd.Series
    timings: dict[str, float]
    cache_hits: dict[str, bool]
    fallback_used: bool
    suggestions: list[str]
    candidates: list[dict[str, Any]]
    mode: str

    def to_cache(self) -> dict[str, Any]:
        """JSON-safe form stored in the recipe cache."""
        return {
            "recipe": self.recipe.model_dump() if self.recipe else None,
            "scores": self.scores.to_dict(),
            "fallback_used": self.fallback_used,
            "suggestions": self.suggestions,
            "candidates": self.candidates,
            "mode": self.mode,
        }
