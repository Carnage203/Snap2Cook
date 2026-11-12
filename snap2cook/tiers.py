"""Kitchen tiers (Strategy pattern) and the pantry they draw staples from."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from snap2cook.exceptions import UnsupportedTierError
from snap2cook.fileio.safe_io import SafeFile
from snap2cook.models import ALL_TECHNIQUES, HOSTEL_TECHNIQUES, Conditions, Recipe

_DIET_OK = {
    "any": {"veg", "vegan", "egg", "non-veg"},
    "non-veg": {"veg", "vegan", "egg", "non-veg"},
    "egg": {"veg", "vegan", "egg"},
    "veg": {"veg", "vegan"},
    "vegan": {"vegan"},
}


@dataclass(frozen=True)
class Pantry:
    """Staples per tier and the Home 'extra supplies' checklist (data/pantry.json)."""

    staples: dict[str, tuple[str, ...]]
    extras: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> "Pantry":
        """Read the pantry file via SafeFile."""
        raw = SafeFile(path).read_json()
        return cls({k: tuple(v) for k, v in raw["staples"].items()}, tuple(raw["extras"]))


class Tier(ABC):
    """A kitchen type: decides which recipes and ingredients are allowed."""

    _equipment_choices: frozenset[str] = frozenset()

    def __init__(self, pantry: Pantry, equipment: Iterable[str] = ()) -> None:
        chosen = frozenset(equipment)
        self._validate_equipment(chosen)
        self._equipment = chosen | {"kettle"}
        self._staples = frozenset(pantry.staples[self.name])
        self._extras_checklist = pantry.extras

    @property
    @abstractmethod
    def name(self) -> str:
        """Tier id: home, pg or hostel."""

    @property
    @abstractmethod
    def max_missing(self) -> int:
        """Most non-staple ingredients the cook may be missing."""

    @property
    def allow_extra_supplies(self) -> bool:
        """Whether user-ticked extra supplies count as available."""
        return False

    @property
    def allowed_techniques(self) -> frozenset[str]:
        """Cooking techniques this kitchen can perform."""
        return frozenset(ALL_TECHNIQUES)

    @property
    def allowed_equipment(self) -> frozenset[str]:
        """Equipment the cook has (a kettle is always assumed)."""
        return self._equipment

    @property
    def staples(self) -> frozenset[str]:
        """Always-available ingredients."""
        return self._staples

    @property
    def extras_checklist(self) -> tuple[str, ...]:
        """Extra supplies offered in the UI (Home only)."""
        return self._extras_checklist if self.allow_extra_supplies else ()

    def _validate_equipment(self, chosen: frozenset[str]) -> None:
        if not chosen <= self._equipment_choices:
            raise UnsupportedTierError(
                f"Invalid equipment {sorted(chosen - self._equipment_choices)} for tier", tier=self.name)

    def available(self, detected: Iterable[str], extras: Iterable[str] = ()) -> set[str]:
        """Ingredients the cook can use: photo items + staples (+ extras if allowed)."""
        pool = set(detected) | self._staples
        if self.allow_extra_supplies:
            pool |= set(extras)
        return pool

    def accepts(self, recipe: Recipe, conditions: Conditions) -> bool:
        """Hard filter: tier, equipment, techniques, diet and time."""
        if self.name not in recipe.tiers:
            return False
        if not set(recipe.equipment_any) & self._equipment:
            return False
        if not set(recipe.techniques) <= self.allowed_techniques:
            return False
        if recipe.diet not in _DIET_OK.get(conditions.diet, _DIET_OK["any"]):
            return False
        return conditions.max_time is None or recipe.time_min <= conditions.max_time

    @abstractmethod
    def prompt_rules(self) -> str:
        """Tier rules inserted into the generation prompt."""


class HomeTier(Tier):
    """Full kitchen: microwave and/or gas stove, plus extra supplies."""

    _equipment_choices = frozenset({"microwave", "gas_stove"})
    name = "home"  # type: ignore[assignment]
    max_missing = 2  # type: ignore[assignment]
    allow_extra_supplies = True  # type: ignore[assignment]

    def prompt_rules(self) -> str:
        return ("Home kitchen. Use only the equipment listed. Extra supplies ticked by the user are "
                "allowed. Beginner friendly.")


class PGTier(Tier):
    """Paying-guest room: one of oven or gas stove, photo items only."""

    _equipment_choices = frozenset({"oven", "gas_stove"})
    name = "pg"  # type: ignore[assignment]
    max_missing = 1  # type: ignore[assignment]

    def _validate_equipment(self, chosen: frozenset[str]) -> None:
        super()._validate_equipment(chosen)
        if len(chosen) != 1:
            raise UnsupportedTierError("PG needs exactly one of oven or gas_stove", tier=self.name)

    def prompt_rules(self) -> str:
        return ("PG room with ONE appliance only. Use only photo ingredients plus water and salt; "
                "at most one missing ingredient may be listed in extra_needed.")


class HostelTier(Tier):
    """Hostel room: kettle only."""

    _equipment_choices = frozenset({"kettle"})
    name = "hostel"  # type: ignore[assignment]
    max_missing = 0  # type: ignore[assignment]

    @property
    def allowed_techniques(self) -> frozenset[str]:
        return HOSTEL_TECHNIQUES

    def prompt_rules(self) -> str:
        return ("Hostel room with an electric kettle ONLY. Allowed techniques: boil water, steep, soak, mix, "
                "pour, mash, chop. No frying, no stove, no microwave. The kettle only boils plain water that "
                "is then poured over food; never heat food inside the kettle.")


class TierFactory:
    """Registry-based factory: ``TierFactory.create('pg', pantry, ['oven'])``."""

    _registry: dict[str, Callable[..., Tier]] = {"home": HomeTier, "pg": PGTier, "hostel": HostelTier}

    @classmethod
    def names(cls) -> list[str]:
        """Registered tier ids."""
        return list(cls._registry)

    @classmethod
    def create(cls, name: str, pantry: Pantry, equipment: Iterable[str] = ()) -> Tier:
        """Build a tier by name or raise UnsupportedTierError."""
        try:
            builder = cls._registry[name]
        except KeyError as exc:
            raise UnsupportedTierError(f"Unknown tier {name!r}", tier=name) from exc
        return builder(pantry, equipment)
