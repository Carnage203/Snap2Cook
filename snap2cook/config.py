"""Application settings read from Streamlit secrets, environment variables or .env."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent


def _from_streamlit_secrets(key: str) -> str | None:
    """Return a value from ``st.secrets`` or None if unavailable."""
    try:
        import streamlit as st

        value = st.secrets.get(key)
    except Exception:  # no secrets file / not running under Streamlit
        return None
    return str(value) if value not in (None, "") else None


@dataclass(frozen=True)
class Settings:
    """Immutable configuration. The API key is hidden from ``repr``."""

    gemini_api_key: str = field(default="", repr=False)
    vision_model: str = "gemini-3.7-flash"
    gen_model: str = "gemini-3.7-flash"
    fallback_models: tuple[str, ...] = ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash")
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    hf_dataset_repo: str = ""
    data_dir: Path = ROOT / "data"
    cache_dir: Path = Path(tempfile.gettempdir()) / "snap2cook_cache"
    log_dir: Path = ROOT / "logs"
    w_similarity: float = 0.6
    w_coverage: float = 0.4
    w_missing: float = 0.1
    rate_limit_calls: int = 6
    rate_limit_window_s: int = 600
    daily_limit: int = 300
    api_timeout_s: int = 30
    api_retries: int = 2
    vision_ttl_s: int = 7 * 24 * 3600
    recipe_ttl_s: int = 24 * 3600

    @property
    def live_mode(self) -> bool:
        """True when a Gemini key is configured."""
        return bool(self.gemini_api_key)

    @classmethod
    def load(cls, overrides: Mapping[str, str] | None = None) -> "Settings":
        """Build settings: st.secrets, then os.environ, then .env (local only)."""
        dotenv = dotenv_values(ROOT / ".env")

        def get(key: str, default: str = "") -> str:
            if overrides and overrides.get(key):
                return str(overrides[key])
            for source in (_from_streamlit_secrets(key), os.environ.get(key), dotenv.get(key)):
                if source:
                    return str(source)
            return default

        defaults = cls()
        return cls(
            gemini_api_key=get("GEMINI_API_KEY"),
            vision_model=get("VISION_MODEL", defaults.vision_model),
            gen_model=get("GEN_MODEL", defaults.gen_model),
            fallback_models=tuple(m.strip() for m in get("FALLBACK_MODELS", ",".join(defaults.fallback_models)).split(",") if m.strip()),
            embed_model=get("EMBED_MODEL", defaults.embed_model),
            embed_dim=int(get("EMBED_DIM", str(defaults.embed_dim))),
            hf_dataset_repo=get("HF_DATASET_REPO"),
        )
