"""Snap2Cook Streamlit UI. Presentation only: all logic lives in the snap2cook package."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from snap2cook.bootstrap import App, build_app
from snap2cook.config import Settings
from snap2cook.exceptions import Snap2CookError
from snap2cook.logging_setup import new_request_id, setup_logging
from snap2cook.models import Conditions, PipelineResult
from snap2cook.tiers import Tier, TierFactory
from snap2cook.vision.image_utils import ImageValidator

log = logging.getLogger("snap2cook.app")

st.set_page_config(page_title="Snap2Cook", page_icon="🍳", layout="centered")

TIER_LABELS = {"hostel": "Hostel (kettle only)", "pg": "PG (one appliance)", "home": "Home (full kitchen)"}
TIME_OPTIONS = [10, 15, 20, 30, 45, 60, 90, "Any"]


def code_version() -> str:
    """Changes whenever a source file changes, so a redeploy never reuses objects built by old code."""
    root = Path(__file__).parent
    return str(max(f.stat().st_mtime_ns for f in [Path(__file__), *(root / "snap2cook").rglob("*.py")]))


@st.cache_resource(show_spinner="Loading recipe library...")
def get_app(version: str) -> App:
    """Build the long-lived objects once per server process (and once per code version)."""
    settings = Settings.load()
    setup_logging(settings)
    return build_app(settings)


def sidebar(app: App) -> tuple[Tier, Conditions, list[str]]:
    """Render the sidebar and return the chosen tier, conditions and extra supplies."""
    st.sidebar.title("🍳 Snap2Cook")
    st.sidebar.caption("Mode: **" + ("Live (Gemini)" if app.settings.live_mode else "Mock (no API key)") + "**")
    tier_name = st.sidebar.radio("Your kitchen", TierFactory.names(), format_func=TIER_LABELS.get)
    equipment: list[str] = []
    extras: list[str] = []
    if tier_name == "home":
        equipment = st.sidebar.multiselect("Equipment", ["gas_stove", "microwave"], default=["gas_stove"])
        extras = st.sidebar.multiselect("Extra supplies you have", list(app.pantry.extras))
    elif tier_name == "pg":
        equipment = [st.sidebar.radio("Appliance (pick one)", ["gas_stove", "oven"])]
    diet = st.sidebar.selectbox("Diet", ["any", "veg", "vegan", "egg", "non-veg"])
    limit = st.sidebar.select_slider("Max cooking time (min)", TIME_OPTIONS, value="Any")
    st.sidebar.caption(f"Cache hit rate: {app.cache.hit_rate:.0%} | Corpus: {app.corpus_source}")
    tier = TierFactory.create(tier_name, app.pantry, equipment)
    return tier, Conditions(diet, None if limit == "Any" else int(limit)), extras


def detect_from_photo(app: App) -> None:
    """Handle upload/camera input (up to max_photos photos) and store detected ingredients."""
    limit = app.settings.max_photos
    source = st.radio("Photo source", ["Upload", "Camera"], horizontal=True)
    if source == "Upload":
        photos = st.file_uploader(f"Photos of your ingredients (up to {limit}, one item per photo works well)",
                                  type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    else:
        single = st.camera_input("Take a photo")
        photos = [single] if single is not None else []
    if len(photos) > limit:
        st.warning(f"Only the first {limit} photos are used.")
        photos = photos[:limit]
    if not photos:
        return
    try:
        images = [ImageValidator().prepare(photo.getvalue()) for photo in photos]
        if len(images) == 1:
            st.image(images[0].data, width="stretch")
        else:
            st.image([image.data for image in images], width=150)
        key = "|".join(sorted(image.sha256 for image in images))
        if st.session_state.get("image_hash") != key:
            with st.spinner("Looking at your ingredients..."):
                series, empty = app.pipeline.detect_many(images)
            st.session_state.update(image_hash=key, detected=series)
            if empty:
                st.info(f"{empty} photo(s) showed no recognisable food and were skipped.")
    except Snap2CookError as exc:
        log.warning("UI error: %s", exc.message)
        st.session_state["image_hash"] = None
        st.warning(exc.user_message)


def ingredient_editor(app: App) -> list[str]:
    """Editable ingredient table (vision is fallible) plus a manual picker."""
    detected: pd.Series = st.session_state.get("detected", pd.Series(dtype=float, name="confidence"))
    frame = pd.DataFrame({"ingredient": detected.index, "confidence": detected.values})
    edited = st.data_editor(
        frame, num_rows="dynamic", width="stretch", key=f"editor_{st.session_state.get('image_hash')}",
        column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0)})
    manual = st.multiselect("Add more ingredients", app.vocabulary)
    names = [str(n) for n in edited["ingredient"].dropna()] + manual
    return [n for n in names if n.strip()]


def show_result(result: PipelineResult) -> None:
    """Render the recipe card or the "nothing to make" message."""
    recipe = result.recipe
    if recipe is None:
        st.info("You can't make a proper dish with this kitchen and these items yet.")
        if result.suggestions:
            st.write("**Add one of these and try again:** " + ", ".join(result.suggestions))
    else:
        st.subheader(recipe.title)
        st.caption(f"⏱ {recipe.time_min} min | uses: {', '.join(recipe.ingredients_used)}")
        if recipe.extra_needed:
            st.warning("You will also need: " + ", ".join(recipe.extra_needed))
        for number, step in enumerate(recipe.steps, 1):
            st.markdown(f"**{number}.** {step}")
        if recipe.substitutions:
            st.markdown("**Substitutions / tips**\n" + "\n".join(f"- {s}" for s in recipe.substitutions))
        st.warning("Safety: " + recipe.safety_note, icon="⚠️")
        st.caption("Based on recipes: " + ", ".join(recipe.source_recipe_ids))
        st.download_button("Download recipe (.txt)", str(recipe), file_name="snap2cook_recipe.txt")


def cook_section(app: App, tier: Tier, conditions: Conditions, extras: list[str]) -> None:
    """The main flow: photo -> editable list -> recipe."""
    detect_from_photo(app)
    names = ingredient_editor(app)
    if st.button("Get recipe", type="primary", disabled=not names):
        new_request_id()
        try:
            with st.spinner("Cooking something up..."):
                result = app.pipeline.recommend(names, tier, conditions, extras, st.session_state["sid"])
            st.session_state["result"] = result
        except Snap2CookError as exc:
            log.warning("UI error: %s", exc.message)
            st.session_state.pop("result", None)
            st.error(exc.user_message)
    if "result" in st.session_state:
        show_result(st.session_state["result"])


def main() -> None:
    """App entry point."""
    st.session_state.setdefault("sid", uuid.uuid4().hex)
    try:
        app = get_app(code_version())
        tier, conditions, extras = sidebar(app)
    except Snap2CookError as exc:
        log.warning("UI error: %s", exc.message)
        st.error(exc.user_message)
        st.stop()
    st.title("Snap2Cook")
    st.caption("Photo of your ingredients in, a recipe that fits your kitchen out.")
    cook_section(app, tier, conditions, extras)


main()
