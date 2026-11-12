"""Snap2Cook Streamlit UI. Presentation only: all logic lives in the snap2cook package."""
from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st

from snap2cook.bootstrap import App, build_app
from snap2cook.config import Settings
from snap2cook.exceptions import Snap2CookError
from snap2cook.fileio.text_tools import FileTextProcessor
from snap2cook.logging_setup import new_request_id, setup_logging
from snap2cook.models import Conditions, PipelineResult
from snap2cook.tiers import Tier, TierFactory
from snap2cook.vision.image_utils import ImageValidator

st.set_page_config(page_title="Snap2Cook", page_icon="🍳", layout="centered")

TIER_LABELS = {"hostel": "Hostel (kettle only)", "pg": "PG (one appliance)", "home": "Home (full kitchen)"}
TIME_OPTIONS = [10, 15, 20, 30, 45, 60, 90, "Any"]


@st.cache_resource(show_spinner="Loading recipe library...")
def get_app() -> App:
    """Build the long-lived objects once per server process."""
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
    """Handle upload/camera input and store detected ingredients in session state."""
    source = st.radio("Photo source", ["Upload", "Camera"], horizontal=True)
    photo = st.file_uploader("Photo of your ingredients", type=["jpg", "jpeg", "png", "webp"]) \
        if source == "Upload" else st.camera_input("Take a photo")
    if photo is None:
        return
    try:
        image = ImageValidator().prepare(photo.getvalue())
        st.image(image.data, width="stretch")
        if st.session_state.get("image_hash") != image.sha256:
            with st.spinner("Looking at your ingredients..."):
                series, hit = app.pipeline.detect(image)
            st.session_state.update(image_hash=image.sha256, detected=series, detect_cached=hit)
    except Snap2CookError as exc:
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
    """Render the recipe card and the 'how this was made' expander."""
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
    with st.expander("How this was made"):
        st.write({"mode": result.mode, "fallback_used": result.fallback_used, "cache_hits": result.cache_hits,
                  "timings_s": {k: round(v, 3) for k, v in result.timings.items()}})
        st.write("Ingredients (pandas Series)")
        st.dataframe(result.ingredients)
        if result.candidates:
            st.write("Retrieved candidates")
            st.dataframe(pd.DataFrame(result.candidates).set_index("id"))


def cook_tab(app: App, tier: Tier, conditions: Conditions, extras: list[str]) -> None:
    """The main flow: photo -> editable list -> recipe."""
    detect_from_photo(app)
    names = ingredient_editor(app)
    if st.button("Get recipe", type="primary", disabled=not names):
        new_request_id()
        try:
            with st.spinner("Cooking something up..."):
                result = app.pipeline.recommend(names, tier, conditions, extras, st.session_state["sid"])
            st.session_state["result"] = result
            st.session_state["recipe_text"] = str(result.recipe) if result.recipe else ""
        except Snap2CookError as exc:
            st.session_state.pop("result", None)
            st.error(exc.user_message)
    if "result" in st.session_state:
        show_result(st.session_state["result"])


def text_tools_tab() -> None:
    """Syllabus Q6 on an uploaded .txt or the last recipe."""
    st.write("Run the four text operations on a `.txt` file or on your last recipe.")
    upload = st.file_uploader("Upload a .txt file", type=["txt"])
    text = st.session_state.get("recipe_text", "")
    if upload is not None:
        try:
            text = upload.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.error("That file is not readable UTF-8 text.")
            return
    text = st.text_area("Text", value=text, height=150)
    proc = FileTextProcessor(text)
    actions = {"Reverse each word": proc.reverse_each_word, "Reverse word order per line": proc.reverse_word_order_per_line,
               "Reverse line order": proc.reverse_line_order, "Remove whitespace": proc.strip_whitespace}
    for label, action in actions.items():
        if st.button(label):
            st.code(action() or "(empty)")


def about_tab() -> None:
    """Map syllabus questions and OOP concepts to code (helps the viva)."""
    st.table(pd.DataFrame([
        ("Q6 File text processing", "fileio/text_tools.py - FileTextProcessor", "Text Tools tab"),
        ("Q8 File error handling", "fileio/safe_io.py - SafeFile", "corpus, cache, logs, config files"),
        ("Q9 User-defined exceptions", "exceptions.py", "every error message you see"),
        ("Q10 Pandas Series", "analytics/series.py", "ingredient confidences and scores"),
        ("OOP: Strategy + Factory", "tiers.py - Tier, TierFactory", "sidebar kitchen choice"),
        ("OOP: ABCs / polymorphism", "VisionProvider, Embedder, CorpusStore, Generator", "Live vs Mock mode"),
        ("OOP: Dependency injection", "pipeline.py / bootstrap.py", "Pipeline(...) constructor"),
    ], columns=["Topic", "Where", "In the app"]))


def main() -> None:
    """App entry point."""
    st.session_state.setdefault("sid", uuid.uuid4().hex)
    try:
        app = get_app()
        tier, conditions, extras = sidebar(app)
    except Snap2CookError as exc:
        st.error(exc.user_message)
        st.stop()
    st.title("Snap2Cook")
    st.caption("Photo of your ingredients in, a recipe that fits your kitchen out.")
    cook, tools, about = st.tabs(["Cook", "Text Tools", "About"])
    with cook:
        cook_tab(app, tier, conditions, extras)
    with tools:
        text_tools_tab()
    with about:
        about_tab()


main()
