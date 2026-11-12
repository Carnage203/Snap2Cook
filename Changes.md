# Changes

Every change made while building Snap2Cook, with the reason. Newest decisions are at the bottom of each section.

## 1. Environment and secrets

| Change | Why |
|---|---|
| Created `.venv` with Python 3.11 (`py -3.11`) | You asked for all installs to be inside a venv, and CLAUDE.md requires Python 3.11 (the PC default is 3.14, which Streamlit Cloud will not use). |
| Installed every package only through `.\.venv\Scripts\python.exe -m pip` | No global installs, as requested. |
| `requirements.txt` pinned to the versions that actually installed; `huggingface_hub` only in `requirements-dev.txt` | CLAUDE.md §5: reproducible builds and a light deployed image. |
| Created `.env` (gitignored) and `.env.example` | All secrets and model names live in `.env`. `GEMINI_API_KEY` and `HF_TOKEN` are left empty for you to fill. |
| `.gitignore` covers `.venv/`, `.env`, `.streamlit/secrets.toml`, logs, caches | Secrets and the venv must never be committed. |
| Model defaults: `gemini-3.7-flash` (vision, generation) and `gemini-embedding-001` (embeddings, 768 dims) | CLAUDE.md says to check Google's docs. Search results showed `gemini-3.7-flash` as the newest stable Flash code and `gemini-embedding-001` as the text embedding model. The docs disagree between pages, so these are config values in `.env`, not hardcoded. Verify them. |

## 2. Architecture decisions that go beyond or interpret CLAUDE.md

| Decision | Why |
|---|---|
| A kettle is always assumed in every tier (`Tier.allowed_equipment` = chosen equipment + kettle) | A PG or home cook with a stove can boil water, so kettle-only hostel recipes must also be available to them. |
| `max_missing`: hostel 0, PG 1, home 2 | The spec fixes PG at 1 and says hostel uses photo items only. Home had no number, 2 is a sensible small allowance. |
| `GeneratedRecipe.extra_needed` field | The spec allows one missing ingredient in PG but also says never to invent ingredients. This field names the missing item openly, and the validator checks it comes from the source recipes. |
| Retriever keeps a separate in-memory lexical matrix and tries the Gemini matrix first | Gemini vectors have 768 dimensions and lexical vectors have vocabulary-size dimensions, so they cannot share one matrix. This also gives the "embedding API fails -> lexical" fallback. |
| `LocalStore.vectors_optional = True`, cloud store requires vectors | You have no Gemini key in `.env` yet, so embeddings cannot be built. The bundled copy still works (lexical search). When vectors exist, `meta.json` is checked against config and a mismatch raises `CorpusLoadError`. |
| Retrieval requires at least one photo ingredient in the recipe (`overlap > 0`) | Without it, a recipe made only of water and salt would score as 100% covered. |
| `VerbatimGenerator` class | Used as the no-key generator and as the fallback when Gemini output is invalid twice. Same `Generator` interface (polymorphism). |
| `snap2cook/gemini.py` (client factory, retry with backoff and jitter, 429 handling) | One retry policy shared by vision, embeddings and generation instead of three copies. |
| `snap2cook/bootstrap.py` (composition root) | Keeps `app.py` UI-only and lets the pipeline get all dependencies by injection. |
| `Pantry` lives in `tiers.py`; `data/pantry.json` holds staples and the Home extras checklist | Staples are a tier property, so the loader sits next to the tiers. |
| `LRUCache` also supports TTL | The spec wants 7-day and 24-hour TTLs; a long-running server would otherwise serve stale in-memory entries. |
| `Pipeline.recommend` checks the rate limiter only on a cache miss, just before generating | Cache hits cost no quota, so they should never count against the limit. |

## 3. Data

| Change | Why |
|---|---|
| `data/recipes.jsonl`: 90 hand-written recipes (30 hostel, 30 PG, 30 home), validated by pydantic on load | CLAUDE.md §7, written by me so it costs no Gemini quota. Generated once from a throwaway script kept outside the repo. |
| Hostel recipes contain no egg or meat dishes | A kettle cannot cook an egg safely. An egg photo in hostel mode honestly returns "nothing to make" plus suggestions. |
| Pydantic check: a recipe tagged `hostel` may only use boil/steep/soak/mix/pour/mash/chop | Mis-tagged data is rejected at load, not at serving time. |
| `data/synonyms.json`, `data/pantry.json` | Indian and common vision names mapped to corpus names; staples and extras per tier. |

## 4. Problems found while building, and fixes

| Problem | Fix |
|---|---|
| `capsicum -> bell pepper` and `bell pepper -> capsicum` made a swap loop in the synonyms file | Single direction: everything maps to `capsicum`. Also removed useless self-maps and a wrong `ketchup -> tomato`. |
| `use_container_width` is deprecated in Streamlit 1.67 | Replaced with `width="stretch"`. |
| A test asserted `a or b` and could never fail | Split into two precise tests (no recipe; suggestions on a near miss). |
| Safety notes rendered as red error boxes | Changed to `st.warning`. |
| A retrieval loop could in theory leave variables unbound if every embedder failed | Added `for/else` that raises `RetrievalError`. |

## 5. Things you should know

- **A Gemini key exists as an OS-level environment variable on this PC** (`GEMINI_API_KEY`), although `.env` is empty. CLAUDE.md's read order is `st.secrets` -> `os.environ` -> `.env`, so my first smoke test used it and made real Gemini calls (generation returned a server error three times and the app correctly fell back to the verbatim recipe). I never printed the key. All later tests ran with the key blanked. If you want secrets to live only in `.env`, delete that Windows environment variable and put the key in `.env`.
- Gemini returned `ServerError` for `gemini-3.7-flash` generation. The cause is unknown (outage, key problem or wrong model name). Check the model name in `.env` before relying on live mode.
- Not yet done because they need you: building `embeddings.npy` (needs the key in `.env`), uploading to Hugging Face, GitHub push, Streamlit Cloud deploy, phone test with a real photo.
- Not written yet: `docs/slides_outline.md` (CLAUDE.md §14 says build when asked).
