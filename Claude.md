# CLAUDE.md — Snap2Cook

> Photo of ingredients → cooking recipe that fits the cook's kitchen (Home / PG / Hostel).
> Gemini vision → ingredient list → cloud RAG (cloud-stored synthetic recipe corpus + cloud embeddings, filter → similarity → rerank → cache) → grounded recipe.
> Course: Advanced Python Programming (MTech DS). Must demonstrate OOP + file handling + exception handling, production-grade, **deployed on Streamlit Community Cloud within 4 hours**. No Docker.

You are building this with me, the student (Satyaki). Read this whole file first. Follow the plan in order. Do not gold-plate.

---

## 0. Hard constraints

1. **4-hour time box** including deployment. Anything not on the "must" list is cut first (§11).
2. **No Docker, no local ML models.** Streamlit Community Cloud has ~1 GB RAM and no GPU: all heavy AI is via the **Gemini API** (vision, embeddings, generation). Dependencies must stay light (no torch, no onnx, no sentence-transformers).
3. **Python 3.11**, type hints everywhere, docstrings on public classes/functions (I'll be asked about them in a viva), no dead code, no TODO stubs.
4. **OOP is the backbone**: ABCs, inheritance, polymorphism, composition, encapsulation (private attrs + properties), dunder methods, a factory/registry, dependency injection.
5. **Secrets only via Streamlit secrets / env vars.** Read order: `st.secrets` → `os.environ` → `.env` (local only). Never print, log or commit a key. `.streamlit/secrets.toml` and `.env` are gitignored.
6. **Never crash the UI.** Every domain failure is a custom exception caught at the UI boundary with a friendly message. Every external dependency has a fallback (§6).
7. Keep modules small (<200 lines). `app.py` is UI only — no business logic.
8. `git init` first; commit after every working milestone; `pytest -q` green before each commit.
9. If a library API or model name differs from this file, check the installed version's docs / `help()` and adapt; tell me what changed. Don't guess model IDs — Gemini model names change; they come from config.

## 1. Product spec

**User:** a bachelor / college fresher who cannot cook. Uploads (or takes) a photo of the food they have. Gets one concrete beginner-friendly recipe they can actually make.

**Three kitchen tiers (core domain concept):**

| Tier | Equipment | Ingredients allowed | Notes |
|---|---|---|---|
| `home` | user ticks microwave and/or gas stove | photo items + user-ticked "extra supplies" (oil, spices, rice, flour, milk, …) + staples | richest recipes |
| `pg` | user picks ONE: oven **or** gas stove | photo items only + staples (water, salt) | max 1 missing non-staple ingredient |
| `hostel` | kettle only | photo items only + staples (water, salt) | only boil / steep / soak / mix / pour / mash / chop; **no frying, no direct heat** |

Optional extra **conditions** the user can set: diet (`veg|egg|non-veg|vegan|any`), max cooking time. Hostel/PG recipes must be honest: if nothing sensible can be made, say so and suggest the single cheapest ingredient to add. Never invent ingredients.

**Flow:**
1. Sidebar: choose tier (+ equipment / extras), diet, max time.
2. Upload / camera photo → validated, resized, hashed.
3. Gemini returns ingredients + confidence → shown as an **editable list** (vision is fallible; human-in-the-loop).
4. Pipeline retrieves + reranks recipes from the cloud-hosted corpus and generates a grounded recipe: title, time, steps, substitutions, safety note, source recipe IDs.
5. Download recipe as `.txt`.

## 2. Architecture

```
                         ┌──────────── Cloud storage (Hugging Face Datasets repo, public) ───────────┐
                         │  recipes.jsonl   embeddings.npy   meta.json (embed model, dim, count)     │
                         └──────────────────────────────────┬─────────────────────────────────────────┘
                                                            │ fetched once at startup (st.cache_resource)
photo ─► ImageValidator ─► GeminiVisionProvider ──(cache: image sha256)                 ▼
              │                    │                                         CorpusStore (Cloud → Local fallback)
              │                    ▼
              │        IngredientNormalizer (synonyms) ─► pd.Series(name → confidence)   [Q10]
              │                    │  user edits list
              │                    ▼
              └────────► Tier (Home/PG/Hostel) + conditions ─► Retriever
                           1. hard filter: tier, equipment, diet, max time  (metadata)
                           2. embed query via Gemini Embeddings API (RETRIEVAL_QUERY)  ← cloud
                           3. cosine vs precomputed doc vectors (numpy, in-process) → top-20
                           4. Reranker: ingredient-coverage + similarity − missing penalty
                                                            │ top-3
                                                            ▼
                      Generator (Gemini, grounded on top-3, JSON schema) ─► validate ─► UI
                      cache: L1 in-memory LRU + L2 file (TTL)         fallback: top-1 corpus recipe verbatim
```

Fallback chain: generation fails / invalid JSON twice → show the top reranked corpus recipe verbatim. Cloud corpus unreachable → bundled local copy in `data/`. Embedding API fails → lexical (ingredient-overlap) retrieval. Vision fails or no key → manual ingredient multiselect. No Gemini key at all → `MockVisionProvider` + lexical retrieval + verbatim recipes (app still demos fully).

## 3. Syllabus coverage (kept: Q6, Q8, Q9, Q10; Q5 and Q7 intentionally dropped by me)

Each kept item lives in its own named module, tagged `# SYLLABUS Q<n>` at the top, is used by the real app, has a `__main__` demo, and is printed by `python -m assignments.run_all` (for the viva).

| Q | Task | Module | Real use in Snap2Cook |
|---|---|---|---|
| 6 | Read file; (i) each word reversed (ii) each line reversed — sample `Python Programming` → `Programming Python` (iii) content without whitespace | `fileio/text_tools.py` → `FileTextProcessor` | "Text Tools" expander on the exported recipe / uploaded `.txt`. **Wording is ambiguous — implement all:** `reverse_each_word()` (letters per word), `reverse_word_order_per_line()` (matches the sample), `reverse_line_order()`, `strip_whitespace()`. |
| 8 | Handle run-time errors in file handling | `fileio/safe_io.py` → `SafeFile` | ALL disk access (local corpus fallback, cache files, logs, exports) goes through it. Handles `FileNotFoundError, PermissionError, IsADirectoryError, UnicodeDecodeError, json.JSONDecodeError, OSError`; atomic write (temp file + `os.replace`); re-raises as our `FileOperationError` with path/op context. |
| 9 | Create and raise user-defined exceptions | `exceptions.py` | Hierarchy below; really raised in pipeline / vision / storage / IO, caught at the UI boundary. |
| 10 | Pandas Series with labels | `analytics/series.py` + vision/retriever | Ingredient confidences = `pd.Series(conf, index=names, name="confidence")`; retrieval scores = `pd.Series(scores, index=recipe_ids, name="score")`. Shown in UI as tables. `__main__` demo shows `.loc`, label slicing, `.sort_values`, boolean masks. |

```
Snap2CookError(Exception)               # base: message + user_message + context dict
├─ FileOperationError                   # path, op
├─ InvalidImageError                    # bad type / too big / corrupt
├─ NoIngredientsDetectedError
├─ VisionProviderError                  # API down, auth, quota (429), timeout, bad JSON
├─ EmbeddingError
├─ CorpusLoadError                      # cloud fetch failed / schema or embedding-dim mismatch
├─ RetrievalError
├─ GenerationError
├─ UnsupportedTierError
└─ RateLimitExceededError(retry_after)
```

## 4. OOP design (be able to explain each in one sentence)

- `Tier(ABC)` → `HomeTier`, `PGTier`, `HostelTier` — **Strategy**. Properties: `name`, `allowed_equipment`, `staples`, `allow_extra_supplies`, `max_missing`, `allowed_techniques`. Methods `accepts(recipe, conditions) -> bool`, `prompt_rules() -> str`. `TierFactory.create(name, **opts)` registry (raises `UnsupportedTierError`).
- `VisionProvider(ABC)` → `GeminiVisionProvider`, `MockVisionProvider`. `detect(image_bytes) -> list[DetectedIngredient]`.
- `Embedder(ABC)` → `GeminiEmbedder`, `LexicalEmbedder` (fallback: bag-of-ingredients overlap, no network). Methods `embed_query`, `embed_documents`.
- `CorpusStore(ABC)` → `HFDatasetStore` (cloud, plain HTTPS GET of public files with timeout + retry), `LocalStore` (bundled copy via `SafeFile`). `load() -> Corpus`.
- `Corpus` — validated recipes + vector matrix; `__len__`, `__iter__`, `__getitem__`, `filter(tier, conditions)`.
- `Reranker(ABC)` → `CoverageReranker`.
- `Retriever` composes `Corpus + Embedder + Reranker`.
- `Generator` (Gemini, schema-constrained) with `repair_retry` and verbatim fallback.
- `Cache`: `LRUCache` (L1, `OrderedDict`) + `FileCache` (L2, JSON via `SafeFile`, TTL) composed in `TieredCache`; `hit_rate` property.
- `RateLimiter` — per-session sliding window (default 6 generations/10 min) + global daily ceiling (in-memory shared via `st.cache_resource`); raises `RateLimitExceededError`. Protects the free Gemini quota.
- `Pipeline` — orchestrator, all collaborators injected via constructor (mockable). Returns `PipelineResult` (recipe, ingredients Series, scores Series, timings dict, cache-hit flags, fallback_used).
- `Settings` — frozen dataclass from secrets/env (`config.py`).
- `Recipe`, `DetectedIngredient`, `GeneratedRecipe` — pydantic models with `__str__`.

## 5. Tech stack (final)

- **UI + hosting:** Streamlit, deployed on **Streamlit Community Cloud** from a GitHub repo (`st.camera_input` works because it's HTTPS).
- **AI:** official `google-genai` SDK. One API key for all three. Config (not hardcoded): `VISION_MODEL`, `GEN_MODEL` (fast Flash-class model), `EMBED_MODEL` (Gemini embedding model), `EMBED_DIM` (use 768 via output dimensionality to keep the vector file small; **L2-normalise vectors yourself** when truncating). Use `RETRIEVAL_DOCUMENT` for corpus vectors and `RETRIEVAL_QUERY` for queries. Gemini JSON mode with a pydantic `response_schema` for vision and generation. **Check current model names in Google's docs before writing config defaults.**
- **Vector search:** plain `numpy` matrix (corpus ≈ 100–300 docs); cosine = dot product. No FAISS/Chroma.
- **Cloud storage:** a **public Hugging Face Datasets repo** holding `recipes.jsonl`, `embeddings.npy`, `meta.json`. The app reads it by plain HTTPS (`…/datasets/<user>/<repo>/resolve/main/<file>`) → no token needed in Streamlit; uploads happen once from my machine via `huggingface_hub`. (Synthetic data, harmless to be public.) `meta.json` records embed model + dim + count; **on load, a mismatch with current config raises `CorpusLoadError`** (prevents silently comparing vectors from different models).
- **Other:** pandas, numpy, pillow, pydantic v2, requests, python-dotenv, huggingface_hub (scripts only), pytest.
- `requirements.txt`: streamlit, google-genai, pandas, numpy, pillow, pydantic, requests, python-dotenv, pytest. Pin versions that actually installed. Keep `huggingface_hub` in `requirements-dev.txt` so the deployed image stays light.

## 6. Production-grade checklist (what "production" means here)

- **Config:** `Settings` with defaults; fail fast with a clear message only in live mode if the key is missing; otherwise drop to mock mode and show a banner.
- **Logging:** `logging` with `RotatingFileHandler` (via `SafeFile`-created dir) + console; request id per run; log stage latencies; never log keys or image bytes.
- **Image validation:** ≤ 5 MB; JPEG/PNG/WEBP verified by Pillow (not extension); strip EXIF; downscale longest side to 1024 px; re-encode JPEG q=85; sha256 of resized bytes = cache key.
- **API calls:** timeout, up to 2 retries with exponential backoff + jitter, special handling of 429/quota → `VisionProviderError` with friendly "try again in a minute" text. Strict JSON, pydantic validation, one repair retry, then fallback.
- **Caching:** vision keyed by image hash (TTL 7 d); embeddings of the query keyed by normalised text; final recipe keyed by `hash(tier, equipment, sorted ingredients, extras, diet, max_time)` (TTL 24 h). Streamlit `st.cache_resource` for the loaded corpus + clients. L2 file cache lives in a temp dir and is allowed to vanish (Community Cloud is ephemeral).
- **Grounding:** the generator prompt includes ONLY the retrieved recipes + tier rules + the allowed-ingredient list. Output must include `source_recipe_ids`. Post-validate: every ingredient used ∈ (detected ∪ extras ∪ staples) and every technique ∈ `tier.allowed_techniques`; else reject → repair retry → verbatim fallback.
- **Safety note** field always present (hot water / kettle, cook eggs/meat through, obvious allergens). Hostel recipes: kettle only boils water that is then poured; never heat food in the kettle.
- **Tests (pytest, ~15–20, fast, NO network):** Q6 four functions incl. the sample and empty / whitespace-only files; `SafeFile` on missing / permission / corrupt-JSON / atomic write; each custom exception raised where expected; Series label behaviour; tier filtering (hostel never gets a frying recipe; PG respects `max_missing`); `CorpusStore` meta mismatch raises `CorpusLoadError`; cloud store failure falls back to local; retriever returns ≤ k; pipeline end-to-end with `MockVisionProvider` + `LexicalEmbedder` + fake generator; cache hit skips the LLM call; rate limiter blocks the 7th call.

## 7. Data: synthetic recipe corpus

- **You (Claude Code) write `data/recipes.jsonl` yourself — it costs no Gemini quota.** Minimum **90 recipes: 30 hostel (kettle-only), 30 PG, 30 home**; cheap, beginner, student-pantry ingredients (eggs, bread, rice, oats, instant noodles, poha, curd, banana, peanut butter, onion, tomato, potato, paneer, dal, atta, milk, tea, biscuits, …). Mix Indian and global; vegetarian-heavy. Validate every row with pydantic on load.
- Row schema:

```json
{"id":"r001","title":"Kettle Masala Oats","tiers":["hostel","pg","home"],
 "equipment_any":["kettle","gas_stove","microwave","oven"],
 "ingredients":[{"name":"oats","required":true},{"name":"salt","required":false}],
 "techniques":["boil","soak","mix"],"time_min":10,"calories":280,
 "cuisine":"indian","diet":"veg","difficulty":"easy",
 "steps":["Pour boiled water over oats…","…"],"tips":"…","doc_text":"title + ingredients + short description used for embedding"}
```

- Hostel recipes may only use `boil|steep|soak|mix|pour|mash|chop` — tag accurately.
- `data/synonyms.json` (aloo→potato, anda→egg, dahi→curd, atta→wheat flour, capsicum↔bell pepper, plurals, common vision variants). `data/pantry.json` (staples per tier + Home "extra supplies" checklist).
- `scripts/build_embeddings.py`: reads `recipes.jsonl`, embeds `doc_text` in batches via `GeminiEmbedder` (respect rate limits, sleep/backoff), L2-normalises, writes `data/embeddings.npy` + `data/meta.json`.
- `scripts/upload_dataset.py`: uploads the three files to the HF dataset repo using `HF_TOKEN` from my environment (**never** written to a file). Local copies stay in `data/` as the offline fallback.
- Stretch only if ahead: more rows.

## 8. Retrieval details

1. Query text: `"Recipe using: egg, bread, onion"` (+ tier hint).
2. Hard filter by tier, equipment, diet, max time.
3. Embed the query via Gemini (cached) → cosine against filtered rows → top-20.
4. Rerank: `final = 0.6·similarity + 0.4·coverage − 0.1·missing_count` (weights in `Settings`), `coverage` = fraction of required ingredients available (detected ∪ extras ∪ staples). Drop recipes with missing required non-staple count > `tier.max_missing`.
5. Return top-3 with scores as a labelled `pd.Series` (Q10). If none survive, return a "can't make a proper dish — add one of: …" result (items computed from the nearest recipes' missing ingredients), not an error.

## 9. UI (Streamlit, `app.py`)

Sidebar: tier radio + equipment / extras widgets, diet, max time, small "mode" badge (Live / Mock), cache hit-rate. Tabs:
1. **Cook** — upload / camera → detected ingredients (editable via `st.data_editor`, confidence column from the Series) → "Get recipe" → recipe card (title, time, numbered steps, substitutions, safety note, source recipes) + `.txt` download + expander "How this was made" (stage timings, cache hits, retrieved candidates with scores, fallback flag).
2. **Text Tools** — Q6 on uploaded `.txt` / the exported recipe; four buttons.
3. **About** — short table mapping Q6, Q8, Q9, Q10 + the OOP classes to where they appear (helps the viva).

Clean, minimal, mobile-first (students use phones). One accent colour. Friendly messages from each exception's `user_message`.

## 10. Deployment (Streamlit Community Cloud — no Docker)

- Repo layout must have `app.py` and `requirements.txt` at the root. Push to GitHub (public or private; private needs the Streamlit–GitHub link).
- Streamlit Cloud → New app → pick repo/branch → `app.py` → Advanced settings: Python 3.11 + paste secrets (TOML): `GEMINI_API_KEY`, `VISION_MODEL`, `GEN_MODEL`, `EMBED_MODEL`, `HF_DATASET_REPO`.
- **Deploy a hello-world at ~0:25** to flush out repo/secrets/requirements problems early; redeploy by pushing.
- Filesystem is ephemeral: cache and logs reset on restart. Acceptable; list it as a limitation in README and slides.
- Local run must also work: `pip install -r requirements.txt && streamlit run app.py` (with `.env`).
- Tell me the exact steps for the parts only I can do: get the Gemini key, create the HF account + public dataset repo + token, create the GitHub repo, connect Streamlit Cloud. Never ask me to paste tokens into files or chat.

## 11. Schedule and cut list

| Time | Deliverable |
|---|---|
| 0:00–0:20 | scaffold, `config`, `exceptions` (Q9), `safe_io` (Q8), logging, git, `.gitignore`, tests |
| 0:20–0:30 | **hello-world deploy on Streamlit Cloud** |
| 0:30–1:00 | corpus (90 recipes) + synonyms + pantry; validated `Corpus` loader |
| 1:00–1:30 | `GeminiEmbedder`, `build_embeddings.py`, upload to HF dataset, `HFDatasetStore` + `LocalStore` fallback |
| 1:30–2:00 | tiers, filter, retriever, `CoverageReranker`, caches, retrieval tests |
| 2:00–2:30 | vision provider + mock, image utils, normalizer, Series (Q10), generator + validation + fallback, `RateLimiter` |
| 2:30–3:10 | `Pipeline`, Streamlit UI, Text Tools (Q6), About tab |
| 3:10–3:30 | `assignments/run_all.py`, README, final deploy, phone smoke test with a real photo |
| 3:30–4:00 | buffer, screenshots, `docs/slides_outline.md` |

**If behind, cut in this order:** (1) L2 file cache (keep LRU); (2) camera input (keep upload); (3) diet / max-time conditions; (4) Text Tools tab polish; (5) corpus 90 → 60 recipes. **Never cut:** Q6/Q8/Q9/Q10 modules, tier filtering, cloud corpus fetch + similarity retrieval, editable ingredient list, fallbacks, deployment.

## 12. Working rules for Claude Code

- Build in §11 order. After each module: tests → `pytest -q` → commit. Report progress in one line each.
- Don't ask what this file answers. Ask only for what only I can do (keys, accounts, a real photo).
- Boring, readable code over clever — I must explain it live.
- When done: run the full tests, `python -m assignments.run_all`, run the app locally once in mock mode and once live, then give me (a) deploy steps, (b) a 10-line viva demo script, (c) real measured numbers for the slides (test count, cold vs cached latency, cache hit-rate on a repeat run).

## 13. Repo layout

```
snap2cook/
├─ CLAUDE.md  README.md  requirements.txt  requirements-dev.txt  .env.example  .gitignore
├─ .streamlit/config.toml            # theme only; secrets.toml is gitignored
├─ app.py                            # UI only
├─ snap2cook/
│  ├─ config.py  exceptions.py  models.py  tiers.py  cache.py  limiter.py  pipeline.py  logging_setup.py
│  ├─ fileio/      safe_io.py  text_tools.py                  # Q8, Q6
│  ├─ analytics/   series.py                                  # Q10
│  ├─ vision/      base.py  gemini_provider.py  mock_provider.py  normalizer.py  image_utils.py
│  └─ rag/         store.py  corpus.py  embedder.py  retriever.py  reranker.py  generator.py
├─ data/           recipes.jsonl  embeddings.npy  meta.json  synonyms.json  pantry.json   # committed (offline fallback)
├─ scripts/        build_embeddings.py  upload_dataset.py
├─ assignments/    run_all.py
├─ tests/
└─ docs/           slides_outline.md
```

## 14. Slide deck (max 7) — outline only; build later when I ask

1. Problem, target user, one-line solution (app screenshot)
2. Architecture: Gemini vision → cloud corpus + cloud embeddings → rerank → grounded generation (§2 diagram, simplified) + tier table
3. OOP design: class diagram (Tier strategy, VisionProvider / Embedder / CorpusStore ABCs, Pipeline DI, exception hierarchy)
4. RAG details: synthetic corpus in cloud storage, conditions filter, similarity, rerank, two-level cache, fallbacks
5. Syllabus coverage map: Q6, Q8, Q9, Q10 ↔ module ↔ where it shows in the app
6. Demo + results: screenshots per tier, measured latency cold vs cached, tests passing
7. Deployment on Streamlit Cloud, limitations (free-tier quota, ephemeral storage, small corpus, vision errors), future work

## 15. Definition of done

- [ ] Public Streamlit URL works on a phone: photo → recipe for all 3 tiers
- [ ] Corpus really comes from cloud storage (log line proves it) and falls back to local when blocked
- [ ] Hostel result never needs heat beyond a kettle; PG never needs unlisted ingredients
- [ ] `pytest -q` green; `python -m assignments.run_all` prints Q6/Q8/Q9/Q10 demos
- [ ] Works without a key (mock mode) and degrades gracefully on Gemini errors / quota
- [ ] No secrets in repo; README documents setup, secrets, limitations