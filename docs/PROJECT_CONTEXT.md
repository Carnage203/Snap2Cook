# Snap2Cook: project context (for planning the presentation)

Give this file to an AI assistant and ask it to plan the slides. Facts below describe the **deployed** app. Items marked
**[fill in]** are for the author to complete. Nothing here contains secrets.

## 1. One-line pitch

Snap a photo (up to 3) of the food you have, and get one beginner-friendly recipe that fits your kitchen: Hostel (kettle only), PG (one appliance) or Home (full kitchen).

## 2. Context

- **Course:** Advanced Python Programming, MTech Data Science. Marking looks at OOP, file handling, exception handling and pandas, plus a working deployed app.
- **Author:** Satyaki (GitHub: Carnage203). Repo: https://github.com/Carnage203/Snap2Cook
- **Live app:** Streamlit Community Cloud, URL **[fill in]**. Corpus dataset: https://huggingface.co/datasets/sohammandal1/snap2cook-corpus
- **Target user:** a college fresher or bachelor who cannot cook and has a few random items.
- **Core idea:** most recipe apps assume a full kitchen. Snap2Cook treats the *kitchen* as a first-class input, so a hostel student with only a kettle gets only kettle-safe recipes, and is told honestly when nothing sensible can be made.

## 3. What the user sees

1. Sidebar: pick kitchen (Hostel / PG / Home), equipment, extra supplies (Home), diet, max time. A badge shows Live or Mock mode. Light and dark theme follow the device.
2. Upload up to 3 photos (single-item photos work well) or take one camera photo.
3. Gemini vision lists ingredients with confidence; shown in an **editable table** (vision is fallible, so a human confirms) plus a manual picker.
4. "Get recipe" returns title, time, numbered steps, substitutions, a safety note, source recipe ids, and a `.txt` download.
5. If nothing can be made: an honest message and up to 3 suggested ingredients to add.

## 4. Kitchen tiers (core domain concept)

| Tier | Equipment | Ingredients allowed | Missing allowed | Techniques |
|---|---|---|---|---|
| Hostel | kettle | photo items + water, salt | 0 | boil water, steep, soak, mix, pour, mash, chop. No frying or direct heat |
| PG | one of gas stove / oven (kettle assumed) | photo items + water, salt | 1 | what that appliance supports |
| Home | gas stove and/or microwave | photo items + ticked extra supplies + staples | 2 | anything |

Design choice worth a slide note: hostel recipes contain no egg or meat dishes, because a kettle cannot cook them safely. An egg photo in Hostel mode returns "can't make this" plus suggestions (honesty over fake recipes).

## 5. Architecture and data flow

```
photos -> ImageValidator -> VisionProvider (Gemini | Mock) -> IngredientNormalizer -> pd.Series(name -> confidence)
                                                                  | user edits list
Hugging Face dataset -> CorpusStore (Cloud -> Local fallback)     v
          \                                                  Tier + Conditions
           +--------------------> Retriever <------------------------+
             1 hard filter (tier, equipment, diet, time)
             2 Gemini query embedding (RETRIEVAL_QUERY)
             3 cosine vs precomputed doc vectors (numpy) -> top 20
             4 CoverageReranker: 0.6*similarity + 0.4*coverage - 0.1*missing -> top 3
                              |
            Generator (Gemini, JSON schema, grounded on top 3) -> validate -> UI
            cache: in-memory LRU + file cache (TTL)
```

- **Cloud pieces:** Gemini API (vision, embeddings, generation); Hugging Face public dataset (recipes.jsonl, embeddings.npy, meta.json); Streamlit Community Cloud (hosting, HTTPS so the camera works).
- **No Docker, no local ML models.** Heavy AI is all API calls, so the deployed image stays light.
- **Grounding:** the prompt contains only retrieved recipes, tier rules and the allowed-ingredient list. Output is checked: every ingredient must be allowed, every technique must be allowed for the tier, and `extra_needed` must respect the tier limit. A failed check triggers one repair retry, then the stored recipe is shown verbatim.
- **Embedding safety:** `meta.json` stores embedding model, dimension and count. A mismatch with the current config raises `CorpusLoadError`, which prevents silently comparing vectors from different models.

## 6. Fallbacks (a strong "production-grade" slide)

| Failure | Behaviour |
|---|---|
| Main Gemini model overloaded or invalid | Next model in `FALLBACK_MODELS` (3.6 -> 3.8 -> 3.5 -> 2.5 flash). Real example: `gemini-3.8-flash` returned 503 "high demand" and the chain recovered |
| Generation invalid twice or all models fail | Top corpus recipe shown verbatim |
| Embedding API fails | Lexical (ingredient-overlap) retrieval |
| Cloud corpus unreachable | Bundled copy in `data/` |
| Vision fails or no key | Manual ingredient picker; with no key at all, Mock mode (fake detector, lexical search, stored recipes) |
| Quota (429) | Friendly "try again in a minute" message; rate limiter protects the free quota (6 generations per 10 min per session, 300 per day) |

## 7. OOP design (one sentence each)

- `Tier` (ABC) -> `HomeTier`, `PGTier`, `HostelTier`, built by `TierFactory`: **Strategy + Factory**.
- `VisionProvider` (ABC) -> `GeminiVisionProvider`, `MockVisionProvider`: **polymorphism**, swap Live/Mock.
- `Embedder` (ABC) -> `GeminiEmbedder`, `LexicalEmbedder`; `CorpusStore` (ABC) -> `HFDatasetStore`, `LocalStore`; `Generator` -> `GeminiGenerator`, `VerbatimGenerator`; `Reranker` -> `CoverageReranker`.
- `Pipeline` receives all collaborators through its constructor: **dependency injection**, so tests use fakes. `bootstrap.py` is the composition root.
- `Retriever` composes Corpus + Embedder + Reranker: **composition**.
- `TieredCache` composes `LRUCache` and `FileCache`; `Settings` is a frozen dataclass whose API key is hidden from `repr`: **encapsulation**.
- `Recipe`, `DetectedIngredient`, `GeneratedRecipe` are pydantic models with `__str__`; `Corpus` and caches use dunder methods.

## 8. Syllabus coverage (Q5 and Q7 intentionally dropped)

| Q | Module | Real use in app |
|---|---|---|
| Q6 text file operations (word reversal, line reversal, strip whitespace) | `snap2cook/fileio/text_tools.py` (`FileTextProcessor`) | **Not visible in the UI any more** (Text Tools tab removed). Demonstrated via `python -m assignments.run_all` |
| Q8 file-handling errors | `snap2cook/fileio/safe_io.py` (`SafeFile`) | All disk access (corpus fallback, cache files, logs). Handles missing file, permission, directory, bad encoding, bad JSON; atomic write via temp file + `os.replace`; re-raises as `FileOperationError` |
| Q9 user-defined exceptions | `snap2cook/exceptions.py` | 11-class hierarchy under `Snap2CookError`, each with a technical message and a friendly `user_message`; raised in pipeline/vision/storage/IO and caught only at the UI boundary |
| Q10 labelled pandas Series | `snap2cook/analytics/series.py` | Ingredient confidences and retrieval scores are labelled Series; multi-photo results are merged with a groupby-max |

Exception tree: `Snap2CookError` -> `FileOperationError`, `InvalidImageError`, `NoIngredientsDetectedError`, `VisionProviderError`, `EmbeddingError`, `CorpusLoadError`, `RetrievalError`, `GenerationError`, `UnsupportedTierError`, `RateLimitExceededError`.

## 9. Data

- **Corpus:** 90 synthetic recipes written by hand (30 Hostel, 30 PG, 30 Home); cheap student-pantry ingredients, Indian and global, vegetarian-heavy. Validated by pydantic on load; a recipe tagged hostel may only use hostel techniques.
- **Embeddings:** 90 x 768 float32 vectors, L2-normalised, from `gemini-embedding-001`.
- Also `synonyms.json` (aloo -> potato, dahi -> curd, anda -> egg...) and `pantry.json` (staples and Home extras).
- The data is synthetic and harmless, so the Hugging Face dataset is public and the deployed app needs no HF token.

## 10. Production-grade practices to mention

- Secrets only in `.env` locally or Streamlit secrets in the cloud (read order: `st.secrets` -> environment -> `.env`); never logged or committed; `.gitignore` covers `.env`, `.venv`, caches, logs.
- Image validation: size limit 5 MB, type verified by Pillow (not extension), EXIF stripped, resized to 1024 px, re-encoded; SHA-256 of the result is the cache key.
- API calls: timeout, retries with exponential backoff and jitter, 429 handled separately, strict JSON with pydantic validation.
- Caching: vision by image hash (7 days), query embeddings by text, final recipe by hash of (tier, equipment, ingredients, extras, diet, time) (24 hours). Cache hits do not use API quota and do not count against the rate limit.
- Logging with a rotating file handler and a request id per run.
- Pinned dependencies; `huggingface_hub` only in `requirements-dev.txt` to keep the deployed image light.
- Tests: pytest, offline, fast.

## 11. Measured results

| Metric | Value |
|---|---|
| Automated tests | 45 passing, about 0.3 s, no network |
| Corpus | 90 recipes; loaded from the cloud dataset (log line: `Corpus loaded from cloud:hf:sohammandal1/snap2cook-corpus (90 recipes)`) |
| Cold recipe request (live Gemini, hostel oats + banana) | about 6 to 14 s across runs |
| Repeat identical request | about 1 ms (served from cache, no API call) |
| Cache hit rate on a repeat run | 33% over the two-request test (1 hit of 3 lookups); higher with more repeats |
| Model fallback | demonstrated: invalid primary model recovered via the fallback chain |

**[fill in]** screenshots per tier on a phone (Hostel, PG, Home) and one "can't make this" screenshot.

## 12. Honest limitations and lessons learned

- Free-tier Gemini quota and occasional model overload (503), mitigated by the fallback chain.
- Streamlit Community Cloud filesystem is ephemeral: caches and logs reset on restart.
- Small synthetic corpus (90 recipes); vision can misidentify items, so the user reviews the list.
- Model names change often; they are configuration, not code. One search source gave an inconsistent name, so names were verified against the real API model list.
- A real bug worth telling: after adding a new setting, the deployed app crashed because Streamlit's cached app object still held the old settings class. Fixed by keying the cache on a code version.
- The Q6 text tools were removed from the interface by choice; they remain in code and in the demo script, so be ready to explain where Q6 lives.

## 13. Suggested deck (max 7 slides)

1. Problem, user, one-line solution, app screenshot.
2. Architecture: photo -> Gemini vision -> cloud corpus + embeddings -> rerank -> grounded generation, plus the tier table.
3. OOP design: class diagram (Tier strategy, the ABC families, Pipeline with dependency injection, exception hierarchy).
4. RAG details: synthetic corpus in cloud storage, filters, similarity, rerank formula, two-level cache, grounding validation, fallback chain.
5. Syllabus map: Q6, Q8, Q9, Q10 -> module -> where it shows in the app.
6. Demo and results: screenshots per tier, cold vs cached latency, 45 tests.
7. Deployment (GitHub -> Streamlit Cloud, secrets, Hugging Face dataset), limitations, future work (bigger corpus, user accounts, multi-photo camera, nutrition info).

## 14. Viva demo script (about 10 lines)

1. Open the live URL on a phone; point out the Live badge and "Corpus: cloud:hf..." in the sidebar.
2. Hostel, upload 3 single photos: oats, banana, peanut butter -> edit the table -> Get recipe.
3. Explain the retrieval steps and why the recipe only uses allowed items and kettle techniques.
4. Switch to Hostel with an egg photo -> show the honest "can't make this" and suggestions.
5. Switch to Home with a stove and extras -> richer recipe.
6. Repeat a request -> instant response, explain the cache.
7. Show `snap2cook/tiers.py` (Strategy + Factory) and `pipeline.py` (dependency injection).
8. Run `python -m assignments.run_all` -> Q6, Q8, Q9, Q10 demos.
9. Run `pytest -q` -> 45 passed.
10. Explain the fallback chain, using the real 503 example.

## 15. Repository map

```
app.py                      UI only
snap2cook/                  config, exceptions, tiers, pipeline, bootstrap, gemini (retry + model fallback), cache, limiter, logging_setup, models
  fileio/ (Q8, Q6)  analytics/ (Q10)  vision/  rag/
data/                       recipes.jsonl, embeddings.npy, meta.json, synonyms.json, pantry.json
scripts/                    build_embeddings.py, upload_dataset.py
assignments/run_all.py      syllabus demos
tests/                      45 offline tests
docs/PROJECT_CONTEXT.md     this file
```

Stack: Python 3.11, Streamlit, google-genai, pandas, numpy, pillow, pydantic v2, requests, python-dotenv, pytest.
