# Snap2Cook

> Snap a photo of the food you have. Get one beginner-friendly recipe that fits your kitchen: **Hostel**, **PG** or **Home**.

Snap2Cook is a Streamlit app for students who cannot cook. Gemini vision reads the photo, a retrieval-augmented
generation (RAG) pipeline finds matching recipes in a cloud-hosted corpus, and Gemini writes a recipe that is
**grounded on those recipes and limited to the kitchen you actually have**. Every external dependency has a fallback,
so the app also runs with no API key (Mock mode).

Built for the MTech DS course *Advanced Python Programming* (OOP, file handling, exception handling, pandas).

## Features

- **Three kitchen tiers** (Strategy pattern): Hostel (kettle only), PG (one appliance), Home (stove and/or microwave plus extra supplies).
- **Photo to ingredients** with Gemini vision, shown as an editable table (vision can be wrong, so a human confirms).
- **RAG pipeline**: hard filter (tier, equipment, diet, time), embedding similarity, coverage rerank, two-level cache.
- **Grounded generation**: output is validated against the allowed ingredients and techniques; invalid output is repaired once, then replaced by the stored recipe.
- **Model fallback chain**: if the main Gemini model is overloaded, the next model in `FALLBACK_MODELS` is tried.
- **Cloud corpus** on a public Hugging Face dataset, with a bundled local copy as offline fallback.
- **Safe by default**: image validation, rate limiting, no secrets in logs, friendly messages for every domain error.

## Architecture

```
photo -> ImageValidator -> VisionProvider (Gemini | Mock) -> IngredientNormalizer -> pd.Series(name -> confidence)
                                                                      | (user edits list)
                                                                      v
 Hugging Face dataset --> CorpusStore (Cloud -> Local fallback)   Tier + Conditions
                                   |                                  |
                                   +--------------> Retriever <-------+
                                     1. hard filter  2. Gemini query embedding  3. cosine top-20
                                     4. CoverageReranker  -> top-3
                                                      |
                                    Generator (Gemini, JSON schema) -> validate -> UI
                                    cache: LRU (memory) + file (TTL)   fallback: top recipe verbatim
```

| Failure | Fallback |
|---|---|
| Main Gemini model overloaded or invalid | Next model in `FALLBACK_MODELS` |
| Generation invalid twice or all models fail | Top corpus recipe shown verbatim |
| Embedding API fails | Lexical (ingredient overlap) retrieval |
| Cloud corpus unreachable | Bundled copy in `data/` |
| Vision fails or no key | Manual ingredient picker / Mock provider |

### Kitchen tiers

| Tier | Equipment | Ingredients | Techniques |
|---|---|---|---|
| Hostel | kettle | photo items + water, salt (0 missing) | boil water, steep, soak, mix, pour, mash, chop |
| PG | one of gas stove / oven (kettle assumed) | photo items + water, salt (max 1 missing) | whatever that appliance supports |
| Home | gas stove and/or microwave | photo items + ticked extras + staples (max 2 missing) | anything |

## Project layout

```
app.py                  Streamlit UI only
snap2cook/              application package
  config.py             Settings (st.secrets -> os.environ -> .env)
  exceptions.py         custom exception hierarchy (Q9)
  tiers.py              Tier ABC, Home/PG/Hostel, TierFactory, Pantry
  pipeline.py           orchestrator (dependency injection)
  bootstrap.py          composition root
  gemini.py             client, retry policy, model fallback
  cache.py  limiter.py  logging_setup.py  models.py
  fileio/               safe_io.py (Q8), text_tools.py (Q6)
  analytics/            series.py (Q10)
  vision/               providers, image validation, normalizer
  rag/                  store, embedder, retriever, reranker, generator
data/                   recipes.jsonl, embeddings.npy, meta.json, synonyms.json, pantry.json
scripts/                build_embeddings.py, upload_dataset.py
assignments/            run_all.py (syllabus demos)
tests/                  pytest suite (no network)
```

## Getting started (local)

Requires Python 3.11. Always work inside the virtual environment.

```powershell
git clone https://github.com/Carnage203/Snap2Cook.git
cd Snap2Cook
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
copy .env.example .env          # then edit .env (it is gitignored)
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Leave `GEMINI_API_KEY` empty to run in **Mock mode** (no network, fake photo detector, stored recipes).

## Configuration

Read order: `st.secrets` -> OS environment variables -> `.env` (local only). Secrets are never logged or committed.

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Google AI Studio key; empty means Mock mode | empty |
| `VISION_MODEL` | Model for photo analysis | `gemini-3.6-flash` |
| `GEN_MODEL` | Model for recipe writing | `gemini-3.6-flash` |
| `FALLBACK_MODELS` | Comma-separated models tried in order if the main one fails | `gemini-3.8-flash,gemini-3.5-flash,gemini-2.5-flash` |
| `EMBED_MODEL` / `EMBED_DIM` | Embedding model and vector size | `gemini-embedding-001` / `768` |
| `HF_DATASET_REPO` | Public dataset holding the corpus, e.g. `user/snap2cook-corpus` | empty (local copy used) |
| `HF_TOKEN` | Write token, used only by `scripts.upload_dataset` on your machine | empty |

Model names change over time; check them in Google AI Studio and edit `.env`.

## Data pipeline (one-time)

```powershell
.\.venv\Scripts\python.exe -m scripts.build_embeddings   # embeds data/recipes.jsonl -> embeddings.npy + meta.json
.\.venv\Scripts\python.exe -m scripts.upload_dataset     # pushes the three files to the Hugging Face dataset
```

`meta.json` records the embedding model, dimension and count. If it does not match the current config the app raises
`CorpusLoadError` instead of comparing vectors from different models.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q                 # fast, offline
.\.venv\Scripts\python.exe -m assignments.run_all       # prints the syllabus demos
```

| Syllabus | Module |
|---|---|
| Q6 text file operations | `snap2cook/fileio/text_tools.py` |
| Q8 safe file handling | `snap2cook/fileio/safe_io.py` |
| Q9 user-defined exceptions | `snap2cook/exceptions.py` |
| Q10 labelled pandas Series | `snap2cook/analytics/series.py` |

## Deployment (Streamlit Community Cloud)

1. Push the repository to GitHub (`.env`, `.venv` and secrets are gitignored).
2. Go to share.streamlit.io, choose **New app**, pick the repo and branch, and set the main file to `app.py`.
3. Under **Advanced settings** choose Python 3.11 and paste your secrets (TOML):

   ```toml
   GEMINI_API_KEY = "your-key"
   VISION_MODEL = "gemini-3.6-flash"
   GEN_MODEL = "gemini-3.6-flash"
   FALLBACK_MODELS = "gemini-3.8-flash,gemini-3.5-flash,gemini-2.5-flash"
   EMBED_MODEL = "gemini-embedding-001"
   EMBED_DIM = "768"
   HF_DATASET_REPO = "your-username/snap2cook-corpus"
   ```

   `HF_TOKEN` is not needed in the cloud because the dataset is public.

## Security

- Secrets live only in `.env` (local) or Streamlit secrets (cloud); `.env` is gitignored and the key is hidden from `repr` and logs.
- Uploads are verified by Pillow (not by extension), limited to 5 MB, stripped of EXIF and downscaled.
- A per-session and daily rate limiter protects the free Gemini quota.
- Recipes always carry a safety note; Hostel recipes never heat food in the kettle.

## Limitations

- Free-tier Gemini quota and occasional model overload (the fallback chain mitigates this).
- Ephemeral filesystem on Community Cloud: caches and logs reset on restart.
- Small synthetic corpus (90 recipes); vision can misidentify items, so review the detected list.

## License

Coursework project. No license has been chosen yet.
