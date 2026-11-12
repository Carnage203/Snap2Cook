# Snap2Cook

Photo of your ingredients in, a beginner-friendly recipe that fits your kitchen out (Hostel / PG / Home).

Gemini vision finds the ingredients, a RAG pipeline (cloud-hosted synthetic recipe corpus, embeddings, filter,
similarity, rerank, cache) finds matching recipes, and Gemini writes a recipe grounded on them. Everything has a fallback,
so the app also works with **no API key** (Mock mode).

## Kitchen tiers

| Tier | Equipment | Ingredients | Techniques |
|---|---|---|---|
| Hostel | kettle | photo items + water, salt | boil water, steep, soak, mix, pour, mash, chop |
| PG | one of oven / gas stove | photo items + water, salt (max 1 missing) | anything that appliance supports |
| Home | gas stove and/or microwave | photo items + ticked extra supplies + staples | anything |

## Setup (local, always inside the virtual environment)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
copy .env.example .env        # then edit .env (it is gitignored)
.\.venv\Scripts\python.exe -m streamlit run app.py
```

All secrets live in `.env` (or Streamlit secrets in the cloud): `GEMINI_API_KEY`, `VISION_MODEL`, `GEN_MODEL`,
`FALLBACK_MODELS` (tried in order if the main model fails), `EMBED_MODEL`, `EMBED_DIM`, `HF_DATASET_REPO`, `HF_TOKEN` (HF token is only used by the upload script).
Config read order: `st.secrets` -> OS environment -> `.env`. Leave `GEMINI_API_KEY` empty for Mock mode.

## One-time data steps (need your accounts)

1. Put `GEMINI_API_KEY` in `.env`, then `.\.venv\Scripts\python.exe -m scripts.build_embeddings`
   (writes `data/embeddings.npy` and `data/meta.json`).
2. Create a Hugging Face account and a token with write access; put `HF_TOKEN` and `HF_DATASET_REPO`
   (`your-username/snap2cook-corpus`) in `.env`, then `.\.venv\Scripts\python.exe -m scripts.upload_dataset`.
3. The app loads the corpus from that public dataset and falls back to the copy in `data/` when it is unreachable.

## Tests and syllabus demos

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m assignments.run_all     # prints Q6, Q8, Q9, Q10 demos
```

| Q | Module |
|---|---|
| 6 text tools | `snap2cook/fileio/text_tools.py` |
| 8 safe file handling | `snap2cook/fileio/safe_io.py` |
| 9 user-defined exceptions | `snap2cook/exceptions.py` |
| 10 labelled pandas Series | `snap2cook/analytics/series.py` |

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (`.env` and `.venv` are gitignored).
2. share.streamlit.io -> New app -> repo, branch, main file `app.py`.
3. Advanced settings: Python 3.11, and paste secrets (TOML):
   ```toml
   GEMINI_API_KEY = "..."
   VISION_MODEL = "gemini-3.6-flash"
   GEN_MODEL = "gemini-3.6-flash"
   FALLBACK_MODELS = "gemini-3.8-flash,gemini-3.5-flash,gemini-2.5-flash"
   EMBED_MODEL = "gemini-embedding-001"
   EMBED_DIM = "768"
   HF_DATASET_REPO = "your-username/snap2cook-corpus"
   ```

## Limitations

Free-tier Gemini quota (a rate limiter protects it), ephemeral filesystem on Community Cloud (cache and logs reset on
restart), a small synthetic corpus (90 recipes), and vision mistakes (you can edit the detected list).
