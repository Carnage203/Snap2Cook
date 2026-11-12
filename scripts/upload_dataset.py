"""Upload the three corpus files to the public Hugging Face dataset repo.

Run from the project root:  python -m scripts.upload_dataset
Needs HF_TOKEN (write access) and HF_DATASET_REPO in .env. The token is never written anywhere.
"""
from __future__ import annotations

import os
import sys

from dotenv import dotenv_values
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from snap2cook.config import ROOT, Settings
from snap2cook.rag.store import FILES


def main() -> int:
    """Create the dataset repo if needed and upload the files."""
    settings = Settings.load()
    token = os.environ.get("HF_TOKEN") or dotenv_values(ROOT / ".env").get("HF_TOKEN")
    if not token or not settings.hf_dataset_repo:
        print("Set HF_TOKEN and HF_DATASET_REPO in .env first.")
        return 1
    missing = [f for f in FILES if not (settings.data_dir / f).exists()]
    if missing:
        print(f"Missing files in data/: {missing}. Run `python -m scripts.build_embeddings` first.")
        return 1
    api = HfApi(token=token)
    try:
        api.create_repo(settings.hf_dataset_repo, repo_type="dataset", private=False, exist_ok=True)
        for name in FILES:
            api.upload_file(path_or_fileobj=str(settings.data_dir / name), path_in_repo=name,
                            repo_id=settings.hf_dataset_repo, repo_type="dataset")
            print(f"uploaded {name}")
    except HfHubHTTPError as exc:
        print(f"Upload failed: {exc}")
        return 1
    print(f"Done: https://huggingface.co/datasets/{settings.hf_dataset_repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
