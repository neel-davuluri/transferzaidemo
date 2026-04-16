"""
Download pre-trained artifacts and fine-tuned BGE model from HuggingFace Hub.
Called automatically by predict.py on first run if local paths are missing.

Manual usage:
  python download_artifacts.py
"""

import os
from pathlib import Path

ARTIFACTS_HF_REPO = "hyperalpha/transferzai-artifacts"
BGE_HF_REPO       = "hyperalpha/transferzai-bge"
ARTIFACTS_DIR     = "./artifacts"
BGE_LOCAL_PATH    = "./finetuned_bge_three"


def download_artifacts():
    """Download artifacts and BGE model from HF Hub if not present locally."""
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")

    if not Path(ARTIFACTS_DIR).exists():
        print(f"Downloading artifacts from {ARTIFACTS_HF_REPO} ...")
        snapshot_download(
            repo_id=ARTIFACTS_HF_REPO,
            repo_type="dataset",
            local_dir=ARTIFACTS_DIR,
            token=token,
            ignore_patterns=["*.git*", ".gitattributes", "README.md"],
        )
        print(f"Artifacts saved to {ARTIFACTS_DIR}")
    else:
        print(f"{ARTIFACTS_DIR} already exists, skipping.")

    if not Path(BGE_LOCAL_PATH).exists():
        print(f"Downloading fine-tuned BGE model from {BGE_HF_REPO} ...")
        snapshot_download(
            repo_id=BGE_HF_REPO,
            repo_type="model",
            local_dir=BGE_LOCAL_PATH,
            token=token,
            ignore_patterns=["*.git*", ".gitattributes", "README.md"],
        )
        print(f"BGE model saved to {BGE_LOCAL_PATH}")
    else:
        print(f"{BGE_LOCAL_PATH} already exists, skipping.")


if __name__ == "__main__":
    download_artifacts()
