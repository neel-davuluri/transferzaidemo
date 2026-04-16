"""
Build institution-specific lookup artifacts from course catalog CSVs.

Usage:
  python build_institution_artifacts.py

Reads: northeastern_courses.csv
Writes: artifacts/northeastern_lookup.pkl
        artifacts/northeastern_codes.pkl
        artifacts/northeastern_embeddings.npy

The W&M artifacts were already built by the training pipeline.
This script handles any additional institution catalogs.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from config import BGE_MODEL_PATH, QUERY_PREFIX, ARTIFACTS_DIR
from paths import NEU_CATALOG, VT_CATALOG, UCSC_CATALOG

INSTITUTIONS_TO_BUILD = {
    "northeastern": {
        "csv": str(NEU_CATALOG),
        "name": "Northeastern University",
    },
    "vt": {
        "csv": str(VT_CATALOG),
        "name": "Virginia Tech",
    },
    "ucsc": {
        "csv": str(UCSC_CATALOG),
        "name": "UC Santa Cruz",
    },
}


def clean_text(text):
    if not text or str(text) in ("", "nan", "Description not found"):
        return ""
    text = str(text).lower()
    text = re.sub(r"prerequisite\(s\):.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_lookup_from_csv(csv_path):
    """
    Parse a course catalog CSV into a lookup dict.
    Expected columns: department, course_code, course_title, course_description
    Returns: lookup dict {code: {"code": ..., "title": ..., "description": ...}}
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["course_code", "course_title"])
    df["course_code"] = df["course_code"].str.strip()
    df["course_title"] = df["course_title"].str.strip()
    df["course_description"] = df["course_description"].fillna("").str.strip()

    lookup = {}
    for _, row in df.iterrows():
        code = row["course_code"]
        lookup[code] = {
            "code": code,
            "title": row["course_title"],
            "description": row["course_description"],
        }
    return lookup


def build_institution_artifacts(key, config, bge_model, tfidf=None):
    csv_path = config["csv"]
    if not Path(csv_path).exists():
        print(f"  Skipping {key}: {csv_path} not found")
        return False

    print(f"\nBuilding artifacts for: {config['name']}")
    lookup = build_lookup_from_csv(csv_path)
    codes = sorted(lookup.keys())
    print(f"  Loaded {len(codes)} courses")

    texts = [
        f"{lookup[c]['title']} {clean_text(lookup[c]['description'])}"
        for c in codes
    ]

    print(f"  Computing BGE embeddings ({len(codes)} courses)...")
    embeddings = bge_model.encode(
        [f"{QUERY_PREFIX}{t}" for t in texts],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )

    out_dir = Path(ARTIFACTS_DIR)
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / f"{key}_lookup.pkl", "wb") as f:
        pickle.dump(lookup, f)
    with open(out_dir / f"{key}_codes.pkl", "wb") as f:
        pickle.dump(codes, f)
    np.save(out_dir / f"{key}_embeddings.npy", embeddings)

    print(f"  Saved {key}_lookup.pkl, {key}_codes.pkl, {key}_embeddings.npy")
    return True


def main():
    print("Loading BGE model...")
    bge_model = SentenceTransformer(
        BGE_MODEL_PATH, device="cpu",
        token=os.environ.get("HF_TOKEN")
    )
    print("Model loaded.")

    built = 0
    for key, config in INSTITUTIONS_TO_BUILD.items():
        out_path = Path(ARTIFACTS_DIR) / f"{key}_embeddings.npy"
        if out_path.exists():
            print(f"Skipping {key}: artifacts already exist (delete to rebuild)")
            continue
        if build_institution_artifacts(key, config, bge_model):
            built += 1

    print(f"\nDone. Built artifacts for {built} institution(s).")


if __name__ == "__main__":
    main()
