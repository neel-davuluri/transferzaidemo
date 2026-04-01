"""
Download pre-trained artifacts from GitHub Releases.
Called automatically by predict.py if artifacts/ doesn't exist.

Usage:
  python download_artifacts.py

If this fails (e.g., on Streamlit Cloud), download manually from:
  https://github.com/neel-davuluri/transferzaidemo/releases
and extract into the repo root.
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = "neel-davuluri/transferzaidemo"
RELEASE_TAG = "v3.0"
ARTIFACT_URL = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/artifacts.tar.gz"
ARTIFACTS_DIR = "./artifacts"

def download_artifacts():
    """Download and extract artifacts if they don't exist."""
    if Path(ARTIFACTS_DIR).exists():
        print(f"{ARTIFACTS_DIR} already exists, skipping download.")
        return

    print(f"Artifacts not found. Attempting to download from {ARTIFACT_URL}...")
    try:
        # Download
        tar_path = "/tmp/artifacts.tar.gz"
        urllib.request.urlretrieve(ARTIFACT_URL, tar_path)
        print(f"Downloaded to {tar_path}")

        # Extract
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(".")
        os.remove(tar_path)
        print(f"✓ Extracted to {ARTIFACTS_DIR}")
    except Exception as e:
        print(f"✗ Failed to download artifacts: {e}", file=sys.stderr)
        print(f"\nTo fix this:", file=sys.stderr)
        print(f"1. Visit: https://github.com/{REPO}/releases/tag/{RELEASE_TAG}", file=sys.stderr)
        print(f"2. Download: artifacts.tar.gz", file=sys.stderr)
        print(f"3. Extract to repo root: tar -xzf artifacts.tar.gz", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    download_artifacts()
