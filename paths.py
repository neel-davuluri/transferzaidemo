"""
Centralized data path definitions for TransferZAI.

All scripts import from here so paths are defined in one place.
Run scripts from the project root: python pipeline/step1_split.py
"""

from pathlib import Path

ROOT = Path(__file__).parent

# ── Full course catalogs ─────────────────────────────────────────────────
WM_CATALOG      = ROOT / "data" / "catalogs" / "wm_courses_2025.csv"
VT_CATALOG      = ROOT / "data" / "catalogs" / "vt_courses_2025.csv"
UCSC_CATALOG    = ROOT / "data" / "catalogs" / "ucsc_courses_2025.csv"
NEU_CATALOG     = ROOT / "data" / "catalogs" / "northeastern_courses.csv"

# ── Transfer equivalency tables ─────────────────────────────────────────
WM_MERGED       = ROOT / "data" / "equivalency" / "vccs_wm_merged.csv"
VT_MERGED       = ROOT / "data" / "equivalency" / "vccs_vt_merged.csv"
CCC_UCSC_MERGED = ROOT / "data" / "equivalency" / "ccc_ucsc_merged.csv"
CCC_UCSC_CLEAN  = ROOT / "data" / "equivalency" / "ccc_ucsc_clean.csv"

# ── Pipeline intermediate outputs ────────────────────────────────────────
TRAIN_PAIRS     = ROOT / "data" / "_train_pairs.csv"
TEST_POS        = ROOT / "data" / "_test_pos.csv"
SYNTH_NEG_CACHE = ROOT / "data" / "_cache_synthetic_negatives.json"
