"""Centralized data path definitions for TransferZAI."""

from pathlib import Path

ROOT = Path(__file__).parent

# ── Full course catalogs ─────────────────────────────────────────────────
WM_CATALOG      = ROOT / "data" / "catalogs" / "wm_courses_2025.csv"
VT_CATALOG      = ROOT / "data" / "catalogs" / "vt_courses_2025.csv"
UCSC_CATALOG    = ROOT / "data" / "catalogs" / "ucsc_courses_2025.csv"

# ── Transfer equivalency tables ─────────────────────────────────────────
WM_MERGED       = ROOT / "data" / "equivalency" / "vccs_wm_merged.csv"
VT_MERGED       = ROOT / "data" / "equivalency" / "vccs_vt_merged.csv"
CCC_UCSC_MERGED = ROOT / "data" / "equivalency" / "ccc_ucsc_merged.csv"
CCC_UCSC_CLEAN  = ROOT / "data" / "equivalency" / "ccc_ucsc_clean.csv"

# ── Eval ─────────────────────────────────────────────────────────────────
TEST_POS        = ROOT / "data" / "_test_pos.csv"
