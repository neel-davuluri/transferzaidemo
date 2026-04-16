"""
Step 1 — Leakage-free data split
Stratified 80/20 on pos_df by W&M department. dept_map from train_pos only.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import WM_MERGED, WM_CATALOG

import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

np.random.seed(42)

# ── Load data exactly as in transferzaieval.py ──────────────────────────
df = pd.read_csv(WM_MERGED)
wm_catalog = pd.read_csv(WM_CATALOG, encoding="latin-1")

df.columns = df.columns.str.strip()
df = df.rename(columns={"Unnamed: 0": "idx"})

has_match = df["W&M Course Code"].notna() & (df["W&M Course Code"].str.strip() != "")
pos_df = df[has_match].copy()
neg_df = df[~has_match].copy()

wm_catalog = wm_catalog.dropna(subset=["course_code"])
wm_lookup = {}
for _, r in wm_catalog.iterrows():
    code = str(r["course_code"]).strip()
    wm_lookup[code] = {
        "code": code,
        "title": str(r.get("course_title", "")),
        "description": str(r.get("course_description", "")) if pd.notna(r.get("course_description")) else "",
    }

# Fill missing W&M descriptions from catalog
for idx, row in pos_df.iterrows():
    wm_code = str(row["W&M Course Code"]).strip()
    if pd.isna(row["W&M Description"]) or row["W&M Description"] == "":
        if wm_code in wm_lookup and wm_lookup[wm_code]["description"]:
            pos_df.at[idx, "W&M Description"] = wm_lookup[wm_code]["description"]
            if pd.isna(row.get("W&M Course Title")) or row.get("W&M Course Title") == "":
                pos_df.at[idx, "W&M Course Title"] = wm_lookup[wm_code]["title"]

print(f"Total rows: {len(df)}")
print(f"Positive (transfers): {len(pos_df)}  |  Negative (no transfer): {len(neg_df)}")
print(f"W&M catalog: {len(wm_lookup)} courses")


# ── Parse course codes (from transferzaieval.py) ────────────────────────
def parse_wm_course(code_str):
    if pd.isna(code_str):
        return None
    m = re.match(r"([A-Z]{2,4})\s+(\d{3})", str(code_str).strip())
    if m:
        return {"dept": m.group(1), "number": int(m.group(2)), "full": f"{m.group(1)} {m.group(2)}"}
    return None


def parse_vccs_course(raw):
    raw = str(raw).strip()
    parts = re.split(r"\s*TAKEN WITH\s*", raw, flags=re.IGNORECASE)
    courses = []
    for part in parts:
        m = re.match(r"([A-Z]{2,4})\s+(\d{3})\s+(.*)", part.strip())
        if m:
            courses.append({
                "dept": m.group(1),
                "number": int(m.group(2)),
                "title": m.group(3).strip(),
                "full": f"{m.group(1)} {m.group(2)}",
            })
    return courses


# ── Extract W&M department for stratification ───────────────────────────
pos_df["wm_dept"] = pos_df["W&M Course Code"].apply(
    lambda x: parse_wm_course(x)["dept"] if parse_wm_course(x) else "UNK"
)

# Departments with only 1 sample can't be stratified — group them
dept_counts = pos_df["wm_dept"].value_counts()
rare_depts = dept_counts[dept_counts < 2].index.tolist()
pos_df["strat_dept"] = pos_df["wm_dept"].apply(lambda d: "RARE" if d in rare_depts else d)

print(f"\nW&M departments in positive pairs: {pos_df['wm_dept'].nunique()}")
print(f"Departments with <2 samples (grouped as RARE): {len(rare_depts)}")

# ── Stratified 80/20 split ──────────────────────────────────────────────
train_pos, test_pos = train_test_split(
    pos_df, test_size=0.20, random_state=42, stratify=pos_df["strat_dept"]
)

print(f"\n{'='*50}")
print(f"SPLIT SIZES")
print(f"{'='*50}")
print(f"  train_pos: {len(train_pos)}")
print(f"  test_pos:  {len(test_pos)}  (HELD OUT — untouched until Step 6)")
print(f"  neg_df:    {len(neg_df)}    (used for hard negative mining in training)")

# ── Build dept_map from train_pos ONLY ──────────────────────────────────
dept_map = defaultdict(lambda: defaultdict(int))
for _, r in train_pos.iterrows():
    vccs_courses = parse_vccs_course(r["VCCS Course"])
    wm_parsed = parse_wm_course(r["W&M Course Code"])
    if vccs_courses and wm_parsed:
        for vc in vccs_courses:
            dept_map[vc["dept"]][wm_parsed["dept"]] += 1

print(f"\n{'='*50}")
print(f"DEPARTMENT MAP (from train_pos only, {len(dept_map)} VCCS depts)")
print(f"{'='*50}")
for vd in sorted(dept_map.keys())[:10]:
    targets = sorted(dept_map[vd].items(), key=lambda x: -x[1])[:3]
    print(f"  {vd} -> {', '.join(f'{d}({c})' for d, c in targets)}")

# ── Class balance ───────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"CLASS BALANCE CHECK")
print(f"{'='*50}")
print(f"  train_pos label=1: {len(train_pos)}")
print(f"  test_pos  label=1: {len(test_pos)}")

# Verify dept distribution preserved
train_dept_dist = train_pos["wm_dept"].value_counts(normalize=True).sort_index()
test_dept_dist = test_pos["wm_dept"].value_counts(normalize=True).sort_index()
print(f"\n  Top departments (train vs test proportion):")
for dept in train_dept_dist.head(8).index:
    tr = train_dept_dist.get(dept, 0)
    te = test_dept_dist.get(dept, 0)
    print(f"    {dept:>6}: train={tr:.3f}  test={te:.3f}")

# ── Verify no leakage ──────────────────────────────────────────────────
train_idx_set = set(train_pos.index)
test_idx_set = set(test_pos.index)
assert train_idx_set.isdisjoint(test_idx_set), "LEAKAGE: overlapping indices!"
print(f"\n  Leakage check: PASSED (0 overlapping indices)")
print(f"  test_pos indices stored — will not be touched until Step 6.")
