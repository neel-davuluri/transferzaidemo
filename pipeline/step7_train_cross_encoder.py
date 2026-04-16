"""
Step 7 — Fine-tune a cross-encoder reranker on transfer equivalency pairs.

Architecture:
  BGE bi-encoder  → retrieves top-50 candidates  (fast, stays)
  LogReg          → filters to top-10             (fast, stays)
  Cross-encoder   → re-ranks top-10               (slower, much more accurate)

The cross-encoder reads query + candidate *together* as one input, so it
captures fine-grained interactions that a bi-encoder misses when courses
are semantically very similar (the top-1 / top-3 gap problem).

Base model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M params, fast inference (~10ms per pair on CPU)
  - Pre-trained on passage re-ranking, fine-tuned here on transfer pairs

Output: ./cross_encoder/
"""

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # disable MPS memory cap

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import WM_MERGED, WM_CATALOG, VT_MERGED, VT_CATALOG, CCC_UCSC_CLEAN, UCSC_CATALOG
from config import CROSS_ENCODER_PATH

import re
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

# ── Helpers for reproducible train/test splits ───────────────────────────────
# These must match test_cross_encoder.py exactly (same random_state, test_size,
# stratification logic) so that CE training never sees held-out eval rows.

def wm_train_split(df):
    has = df["W&M Course Code"].notna() & (df["W&M Course Code"].str.strip() != "")
    pos = df[has].copy()
    def _dept(c):
        m = re.match(r"([A-Z]{2,4})\s+\d+", str(c).strip())
        return m.group(1) if m else "UNK"
    pos["_dept"] = pos["W&M Course Code"].apply(_dept)
    dc = pos["_dept"].value_counts()
    pos["_strat"] = pos["_dept"].apply(lambda d: "RARE" if dc[d] < 2 else d)
    train, _ = train_test_split(pos, test_size=0.20, random_state=42, stratify=pos["_strat"])
    return train

def vt_train_split(df):
    def _dept(c):
        m = re.match(r"([A-Z]+)\s+\d+", str(c).strip())
        return m.group(1) if m else "UNK"
    df = df.copy()
    df["_dept"] = df["VT Course Code"].apply(lambda x: _dept(x.split("+")[0].strip()))
    dc = df["_dept"].value_counts()
    df["_strat"] = df["_dept"].apply(lambda d: "RARE" if dc[d] < 2 else d)
    train, _ = train_test_split(df, test_size=0.20, random_state=42, stratify=df["_strat"])
    return train

def ccc_train_split(df):
    def _dept(c):
        m = re.match(r"([A-Z]+)\s+", str(c).strip())
        return m.group(1) if m else "UNK"
    df = df.copy()
    df["_dept"] = df["UCSC Course Code"].apply(_dept)
    dc = df["_dept"].value_counts()
    df["_strat"] = df["_dept"].apply(lambda d: "RARE" if dc[d] < 2 else d)
    train, _ = train_test_split(df, test_size=0.20, random_state=42, stratify=df["_strat"])
    return train

random.seed(42)
np.random.seed(42)

RERANK_K    = 10    # how many LogReg candidates the CE will see at inference
NEGS_PER_POS = 4    # hard negatives per positive in CE training


def clean_text(text):
    if pd.isna(text) or str(text) in ("Description not found", "nan", ""):
        return ""
    text = str(text).lower()
    text = re.sub(r"prerequisite\(s\):.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(csi|alv|nqr|additional)\b", "", text)
    text = re.sub(r"cross-listed with:.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def course_text(title, desc):
    return f"{title} {clean_text(desc)}".strip()


# ══════════════════════════════════════════════════════════════
# LOAD CATALOGS
# ══════════════════════════════════════════════════════════════
print("Loading catalogs...")

def load_catalog(path, encoding="utf-8"):
    df = pd.read_csv(path, encoding=encoding).dropna(subset=["course_code"])
    lk = {}
    for _, r in df.iterrows():
        code = str(r["course_code"]).strip()
        lk[code] = {
            "title":       str(r.get("course_title", "")),
            "description": str(r.get("course_description", "")) if pd.notna(r.get("course_description")) else "",
        }
    return lk

wm_lookup   = load_catalog(WM_CATALOG, encoding="latin-1")
vt_lookup   = load_catalog(VT_CATALOG)
ucsc_lookup = load_catalog(UCSC_CATALOG)


# ══════════════════════════════════════════════════════════════
# BUILD TRAINING PAIRS
# ══════════════════════════════════════════════════════════════
# For each positive (query, correct_target), we pair it with NEGS_PER_POS
# hard negatives sampled from same-department courses in the catalog.
# These simulate the LogReg top-K pool the CE will see at inference.

def parse_vccs(raw):
    parts = re.split(r"\s*TAKEN WITH\s*", str(raw).strip(), flags=re.IGNORECASE)
    courses = []
    for p in parts:
        m = re.match(r"([A-Z]{2,4})\s+(\d{3})\s+(.*)", p.strip())
        if m:
            courses.append({"dept": m.group(1), "title": m.group(3).strip()})
    return courses

def vccs_query_text(row, course_col="VCCS Course", desc_col="VCCS Description"):
    courses = parse_vccs(row[course_col])
    titles  = " ".join(c["title"] for c in courses)
    return f"{titles} {clean_text(row.get(desc_col, ''))}".strip()

def sample_negatives(target_code, lookup, dept_prefix, n):
    """Sample n courses from the same department, excluding target."""
    same_dept = [c for c in lookup if c.startswith(dept_prefix) and c != target_code]
    if not same_dept:
        same_dept = [c for c in lookup if c != target_code]
    return random.sample(same_dept, min(n, len(same_dept)))


examples = []   # InputExample(texts=[query, candidate], label=1.0 or 0.0)

# ── W&M (train split only — 80% of positives) ───────────────────────────
df_wm = pd.read_csv(WM_MERGED)
df_wm.columns = df_wm.columns.str.strip()
wm_train = wm_train_split(df_wm)
print(f"W&M  train rows: {len(wm_train)}")

for _, row in wm_train.iterrows():
    target = str(row["W&M Course Code"]).strip()
    if target not in wm_lookup:
        continue
    query = vccs_query_text(row)
    pos_text = course_text(wm_lookup[target]["title"], wm_lookup[target]["description"])
    examples.append(InputExample(texts=[query, pos_text], label=1.0))

    dept = re.match(r"([A-Z]{2,4})", target)
    dept_prefix = dept.group(1) if dept else ""
    for neg_code in sample_negatives(target, wm_lookup, dept_prefix, NEGS_PER_POS):
        neg_text = course_text(wm_lookup[neg_code]["title"], wm_lookup[neg_code]["description"])
        examples.append(InputExample(texts=[query, neg_text], label=0.0))

# ── VT (train split only — 80% of positives) ────────────────────────────
df_vt = pd.read_csv(VT_MERGED)
vt_train = vt_train_split(df_vt)
print(f"VT   train rows: {len(vt_train)}")

for _, row in vt_train.iterrows():
    target = row["VT Course Code"].split("+")[0].strip()
    if target not in vt_lookup:
        continue
    query = vccs_query_text(row)
    pos_text = course_text(
        str(row.get("VT Course Title", "")),
        str(row.get("VT Description", "")).split("|")[0]
    )
    examples.append(InputExample(texts=[query, pos_text], label=1.0))

    dept = re.match(r"([A-Z]+)", target)
    dept_prefix = dept.group(1) if dept else ""
    for neg_code in sample_negatives(target, vt_lookup, dept_prefix, NEGS_PER_POS):
        neg_text = course_text(vt_lookup[neg_code]["title"], vt_lookup[neg_code]["description"])
        examples.append(InputExample(texts=[query, neg_text], label=0.0))

# ── CCC → UCSC (train split only — 80% of positives) ────────────────────
df_ccc = pd.read_csv(CCC_UCSC_CLEAN)
df_ccc.columns = df_ccc.columns.str.strip()
ccc_train = ccc_train_split(df_ccc)
print(f"CCC  train rows: {len(ccc_train)}")

def ccc_query_text(row):
    title_raw = str(row.get("CCC Course", "")).strip()
    m = re.match(r"^[A-Z][A-Z0-9 ]*?\s+\S+\s+(.*)", title_raw)
    title = m.group(1).strip() if m else title_raw
    return f"{title} {clean_text(row.get('CCC Description', ''))}".strip()

for _, row in ccc_train.iterrows():
    target = str(row["UCSC Course Code"]).strip()
    if target not in ucsc_lookup:
        continue
    query = ccc_query_text(row)
    pos_text = course_text(
        str(row.get("UCSC Course Title", "")),
        str(row.get("UCSC Description", ""))
    )
    examples.append(InputExample(texts=[query, pos_text], label=1.0))

    dept = re.match(r"([A-Z]+)", target)
    dept_prefix = dept.group(1) if dept else ""
    for neg_code in sample_negatives(target, ucsc_lookup, dept_prefix, NEGS_PER_POS):
        neg_text = course_text(ucsc_lookup[neg_code]["title"], ucsc_lookup[neg_code]["description"])
        examples.append(InputExample(texts=[query, neg_text], label=0.0))

random.shuffle(examples)
n_pos = sum(1 for e in examples if e.label == 1.0)
n_neg = sum(1 for e in examples if e.label == 0.0)
print(f"\nTraining examples: {len(examples)}  (pos={n_pos}, neg={n_neg}, ratio={n_neg/n_pos:.1f}:1)")

# 90/10 split for validation
split = int(len(examples) * 0.9)
train_examples = examples[:split]
val_examples   = examples[split:]
print(f"Train: {len(train_examples)}  Val: {len(val_examples)}")


# ══════════════════════════════════════════════════════════════
# FINE-TUNE CROSS-ENCODER
# ══════════════════════════════════════════════════════════════
import torch
device = "cpu"  # L-2 on CPU is faster than L-6 OOM-ing on MPS
print(f"\nDevice: {device}")
print("Loading base cross-encoder (ms-marco-MiniLM-L-2-v2)...")
ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2", num_labels=1, device=device)

train_loader = DataLoader(train_examples, shuffle=True, batch_size=16)

EPOCHS = 3
warmup = int(len(train_loader) * EPOCHS * 0.1)
print(f"Fine-tuning: {EPOCHS} epochs, {len(train_loader)} steps/epoch, warmup={warmup}")

ce_model.fit(
    train_dataloader=train_loader,
    epochs=EPOCHS,
    warmup_steps=warmup,
    show_progress_bar=True,
    use_amp=False,
)
import os
os.makedirs(CROSS_ENCODER_PATH, exist_ok=True)
ce_model.save(CROSS_ENCODER_PATH)
print(f"\nCross-encoder saved to {CROSS_ENCODER_PATH}")


# ══════════════════════════════════════════════════════════════
# QUICK EVAL — does reranking help on W&M?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("QUICK EVAL — cross-encoder reranking on W&M val pairs")
print("=" * 60)

# Use val positives: for each, the CE should rank the positive above negatives
hits_before = hits_after = 0
val_pos = [(e, i) for i, e in enumerate(val_examples) if e.label == 1.0]

for pos_ex, _ in val_pos[:100]:
    query = pos_ex.texts[0]
    # Pair with random same-dept negatives to simulate top-10 LogReg pool
    neg_pool = random.sample([e for e in val_examples if e.label == 0.0],
                              min(RERANK_K - 1, len(val_examples)))
    pool = [pos_ex] + neg_pool
    random.shuffle(pool)

    pairs  = [[query, e.texts[1]] for e in pool]
    scores = ce_model.predict(pairs)
    best_idx = int(np.argmax(scores))
    if pool[best_idx].label == 1.0:
        hits_after += 1

print(f"  CE top-1 accuracy on val (simulated pool of {RERANK_K}): {hits_after/min(100,len(val_pos)):.3f}")
print("\nDone.")
