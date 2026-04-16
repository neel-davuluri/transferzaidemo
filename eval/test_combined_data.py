"""
Quick comparison: W&M-only fine-tuning vs W&M+VT combined fine-tuning.

Conditions:
  A (baseline): finetuned_bge trained on 267 W&M pairs (already saved)
  B (combined): BGE re-fine-tuned on W&M train + VT train pairs

Both evaluated on the SAME fixed W&M held-out test set (_test_pos.csv).
Also reports VT retrieval recall as a bonus.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import WM_MERGED, WM_CATALOG, VT_MERGED, VT_CATALOG

import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

np.random.seed(42)

QUERY_PREFIX = "Represent this course for finding transfer equivalents: "

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

# ── W&M data (replicate existing split exactly) ─────────────────
df_wm = pd.read_csv(WM_MERGED)
wm_catalog = pd.read_csv(WM_CATALOG, encoding="latin-1")
df_wm.columns = df_wm.columns.str.strip()
df_wm = df_wm.rename(columns={"Unnamed: 0": "idx"})

has_match = df_wm["W&M Course Code"].notna() & (df_wm["W&M Course Code"].str.strip() != "")
pos_df = df_wm[has_match].copy()

wm_catalog = wm_catalog.dropna(subset=["course_code"])
wm_lookup = {}
for _, r in wm_catalog.iterrows():
    code = str(r["course_code"]).strip()
    wm_lookup[code] = {
        "code": code,
        "title": str(r.get("course_title", "")),
        "description": str(r.get("course_description", "")) if pd.notna(r.get("course_description")) else "",
    }

for idx, row in pos_df.iterrows():
    wm_code = str(row["W&M Course Code"]).strip()
    if pd.isna(row["W&M Description"]) or row["W&M Description"] == "":
        if wm_code in wm_lookup and wm_lookup[wm_code]["description"]:
            pos_df.at[idx, "W&M Description"] = wm_lookup[wm_code]["description"]

# ── VT data ─────────────────────────────────────────────────────
df_vt = pd.read_csv(VT_MERGED)

# ── Helpers ─────────────────────────────────────────────────────
def parse_wm_course(code_str):
    if pd.isna(code_str): return None
    m = re.match(r"([A-Z]{2,4})\s+(\d{3})", str(code_str).strip())
    return {"dept": m.group(1), "number": int(m.group(2))} if m else None

def parse_vccs_course(raw):
    raw = str(raw).strip()
    parts = re.split(r"\s*TAKEN WITH\s*", raw, flags=re.IGNORECASE)
    courses = []
    for part in parts:
        m = re.match(r"([A-Z]{2,4})\s+(\d{3})\s+(.*)", part.strip())
        if m:
            courses.append({"dept": m.group(1), "number": int(m.group(2)), "title": m.group(3).strip()})
    return courses

def clean_text(text):
    if pd.isna(text) or str(text) in ("Description not found", "nan", ""): return ""
    text = str(text).lower()
    text = re.sub(r"prerequisite\(s\):.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(csi|alv|nqr|additional)\b", "", text)   # fixed: lowercase after .lower()
    text = re.sub(r"cross-listed with:.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def get_vccs_text(row, course_col="VCCS Course", desc_col="VCCS Description"):
    courses = parse_vccs_course(row[course_col])
    titles = " ".join(c["title"] for c in courses)
    desc = clean_text(row.get(desc_col, ""))
    return f"{titles} {desc}"

# ── Reproduce exact W&M train/test split ────────────────────────
pos_df["wm_dept"] = pos_df["W&M Course Code"].apply(
    lambda x: parse_wm_course(x)["dept"] if parse_wm_course(x) else "UNK")
dept_counts = pos_df["wm_dept"].value_counts()
rare_depts = dept_counts[dept_counts < 2].index.tolist()
pos_df["strat_dept"] = pos_df["wm_dept"].apply(lambda d: "RARE" if d in rare_depts else d)
wm_train, wm_test = train_test_split(pos_df, test_size=0.20, random_state=42, stratify=pos_df["strat_dept"])
print(f"W&M  train: {len(wm_train)}  test: {len(wm_test)}")

# ── VT train/test split (80/20, stratify by VT dept prefix) ────
def vt_dept(code_str):
    m = re.match(r"([A-Z]+)\s+\d+", str(code_str).strip())
    return m.group(1) if m else "UNK"

df_vt["vt_dept"] = df_vt["VT Course Code"].apply(lambda x: vt_dept(x.split("+")[0].strip()))
vt_dept_counts = df_vt["vt_dept"].value_counts()
rare_vt = vt_dept_counts[vt_dept_counts < 2].index.tolist()
df_vt["strat_vt"] = df_vt["vt_dept"].apply(lambda d: "RARE" if d in rare_vt else d)
vt_train, vt_test = train_test_split(df_vt, test_size=0.20, random_state=42, stratify=df_vt["strat_vt"])
print(f"VT   train: {len(vt_train)}  test: {len(vt_test)}")
print(f"Combined training positives: {len(wm_train) + len(vt_train)}")


# ══════════════════════════════════════════════════════════════
# BUILD W&M CATALOG INDEX (shared by both conditions)
# ══════════════════════════════════════════════════════════════
wm_codes = list(wm_lookup.keys())
wm_texts = [f"{wm_lookup[c]['title']} {clean_text(wm_lookup[c]['description'])}" for c in wm_codes]

print(f"\nW&M catalog: {len(wm_codes)} courses")

# TF-IDF (fit once, shared)
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95,
                        stop_words="english", sublinear_tf=True)
tfidf.fit(wm_texts)


# ══════════════════════════════════════════════════════════════
# RETRIEVAL EVAL FUNCTION (two-signal RRF on W&M catalog)
# ══════════════════════════════════════════════════════════════
from sklearn.metrics.pairwise import cosine_similarity

def eval_retrieval_wm(model, label, eval_df=None):
    """Evaluate retrieval recall on W&M test set."""
    if eval_df is None:
        eval_df = wm_test

    embs = model.encode(wm_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    tfidf_mat = tfidf.transform(wm_texts)

    hits = {k: 0 for k in [1, 3, 5, 10, 20, 50]}
    for _, row in eval_df.iterrows():
        vccs_text = get_vccs_text(row, "VCCS Course", "VCCS Description")
        target = str(row["W&M Course Code"]).strip()

        q_emb = model.encode([f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True)[0]
        bge_sims = embs @ q_emb
        bge_ranked = np.argsort(bge_sims)[::-1]

        tfidf_vec = tfidf.transform([vccs_text])
        tfidf_sims = cosine_similarity(tfidf_vec, tfidf_mat).flatten()
        tfidf_ranked = np.argsort(tfidf_sims)[::-1]

        RRF_K = 60
        rrf = defaultdict(float)
        for rank, idx in enumerate(bge_ranked[:200]):
            rrf[wm_codes[idx]] += 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(tfidf_ranked[:200]):
            rrf[wm_codes[idx]] += 1.0 / (RRF_K + rank + 1)

        ranked_codes = [c for c, _ in sorted(rrf.items(), key=lambda x: -x[1])]
        for k in hits:
            if target in ranked_codes[:k]:
                hits[k] += 1

    n = len(eval_df)
    recalls = {k: hits[k] / n for k in hits}
    print(f"\n  {label} (n={n}):")
    print(f"    Top-1: {recalls[1]:.3f}  Top-3: {recalls[3]:.3f}  Top-5: {recalls[5]:.3f}"
          f"  Top-10: {recalls[10]:.3f}  Top-50: {recalls[50]:.3f}")
    return recalls


def eval_retrieval_vt(model, label, eval_df=None):
    """Evaluate VT retrieval: catalog = full VT undergraduate catalog (vt_courses_2025.csv)."""
    if eval_df is None:
        eval_df = vt_test

    # Full VT catalog — same approach as W&M uses wm_courses_2025.csv
    vt_full = pd.read_csv(VT_CATALOG)
    vt_full = vt_full.dropna(subset=["course_code"])
    vt_catalog = {}
    for _, row in vt_full.iterrows():
        code = str(row["course_code"]).strip()
        vt_catalog[code] = {
            "title": str(row.get("course_title", "")),
            "description": str(row.get("course_description", "")) if pd.notna(row.get("course_description")) else "",
        }

    vt_codes_list = list(vt_catalog.keys())
    vt_texts_list = [f"{vt_catalog[c]['title']} {clean_text(vt_catalog[c]['description'])}"
                     for c in vt_codes_list]

    vt_embs = model.encode(vt_texts_list, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    vt_tfidf_mat = tfidf.transform(vt_texts_list)

    hits = {k: 0 for k in [1, 3, 5, 10, 20]}
    found = 0
    for _, row in eval_df.iterrows():
        vccs_text = get_vccs_text(row, "VCCS Course", "VCCS Description")
        # Target is the first VT code in the pair
        target = row["VT Course Code"].split("+")[0].strip()
        if target not in vt_codes_list:
            continue
        found += 1

        q_emb = model.encode([f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True)[0]
        bge_sims = vt_embs @ q_emb
        bge_ranked = np.argsort(bge_sims)[::-1]

        tfidf_vec = tfidf.transform([vccs_text])
        tfidf_sims = cosine_similarity(tfidf_vec, vt_tfidf_mat).flatten()
        tfidf_ranked = np.argsort(tfidf_sims)[::-1]

        RRF_K = 60
        rrf = defaultdict(float)
        for rank, idx in enumerate(bge_ranked[:200]):
            rrf[vt_codes_list[idx]] += 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(tfidf_ranked[:200]):
            rrf[vt_codes_list[idx]] += 1.0 / (RRF_K + rank + 1)

        ranked_codes = [c for c, _ in sorted(rrf.items(), key=lambda x: -x[1])]
        for k in hits:
            if target in ranked_codes[:k]:
                hits[k] += 1

    n = found
    if n == 0:
        print(f"\n  {label}: no evaluable rows")
        return {}
    recalls = {k: hits[k] / n for k in hits}
    print(f"\n  {label} (n={n}):")
    print(f"    Top-1: {recalls[1]:.3f}  Top-3: {recalls[3]:.3f}  Top-5: {recalls[5]:.3f}"
          f"  Top-10: {recalls[10]:.3f}  Top-20: {recalls[20]:.3f}")
    return recalls


# ══════════════════════════════════════════════════════════════
# CONDITION A — Baseline (W&M-only fine-tuned BGE, already saved)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CONDITION A: Baseline (finetuned on W&M only, 267 pairs)")
print("=" * 60)

model_a = SentenceTransformer("./finetuned_bge", device="cpu")
recalls_a_wm = eval_retrieval_wm(model_a, "A: W&M test")
recalls_a_vt = eval_retrieval_vt(model_a, "A: VT test (baseline, never seen VT data)")


# ══════════════════════════════════════════════════════════════
# CONDITION B — Combined W&M + VT fine-tuning
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"CONDITION B: Fine-tune on W&M ({len(wm_train)}) + VT ({len(vt_train)}) = "
      f"{len(wm_train)+len(vt_train)} pairs")
print("=" * 60)

# Build training examples from W&M train
train_examples = []
for _, row in wm_train.iterrows():
    vccs_text = get_vccs_text(row, "VCCS Course", "VCCS Description")
    wm_code = str(row["W&M Course Code"]).strip()
    wm_info = wm_lookup.get(wm_code, {})
    wm_text = f"{wm_info.get('title', '')} {clean_text(wm_info.get('description', ''))}"
    if vccs_text.strip() and wm_text.strip():
        train_examples.append(InputExample(texts=[f"{QUERY_PREFIX}{vccs_text}", wm_text]))

# Add VT train pairs
for _, row in vt_train.iterrows():
    vccs_text = get_vccs_text(row, "VCCS Course", "VCCS Description")
    vt_title = str(row.get("VT Course Title", ""))
    # Use first description if multi-course (split on |)
    vt_raw_desc = str(row.get("VT Description", ""))
    vt_desc = vt_raw_desc.split("|")[0].strip()
    vt_text = f"{vt_title} {clean_text(vt_desc)}"
    if vccs_text.strip() and vt_text.strip():
        train_examples.append(InputExample(texts=[f"{QUERY_PREFIX}{vccs_text}", vt_text]))

print(f"Training examples: {len(train_examples)} "
      f"(W&M: {len(wm_train)}, VT: {len(vt_train)})")

# Fine-tune from base BGE (not from the W&M-only checkpoint, to isolate the effect)
print("Loading base BGE-small...")
model_b = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

loader = DataLoader(train_examples, shuffle=True, batch_size=16)
loss_fn = losses.MultipleNegativesRankingLoss(model_b)
epochs = 3
warmup = int(len(loader) * epochs * 0.1)
print(f"Fine-tuning: {epochs} epochs, {len(loader)} steps/epoch, "
      f"{len(loader)*epochs} total steps, warmup={warmup}")

model_b.fit(
    train_objectives=[(loader, loss_fn)],
    epochs=epochs,
    warmup_steps=warmup,
    output_path="./finetuned_bge_combined",
    show_progress_bar=True,
)
print("Done fine-tuning.")

recalls_b_wm = eval_retrieval_wm(model_b, "B: W&M test")
recalls_b_vt = eval_retrieval_vt(model_b, "B: VT test (trained on VT data)")


# ══════════════════════════════════════════════════════════════
# ALSO: Baseline with BASE BGE (no fine-tuning) for reference
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CONDITION C: Off-the-shelf BGE-small (no fine-tuning, reference)")
print("=" * 60)

model_c = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
recalls_c_wm = eval_retrieval_wm(model_c, "C: W&M test")


# ══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RESULTS SUMMARY — W&M held-out test set (n=67)")
print("=" * 70)
print(f"  {'Condition':<42} {'Top-1':>6} {'Top-3':>6} {'Top-5':>6} {'Top-10':>7} {'Top-50':>7}")
print(f"  {'─'*68}")

for label, recalls in [
    ("C: Off-the-shelf BGE (no fine-tune)", recalls_c_wm),
    ("A: Fine-tuned W&M only (267 pairs)", recalls_a_wm),
    ("B: Fine-tuned W&M+VT (combined)",    recalls_b_wm),
]:
    print(f"  {label:<42} {recalls[1]:>6.3f} {recalls[3]:>6.3f} {recalls[5]:>6.3f}"
          f" {recalls[10]:>7.3f} {recalls[50]:>7.3f}")

print(f"\n  Delta B vs A (combined - W&M only):")
for k in [1, 3, 5, 10, 50]:
    d = recalls_b_wm[k] - recalls_a_wm[k]
    print(f"    Top-{k:>2}: {d:+.3f}")

print("\n" + "=" * 70)
print("RESULTS SUMMARY — VT held-out test set (n~60)")
print("=" * 70)
print(f"  {'Condition':<42} {'Top-1':>6} {'Top-3':>6} {'Top-5':>6} {'Top-10':>7}")
print(f"  {'─'*62}")
for label, recalls in [
    ("A: Fine-tuned W&M only (never saw VT)", recalls_a_vt),
    ("B: Fine-tuned W&M+VT (combined)",       recalls_b_vt),
]:
    if recalls:
        print(f"  {label:<42} {recalls[1]:>6.3f} {recalls[3]:>6.3f} {recalls[5]:>6.3f}"
              f" {recalls[10]:>7.3f}")
