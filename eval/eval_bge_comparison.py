"""
Compare all four BGE fine-tune checkpoints on W&M, VT, and CCC→UCSC test sets.

Models evaluated (all loaded from local disk):
  A: finetuned_bge        — bge-small, W&M only (267 pairs), MNRL
  B: finetuned_bge_combined — bge-small, W&M+VT (~509 pairs), MNRL
  C: finetuned_bge_three  — bge-small, W&M+VT+CCC (~1218 pairs), in-batch negatives
  D: finetuned_bge_base   — bge-BASE (768-dim), all three datasets, TripletLoss, 4563 samples

Eval: BGE+TF-IDF RRF retrieval recall on held-out 20% test splits.
No cross-encoder — this measures the retrieval stage only.

Run from project root:
    python eval/eval_bge_comparison.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import WM_MERGED, WM_CATALOG, VT_MERGED, VT_CATALOG, CCC_UCSC_CLEAN, UCSC_CATALOG

import re
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

np.random.seed(42)

QUERY_PREFIX = "Represent this course for finding transfer equivalents: "
RRF_K = 60

# ── Text cleaning ─────────────────────────────────────────────

_CCC_NOISE = re.compile(
    r"may be offered in a distance.learning format"
    r"|transfer information:"
    r"|transferable to (?:both )?uc"
    r"|limitations on enrollment:"
    r"|requisites:"
    r"|minimum credit units|maximum credit units"
    r"|toggle (?:additional|general|learning)"
    r"|grade options:|see open sections"
    r"|connect with an academic counselor"
    r"|some of the class hours for this course",
    re.IGNORECASE,
)

def clean_text(text):
    if pd.isna(text) or str(text) in ("Description not found", "nan", ""):
        return ""
    text = str(text)
    m = _CCC_NOISE.search(text)
    if m:
        text = text[:m.start()]
    text = text.lower()
    text = re.sub(r"prerequisite\(s\):.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(csi|alv|nqr|additional)\b", "", text)
    text = re.sub(r"cross-listed with:.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Data loading & splits ─────────────────────────────────────

def parse_vccs(raw):
    parts = re.split(r"\s*TAKEN WITH\s*", str(raw).strip(), flags=re.IGNORECASE)
    courses = []
    for p in parts:
        m = re.match(r"([A-Z]{2,4})\s+(\d{3})\s+(.*)", p.strip())
        if m:
            courses.append({"dept": m.group(1), "title": m.group(3).strip()})
    return courses

def vccs_text(row, course_col="VCCS Course", desc_col="VCCS Description"):
    courses = parse_vccs(row[course_col])
    titles = " ".join(c["title"] for c in courses)
    desc = clean_text(row.get(desc_col, ""))
    return f"{titles} {desc}".strip()

# W&M
df_wm = pd.read_csv(WM_MERGED)
wm_catalog = pd.read_csv(WM_CATALOG, encoding="latin-1")
df_wm.columns = df_wm.columns.str.strip()
df_wm = df_wm.rename(columns={"Unnamed: 0": "idx"})

has_match = df_wm["W&M Course Code"].notna() & (df_wm["W&M Course Code"].str.strip() != "")
pos_wm = df_wm[has_match].copy()

wm_catalog = wm_catalog.dropna(subset=["course_code"])
wm_lookup = {}
for _, r in wm_catalog.iterrows():
    code = str(r["course_code"]).strip()
    wm_lookup[code] = {
        "title": str(r.get("course_title", "")),
        "description": str(r.get("course_description", "")) if pd.notna(r.get("course_description")) else "",
    }

def parse_wm_code(c):
    m = re.match(r"([A-Z]{2,4})\s+(\d{3})", str(c).strip())
    return {"dept": m.group(1)} if m else None

pos_wm["_dept"] = pos_wm["W&M Course Code"].apply(
    lambda x: parse_wm_code(x)["dept"] if parse_wm_code(x) else "UNK")
dc = pos_wm["_dept"].value_counts()
pos_wm["_strat"] = pos_wm["_dept"].apply(lambda d: "RARE" if dc[d] < 2 else d)
_, wm_test = train_test_split(pos_wm, test_size=0.20, random_state=42, stratify=pos_wm["_strat"])
print(f"W&M test: {len(wm_test)}")

# VT
df_vt = pd.read_csv(VT_MERGED)

def vt_dept(c):
    m = re.match(r"([A-Z]+)\s+\d+", str(c).strip())
    return m.group(1) if m else "UNK"

df_vt["_dept"] = df_vt["VT Course Code"].apply(lambda x: vt_dept(x.split("+")[0].strip()))
dc2 = df_vt["_dept"].value_counts()
df_vt["_strat"] = df_vt["_dept"].apply(lambda d: "RARE" if dc2[d] < 2 else d)
_, vt_test = train_test_split(df_vt, test_size=0.20, random_state=42, stratify=df_vt["_strat"])

vt_full = pd.read_csv(VT_CATALOG)
vt_full = vt_full.dropna(subset=["course_code"])
vt_lookup = {}
for _, r in vt_full.iterrows():
    code = str(r["course_code"]).strip()
    vt_lookup[code] = {
        "title": str(r.get("course_title", "")),
        "description": str(r.get("course_description", "")) if pd.notna(r.get("course_description")) else "",
    }
print(f"VT  test: {len(vt_test)}  catalog: {len(vt_lookup)}")

# CCC → UCSC
df_ccc = pd.read_csv(CCC_UCSC_CLEAN)
df_ccc.columns = df_ccc.columns.str.strip()

ucsc_full = pd.read_csv(UCSC_CATALOG)
ucsc_full = ucsc_full.dropna(subset=["course_code"])
ucsc_lookup = {}
for _, r in ucsc_full.iterrows():
    code = str(r["course_code"]).strip()
    ucsc_lookup[code] = {
        "title": str(r.get("course_title", "")),
        "description": str(r.get("course_description", "")) if pd.notna(r.get("course_description")) else "",
    }

def ucsc_dept(c):
    m = re.match(r"([A-Z]+)\s+", str(c).strip())
    return m.group(1) if m else "UNK"

df_ccc["_dept"] = df_ccc["UCSC Course Code"].apply(ucsc_dept)
dc3 = df_ccc["_dept"].value_counts()
df_ccc["_strat"] = df_ccc["_dept"].apply(lambda d: "RARE" if dc3[d] < 2 else d)
_, ccc_test = train_test_split(df_ccc, test_size=0.20, random_state=42, stratify=df_ccc["_strat"])
print(f"CCC test: {len(ccc_test)}  UCSC catalog: {len(ucsc_lookup)}")


# ── TF-IDF (fit once on all catalogs) ────────────────────────
wm_codes   = list(wm_lookup.keys())
vt_codes   = list(vt_lookup.keys())
ucsc_codes = list(ucsc_lookup.keys())

wm_texts   = [f"{wm_lookup[c]['title']} {clean_text(wm_lookup[c]['description'])}" for c in wm_codes]
vt_texts   = [f"{vt_lookup[c]['title']} {clean_text(vt_lookup[c]['description'])}" for c in vt_codes]
ucsc_texts = [f"{ucsc_lookup[c]['title']} {clean_text(ucsc_lookup[c]['description'])}" for c in ucsc_codes]

print("Fitting TF-IDF on all catalogs...")
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=1, max_df=0.95,
                        stop_words="english", sublinear_tf=True)
tfidf.fit(wm_texts + vt_texts + ucsc_texts)
print(f"TF-IDF vocab: {len(tfidf.vocabulary_)} terms")

wm_tfidf   = tfidf.transform(wm_texts)
vt_tfidf   = tfidf.transform(vt_texts)
ucsc_tfidf = tfidf.transform(ucsc_texts)


# ── RRF retrieval ─────────────────────────────────────────────

def rrf_retrieve(query_text, query_emb, cat_codes, cat_embs, cat_tfidf_mat, k=50):
    bge_sims   = cat_embs @ query_emb
    tfidf_sims = cosine_similarity(tfidf.transform([query_text]), cat_tfidf_mat).flatten()
    bge_ranked   = np.argsort(bge_sims)[::-1]
    tfidf_ranked = np.argsort(tfidf_sims)[::-1]
    rrf = defaultdict(float)
    for rank, idx in enumerate(bge_ranked[:200]):
        rrf[cat_codes[idx]] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(tfidf_ranked[:200]):
        rrf[cat_codes[idx]] += 1.0 / (RRF_K + rank + 1)
    return [c for c, _ in sorted(rrf.items(), key=lambda x: -x[1])[:k]]


# ── Eval function ─────────────────────────────────────────────

def eval_model(model, label, test_sets):
    """
    test_sets: list of (name, test_df, query_fn, target_fn, cat_codes, cat_texts, cat_tfidf)
    Returns dict: {dataset_name: {1: recall, 3: recall, 5: recall, 10: recall}}
    """
    results = {}
    for name, test_df, query_fn, target_fn, cat_codes, cat_texts, cat_tfidf in test_sets:
        t0 = time.time()
        cat_embs = model.encode(cat_texts, batch_size=64, normalize_embeddings=True,
                                show_progress_bar=False)
        hits = {k: 0 for k in [1, 3, 5, 10, 20]}
        found = 0
        for _, row in test_df.iterrows():
            query = query_fn(row)
            target = target_fn(row)
            if target not in cat_codes:
                continue
            found += 1
            q_emb = model.encode([f"{QUERY_PREFIX}{query}"], normalize_embeddings=True)[0]
            ranked = rrf_retrieve(query, q_emb, cat_codes, cat_embs, cat_tfidf)
            for k in hits:
                if target in ranked[:k]:
                    hits[k] += 1

        n = found
        r = {k: hits[k] / n for k in hits} if n > 0 else {k: 0.0 for k in hits}
        elapsed = time.time() - t0
        results[name] = r
        print(f"  {label} | {name:<10} (n={n}, {elapsed:.0f}s)  "
              f"Top-1:{r[1]:.3f}  Top-3:{r[3]:.3f}  Top-5:{r[5]:.3f}  Top-10:{r[10]:.3f}")

    return results


# ── Define test sets ──────────────────────────────────────────

def ccc_query_fn(row):
    title_raw = str(row.get("CCC Course", "")).strip()
    m = re.match(r"^[A-Z][A-Z0-9 ]*?\s+\S+\s+(.*)", title_raw)
    title = m.group(1).strip() if m else title_raw
    desc = clean_text(row.get("CCC Description", ""))
    return f"{title} {desc}".strip()

TEST_SETS = [
    ("W&M",
     wm_test,
     lambda row: vccs_text(row, "VCCS Course", "VCCS Description"),
     lambda row: str(row["W&M Course Code"]).strip(),
     wm_codes, wm_texts, wm_tfidf),

    ("VT",
     vt_test,
     lambda row: vccs_text(row, "VCCS Course", "VCCS Description"),
     lambda row: row["VT Course Code"].split("+")[0].strip(),
     vt_codes, vt_texts, vt_tfidf),

    ("CCC→UCSC",
     ccc_test,
     ccc_query_fn,
     lambda row: str(row["UCSC Course Code"]).strip(),
     ucsc_codes, ucsc_texts, ucsc_tfidf),
]

# ── Run comparisons ───────────────────────────────────────────

MODELS = [
    ("A: finetuned_bge        (small, W&M only, 267 pairs, MNRL)",           "./finetuned_bge"),
    ("B: finetuned_bge_combined (small, W&M+VT, ~509 pairs, MNRL)",          "./finetuned_bge_combined"),
    ("C: finetuned_bge_three  (small, W&M+VT+CCC, ~1218 pairs, in-batch)",   "./finetuned_bge_three"),
    ("D: finetuned_bge_base   (BASE 768-dim, all datasets, TripletLoss, 4563 samples)", "./finetuned_bge_base"),
]

all_results = {}

for label, path in MODELS:
    print(f"\n{'='*70}")
    print(f"Loading {path}...")
    model = SentenceTransformer(path, device="cpu")
    print(f"Evaluating {label}")
    results = eval_model(model, label[:20], TEST_SETS)
    all_results[label] = results
    del model  # free memory between models


# ── Summary tables ────────────────────────────────────────────

for dataset in ["W&M", "VT", "CCC→UCSC"]:
    print(f"\n{'='*72}")
    print(f"RESULTS — {dataset} held-out test set")
    print(f"{'='*72}")
    print(f"  {'Model':<56} {'Top-1':>6} {'Top-3':>6} {'Top-5':>6} {'Top-10':>7}")
    print(f"  {'─'*76}")
    for label, _ in MODELS:
        r = all_results[label][dataset]
        print(f"  {label:<56} {r[1]:>6.3f} {r[3]:>6.3f} {r[5]:>6.3f} {r[10]:>7.3f}")

# Deltas vs A
print(f"\n{'='*72}")
print("DELTAS vs A (finetuned_bge baseline)")
print(f"{'='*72}")
for dataset in ["W&M", "VT", "CCC→UCSC"]:
    print(f"\n  {dataset}:")
    r_a = all_results[MODELS[0][0]][dataset]
    for label, _ in MODELS[1:]:
        r = all_results[label][dataset]
        deltas = {k: r[k] - r_a[k] for k in [1, 3, 5, 10]}
        short = label[:4]
        print(f"    {short}: " + "  ".join(f"Top-{k}:{deltas[k]:+.3f}" for k in [1, 3, 5, 10]))

print("\nDone.")
