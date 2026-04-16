"""
Step 2 — BGE-mined hard negatives across all three institutions.

For each positive training pair (W&M, VT, CCC→UCSC), runs BGE retrieval
against the full institution catalog and takes the top retrieved courses
that are NOT the correct answer as hard negatives.

These are the hardest negatives possible — courses the model already
confuses with the correct answer — making the classifier training maximally
informative.

Output: data/_train_pairs.csv
Schema: query_course, query_desc, target_code, target_title, target_desc,
        label, source, institution
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import (
    WM_MERGED, WM_CATALOG, VT_MERGED, VT_CATALOG,
    CCC_UCSC_CLEAN, UCSC_CATALOG, TRAIN_PAIRS, TEST_POS,
)

import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

np.random.seed(42)

QUERY_PREFIX = "Represent this course for finding transfer equivalents: "
HARD_NEGS_PER_POSITIVE = 3   # hard negatives per positive pair
RRF_K = 60


# ══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════

def clean_text(text):
    if pd.isna(text) or str(text) in ("Description not found", "nan", ""):
        return ""
    text = str(text).lower()
    text = re.sub(r"prerequisite\(s\):.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(csi|alv|nqr|additional)\b", "", text)
    text = re.sub(r"cross-listed with:.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_vccs_course(raw):
    raw = str(raw).strip()
    parts = re.split(r"\s*TAKEN WITH\s*", raw, flags=re.IGNORECASE)
    courses = []
    for part in parts:
        m = re.match(r"([A-Z]{2,4})\s+(\d{3})\s+(.*)", part.strip())
        if m:
            courses.append({"dept": m.group(1), "number": int(m.group(2)),
                            "title": m.group(3).strip()})
    return courses


def build_catalog_index(codes, texts, model, tfidf):
    """Embed and TF-IDF a catalog. Returns (embeddings, tfidf_matrix)."""
    print(f"  Embedding {len(codes)} courses...")
    embs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    tfidf_mat = tfidf.transform(texts)
    return embs, tfidf_mat


def retrieve_hard_negatives(query_text, query_emb, catalog_codes, catalog_embs,
                             tfidf, tfidf_mat, exclude_codes, k=HARD_NEGS_PER_POSITIVE):
    """RRF retrieval; returns top-k codes that are not in exclude_codes."""
    bge_sims = catalog_embs @ query_emb
    bge_ranked = np.argsort(bge_sims)[::-1]

    tfidf_vec = tfidf.transform([query_text])
    tfidf_sims = cosine_similarity(tfidf_vec, tfidf_mat).flatten()
    tfidf_ranked = np.argsort(tfidf_sims)[::-1]

    rrf = defaultdict(float)
    for rank, idx in enumerate(bge_ranked[:200]):
        code = catalog_codes[idx]
        if code not in exclude_codes:
            rrf[code] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(tfidf_ranked[:200]):
        code = catalog_codes[idx]
        if code not in exclude_codes:
            rrf[code] += 1.0 / (RRF_K + rank + 1)

    return [c for c, _ in sorted(rrf.items(), key=lambda x: -x[1])[:k]]


# ══════════════════════════════════════════════════════════════
# LOAD MODEL + FIT SHARED TF-IDF
# ══════════════════════════════════════════════════════════════
print("Loading BGE model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

# Fit TF-IDF on all three catalogs combined for shared vocabulary
print("\nLoading catalogs...")
wm_cat  = pd.read_csv(WM_CATALOG, encoding="latin-1").dropna(subset=["course_code"])
vt_cat  = pd.read_csv(VT_CATALOG).dropna(subset=["course_code"])
ucsc_cat = pd.read_csv(UCSC_CATALOG).dropna(subset=["course_code"])

def cat_to_lookup(df):
    lk = {}
    for _, r in df.iterrows():
        code = str(r["course_code"]).strip()
        lk[code] = {"title": str(r.get("course_title", "")),
                    "description": str(r.get("course_description", "")) if pd.notna(r.get("course_description")) else ""}
    return lk

wm_lookup   = cat_to_lookup(wm_cat)
vt_lookup   = cat_to_lookup(vt_cat)
ucsc_lookup = cat_to_lookup(ucsc_cat)

def lookup_texts(lk):
    codes = list(lk.keys())
    texts = [f"{lk[c]['title']} {clean_text(lk[c]['description'])}" for c in codes]
    return codes, texts

wm_codes,   wm_texts   = lookup_texts(wm_lookup)
vt_codes,   vt_texts   = lookup_texts(vt_lookup)
ucsc_codes, ucsc_texts = lookup_texts(ucsc_lookup)

print(f"  W&M:  {len(wm_codes)} | VT: {len(vt_codes)} | UCSC: {len(ucsc_codes)}")

tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=1, max_df=0.95,
                        stop_words="english", sublinear_tf=True)
tfidf.fit(wm_texts + vt_texts + ucsc_texts)
print(f"TF-IDF vocab: {len(tfidf.vocabulary_)} terms")


# ══════════════════════════════════════════════════════════════
# INSTITUTION 1 — W&M
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("W&M hard negatives")
print("=" * 60)

df_wm = pd.read_csv(WM_MERGED)
df_wm.columns = df_wm.columns.str.strip()
df_wm = df_wm.rename(columns={"Unnamed: 0": "idx"})
has_match = df_wm["W&M Course Code"].notna() & (df_wm["W&M Course Code"].str.strip() != "")
pos_wm  = df_wm[has_match].copy()
neg_wm  = df_wm[~has_match].copy()

# Backfill W&M descriptions from catalog
for idx, row in pos_wm.iterrows():
    wm_code = str(row["W&M Course Code"]).strip()
    if pd.isna(row.get("W&M Description")) or row.get("W&M Description") == "":
        if wm_code in wm_lookup and wm_lookup[wm_code]["description"]:
            pos_wm.at[idx, "W&M Description"] = wm_lookup[wm_code]["description"]

def parse_wm_code(code_str):
    if pd.isna(code_str): return None
    m = re.match(r"([A-Z]{2,4})\s+(\d{3})", str(code_str).strip())
    return {"dept": m.group(1)} if m else None

pos_wm["wm_dept"] = pos_wm["W&M Course Code"].apply(
    lambda x: parse_wm_code(x)["dept"] if parse_wm_code(x) else "UNK")
dc = pos_wm["wm_dept"].value_counts()
pos_wm["strat"] = pos_wm["wm_dept"].apply(lambda d: "RARE" if dc[d] < 2 else d)
wm_train, wm_test = train_test_split(pos_wm, test_size=0.20, random_state=42, stratify=pos_wm["strat"])
print(f"W&M train: {len(wm_train)}  test: {len(wm_test)}")

wm_embs, wm_tfidf_mat = build_catalog_index(wm_codes, wm_texts, model, tfidf)

wm_pairs = []

# Positives
for _, row in wm_train.iterrows():
    wm_pairs.append({
        "query_course": str(row["VCCS Course"]),
        "query_desc":   str(row.get("VCCS Description", "")),
        "target_code":  str(row["W&M Course Code"]).strip(),
        "target_title": str(row.get("W&M Course Title", "")),
        "target_desc":  str(row.get("W&M Description", "")),
        "label": 1, "source": "ground_truth", "institution": "wm",
    })

# Hard negatives from train positives
print("Mining hard negatives from train_pos...")
for _, row in wm_train.iterrows():
    vccs_text = f"{' '.join(c['title'] for c in parse_vccs_course(row['VCCS Course']))} {clean_text(row.get('VCCS Description',''))}"
    vccs_emb  = model.encode([f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True)[0]
    target    = str(row["W&M Course Code"]).strip()
    for neg_code in retrieve_hard_negatives(vccs_text, vccs_emb, wm_codes, wm_embs,
                                             tfidf, wm_tfidf_mat, {target}):
        info = wm_lookup.get(neg_code, {})
        wm_pairs.append({
            "query_course": str(row["VCCS Course"]),
            "query_desc":   str(row.get("VCCS Description", "")),
            "target_code":  neg_code,
            "target_title": info.get("title", ""),
            "target_desc":  info.get("description", ""),
            "label": 0, "source": "retriever_hard", "institution": "wm",
        })

# No-transfer negatives (VCCS courses that genuinely don't transfer)
print("Mining no-transfer negatives from neg_df...")
for _, row in neg_wm.iterrows():
    vccs_text = f"{' '.join(c['title'] for c in parse_vccs_course(row['VCCS Course']))} {clean_text(row.get('VCCS Description',''))}"
    vccs_emb  = model.encode([f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True)[0]
    for neg_code in retrieve_hard_negatives(vccs_text, vccs_emb, wm_codes, wm_embs,
                                             tfidf, wm_tfidf_mat, set(), k=1):
        info = wm_lookup.get(neg_code, {})
        wm_pairs.append({
            "query_course": str(row["VCCS Course"]),
            "query_desc":   str(row.get("VCCS Description", "")),
            "target_code":  neg_code,
            "target_title": info.get("title", ""),
            "target_desc":  info.get("description", ""),
            "label": 0, "source": "no_transfer", "institution": "wm",
        })

print(f"W&M pairs: {len(wm_pairs)}  (pos={sum(p['label'] for p in wm_pairs)}, neg={sum(1-p['label'] for p in wm_pairs)})")


# ══════════════════════════════════════════════════════════════
# INSTITUTION 2 — VT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("VT hard negatives")
print("=" * 60)

df_vt = pd.read_csv(VT_MERGED)

def vt_dept(code_str):
    m = re.match(r"([A-Z]+)\s+\d+", str(code_str).strip())
    return m.group(1) if m else "UNK"

df_vt["vt_dept"] = df_vt["VT Course Code"].apply(lambda x: vt_dept(x.split("+")[0].strip()))
dc_vt = df_vt["vt_dept"].value_counts()
df_vt["strat"] = df_vt["vt_dept"].apply(lambda d: "RARE" if dc_vt[d] < 2 else d)
vt_train, vt_test = train_test_split(df_vt, test_size=0.20, random_state=42, stratify=df_vt["strat"])
print(f"VT train: {len(vt_train)}  test: {len(vt_test)}")

vt_embs, vt_tfidf_mat = build_catalog_index(vt_codes, vt_texts, model, tfidf)

vt_pairs = []

for _, row in vt_train.iterrows():
    vccs_text = f"{' '.join(c['title'] for c in parse_vccs_course(row['VCCS Course']))} {clean_text(row.get('VCCS Description',''))}"
    target_code = row["VT Course Code"].split("+")[0].strip()
    target_info = vt_lookup.get(target_code, {})

    # Positive
    vt_pairs.append({
        "query_course": str(row["VCCS Course"]),
        "query_desc":   str(row.get("VCCS Description", "")),
        "target_code":  target_code,
        "target_title": str(row.get("VT Course Title", "")),
        "target_desc":  str(row.get("VT Description", "")).split("|")[0].strip(),
        "label": 1, "source": "ground_truth", "institution": "vt",
    })

    # Hard negatives
    vccs_emb = model.encode([f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True)[0]
    # Exclude all VT codes in this equivalency row (may be multi-course)
    all_targets = {c.strip() for c in str(row["VT Course Code"]).split("+")}
    for neg_code in retrieve_hard_negatives(vccs_text, vccs_emb, vt_codes, vt_embs,
                                             tfidf, vt_tfidf_mat, all_targets):
        if neg_code not in vt_lookup:
            continue
        info = vt_lookup[neg_code]
        vt_pairs.append({
            "query_course": str(row["VCCS Course"]),
            "query_desc":   str(row.get("VCCS Description", "")),
            "target_code":  neg_code,
            "target_title": info.get("title", ""),
            "target_desc":  info.get("description", ""),
            "label": 0, "source": "retriever_hard", "institution": "vt",
        })

print(f"VT pairs: {len(vt_pairs)}  (pos={sum(p['label'] for p in vt_pairs)}, neg={sum(1-p['label'] for p in vt_pairs)})")


# ══════════════════════════════════════════════════════════════
# INSTITUTION 3 — CCC → UCSC
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CCC→UCSC hard negatives")
print("=" * 60)

df_ccc = pd.read_csv(CCC_UCSC_CLEAN)
df_ccc.columns = df_ccc.columns.str.strip()

def ucsc_dept(code_str):
    m = re.match(r"([A-Z]+)\s+", str(code_str).strip())
    return m.group(1) if m else "UNK"

df_ccc["ucsc_dept"] = df_ccc["UCSC Course Code"].apply(ucsc_dept)
dc_ccc = df_ccc["ucsc_dept"].value_counts()
df_ccc["strat"] = df_ccc["ucsc_dept"].apply(lambda d: "RARE" if dc_ccc[d] < 2 else d)
ccc_train, ccc_test = train_test_split(df_ccc, test_size=0.20, random_state=42, stratify=df_ccc["strat"])
print(f"CCC train: {len(ccc_train)}  test: {len(ccc_test)}")

def get_ccc_text(row):
    title_raw = str(row.get("CCC Course", "")).strip()
    # CCC Course format: "CHEM 1A General Chemistry" — strip code prefix
    m = re.match(r"^[A-Z][A-Z0-9 ]*?\s+\S+\s+(.*)", title_raw)
    title = m.group(1).strip() if m else title_raw
    desc = clean_text(row.get("CCC Description", ""))
    return f"{title} {desc}".strip()

ucsc_embs, ucsc_tfidf_mat = build_catalog_index(ucsc_codes, ucsc_texts, model, tfidf)

ccc_pairs = []

for _, row in ccc_train.iterrows():
    query_text  = get_ccc_text(row)
    target_code = str(row["UCSC Course Code"]).strip()
    target_info = ucsc_lookup.get(target_code, {})

    # Positive
    ccc_pairs.append({
        "query_course": str(row.get("CCC Course", "")),
        "query_desc":   str(row.get("CCC Description", "")),
        "target_code":  target_code,
        "target_title": str(row.get("UCSC Course Title", "")),
        "target_desc":  str(row.get("UCSC Description", "")),
        "label": 1, "source": "ground_truth", "institution": "ucsc",
    })

    # Hard negatives
    if target_code not in ucsc_lookup:
        continue
    query_emb = model.encode([f"{QUERY_PREFIX}{query_text}"], normalize_embeddings=True)[0]
    for neg_code in retrieve_hard_negatives(query_text, query_emb, ucsc_codes, ucsc_embs,
                                             tfidf, ucsc_tfidf_mat, {target_code}):
        if neg_code not in ucsc_lookup:
            continue
        info = ucsc_lookup[neg_code]
        ccc_pairs.append({
            "query_course": str(row.get("CCC Course", "")),
            "query_desc":   str(row.get("CCC Description", "")),
            "target_code":  neg_code,
            "target_title": info.get("title", ""),
            "target_desc":  info.get("description", ""),
            "label": 0, "source": "retriever_hard", "institution": "ucsc",
        })

print(f"CCC pairs: {len(ccc_pairs)}  (pos={sum(p['label'] for p in ccc_pairs)}, neg={sum(1-p['label'] for p in ccc_pairs)})")


# ══════════════════════════════════════════════════════════════
# COMBINE + SAVE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

all_pairs = wm_pairs + vt_pairs + ccc_pairs
np.random.shuffle(all_pairs)
df_out = pd.DataFrame(all_pairs)

print(f"\nTotal pairs: {len(df_out)}")
print(f"\nBy institution:")
for inst, grp in df_out.groupby("institution"):
    pos = (grp["label"] == 1).sum()
    neg = (grp["label"] == 0).sum()
    print(f"  {inst:<6}  pos={pos}  neg={neg}  ratio={neg/max(pos,1):.1f}:1")

print(f"\nBy source:")
for src, cnt in df_out["source"].value_counts().items():
    print(f"  {src:<20} {cnt}")

df_out.to_csv(TRAIN_PAIRS, index=False)
print(f"\nSaved {len(df_out)} pairs to {TRAIN_PAIRS}")

# Save test splits for reproducibility
wm_test.to_csv(TEST_POS, index=False)
print(f"Saved W&M test_pos ({len(wm_test)} rows) to {TEST_POS}")
