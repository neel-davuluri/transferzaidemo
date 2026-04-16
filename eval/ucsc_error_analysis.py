"""
Pull UCSC errors and format them for manual review.

For each wrong CE prediction, shows:
  - CCC query (source course + description)
  - Model's top-1 prediction (what it said)
  - Correct answer (ground truth)
  - Whether correct answer was retrieved at all (recall issue vs ranking issue)
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import CCC_UCSC_CLEAN, UCSC_CATALOG

import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
QUERY_PREFIX = "Represent this course for finding transfer equivalents: "
RRF_K = 60
RERANK_K = 10
N_ERRORS = 30

_CCC_NOISE = re.compile(
    r"may be offered in a distance.learning format"
    r"|transfer information:|transferable to (?:both )?uc"
    r"|limitations on enrollment:|requisites:"
    r"|minimum credit units|maximum credit units"
    r"|toggle (?:additional|general|learning)"
    r"|grade options:|see open sections|connect with an academic counselor"
    r"|some of the class hours for this course",
    re.IGNORECASE,
)

def clean_text(text):
    if pd.isna(text) or str(text) in ("Description not found", "nan", ""): return ""
    text = str(text)
    m = _CCC_NOISE.search(text)
    if m: text = text[:m.start()]
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def load_catalog(path):
    df = pd.read_csv(path).dropna(subset=["course_code"])
    lk = {}
    for _, r in df.iterrows():
        code = str(r["course_code"]).strip()
        lk[code] = {"title": str(r.get("course_title", "")),
                    "description": str(r.get("course_description", "")) if pd.notna(r.get("course_description")) else ""}
    return lk

# ── Load data + split ────────────────────────────────────────
df_ccc = pd.read_csv(CCC_UCSC_CLEAN)
df_ccc.columns = df_ccc.columns.str.strip()

def ucsc_dept(c):
    m = re.match(r"([A-Z]+)\s+", str(c).strip())
    return m.group(1) if m else "UNK"

df_ccc["dept"] = df_ccc["UCSC Course Code"].apply(ucsc_dept)
dc = df_ccc["dept"].value_counts()
df_ccc["strat"] = df_ccc["dept"].apply(lambda d: "RARE" if dc[d] < 2 else d)
_, ccc_test = train_test_split(df_ccc, test_size=0.20, random_state=42, stratify=df_ccc["strat"])

ucsc_lk = load_catalog(UCSC_CATALOG)

# ── Load models ───────────────────────────────────────────────
print("Loading models...")
bge = SentenceTransformer("./finetuned_bge_three", device=device)
ce  = CrossEncoder("./cross_encoder", device=device)

codes = list(ucsc_lk.keys())
texts = [f"{ucsc_lk[c]['title']} {clean_text(ucsc_lk[c]['description'])}" for c in codes]
embs  = bge.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,2), min_df=1, max_df=0.95,
                        stop_words="english", sublinear_tf=True)
tfidf_mat = tfidf.fit_transform(texts)

# ── Run inference + collect errors ───────────────────────────
errors = []

for _, row in ccc_test.iterrows():
    title_raw = str(row.get("CCC Course", "")).strip()
    m = re.match(r"^[A-Z][A-Z0-9 ]*?\s+\S+\s+(.*)", title_raw)
    ccc_title = m.group(1).strip() if m else title_raw
    ccc_desc  = str(row.get("CCC Description", ""))
    query_text = f"{ccc_title} {clean_text(ccc_desc)}".strip()

    target = str(row["UCSC Course Code"]).strip()
    if target not in codes: continue

    q_emb = bge.encode([f"{QUERY_PREFIX}{query_text}"], normalize_embeddings=True)[0]

    bge_sims   = embs @ q_emb
    tfidf_sims = cosine_similarity(tfidf.transform([query_text]), tfidf_mat).flatten()
    bge_ranked   = np.argsort(bge_sims)[::-1]
    tfidf_ranked = np.argsort(tfidf_sims)[::-1]
    rrf = defaultdict(float)
    for rank, idx in enumerate(bge_ranked[:200]):
        rrf[codes[idx]] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(tfidf_ranked[:200]):
        rrf[codes[idx]] += 1.0 / (RRF_K + rank + 1)
    ranked = [c for c, _ in sorted(rrf.items(), key=lambda x: -x[1])[:50]]

    pool  = ranked[:RERANK_K]
    pairs = [[query_text, f"{ucsc_lk.get(c,{}).get('title','')} {clean_text(ucsc_lk.get(c,{}).get('description',''))}"]
             for c in pool]
    ce_scores = ce.predict(pairs)
    ce_ranked = [pool[i] for i in np.argsort(ce_scores)[::-1]]

    predicted = ce_ranked[0] if ce_ranked else None

    if predicted != target:
        target_rank_in_pool = ce_ranked.index(target) + 1 if target in ce_ranked else None
        target_rank_in_rrf  = ranked.index(target) + 1 if target in ranked else None

        errors.append({
            "ccc_course":     title_raw,
            "ccc_desc_clean": clean_text(ccc_desc)[:300],
            "correct_code":   target,
            "correct_title":  ucsc_lk.get(target, {}).get("title", ""),
            "correct_desc":   ucsc_lk.get(target, {}).get("description", "")[:200],
            "predicted_code": predicted,
            "predicted_title":ucsc_lk.get(predicted, {}).get("title", "") if predicted else "",
            "predicted_desc": ucsc_lk.get(predicted, {}).get("description", "")[:200] if predicted else "",
            "target_rank_ce": target_rank_in_pool,   # rank in CE reranked pool (None = not in top-10)
            "target_rank_rrf": target_rank_in_rrf,   # rank in RRF top-50 (None = not retrieved)
        })

print(f"\nTotal test queries: {len(ccc_test)}")
print(f"Errors (CE wrong):  {len(errors)}")
print(f"Printing first {N_ERRORS}...\n")

print("=" * 80)
for i, e in enumerate(errors[:N_ERRORS]):
    not_retrieved = e["target_rank_rrf"] is None
    not_in_pool   = e["target_rank_ce"] is None and not not_retrieved
    in_pool_wrong = e["target_rank_ce"] is not None

    if not_retrieved:
        failure_type = "RETRIEVAL MISS (correct not in top-50)"
    elif not_in_pool:
        failure_type = f"POOL MISS (correct at RRF rank {e['target_rank_rrf']}, outside top-{RERANK_K})"
    else:
        failure_type = f"RERANK MISS (correct at CE rank {e['target_rank_ce']} in pool)"

    print(f"[{i+1:02d}] {failure_type}")
    print(f"     CCC:       {e['ccc_course']}")
    print(f"     CCC desc:  {e['ccc_desc_clean'][:200]}")
    print(f"     PREDICTED: {e['predicted_code']} — {e['predicted_title']}")
    print(f"                {e['predicted_desc'][:150]}")
    print(f"     CORRECT:   {e['correct_code']} — {e['correct_title']}")
    print(f"                {e['correct_desc'][:150]}")
    print()

# ── Failure type breakdown ────────────────────────────────────
print("=" * 80)
print("FAILURE TYPE BREAKDOWN")
print("=" * 80)
retrieval_misses = sum(1 for e in errors if e["target_rank_rrf"] is None)
pool_misses      = sum(1 for e in errors if e["target_rank_ce"] is None and e["target_rank_rrf"] is not None)
rerank_misses    = sum(1 for e in errors if e["target_rank_ce"] is not None)
print(f"  Retrieval miss (not in top-50):           {retrieval_misses:>3} ({retrieval_misses/len(errors)*100:.0f}%)")
print(f"  Pool miss (top-50 but not top-{RERANK_K}):       {pool_misses:>3} ({pool_misses/len(errors)*100:.0f}%)")
print(f"  Rerank miss (in pool, CE ranked it wrong): {rerank_misses:>3} ({rerank_misses/len(errors)*100:.0f}%)")
print(f"\nNote: only rerank misses are fixable by improving the CE.")
print(f"      retrieval+pool misses require improving BGE or expanding pool size.")
