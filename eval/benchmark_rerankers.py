"""
Benchmark: custom cross-encoder vs BAAI/bge-reranker-v2-m3

Runs both rerankers on the same RRF-retrieved candidates across all 3 institutions.
Reports Top-1 and Top-3 deltas vs RRF-only baseline.

Usage:
    source .venv/bin/activate
    python eval/benchmark_rerankers.py
"""

import sys, re, os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import WM_MERGED, WM_CATALOG, VT_MERGED, VT_CATALOG, CCC_UCSC_CLEAN, UCSC_CATALOG
from eval.sequence_features import seq_token_from_text, augment_text, lab_token_from_query

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
print(f"Device: {device}")

RRF_K    = 60
RERANK_K = 50   # candidates passed to reranker


# ── text helpers ──────────────────────────────────────────────────────────────

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
    text = re.sub(r"prerequisite\(s\):.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(csi|alv|nqr|additional)\b", "", text)
    text = re.sub(r"cross-listed with:.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def load_catalog(path, encoding="utf-8"):
    df = pd.read_csv(path, encoding=encoding).dropna(subset=["course_code"])
    return {str(r["course_code"]).strip(): {
        "title": str(r.get("course_title", "")),
        "description": str(r.get("course_description", "")) if pd.notna(r.get("course_description")) else ""
    } for _, r in df.iterrows()}

def rrf_retrieve(query_text, query_emb, codes, embs, tfidf_mat, query_dept=None, code_to_dept=None, k=50):
    bge_sims   = embs @ query_emb
    tfidf_sims = cosine_similarity(tfidf.transform([query_text]), tfidf_mat).flatten()
    rrf = defaultdict(float)
    for rank, idx in enumerate(np.argsort(bge_sims)[::-1][:200]):
        rrf[codes[idx]] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(np.argsort(tfidf_sims)[::-1][:200]):
        rrf[codes[idx]] += 1.0 / (RRF_K + rank + 1)
    if query_dept and code_to_dept:
        dept_sc = sorted(range(len(codes)),
                         key=lambda i: -__import__('difflib').SequenceMatcher(
                             None, query_dept.lower(),
                             code_to_dept.get(codes[i], "UNK").lower()).ratio())
        for rank, idx in enumerate(dept_sc[:200]):
            rrf[codes[idx]] += 0.5 * (1.0 / (RRF_K + rank + 1))
    return [c for c, _ in sorted(rrf.items(), key=lambda x: -x[1])[:k]]


# ── core eval ─────────────────────────────────────────────────────────────────

def eval_institution(name, test_df, query_fn, target_fn, catalog_lk,
                     rerankers, query_dept_fn=None):
    codes = list(catalog_lk.keys())
    code_to_dept = {c: (re.match(r"([A-Z]+)", c).group(1)
                        if re.match(r"([A-Z]+)", c) else "UNK") for c in codes}
    candidate_texts = [
        f"{catalog_lk[c]['title']} {clean_text(catalog_lk[c]['description'])}"
        for c in codes
    ]
    embs      = bge.encode(candidate_texts, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False)
    tfidf_mat = tfidf.transform(candidate_texts)

    counts = {"rrf": [0, 0]}
    for name_r in rerankers:
        counts[name_r] = [0, 0]  # [top1, top3]
    found = 0

    for _, row in test_df.iterrows():
        query_text = query_fn(row)
        target     = target_fn(row)
        if target not in codes: continue
        found += 1

        q_emb      = bge.encode([f"Represent this course for finding transfer equivalents: {query_text}"],
                                 normalize_embeddings=True)[0]
        query_dept = query_dept_fn(row) if query_dept_fn else None
        ranked     = rrf_retrieve(query_text, q_emb, codes, embs, tfidf_mat,
                                  query_dept=query_dept, code_to_dept=code_to_dept)

        counts["rrf"][0] += int(ranked[0] == target)
        counts["rrf"][1] += int(target in ranked[:3])

        pool = ranked[:RERANK_K]
        pool_texts = [candidate_texts[codes.index(c)] for c in pool]
        pairs = [[query_text, ct] for ct in pool_texts]

        for rname, ce_model in rerankers.items():
            scores    = ce_model.predict(pairs, show_progress_bar=False)
            ce_ranked = [pool[i] for i in np.argsort(scores)[::-1]]
            counts[rname][0] += int(ce_ranked[0] == target)
            counts[rname][1] += int(target in ce_ranked[:3])

    n = found
    print(f"\n  {name}  (n={n}, rerank top-{RERANK_K})")
    print(f"    {'Model':<35} {'Top-1':>6}  {'Top-3':>6}  {'ΔTop-1':>7}")
    rrf_t1 = counts['rrf'][0] / n
    rrf_t3 = counts['rrf'][1] / n
    print(f"    {'RRF baseline':<35} {rrf_t1:>6.3f}  {rrf_t3:>6.3f}")
    results = {"rrf": (rrf_t1, rrf_t3)}
    for rname in rerankers:
        t1 = counts[rname][0] / n
        t3 = counts[rname][1] / n
        print(f"    {rname:<35} {t1:>6.3f}  {t3:>6.3f}  {t1-rrf_t1:>+7.3f}")
        results[rname] = (t1, t3)
    return results


# ── load models ───────────────────────────────────────────────────────────────

print("\nLoading BGE bi-encoder...")
bge = SentenceTransformer("./finetuned_bge_three", device=device)

print("Loading TF-IDF...")
wm_lk   = load_catalog(WM_CATALOG, encoding="latin-1")
vt_lk   = load_catalog(VT_CATALOG)
ucsc_lk = load_catalog(UCSC_CATALOG)
all_texts = (
    [f"{v['title']} {clean_text(v['description'])}" for v in wm_lk.values()] +
    [f"{v['title']} {clean_text(v['description'])}" for v in vt_lk.values()] +
    [f"{v['title']} {clean_text(v['description'])}" for v in ucsc_lk.values()]
)
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,2), min_df=1,
                        max_df=0.95, stop_words="english", sublinear_tf=True)
tfidf.fit(all_texts)

print("Loading custom cross-encoder (ms-marco-MiniLM fine-tuned)...")
ce_custom = CrossEncoder("./cross_encoder", device=device)

print("Loading BAAI/bge-reranker-v2-m3 (downloading if needed ~1.1GB)...")
ce_bge = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)

rerankers = {
    "Custom CE (MiniLM fine-tuned)":   ce_custom,
    "BGE-Reranker-v2-m3 (zero-shot)":  ce_bge,
}


# ── data splits (must match training splits exactly) ──────────────────────────

def dept_from_code(c):
    m = re.match(r"([A-Z]+)", str(c).strip())
    return m.group(1) if m else "UNK"

df_wm  = pd.read_csv(WM_MERGED)
df_wm.columns = df_wm.columns.str.strip()
pos_wm = df_wm[df_wm["W&M Course Code"].notna() & (df_wm["W&M Course Code"].str.strip() != "")].copy()
pos_wm["_d"] = pos_wm["W&M Course Code"].apply(lambda x: re.match(r"([A-Z]{2,4})", str(x)).group(1)
                                                if re.match(r"([A-Z]{2,4})", str(x)) else "UNK")
dc = pos_wm["_d"].value_counts()
pos_wm["_s"] = pos_wm["_d"].apply(lambda d: "RARE" if dc[d] < 2 else d)
_, wm_test = train_test_split(pos_wm, test_size=0.20, random_state=42, stratify=pos_wm["_s"])

df_vt = pd.read_csv(VT_MERGED)
df_vt["_d"] = df_vt["VT Course Code"].apply(lambda x: dept_from_code(x.split("+")[0].strip()))
dc2 = df_vt["_d"].value_counts()
df_vt["_s"] = df_vt["_d"].apply(lambda d: "RARE" if dc2[d] < 2 else d)
_, vt_test = train_test_split(df_vt, test_size=0.20, random_state=42, stratify=df_vt["_s"])

df_ccc = pd.read_csv(CCC_UCSC_CLEAN)
df_ccc.columns = df_ccc.columns.str.strip()
df_ccc["_d"] = df_ccc["UCSC Course Code"].apply(dept_from_code)
dc3 = df_ccc["_d"].value_counts()
df_ccc["_s"] = df_ccc["_d"].apply(lambda d: "RARE" if dc3[d] < 2 else d)
_, ccc_test = train_test_split(df_ccc, test_size=0.20, random_state=42, stratify=df_ccc["_s"])


# ── query functions ───────────────────────────────────────────────────────────

def parse_vccs(raw):
    parts = re.split(r"\s*TAKEN WITH\s*", str(raw).strip(), flags=re.IGNORECASE)
    courses = []
    for p in parts:
        m = re.match(r"([A-Z]{2,4})\s+(\d{3})\s+(.*)", p.strip())
        if m: courses.append({"dept": m.group(1), "title": m.group(3).strip()})
    return courses

def wm_query(row):
    courses = parse_vccs(row["VCCS Course"])
    titles  = " ".join(c["title"] for c in courses)
    desc    = row.get("VCCS Description", "")
    return f"{titles} {clean_text(desc)}".strip()

def wm_target(row): return str(row["W&M Course Code"]).strip()

def vt_query(row):
    courses = parse_vccs(row["VCCS Course"])
    titles  = " ".join(c["title"] for c in courses)
    desc    = row.get("VCCS Description", "")
    return f"{titles} {clean_text(desc)}".strip()

def vt_target(row): return row["VT Course Code"].split("+")[0].strip()

def ccc_query(row):
    title_raw = str(row.get("CCC Course", "")).strip()
    m = re.match(r"^[A-Z][A-Z0-9 ]*?\s+\S+\s+(.*)", title_raw)
    title = m.group(1).strip() if m else title_raw
    desc  = row.get("CCC Description", "")
    return f"{title} {clean_text(desc)}".strip()

def ccc_target(row): return str(row["UCSC Course Code"]).strip()

def ccc_dept(row):
    m = re.match(r"^([A-Z][A-Z0-9]*)\s+", str(row.get("CCC Course", "")).strip())
    return m.group(1) if m else None

def wm_dept(row):
    courses = parse_vccs(row["VCCS Course"])
    return courses[0]["dept"] if courses else None

def vt_dept(row):
    courses = parse_vccs(row["VCCS Course"])
    return courses[0]["dept"] if courses else None


# ── run ───────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("BENCHMARK: Custom CE vs BGE-Reranker-v2-m3")
print("=" * 65)

r_wm   = eval_institution("W&M",      wm_test,  wm_query,  wm_target,  wm_lk,   rerankers, query_dept_fn=wm_dept)
r_vt   = eval_institution("VT",       vt_test,  vt_query,  vt_target,  vt_lk,   rerankers, query_dept_fn=vt_dept)
r_ucsc = eval_institution("CCC→UCSC", ccc_test, ccc_query, ccc_target, ucsc_lk, rerankers, query_dept_fn=ccc_dept)

print(f"\n{'='*65}")
print("SUMMARY — Top-1 across all institutions")
print(f"{'='*65}")
print(f"  {'Model':<35} {'W&M':>6}  {'VT':>6}  {'UCSC':>6}  {'Avg':>6}")
for rname in ["rrf", "Custom CE (MiniLM fine-tuned)", "BGE-Reranker-v2-m3 (zero-shot)"]:
    wm_t1   = r_wm[rname][0]
    vt_t1   = r_vt[rname][0]
    ucsc_t1 = r_ucsc[rname][0]
    avg     = (wm_t1 + vt_t1 + ucsc_t1) / 3
    label   = "RRF baseline" if rname == "rrf" else rname
    print(f"  {label:<35} {wm_t1:>6.3f}  {vt_t1:>6.3f}  {ucsc_t1:>6.3f}  {avg:>6.3f}")
