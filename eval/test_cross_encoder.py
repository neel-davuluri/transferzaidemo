"""
Eval: BGE+RRF vs BGE+RRF+CE reranking across all three institutions.
Uses the fine-tuned cross-encoder in ./cross_encoder/.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import WM_MERGED, WM_CATALOG, VT_MERGED, VT_CATALOG, CCC_UCSC_CLEAN, UCSC_CATALOG
from eval.sequence_features import (seq_token_from_code, seq_token_from_text, augment_text,
                                    lab_token_from_code_and_desc, lab_token_from_query)

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
print(f"Device: {device}")

QUERY_PREFIX = "Represent this course for finding transfer equivalents: "
RRF_K   = 60
RERANK_K = 50

# ── Helpers ──────────────────────────────────────────────────

_CCC_NOISE = re.compile(
    r"may be offered in a distance.learning format"
    r"|transfer information:"
    r"|transferable to (?:both )?uc"
    r"|limitations on enrollment:"
    r"|requisites:"
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
    lk = {}
    for _, r in df.iterrows():
        code = str(r["course_code"]).strip()
        lk[code] = {"title": str(r.get("course_title", "")),
                    "description": str(r.get("course_description", "")) if pd.notna(r.get("course_description")) else ""}
    return lk

def rrf_retrieve(query_text, query_emb, catalog_codes, catalog_embs, tfidf, tfidf_mat,
                 dept_prior=None, query_dept=None, code_to_dept=None, k=50):
    bge_sims   = catalog_embs @ query_emb
    tfidf_sims = cosine_similarity(tfidf.transform([query_text]), tfidf_mat).flatten()
    bge_ranked   = np.argsort(bge_sims)[::-1]
    tfidf_ranked = np.argsort(tfidf_sims)[::-1]
    rrf = defaultdict(float)
    for rank, idx in enumerate(bge_ranked[:200]):
        rrf[catalog_codes[idx]] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(tfidf_ranked[:200]):
        rrf[catalog_codes[idx]] += 1.0 / (RRF_K + rank + 1)
    # Department prior signal (same pattern as W&M pipeline)
    if dept_prior and query_dept and code_to_dept:
        prior = dept_prior.get(query_dept, {})
        if prior:
            dept_scores = [(i, prior.get(code_to_dept.get(catalog_codes[i], "UNK"), 0.0))
                           for i in range(len(catalog_codes))]
            dept_scores.sort(key=lambda x: (-x[1], x[0]))
            for rank, (idx, _) in enumerate(dept_scores[:200]):
                rrf[catalog_codes[idx]] += 0.5 * (1.0 / (RRF_K + rank + 1))
    return [c for c, _ in sorted(rrf.items(), key=lambda x: -x[1])[:k]]

def eval_institution(name, test_df, query_fn, target_fn, catalog_lk,
                     bge, tfidf, ce,
                     dept_prior=None, query_dept_fn=None,
                     equiv_map=None):
    codes = list(catalog_lk.keys())

    def candidate_text_clean(c):
        """Clean text for CE reranking (matches CE training distribution)."""
        title = catalog_lk[c]['title']
        desc  = catalog_lk[c]['description']
        base  = f"{title} {clean_text(desc)}"
        base  = augment_text(base, seq_token_from_code(c))
        base  = augment_text(base, lab_token_from_code_and_desc(c, title, desc))
        return base

    def candidate_text_augmented(c):
        """Augmented text for BGE/TF-IDF retrieval — appends known CC equivalent
        titles as anchor text, pulling the university embedding toward the CC query
        space. Exploits many-to-one structure (e.g. 48 CCC courses → THEA 30)."""
        base = candidate_text_clean(c)
        # Only augment when 3+ CC sources exist — fewer titles add noise, not signal.
        if equiv_map and c in equiv_map and len(equiv_map[c]) >= 3:
            anchors = ' '.join(equiv_map[c][:8])
            base = f"{base} {anchors}"
        return base

    clean_texts = [candidate_text_clean(c) for c in codes]
    aug_texts   = [candidate_text_augmented(c) for c in codes]
    embs      = bge.encode(aug_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    tfidf_mat = tfidf.transform(aug_texts)

    t1_bge = t3_bge = t1_ce = t3_ce = found = 0

    for _, row in test_df.iterrows():
        query_text = query_fn(row)
        target     = target_fn(row)
        if target not in codes: continue
        found += 1

        q_emb      = bge.encode([f"{QUERY_PREFIX}{query_text}"], normalize_embeddings=True)[0]
        query_dept = query_dept_fn(row) if query_dept_fn else None
        code_to_dept = {c: re.match(r"([A-Z]+)", c).group(1) if re.match(r"([A-Z]+)", c) else "UNK"
                        for c in codes}
        ranked = rrf_retrieve(query_text, q_emb, codes, embs, tfidf, tfidf_mat,
                              dept_prior=dept_prior, query_dept=query_dept,
                              code_to_dept=code_to_dept)

        if target in ranked[:1]: t1_bge += 1
        if target in ranked[:3]: t3_bge += 1

        pool  = ranked[:RERANK_K]
        pairs = [[query_text, clean_texts[codes.index(c)]] for c in pool]
        ce_scores  = ce.predict(pairs)
        ce_ranked  = [pool[i] for i in np.argsort(ce_scores)[::-1]]

        if target in ce_ranked[:1]: t1_ce += 1
        if target in ce_ranked[:3]: t3_ce += 1

    n = found
    print(f"\n  {name} (n={n}), rerank top-{RERANK_K}:")
    print(f"    {'':30} {'Top-1':>6} {'Top-3':>6}")
    print(f"    BGE+RRF only            {t1_bge/n:>6.3f} {t3_bge/n:>6.3f}")
    print(f"    + CE rerank (fine-tuned){t1_ce/n:>6.3f} {t3_ce/n:>6.3f}")
    print(f"    Delta                   {(t1_ce-t1_bge)/n:>+6.3f} {(t3_ce-t3_bge)/n:>+6.3f}")
    return {"t1_bge": t1_bge/n, "t1_ce": t1_ce/n, "t3_bge": t3_bge/n, "t3_ce": t3_ce/n}


# ── Load shared models ────────────────────────────────────────
print("Loading BGE model...")
bge = SentenceTransformer("./finetuned_bge_three", device=device)

print("Loading cross-encoder...")
ce = CrossEncoder("./cross_encoder", device=device)

# Shared TF-IDF fit on all catalogs
wm_lk   = load_catalog(WM_CATALOG, encoding="latin-1")
vt_lk   = load_catalog(VT_CATALOG)
ucsc_lk = load_catalog(UCSC_CATALOG)

all_texts = (
    [f"{v['title']} {clean_text(v['description'])}" for v in wm_lk.values()] +
    [f"{v['title']} {clean_text(v['description'])}" for v in vt_lk.values()] +
    [f"{v['title']} {clean_text(v['description'])}" for v in ucsc_lk.values()]
)
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,2), min_df=1, max_df=0.95,
                        stop_words="english", sublinear_tf=True)
tfidf.fit(all_texts)
print(f"TF-IDF vocab: {len(tfidf.vocabulary_)} terms")


# ── W&M ──────────────────────────────────────────────────────
df_wm = pd.read_csv(WM_MERGED)
df_wm.columns = df_wm.columns.str.strip()
has = df_wm["W&M Course Code"].notna() & (df_wm["W&M Course Code"].str.strip() != "")
pos_wm = df_wm[has].copy()

def parse_wm_code(c):
    m = re.match(r"([A-Z]{2,4})\s+(\d{3})", str(c).strip())
    return {"dept": m.group(1)} if m else None

def parse_vccs(raw):
    parts = re.split(r"\s*TAKEN WITH\s*", str(raw).strip(), flags=re.IGNORECASE)
    courses = []
    for p in parts:
        m = re.match(r"([A-Z]{2,4})\s+(\d{3})\s+(.*)", p.strip())
        if m: courses.append({"title": m.group(3).strip()})
    return courses

pos_wm["dept"] = pos_wm["W&M Course Code"].apply(lambda x: parse_wm_code(x)["dept"] if parse_wm_code(x) else "UNK")
dc = pos_wm["dept"].value_counts()
pos_wm["strat"] = pos_wm["dept"].apply(lambda d: "RARE" if dc[d] < 2 else d)
wm_train, wm_test = train_test_split(pos_wm, test_size=0.20, random_state=42, stratify=pos_wm["strat"])

# Build equiv map: {WM code → [VCCS titles from training rows]}
wm_equiv = defaultdict(list)
for _, row in wm_train.iterrows():
    target = str(row["W&M Course Code"]).strip()
    for c in parse_vccs(row["VCCS Course"]):
        wm_equiv[target].append(c["title"])
wm_equiv = dict(wm_equiv)

def wm_query(row):
    courses = parse_vccs(row["VCCS Course"])
    titles = " ".join(c["title"] for c in courses)
    desc   = row.get("VCCS Description", "")
    base   = f"{titles} {clean_text(desc)}".strip()
    return augment_text(base, seq_token_from_text(titles, str(desc)))

def wm_target(row): return str(row["W&M Course Code"]).strip()


# ── VT ───────────────────────────────────────────────────────
df_vt = pd.read_csv(VT_MERGED)

def vt_dept(c):
    m = re.match(r"([A-Z]+)\s+\d+", str(c).strip())
    return m.group(1) if m else "UNK"

df_vt["dept"] = df_vt["VT Course Code"].apply(lambda x: vt_dept(x.split("+")[0].strip()))
dc2 = df_vt["dept"].value_counts()
df_vt["strat"] = df_vt["dept"].apply(lambda d: "RARE" if dc2[d] < 2 else d)
vt_train, vt_test = train_test_split(df_vt, test_size=0.20, random_state=42, stratify=df_vt["strat"])

# Build equiv map: {VT code → [VCCS titles from training rows]}
vt_equiv = defaultdict(list)
for _, row in vt_train.iterrows():
    target = row["VT Course Code"].split("+")[0].strip()
    for c in parse_vccs(row["VCCS Course"]):
        vt_equiv[target].append(c["title"])
vt_equiv = dict(vt_equiv)

def vt_query(row):
    courses = parse_vccs(row["VCCS Course"])
    titles = " ".join(c["title"] for c in courses)
    desc   = row.get("VCCS Description", "")
    base   = f"{titles} {clean_text(desc)}".strip()
    return augment_text(base, seq_token_from_text(titles, str(desc)))

def vt_target(row): return row["VT Course Code"].split("+")[0].strip()


# ── CCC → UCSC ───────────────────────────────────────────────
df_ccc = pd.read_csv(CCC_UCSC_CLEAN)
df_ccc.columns = df_ccc.columns.str.strip()

def ucsc_dept(c):
    m = re.match(r"([A-Z]+)\s+", str(c).strip())
    return m.group(1) if m else "UNK"

df_ccc["dept"] = df_ccc["UCSC Course Code"].apply(ucsc_dept)
dc3 = df_ccc["dept"].value_counts()
df_ccc["strat"] = df_ccc["dept"].apply(lambda d: "RARE" if dc3[d] < 2 else d)
ccc_train, ccc_test = train_test_split(df_ccc, test_size=0.20, random_state=42, stratify=df_ccc["strat"])

# Build CCC dept prior from training data only
def ccc_dept_from_course(course_str):
    m = re.match(r"^([A-Z][A-Z0-9]*)\s+", str(course_str).strip())
    return m.group(1) if m else "UNK"

from collections import Counter
_ccc_dept_counts = defaultdict(Counter)
for _, r in ccc_train.iterrows():
    _ccc_dept_counts[ccc_dept_from_course(r["CCC Course"])][ucsc_dept(r["UCSC Course Code"])] += 1

ccc_dept_prior = {}
for ccc_d, ucsc_counts in _ccc_dept_counts.items():
    total = sum(ucsc_counts.values())
    ccc_dept_prior[ccc_d] = {u: n / total for u, n in ucsc_counts.items()}

# Sanity check — top-5 mappings
print("\nCCC dept prior (top-5 by volume):")
for ccc_d in sorted(ccc_dept_prior, key=lambda d: -sum(_ccc_dept_counts[d].values()))[:5]:
    total = sum(_ccc_dept_counts[ccc_d].values())
    top   = sorted(ccc_dept_prior[ccc_d].items(), key=lambda x: -x[1])[:3]
    print(f"  {ccc_d:<8} (n={total}) → {' | '.join(f'{u}={p:.2f}' for u,p in top)}")

def ccc_query(row):
    title_raw = str(row.get("CCC Course", "")).strip()
    m = re.match(r"^[A-Z][A-Z0-9 ]*?\s+\S+\s+(.*)", title_raw)
    title = m.group(1).strip() if m else title_raw
    desc  = row.get("CCC Description", "")
    base  = f"{title} {clean_text(desc)}".strip()
    base  = augment_text(base, seq_token_from_text(title, str(desc)))
    base  = augment_text(base, lab_token_from_query(title))
    return base

def ccc_target(row): return str(row["UCSC Course Code"]).strip()

# Build equiv map: {UCSC code → [CCC titles from training rows]}
ccc_equiv = defaultdict(list)
for _, row in ccc_train.iterrows():
    target = str(row["UCSC Course Code"]).strip()
    title_raw = str(row.get("CCC Course", "")).strip()
    m = re.match(r"^[A-Z][A-Z0-9 ]*?\s+\S+\s+(.*)", title_raw)
    title = m.group(1).strip() if m else title_raw
    if title:
        ccc_equiv[target].append(title)
ccc_equiv = dict(ccc_equiv)


# ── Run eval ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS — BGE+RRF vs BGE+RRF+CE (fine-tuned cross-encoder)")
print("=" * 60)

r_wm   = eval_institution("W&M",      wm_test,  wm_query,  wm_target,  wm_lk,   bge, tfidf, ce,
                          equiv_map=wm_equiv)
r_vt   = eval_institution("VT",       vt_test,  vt_query,  vt_target,  vt_lk,   bge, tfidf, ce,
                          equiv_map=vt_equiv)
r_ucsc = eval_institution("CCC→UCSC", ccc_test, ccc_query, ccc_target, ucsc_lk, bge, tfidf, ce,
                          dept_prior=ccc_dept_prior,
                          query_dept_fn=lambda row: ccc_dept_from_course(row["CCC Course"]),
                          equiv_map=ccc_equiv)

print(f"\n{'='*60}")
print(f"SUMMARY — Top-1 with CE reranking")
print(f"{'='*60}")
print(f"  W&M:       {r_wm['t1_ce']:.3f}  (was {r_wm['t1_bge']:.3f}, +{r_wm['t1_ce']-r_wm['t1_bge']:.3f})")
print(f"  VT:        {r_vt['t1_ce']:.3f}  (was {r_vt['t1_bge']:.3f}, +{r_vt['t1_ce']-r_vt['t1_bge']:.3f})")
print(f"  CCC→UCSC:  {r_ucsc['t1_ce']:.3f}  (was {r_ucsc['t1_bge']:.3f}, +{r_ucsc['t1_ce']-r_ucsc['t1_bge']:.3f})")
print(f"\nDevice used: {device}")
