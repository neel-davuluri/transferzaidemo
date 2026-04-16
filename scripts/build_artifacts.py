"""
scripts/build_artifacts.py

Trains the XGBoost reranker on all 3 institutions (W&M, VT, CCC→UCSC) and saves
all artifacts required by predict.py + app.py.

Run from repo root:
    source .venv/bin/activate
    python scripts/build_artifacts.py

Writes to artifacts/:
    classifier.pkl        — XGBoost model (raw uncalibrated, for ranking)
    iso_cal.pkl           — IsotonicRegression calibrator (for Brier/ECE display)
    tfidf.pkl             — TfidfVectorizer (15k features, fitted on all 3 catalogs)
    dept_prior_map.pkl    — {"wm": {...}, "vt": {...}, "ucsc": {...}}
    feature_names.pkl     — list of 13 feature names (matches predict.py order)
    scorecard.pkl         — multi-institution metrics dict for Model Card tab
    wm_lookup.pkl / wm_codes.pkl / wm_embeddings.npy
    vt_lookup.pkl / vt_codes.pkl / vt_embeddings.npy
    ucsc_lookup.pkl / ucsc_codes.pkl / ucsc_embeddings.npy
"""

import sys
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from scipy.special import softmax as scipy_softmax

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import torch

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paths import WM_MERGED, WM_CATALOG, VT_MERGED, VT_CATALOG, CCC_UCSC_CLEAN, UCSC_CATALOG
from eval.sequence_features import (
    seq_token_from_code, augment_text,
    lab_token_from_code_and_desc,
    seq_token_from_text, lab_token_from_query,
)

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

QUERY_PREFIX = "Represent this course for finding transfer equivalents: "
RRF_K    = 60
RERANK_K = 50

FEATURE_NAMES = [
    "bge_sim", "tfidf_sim", "tfidf_title_sim",
    "dept_sim", "title_sim", "level_ratio", "same_level",
    "rrf_score",
    "bge_x_dept", "bge_x_title", "bge_x_tfidf",
    "dept_x_title", "dept_x_level",
]

# ── text helpers ───────────────────────────────────────────────────────────────

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
        lk[code] = {
            "title": str(r.get("course_title", "")),
            "description": str(r.get("course_description", ""))
                           if pd.notna(r.get("course_description")) else "",
        }
    return lk

def parse_vccs(raw):
    parts = re.split(r"\s*TAKEN WITH\s*", str(raw).strip(), flags=re.IGNORECASE)
    courses = []
    for p in parts:
        m = re.match(r"([A-Z]{2,4})\s+(\d{3})\s+(.*)", p.strip())
        if m:
            courses.append({"dept": m.group(1), "number": int(m.group(2)),
                            "title": m.group(3).strip()})
    return courses

def parse_ccc_course(raw):
    m = re.match(r"^([A-Z][A-Z0-9]*)\s+(\d+[A-Za-z]*)\s+(.*)", str(raw).strip())
    if m:
        num_str = re.sub(r"[^0-9]", "", m.group(2))
        return {"dept": m.group(1), "number": int(num_str) if num_str else 0,
                "title": m.group(3).strip()}
    return None

def extract_number(code_str):
    m = re.search(r"(\d+)", str(code_str))
    return int(m.group(1)) if m else 0

def academic_level(number, inst):
    if number <= 0: return 0
    if inst in ("vccs", "wm", "ucsc_ccc"):
        return max(1, min(4, number // 100)) if number >= 100 else 1
    if inst in ("vt", "neu"):
        return max(1, min(4, number // 1000))
    if inst == "ucsc":
        if number < 50:  return 1
        if number < 100: return 2
        return 3
    return max(1, min(4, number // 100))


# ── load models ────────────────────────────────────────────────────────────────

print("\nLoading BGE model...")
bge = SentenceTransformer("./finetuned_bge_three", device=device)

print("Loading catalogs...")
wm_lk   = load_catalog(WM_CATALOG, encoding="latin-1")
vt_lk   = load_catalog(VT_CATALOG)
ucsc_lk = load_catalog(UCSC_CATALOG)

all_texts = (
    [f"{v['title']} {clean_text(v['description'])}" for v in wm_lk.values()] +
    [f"{v['title']} {clean_text(v['description'])}" for v in vt_lk.values()] +
    [f"{v['title']} {clean_text(v['description'])}" for v in ucsc_lk.values()]
)
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=1, max_df=0.95,
                        stop_words="english", sublinear_tf=True)
tfidf.fit(all_texts)
print(f"TF-IDF vocab: {len(tfidf.vocabulary_)} terms")


# ── data splits ────────────────────────────────────────────────────────────────

df_wm = pd.read_csv(WM_MERGED)
df_wm.columns = df_wm.columns.str.strip()
pos_wm = df_wm[df_wm["W&M Course Code"].notna() & (df_wm["W&M Course Code"].str.strip() != "")].copy()
pos_wm["dept"] = pos_wm["W&M Course Code"].apply(
    lambda x: re.match(r"([A-Z]{2,4})", str(x)).group(1) if re.match(r"([A-Z]{2,4})", str(x)) else "UNK")
dc = pos_wm["dept"].value_counts()
pos_wm["strat"] = pos_wm["dept"].apply(lambda d: "RARE" if dc[d] < 2 else d)
wm_train, wm_test = train_test_split(pos_wm, test_size=0.20, random_state=42, stratify=pos_wm["strat"])

df_vt = pd.read_csv(VT_MERGED)
df_vt["dept"] = df_vt["VT Course Code"].apply(
    lambda x: re.match(r"([A-Z]+)", str(x).split("+")[0].strip()).group(1)
              if re.match(r"([A-Z]+)", str(x).split("+")[0].strip()) else "UNK")
dc2 = df_vt["dept"].value_counts()
df_vt["strat"] = df_vt["dept"].apply(lambda d: "RARE" if dc2[d] < 2 else d)
vt_train, vt_test = train_test_split(df_vt, test_size=0.20, random_state=42, stratify=df_vt["strat"])

df_ccc = pd.read_csv(CCC_UCSC_CLEAN)
df_ccc.columns = df_ccc.columns.str.strip()
df_ccc["dept"] = df_ccc["UCSC Course Code"].apply(
    lambda x: re.match(r"([A-Z]+)", str(x)).group(1) if re.match(r"([A-Z]+)", str(x)) else "UNK")
dc3 = df_ccc["dept"].value_counts()
df_ccc["strat"] = df_ccc["dept"].apply(lambda d: "RARE" if dc3[d] < 2 else d)
ccc_train, ccc_test = train_test_split(df_ccc, test_size=0.20, random_state=42, stratify=df_ccc["strat"])


# ── dept priors ────────────────────────────────────────────────────────────────

def build_vccs_prior(train_df, vccs_col, target_col):
    counts = defaultdict(Counter)
    for _, r in train_df.iterrows():
        for vc in parse_vccs(r[vccs_col]):
            tgt = re.match(r"([A-Z]+)", str(r[target_col]).split("+")[0].strip())
            if tgt: counts[vc["dept"]][tgt.group(1)] += 1
    return {d: {t: n/sum(c.values()) for t, n in c.items()} for d, c in counts.items()}

def build_ccc_prior(train_df):
    counts = defaultdict(Counter)
    for _, r in train_df.iterrows():
        p = parse_ccc_course(r["CCC Course"])
        ud = re.match(r"([A-Z]+)", str(r["UCSC Course Code"]))
        if p and ud: counts[p["dept"]][ud.group(1)] += 1
    return {d: {t: n/sum(c.values()) for t, n in c.items()} for d, c in counts.items()}

wm_dept_prior  = build_vccs_prior(wm_train,  "VCCS Course", "W&M Course Code")
vt_dept_prior  = build_vccs_prior(vt_train,  "VCCS Course", "VT Course Code")
ccc_dept_prior = build_ccc_prior(ccc_train)


# ── equiv maps ─────────────────────────────────────────────────────────────────

def build_wm_equiv(df):
    em = defaultdict(list)
    for _, r in df.iterrows():
        for c in parse_vccs(r["VCCS Course"]):
            em[str(r["W&M Course Code"]).strip()].append(c["title"])
    return dict(em)

def build_vt_equiv(df):
    em = defaultdict(list)
    for _, r in df.iterrows():
        for c in parse_vccs(r["VCCS Course"]):
            em[str(r["VT Course Code"]).split("+")[0].strip()].append(c["title"])
    return dict(em)

def build_ccc_equiv(df):
    em = defaultdict(list)
    for _, r in df.iterrows():
        p = parse_ccc_course(r["CCC Course"])
        if p: em[str(r["UCSC Course Code"]).strip()].append(p["title"])
    return dict(em)

wm_equiv  = build_wm_equiv(wm_train)
vt_equiv  = build_vt_equiv(vt_train)
ccc_equiv = build_ccc_equiv(ccc_train)


# ── embed catalogs ─────────────────────────────────────────────────────────────

def cand_text_clean(code, lk):
    title = lk[code]["title"]; desc = lk[code]["description"]
    base = f"{title} {clean_text(desc)}"
    base = augment_text(base, seq_token_from_code(code))
    base = augment_text(base, lab_token_from_code_and_desc(code, title, desc))
    return base

def cand_text_aug(code, lk, equiv_map):
    base = cand_text_clean(code, lk)
    if equiv_map and code in equiv_map and len(equiv_map[code]) >= 3:
        base = f"{base} {' '.join(equiv_map[code][:8])}"
    return base

def embed_catalog(lk, equiv_map, label):
    codes = list(lk.keys())
    aug_texts   = [cand_text_aug(c, lk, equiv_map)   for c in codes]
    clean_texts = [cand_text_clean(c, lk)             for c in codes]
    print(f"  Embedding {label} ({len(codes)} courses)...")
    embs      = bge.encode(aug_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    tfidf_mat = tfidf.transform(aug_texts)
    return {
        "codes": codes,
        "embs": np.array(embs, dtype=np.float32),
        "tfidf_mat": tfidf_mat,
        "code_to_idx":  {c: i for i, c in enumerate(codes)},
        "code_to_dept": {
            c: (re.match(r"([A-Z]+)", c).group(1) if re.match(r"([A-Z]+)", c) else "UNK")
            for c in codes
        },
        "clean_texts": clean_texts,
        "lk": lk,
    }

print("\nEmbedding catalogs...")
wm_cat   = embed_catalog(wm_lk,   wm_equiv,  "W&M")
vt_cat   = embed_catalog(vt_lk,   vt_equiv,  "VT")
ucsc_cat = embed_catalog(ucsc_lk, ccc_equiv, "UCSC")


# ── retrieval + features ───────────────────────────────────────────────────────

def rrf_retrieve(query_text, query_emb, codes, embs, tfidf_mat,
                 query_dept=None, code_to_dept=None, k=50):
    bge_sims   = embs @ query_emb
    tfidf_sims = cosine_similarity(tfidf.transform([query_text]), tfidf_mat).flatten()
    bge_ranked   = np.argsort(bge_sims)[::-1]
    tfidf_ranked = np.argsort(tfidf_sims)[::-1]
    rrf = defaultdict(float)
    for rank, idx in enumerate(bge_ranked[:200]):
        rrf[codes[idx]] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(tfidf_ranked[:200]):
        rrf[codes[idx]] += 1.0 / (RRF_K + rank + 1)
    if query_dept and code_to_dept:
        dept_sc = sorted(range(len(codes)),
                         key=lambda i: -SequenceMatcher(
                             None, query_dept.lower(),
                             code_to_dept.get(codes[i], "UNK").lower()
                         ).ratio())
        for rank, idx in enumerate(dept_sc[:200]):
            rrf[codes[idx]] += 0.5 * (1.0 / (RRF_K + rank + 1))
    return {c: sc for c, sc in sorted(rrf.items(), key=lambda x: -x[1])[:k]}

def extract_features(query_emb, query_text, query_title,
                     query_dept, query_number, query_inst,
                     cand_code, rrf_score, cat):
    idx      = cat["code_to_idx"][cand_code]
    cand_emb = cat["embs"][idx]
    bge_sim  = float(query_emb @ cand_emb)

    cand_clean = cat["clean_texts"][idx]
    vecs = tfidf.transform([query_text, cand_clean])
    tfidf_sim = float(cosine_similarity(vecs[0:1], vecs[1:2])[0, 0])

    cand_title = cat["lk"][cand_code]["title"]
    if query_title.strip() and cand_title.strip():
        tv = tfidf.transform([clean_text(query_title), clean_text(cand_title)])
        tfidf_title_sim = float(cosine_similarity(tv[0:1], tv[1:2])[0, 0])
    else:
        tfidf_title_sim = 0.0

    cand_dept = cat["code_to_dept"].get(cand_code, "UNK")
    dept_sim  = SequenceMatcher(None, query_dept.lower(), cand_dept.lower()).ratio() \
                if query_dept else 0.0
    title_sim = SequenceMatcher(None, clean_text(query_title), clean_text(cand_title)).ratio()

    cand_num = extract_number(cand_code)
    tgt_inst = ("wm" if cand_code in wm_lk else "vt" if cand_code in vt_lk else "ucsc")
    q_lvl    = academic_level(query_number, query_inst)
    c_lvl    = academic_level(cand_num, tgt_inst)
    if q_lvl > 0 and c_lvl > 0:
        diff        = abs(q_lvl - c_lvl)
        level_ratio = 1.0 - diff / 3.0
        same_level  = float(diff == 0)
    else:
        level_ratio = 0.5  # neutral: level unknown
        same_level  = 0.0

    return [
        bge_sim, tfidf_sim, tfidf_title_sim,
        dept_sim, title_sim, level_ratio, same_level,
        rrf_score,
        bge_sim * dept_sim,
        bge_sim * title_sim,
        bge_sim * tfidf_sim,
        dept_sim * title_sim,
        dept_sim * level_ratio,
    ]

def vccs_query(row):
    courses = parse_vccs(row["VCCS Course"])
    titles  = " ".join(c["title"] for c in courses)
    desc    = row.get("VCCS Description", "")
    base    = f"{titles} {clean_text(desc)}".strip()
    return augment_text(base, seq_token_from_text(titles, str(desc))), titles

def ccc_query(row):
    raw   = str(row.get("CCC Course", "")).strip()
    p     = parse_ccc_course(raw)
    title = p["title"] if p else raw
    desc  = row.get("CCC Description", "")
    base  = f"{title} {clean_text(desc)}".strip()
    base  = augment_text(base, seq_token_from_text(title, str(desc)))
    base  = augment_text(base, lab_token_from_query(title))
    return base, title

def wm_target(row):  return str(row["W&M Course Code"]).strip()
def vt_target(row):  return str(row["VT Course Code"]).split("+")[0].strip()
def ccc_target(row): return str(row["UCSC Course Code"]).strip()

def wm_dept_num(row):
    cs = parse_vccs(row["VCCS Course"])
    return (cs[0]["dept"] if cs else "UNK", cs[0]["number"] if cs else 0)

def ccc_dept_num(row):
    p = parse_ccc_course(row["CCC Course"])
    return (p["dept"] if p else "UNK", p["number"] if p else 0)


def collect(train_df, get_query, get_target, get_dept_num, query_inst, cat, label):
    rows, labels = [], []
    total = len(train_df)
    for i, (_, row) in enumerate(train_df.iterrows()):
        qt, q_title = get_query(row)
        target      = get_target(row)
        if target not in cat["code_to_idx"]: continue

        q_dept, q_number = get_dept_num(row)
        q_emb  = bge.encode([f"{QUERY_PREFIX}{qt}"], normalize_embeddings=True)[0]
        ranked = rrf_retrieve(qt, q_emb, cat["codes"], cat["embs"], cat["tfidf_mat"],
                              query_dept=q_dept, code_to_dept=cat["code_to_dept"])

        for code, rrf_sc in ranked.items():
            feats = extract_features(q_emb, qt, q_title, q_dept, q_number, query_inst,
                                     code, rrf_sc, cat)
            rows.append(feats)
            labels.append(int(code == target))

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{total}")
    return rows, labels


print("\n--- Collecting W&M training features ---")
wm_r,  wm_l  = collect(wm_train,  vccs_query, wm_target,  wm_dept_num,  "vccs",     wm_cat,   "W&M")
print("\n--- Collecting VT training features ---")
vt_r,  vt_l  = collect(vt_train,  vccs_query, vt_target,  wm_dept_num,  "vccs",     vt_cat,   "VT")
print("\n--- Collecting CCC training features ---")
ccc_r, ccc_l = collect(ccc_train, ccc_query,  ccc_target, ccc_dept_num, "ucsc_ccc", ucsc_cat, "CCC")

X_train = np.array(wm_r  + vt_r  + ccc_r,  dtype=np.float64)
y_train = np.array(wm_l  + vt_l  + ccc_l,  dtype=int)
print(f"\nTotal training candidates: {len(X_train)}  positives: {y_train.sum()}")


# ── train XGBoost ──────────────────────────────────────────────────────────────

print("\n--- Training XGBoost ---")
pos_rate = y_train.sum() / len(y_train)
scale_pw = (1 - pos_rate) / pos_rate

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=1.0,
    scale_pos_weight=scale_pw,
    tree_method="hist",
    device="cpu",
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)
xgb_model.fit(X_train, y_train)

# Fit calibrator using sigmoid(margins) — avoids predict_proba segfault w/ PyTorch
dmat_tr   = xgb.DMatrix(X_train)
margins_tr = xgb_model.get_booster().predict(dmat_tr, output_margin=True)
raw_probs_tr = 1.0 / (1.0 + np.exp(-margins_tr))
iso_cal = IsotonicRegression(out_of_bounds="clip")
iso_cal.fit(raw_probs_tr, y_train)
print("  XGBoost trained. Calibrator fitted.")


# ── quick eval helper ──────────────────────────────────────────────────────────

def compute_ece(y_true, y_prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        mask = (y_prob >= edges[i]) & (y_prob < edges[i+1])
        if i == n_bins - 1: mask |= (y_prob == 1.0)
        if mask.sum() == 0: continue
        err += (mask.sum() / len(y_true)) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return err

def quick_eval(test_df, get_query, get_target, get_dept_num, query_inst, cat, label):
    t1_rrf = t1_lr = t3_lr = found = 0
    all_y, all_p = [], []
    query_confs  = []

    for _, row in test_df.iterrows():
        qt, q_title  = get_query(row)
        target       = get_target(row)
        if target not in cat["code_to_idx"]: continue
        found += 1

        q_dept, q_number = get_dept_num(row)
        q_emb  = bge.encode([f"{QUERY_PREFIX}{qt}"], normalize_embeddings=True)[0]
        ranked = rrf_retrieve(qt, q_emb, cat["codes"], cat["embs"], cat["tfidf_mat"],
                              query_dept=q_dept, code_to_dept=cat["code_to_dept"])

        rrf_order = list(ranked.keys())
        if target in rrf_order[:1]: t1_rrf += 1

        pool = rrf_order[:RERANK_K]
        feat_mat = [
            extract_features(q_emb, qt, q_title, q_dept, q_number, query_inst,
                             code, ranked.get(code, 0.0), cat)
            for code in pool
        ]
        X_q      = np.array(feat_mat, dtype=np.float64)
        dmat_q   = xgb.DMatrix(X_q)
        margins  = xgb_model.get_booster().predict(dmat_q, output_margin=True)
        raw_probs = 1.0 / (1.0 + np.exp(-margins))
        cal_probs = iso_cal.predict(raw_probs)
        conf      = scipy_softmax(margins)

        lr_order    = [pool[i] for i in np.argsort(conf)[::-1]]
        top1_correct = int(lr_order[0] == target)
        if top1_correct: t1_lr += 1
        if target in lr_order[:3]: t3_lr += 1

        for code, cp in zip(pool, cal_probs):
            all_y.append(int(code == target))
            all_p.append(float(cp))

        query_confs.append((float(conf.max()), top1_correct))

    n  = found
    ya = np.array(all_y); pa = np.array(all_p)
    brier   = float(brier_score_loss(ya, pa))
    ece     = float(compute_ece(ya, pa))
    top1_rrf_r = t1_rrf / n
    top1_lr_r  = t1_lr  / n
    top3_lr_r  = t3_lr  / n

    # threshold sweep
    confs  = np.array([x[0] for x in query_confs])
    labels = np.array([x[1] for x in query_confs])
    thresholds = np.arange(0.05, 0.96, 0.05)
    sweep = []
    for tau in thresholds:
        mask = confs >= tau
        if mask.sum() == 0:
            sweep.append((tau, 0.0, 0.0, 0.0)); continue
        sweep.append((tau, float(labels[mask].mean()),
                      float(labels[mask].sum() / n), float(mask.sum() / n)))

    op_tau = op_prec = op_rec = op_cov = 0.0
    for tau, prec, recall, coverage in sweep:
        if prec >= 0.90 and coverage > 0:
            op_tau = tau; op_prec = prec; op_rec = recall; op_cov = coverage
            break  # take the LOWEST tau that achieves >= 90% precision

    print(f"\n  {label}  (n={n})")
    print(f"    Top-1 RRF={top1_rrf_r:.3f}  Top-1 XGB={top1_lr_r:.3f}  Top-3={top3_lr_r:.3f}")
    print(f"    Prec@τ={op_prec:.3f}  Rec@τ={op_rec:.3f}  Cov@τ={op_cov:.3f}  τ={op_tau:.2f}")
    print(f"    Brier={brier:.4f}  ECE={ece:.4f}")

    return {
        "label": label, "n": n,
        "top1_rrf": top1_rrf_r, "top1_lr": top1_lr_r, "top3_lr": top3_lr_r,
        "precision_tau": op_prec, "recall_tau": op_rec, "coverage_tau": op_cov,
        "op_tau": op_tau, "brier": brier, "ece": ece,
        "sweep": sweep,
    }

print("\n--- Evaluating (test sets) ---")
sc_wm   = quick_eval(wm_test,  vccs_query, wm_target,  wm_dept_num,  "vccs",     wm_cat,   "W&M")
sc_vt   = quick_eval(vt_test,  vccs_query, vt_target,  wm_dept_num,  "vccs",     vt_cat,   "VT")
sc_ucsc = quick_eval(ccc_test, ccc_query,  ccc_target, ccc_dept_num, "ucsc_ccc", ucsc_cat, "CCC→UCSC")

scorecard = {
    "wm":   sc_wm,
    "vt":   sc_vt,
    "ucsc": sc_ucsc,
    "feature_names": FEATURE_NAMES,
    "model_version": "xgboost-v1",
}


# ── save artifacts ─────────────────────────────────────────────────────────────

print("\n--- Saving artifacts ---")

with open(ARTIFACTS / "classifier.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
print("  classifier.pkl")

with open(ARTIFACTS / "iso_cal.pkl", "wb") as f:
    pickle.dump(iso_cal, f)
print("  iso_cal.pkl")

with open(ARTIFACTS / "tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)
print("  tfidf.pkl")

# dept_prior_map.pkl no longer needed — dept_sim replaces the lookup table

with open(ARTIFACTS / "feature_names.pkl", "wb") as f:
    pickle.dump(FEATURE_NAMES, f)
print("  feature_names.pkl")

with open(ARTIFACTS / "scorecard.pkl", "wb") as f:
    pickle.dump(scorecard, f)
print("  scorecard.pkl")

# Per-institution catalogs
for key, cat_data in [("wm", wm_cat), ("vt", vt_cat), ("ucsc", ucsc_cat)]:
    with open(ARTIFACTS / f"{key}_codes.pkl", "wb") as f:
        pickle.dump(cat_data["codes"], f)
    with open(ARTIFACTS / f"{key}_lookup.pkl", "wb") as f:
        pickle.dump(cat_data["lk"], f)
    np.save(ARTIFACTS / f"{key}_embeddings.npy", cat_data["embs"])
    print(f"  {key}_codes.pkl / {key}_lookup.pkl / {key}_embeddings.npy")

print("\nDone. All artifacts saved to artifacts/")
print(f"\nSummary:")
for sc in [sc_wm, sc_vt, sc_ucsc]:
    print(f"  {sc['label']:<12} Top-1={sc['top1_lr']:.3f}  Top-3={sc['top3_lr']:.3f}  "
          f"Prec@τ={sc['precision_tau']:.3f}  Cov@τ={sc['coverage_tau']:.3f}")
