"""
Step 5 — Feature engineering + XGBoost + isotonic calibration
Keep full feature set from v2, remove cross_encoder_score, add rrf_score and bge_finetuned_sim.
Train XGBoost with same hyperparams. Calibrate isotonic cv=5. 5-fold CV on training data only.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import WM_MERGED, WM_CATALOG, TRAIN_PAIRS

import re, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from difflib import SequenceMatcher

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
import xgboost as xgb

np.random.seed(42)

# ══════════════════════════════════════════════════════════════
# DATA LOADING (identical to prior steps)
# ══════════════════════════════════════════════════════════════
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

for idx, row in pos_df.iterrows():
    wm_code = str(row["W&M Course Code"]).strip()
    if pd.isna(row["W&M Description"]) or row["W&M Description"] == "":
        if wm_code in wm_lookup and wm_lookup[wm_code]["description"]:
            pos_df.at[idx, "W&M Description"] = wm_lookup[wm_code]["description"]
            if pd.isna(row.get("W&M Course Title")) or row.get("W&M Course Title") == "":
                pos_df.at[idx, "W&M Course Title"] = wm_lookup[wm_code]["title"]


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
            courses.append({"dept": m.group(1), "number": int(m.group(2)),
                            "title": m.group(3).strip(), "full": f"{m.group(1)} {m.group(2)}"})
    return courses

def clean_text(text):
    if pd.isna(text) or text == "Description not found" or text == "nan":
        return ""
    text = str(text).lower()
    text = re.sub(r"prerequisite\(s\):.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(CSI|ALV|NQR|Additional)\b", "", text)
    text = re.sub(r"cross-listed with:.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_vccs_text(row):
    courses = parse_vccs_course(row["VCCS Course"])
    titles = " ".join(c["title"] for c in courses)
    desc = clean_text(row.get("VCCS Description", ""))
    return f"{titles} {desc}"

# Split
pos_df["wm_dept"] = pos_df["W&M Course Code"].apply(
    lambda x: parse_wm_course(x)["dept"] if parse_wm_course(x) else "UNK")
dept_counts = pos_df["wm_dept"].value_counts()
rare_depts = dept_counts[dept_counts < 2].index.tolist()
pos_df["strat_dept"] = pos_df["wm_dept"].apply(lambda d: "RARE" if d in rare_depts else d)
train_pos, test_pos = train_test_split(
    pos_df, test_size=0.20, random_state=42, stratify=pos_df["strat_dept"])
print(f"train_pos: {len(train_pos)}  |  test_pos: {len(test_pos)} (untouched)")

# dept_map from train_pos only
dept_map = defaultdict(lambda: defaultdict(int))
for _, r in train_pos.iterrows():
    vccs_courses = parse_vccs_course(r["VCCS Course"])
    wm_parsed = parse_wm_course(r["W&M Course Code"])
    if vccs_courses and wm_parsed:
        for vc in vccs_courses:
            dept_map[vc["dept"]][wm_parsed["dept"]] += 1

# Department prior
dept_prior = {}
for vccs_dept in dept_map:
    total = sum(dept_map[vccs_dept].values())
    dept_prior[vccs_dept] = {wm_dept: count / total for wm_dept, count in dept_map[vccs_dept].items()}

# ══════════════════════════════════════════════════════════════
# LOAD MODELS + BUILD INDEXES
# ══════════════════════════════════════════════════════════════
print("\nLoading fine-tuned BGE model...")
ft_model = SentenceTransformer("./finetuned_bge", device="cpu")
QUERY_PREFIX = "Represent this course for finding transfer equivalents: "

wm_codes = list(wm_lookup.keys())
wm_texts = [f"{wm_lookup[c]['title']} {clean_text(wm_lookup[c]['description'])}" for c in wm_codes]

wm_code_to_dept = {}
for code in wm_codes:
    parsed = parse_wm_course(code)
    wm_code_to_dept[code] = parsed["dept"] if parsed else "UNK"

print(f"Embedding {len(wm_texts)} W&M courses...")
wm_embs_ft = ft_model.encode(wm_texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)

# TF-IDF
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95,
                         stop_words="english", sublinear_tf=True)
wm_tfidf_matrix = tfidf.fit_transform(wm_texts)
print(f"TF-IDF vocabulary: {len(tfidf.vocabulary_)} terms")

# Code-to-index mapping for fast lookup
wm_code_to_idx = {code: i for i, code in enumerate(wm_codes)}

# ══════════════════════════════════════════════════════════════
# THREE-SIGNAL RRF RETRIEVAL
# ══════════════════════════════════════════════════════════════
def retrieve_three_signal_rrf(vccs_text, vccs_course_str, vccs_embedding=None, k=50):
    if vccs_embedding is None:
        vccs_embedding = ft_model.encode(
            [f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True)[0]

    bge_sims = wm_embs_ft @ vccs_embedding
    bge_ranked = np.argsort(bge_sims)[::-1]

    tfidf_vec = tfidf.transform([vccs_text])
    tfidf_sims = cosine_similarity(tfidf_vec, wm_tfidf_matrix).flatten()
    tfidf_ranked = np.argsort(tfidf_sims)[::-1]

    vccs_parsed = parse_vccs_course(vccs_course_str)
    vccs_dept = vccs_parsed[0]["dept"] if vccs_parsed else None
    prior_probs = dept_prior.get(vccs_dept, {}) if vccs_dept else {}

    dept_scores = []
    for i, code in enumerate(wm_codes):
        wm_dept = wm_code_to_dept[code]
        prob = prior_probs.get(wm_dept, 0.0)
        dept_scores.append((i, prob))
    dept_scores.sort(key=lambda x: (-x[1], x[0]))
    dept_ranked = [idx for idx, _ in dept_scores]

    RRF_K = 60
    rrf_scores = defaultdict(float)
    for rank, idx in enumerate(bge_ranked[:200]):
        rrf_scores[wm_codes[idx]] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(tfidf_ranked[:200]):
        rrf_scores[wm_codes[idx]] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(dept_ranked[:200]):
        rrf_scores[wm_codes[idx]] += 0.5 * (1.0 / (RRF_K + rank + 1))

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]

    return [
        {"code": code, "rrf_score": rrf_score,
         "bge_sim": float(bge_sims[wm_code_to_idx[code]]),
         "tfidf_sim": float(tfidf_sims[wm_code_to_idx[code]])}
        for code, rrf_score in ranked
    ]


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (v3)
# ══════════════════════════════════════════════════════════════
def extract_features_v3(row, vccs_emb=None, wm_emb=None, rrf_score=0.0, bge_finetuned_sim=0.0):
    """
    v3 feature set:
    - All v2 structural/text/semantic features
    - REMOVE cross_encoder_score
    - ADD rrf_score (raw RRF fusion score)
    - ADD bge_finetuned_sim (fine-tuned encoder cosine)
    """
    features = {}

    vccs_course = str(row["vccs_course"])
    vccs_desc = str(row["vccs_desc"])
    wm_code = str(row["wm_code"])
    wm_title = str(row["wm_title"])
    wm_desc = str(row["wm_desc"])

    vccs_parsed = parse_vccs_course(vccs_course)
    wm_parsed = parse_wm_course(wm_code)

    # ── STRUCTURAL FEATURES ──
    if vccs_parsed and wm_parsed:
        vccs_num = vccs_parsed[0]["number"]
        wm_num = wm_parsed["number"]
        features["number_diff"] = abs(vccs_num - wm_num)
        features["number_ratio"] = min(vccs_num, wm_num) / max(vccs_num, wm_num) if max(vccs_num, wm_num) > 0 else 0
        vccs_level = vccs_num // 100
        wm_level = wm_num // 100
        features["same_level"] = int(vccs_level == wm_level)
        features["level_diff"] = abs(vccs_level - wm_level)
        features["vccs_level"] = vccs_level
        features["wm_level"] = wm_level
    else:
        features.update({"number_diff": 999, "number_ratio": 0, "same_level": 0,
                         "level_diff": 9, "vccs_level": 0, "wm_level": 0})

    if vccs_parsed and wm_parsed:
        vd, wd = vccs_parsed[0]["dept"], wm_parsed["dept"]
        dc = dept_map[vd][wd]
        dt = sum(dept_map[vd].values())
        features["dept_map_count"] = dc
        features["dept_map_prob"] = dc / dt if dt > 0 else 0
        features["dept_map_exists"] = int(dc > 0)
    else:
        features.update({"dept_map_count": 0, "dept_map_prob": 0, "dept_map_exists": 0})

    features["is_multi_course"] = int("TAKEN WITH" in vccs_course.upper())
    features["num_vccs_courses"] = len(vccs_parsed) if vccs_parsed else 1

    # ── TEXT FEATURES ──
    vccs_clean = clean_text(vccs_desc)
    wm_clean = clean_text(wm_desc)
    vccs_title = " ".join(c["title"] for c in vccs_parsed) if vccs_parsed else ""
    vccs_title_clean = clean_text(vccs_title)
    wm_title_clean = clean_text(wm_title)

    if vccs_clean and wm_clean:
        vecs = tfidf.transform([vccs_clean, wm_clean])
        features["tfidf_desc_sim"] = cosine_similarity(vecs[0:1], vecs[1:2])[0, 0]
    else:
        features["tfidf_desc_sim"] = 0.0

    vccs_full = f"{vccs_title} {vccs_clean}"
    wm_full = f"{wm_title} {wm_clean}"
    if vccs_full.strip() and wm_full.strip():
        vecs = tfidf.transform([vccs_full, wm_full])
        features["tfidf_full_sim"] = cosine_similarity(vecs[0:1], vecs[1:2])[0, 0]
    else:
        features["tfidf_full_sim"] = 0.0

    if vccs_title_clean and wm_title_clean:
        features["title_seq_sim"] = SequenceMatcher(None, vccs_title_clean, wm_title_clean).ratio()
        vecs = tfidf.transform([vccs_title_clean, wm_title_clean])
        features["tfidf_title_sim"] = cosine_similarity(vecs[0:1], vecs[1:2])[0, 0]
    else:
        features["title_seq_sim"] = 0.0
        features["tfidf_title_sim"] = 0.0

    # ── SEMANTIC FEATURES ──
    # bge_cosine_sim: off-the-shelf style (kept for backward compat signal diversity)
    if vccs_emb is not None and wm_emb is not None:
        features["bge_cosine_sim"] = float(np.dot(vccs_emb, wm_emb))
    else:
        v_emb, w_emb = ft_model.encode(
            [f"{QUERY_PREFIX}{vccs_full}", wm_full],
            normalize_embeddings=True)
        features["bge_cosine_sim"] = float(np.dot(v_emb, w_emb))

    # NEW v3: fine-tuned BGE similarity (passed in from retrieval)
    features["bge_finetuned_sim"] = bge_finetuned_sim

    # NEW v3: RRF fusion score
    features["rrf_score"] = rrf_score

    # ── WORD OVERLAP ──
    vccs_words = set(vccs_clean.split())
    wm_words = set(wm_clean.split())
    shared = vccs_words & wm_words
    union = vccs_words | wm_words
    features["shared_word_count"] = len(shared)
    features["shared_word_ratio"] = len(shared) / max(len(union), 1)
    features["jaccard_sim"] = len(shared) / len(union) if union else 0.0

    features["vccs_desc_len"] = len(vccs_clean.split())
    features["wm_desc_len"] = len(wm_clean.split())
    features["desc_len_ratio"] = (
        min(features["vccs_desc_len"], features["wm_desc_len"])
        / max(features["vccs_desc_len"], features["wm_desc_len"])
        if max(features["vccs_desc_len"], features["wm_desc_len"]) > 0 else 0)
    features["vccs_has_desc"] = int(len(vccs_clean) > 10)
    features["wm_has_desc"] = int(len(wm_clean) > 10)
    features["both_have_desc"] = int(features["vccs_has_desc"] and features["wm_has_desc"])

    # ── SUBJECT KEYWORDS ──
    subject_keywords = {
        "math_stat": ["calculus", "algebra", "statistics", "equation", "mathematical",
                      "differential", "integral", "linear", "probability"],
        "science": ["biology", "chemistry", "physics", "laboratory", "experiment",
                    "scientific", "cell", "organism", "molecule"],
        "computing": ["programming", "computer", "software", "algorithm", "data",
                      "code", "database", "network", "computing"],
        "humanities": ["literature", "philosophy", "history", "cultural", "ethical",
                       "civilization", "society", "political"],
        "language": ["speaking", "reading", "writing", "conversation", "grammar",
                     "fluency", "pronunciation", "vocabulary"],
        "arts": ["music", "art", "theater", "dance", "performance", "creative",
                 "visual", "studio"],
        "business": ["accounting", "management", "marketing", "finance", "business",
                     "economics", "entrepreneurship"],
    }
    all_vccs = f"{vccs_title_clean} {vccs_clean}".lower()
    all_wm = f"{wm_title_clean} {wm_clean}".lower()
    for subj, kws in subject_keywords.items():
        vm = sum(1 for kw in kws if kw in all_vccs)
        wm_ = sum(1 for kw in kws if kw in all_wm)
        features[f"{subj}_vccs"] = vm
        features[f"{subj}_wm"] = wm_
        features[f"{subj}_both"] = min(vm, wm_)
        features[f"{subj}_diff"] = abs(vm - wm_)

    return features


# ══════════════════════════════════════════════════════════════
# BUILD TRAINING PAIRS WITH RETRIEVAL SCORES
# ══════════════════════════════════════════════════════════════
print("\nLoading training pairs from Step 2...")
train_pairs_df = pd.read_csv(TRAIN_PAIRS)
print(f"Training pairs: {len(train_pairs_df)}")
print(f"Class balance: {train_pairs_df['label'].value_counts().to_dict()}")

# Pre-encode all VCCS texts for training pairs
print("\nEncoding VCCS texts for training pairs...")
pair_vccs_texts = []
pair_vccs_courses = []
for _, row in train_pairs_df.iterrows():
    vc_parsed = parse_vccs_course(row["vccs_course"])
    vc_title = " ".join(c["title"] for c in vc_parsed) if vc_parsed else ""
    vc_desc = clean_text(row["vccs_desc"])
    pair_vccs_texts.append(f"{vc_title} {vc_desc}")
    pair_vccs_courses.append(str(row["vccs_course"]))

pair_wm_texts = []
for _, row in train_pairs_df.iterrows():
    pair_wm_texts.append(f"{row['wm_title']} {clean_text(row['wm_desc'])}")

vccs_embs = ft_model.encode(
    [f"{QUERY_PREFIX}{t}" for t in pair_vccs_texts],
    batch_size=16, show_progress_bar=True, normalize_embeddings=True)
wm_embs = ft_model.encode(
    pair_wm_texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)

# Compute RRF scores and fine-tuned BGE sims for each pair
print("\nComputing RRF scores for training pairs...")
pair_rrf_scores = []
pair_bge_ft_sims = []
for i, (_, row) in enumerate(train_pairs_df.iterrows()):
    wm_code = str(row["wm_code"])
    # Fine-tuned BGE sim is just the dot product of normalized embeddings
    bge_ft_sim = float(np.dot(vccs_embs[i], wm_embs[i]))
    pair_bge_ft_sims.append(bge_ft_sim)

    # RRF score: retrieve candidates and find this pair's score
    # For efficiency, compute the RRF score directly rather than full retrieval
    wm_idx = wm_code_to_idx.get(wm_code)
    if wm_idx is not None:
        bge_sims = wm_embs_ft @ vccs_embs[i]
        bge_ranked = np.argsort(bge_sims)[::-1]

        tfidf_vec = tfidf.transform([pair_vccs_texts[i]])
        tfidf_sims = cosine_similarity(tfidf_vec, wm_tfidf_matrix).flatten()
        tfidf_ranked = np.argsort(tfidf_sims)[::-1]

        vccs_parsed = parse_vccs_course(pair_vccs_courses[i])
        vccs_dept = vccs_parsed[0]["dept"] if vccs_parsed else None
        prior_probs = dept_prior.get(vccs_dept, {}) if vccs_dept else {}

        dept_scores_list = []
        for j, code in enumerate(wm_codes):
            wm_dept = wm_code_to_dept[code]
            prob = prior_probs.get(wm_dept, 0.0)
            dept_scores_list.append((j, prob))
        dept_scores_list.sort(key=lambda x: (-x[1], x[0]))
        dept_ranked = [idx for idx, _ in dept_scores_list]

        RRF_K = 60
        rrf_scores_dict = defaultdict(float)
        for rank, idx in enumerate(bge_ranked[:200]):
            rrf_scores_dict[wm_codes[idx]] += 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(tfidf_ranked[:200]):
            rrf_scores_dict[wm_codes[idx]] += 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(dept_ranked[:200]):
            rrf_scores_dict[wm_codes[idx]] += 0.5 * (1.0 / (RRF_K + rank + 1))

        pair_rrf_scores.append(rrf_scores_dict.get(wm_code, 0.0))
    else:
        pair_rrf_scores.append(0.0)

    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(train_pairs_df)}")

# Extract all features
print("\nExtracting features...")
feature_dicts = []
for i, (_, row) in enumerate(train_pairs_df.iterrows()):
    feats = extract_features_v3(
        row, vccs_emb=vccs_embs[i], wm_emb=wm_embs[i],
        rrf_score=pair_rrf_scores[i],
        bge_finetuned_sim=pair_bge_ft_sims[i])
    feature_dicts.append(feats)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(train_pairs_df)}")

feature_df = pd.DataFrame(feature_dicts)
feature_names = list(feature_df.columns)
X = feature_df.values.astype(np.float32)
y = train_pairs_df["label"].values

print(f"\nFeature matrix: {X.shape}")
print(f"Features ({len(feature_names)}):")
for i, fn in enumerate(feature_names):
    new_tag = " [NEW]" if fn in ("rrf_score", "bge_finetuned_sim") else ""
    print(f"  {i+1:>2}. {fn}{new_tag}")

# ══════════════════════════════════════════════════════════════
# 5-FOLD CV ON TRAINING DATA ONLY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"5-FOLD CROSS-VALIDATION (training data only)")
print(f"{'='*60}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(y))
oof_preds = np.zeros(len(y))
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    base_clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
        eval_metric="logloss",
        random_state=42 + fold,
        verbosity=0,
    )

    calibrated_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
    calibrated_clf.fit(X_train, y_train)

    val_proba = calibrated_clf.predict_proba(X_val)[:, 1]
    val_preds = (val_proba >= 0.5).astype(int)

    oof_proba[val_idx] = val_proba
    oof_preds[val_idx] = val_preds

    prec = precision_score(y_val, val_preds, zero_division=0)
    rec = recall_score(y_val, val_preds, zero_division=0)
    f1 = f1_score(y_val, val_preds, zero_division=0)
    fold_metrics.append({"fold": fold + 1, "precision": prec, "recall": rec, "f1": f1})
    print(f"  Fold {fold+1}: Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

print(f"\n{'='*60}")
print(f"CROSS-VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"  {'Metric':<12} {'Mean':>8} {'± Std':>8}")
print(f"  {'─'*30}")
for m in ["precision", "recall", "f1"]:
    vals = [fm[m] for fm in fold_metrics]
    print(f"  {m.capitalize():<12} {np.mean(vals):>8.3f} {np.std(vals):>8.3f}")

# Overall OOF metrics
from sklearn.metrics import classification_report, brier_score_loss
print(f"\n  Overall OOF (threshold=0.5):")
print(f"    Precision: {precision_score(y, oof_preds, zero_division=0):.3f}")
print(f"    Recall:    {recall_score(y, oof_preds, zero_division=0):.3f}")
print(f"    F1:        {f1_score(y, oof_preds, zero_division=0):.3f}")
print(f"    Brier:     {brier_score_loss(y, oof_proba):.4f}")

neg_probas = oof_proba[y == 0]
fc_rate = (neg_probas > 0.5).mean()
print(f"    False Confidence Rate: {fc_rate:.3f}")
