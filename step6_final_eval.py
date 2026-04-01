"""
Step 6e — Logistic regression over multi-signal features.

Improvement over 6d: Use sklearn LogisticRegression (convex, global optimum)
instead of Nelder-Mead on Platt scaling. Also add interaction features
(bge * dept, bge * title) to capture that BGE sim is more meaningful
when department matches.
"""

import re, pickle, os
import numpy as np
import pandas as pd
from collections import defaultdict
from difflib import SequenceMatcher

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

np.random.seed(42)

# ══════════════════════════════════════════════════════════════
# DATA LOADING + SPLIT (identical boilerplate)
# ══════════════════════════════════════════════════════════════
df = pd.read_csv("vccs_wm_merged.csv")
wm_catalog = pd.read_csv("wm_courses_2025.csv", encoding="latin-1")
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
    if pd.isna(code_str): return None
    m = re.match(r"([A-Z]{2,4})\s+(\d{3})", str(code_str).strip())
    if m: return {"dept": m.group(1), "number": int(m.group(2)), "full": f"{m.group(1)} {m.group(2)}"}
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
    if pd.isna(text) or text == "Description not found" or text == "nan": return ""
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

pos_df["wm_dept"] = pos_df["W&M Course Code"].apply(
    lambda x: parse_wm_course(x)["dept"] if parse_wm_course(x) else "UNK")
dept_counts = pos_df["wm_dept"].value_counts()
rare_depts = dept_counts[dept_counts < 2].index.tolist()
pos_df["strat_dept"] = pos_df["wm_dept"].apply(lambda d: "RARE" if d in rare_depts else d)
train_pos, test_pos = train_test_split(
    pos_df, test_size=0.20, random_state=42, stratify=pos_df["strat_dept"])
print(f"train_pos: {len(train_pos)}  |  test_pos: {len(test_pos)}")

dept_map = defaultdict(lambda: defaultdict(int))
for _, r in train_pos.iterrows():
    vccs_courses = parse_vccs_course(r["VCCS Course"])
    wm_parsed = parse_wm_course(r["W&M Course Code"])
    if vccs_courses and wm_parsed:
        for vc in vccs_courses:
            dept_map[vc["dept"]][wm_parsed["dept"]] += 1

dept_prior = {}
for vccs_dept in dept_map:
    total = sum(dept_map[vccs_dept].values())
    dept_prior[vccs_dept] = {wm_dept: count / total for wm_dept, count in dept_map[vccs_dept].items()}

# ══════════════════════════════════════════════════════════════
# LOAD MODELS + INDEXES
# ══════════════════════════════════════════════════════════════
print("\nLoading fine-tuned BGE model...")
ft_model = SentenceTransformer("./finetuned_bge", device="cpu")
QUERY_PREFIX = "Represent this course for finding transfer equivalents: "

wm_codes = list(wm_lookup.keys())
wm_texts = [f"{wm_lookup[c]['title']} {clean_text(wm_lookup[c]['description'])}" for c in wm_codes]
wm_code_to_dept = {code: (parse_wm_course(code)["dept"] if parse_wm_course(code) else "UNK") for code in wm_codes}
wm_code_to_idx = {code: i for i, code in enumerate(wm_codes)}

print(f"Embedding {len(wm_texts)} W&M courses...")
wm_embs_ft = ft_model.encode(wm_texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95,
                         stop_words="english", sublinear_tf=True)
wm_tfidf_matrix = tfidf.fit_transform(wm_texts)


# ══════════════════════════════════════════════════════════════
# RETRIEVAL
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
    dept_scores = [(i, prior_probs.get(wm_code_to_dept[wm_codes[i]], 0.0)) for i in range(len(wm_codes))]
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
    return [(code, rrf_sc) for code, rrf_sc in ranked]


# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION: base signals + interactions
# ══════════════════════════════════════════════════════════════
def extract_signals(vccs_emb, vccs_text, vccs_course_str, cand_code, rrf_score):
    cand_idx = wm_code_to_idx[cand_code]
    wm_emb = wm_embs_ft[cand_idx]

    bge_sim = float(vccs_emb @ wm_emb)

    cand_text = wm_texts[cand_idx]
    vecs = tfidf.transform([vccs_text, cand_text])
    tfidf_sim = float(cosine_similarity(vecs[0:1], vecs[1:2])[0, 0])

    vccs_parsed = parse_vccs_course(vccs_course_str)
    vccs_dept = vccs_parsed[0]["dept"] if vccs_parsed else None
    wm_dept = wm_code_to_dept.get(cand_code, "UNK")
    dept_prob = dept_prior.get(vccs_dept, {}).get(wm_dept, 0.0) if vccs_dept else 0.0

    vccs_title = " ".join(c["title"] for c in vccs_parsed) if vccs_parsed else ""
    wm_title = wm_lookup.get(cand_code, {}).get("title", "")
    title_sim = SequenceMatcher(None, clean_text(vccs_title), clean_text(wm_title)).ratio()

    # TF-IDF on titles specifically
    if vccs_title.strip() and wm_title.strip():
        tvecs = tfidf.transform([clean_text(vccs_title), clean_text(wm_title)])
        tfidf_title_sim = float(cosine_similarity(tvecs[0:1], tvecs[1:2])[0, 0])
    else:
        tfidf_title_sim = 0.0

    wm_parsed = parse_wm_course(cand_code)
    if vccs_parsed and wm_parsed:
        num_ratio = min(vccs_parsed[0]["number"], wm_parsed["number"]) / max(vccs_parsed[0]["number"], wm_parsed["number"])
        same_level = int(vccs_parsed[0]["number"] // 100 == wm_parsed["number"] // 100)
    else:
        num_ratio = 0.0
        same_level = 0

    # Base features
    feats = {
        "bge_sim": bge_sim,
        "tfidf_sim": tfidf_sim,
        "tfidf_title_sim": tfidf_title_sim,
        "dept_prob": dept_prob,
        "title_sim": title_sim,
        "num_ratio": num_ratio,
        "same_level": float(same_level),
        "rrf_score": rrf_score,
        # Interaction features
        "bge_x_dept": bge_sim * dept_prob,
        "bge_x_title": bge_sim * title_sim,
        "bge_x_tfidf": bge_sim * tfidf_sim,
        "dept_x_title": dept_prob * title_sim,
        "dept_x_num": dept_prob * num_ratio,
    }
    return feats

FEATURE_NAMES = [
    "bge_sim", "tfidf_sim", "tfidf_title_sim", "dept_prob", "title_sim",
    "num_ratio", "same_level", "rrf_score",
    "bge_x_dept", "bge_x_title", "bge_x_tfidf", "dept_x_title", "dept_x_num",
]


# ══════════════════════════════════════════════════════════════
# PHASE 1: Build calibration data from train_pos
# ══════════════════════════════════════════════════════════════
print("\n--- Phase 1: Score train_pos candidates ---")

cal_rows = []

for qi, (_, row) in enumerate(train_pos.iterrows()):
    vccs_text = get_vccs_text(row)
    vccs_course = str(row["VCCS Course"])
    target_code = str(row["W&M Course Code"]).strip()

    vccs_emb = ft_model.encode([f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True)[0]
    candidates = retrieve_three_signal_rrf(vccs_text, vccs_course, vccs_embedding=vccs_emb, k=50)

    for cand_code, rrf_score in candidates:
        sigs = extract_signals(vccs_emb, vccs_text, vccs_course, cand_code, rrf_score)
        sigs["label"] = int(cand_code == target_code)
        cal_rows.append(sigs)

    if (qi + 1) % 50 == 0:
        print(f"  {qi+1}/{len(train_pos)} queries scored")

print(f"  All {len(train_pos)} queries scored.")

cal_df = pd.DataFrame(cal_rows)
X_cal = cal_df[FEATURE_NAMES].values.astype(np.float64)
y_cal = cal_df["label"].values

print(f"\n  Calibration data: {len(cal_df)} candidates")
print(f"  Positive: {y_cal.sum()}  Negative: {(y_cal==0).sum()}")
print(f"  Pos rate: {y_cal.mean():.4f}")

# Signal discrimination
print(f"\n  Signal discrimination:")
for feat in FEATURE_NAMES:
    pos_mean = cal_df[cal_df["label"]==1][feat].mean()
    neg_mean = cal_df[cal_df["label"]==0][feat].mean()
    print(f"    {feat:<20}: pos={pos_mean:.4f}  neg={neg_mean:.4f}  gap={pos_mean-neg_mean:.4f}")


# ══════════════════════════════════════════════════════════════
# PHASE 2: Fit LogisticRegression (convex -> guaranteed global optimum)
# ══════════════════════════════════════════════════════════════
print("\n--- Phase 2: Fit LogisticRegression ---")

scaler = StandardScaler()
X_cal_scaled = scaler.fit_transform(X_cal)

# Try multiple regularization strengths
best_model = None
best_C = None
best_metric = -1

for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    lr = LogisticRegression(C=C, max_iter=5000, solver="lbfgs", random_state=42)
    lr.fit(X_cal_scaled, y_cal)
    probs = lr.predict_proba(X_cal_scaled)[:, 1]

    # Per-query ranking metric
    t1 = t3 = t5 = 0
    qi = 0
    for i in range(0, len(probs), 50):
        chunk_p = probs[i:i+50]
        chunk_l = y_cal[i:i+50]
        if chunk_l.sum() == 0: continue
        qi += 1
        ranked = np.argsort(chunk_p)[::-1]
        tpos = np.where(chunk_l[ranked] == 1)[0]
        if len(tpos) > 0:
            if tpos[0] < 1: t1 += 1
            if tpos[0] < 3: t3 += 1
            if tpos[0] < 5: t5 += 1

    preds_05 = (probs >= 0.5).astype(int)
    p05 = precision_score(y_cal, preds_05, zero_division=0) if preds_05.sum() > 0 else 0

    # Composite: ranking + precision
    composite = t3/qi + p05
    marker = "  <--" if composite > best_metric else ""
    print(f"  C={C:<8.3f}  Top-1={t1/qi:.3f}  Top-3={t3/qi:.3f}  Top-5={t5/qi:.3f}  "
          f"P@0.5={p05:.3f} ({preds_05.sum()} preds){marker}")

    if composite > best_metric:
        best_metric = composite
        best_model = lr
        best_C = C

print(f"\n  Best C={best_C}")

# Print coefficients
print(f"\n  Feature coefficients:")
for i, feat in enumerate(FEATURE_NAMES):
    print(f"    {feat:<20}: {best_model.coef_[0, i]:>8.4f}")
print(f"  Intercept: {best_model.intercept_[0]:.4f}")

# Evaluate calibration on training data
cal_probs = best_model.predict_proba(X_cal_scaled)[:, 1]

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    details = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1: mask = mask | (y_prob == bin_edges[i + 1])
        if mask.sum() == 0: continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        bin_size = mask.sum()
        ece += (bin_size / len(y_true)) * abs(bin_acc - bin_conf)
        details.append({"bin": f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})",
                        "n": int(bin_size), "predicted": float(bin_conf),
                        "actual": float(bin_acc), "gap": float(abs(bin_acc - bin_conf))})
    return ece, details

cal_ece, _ = expected_calibration_error(y_cal, cal_probs)
print(f"\n  Cal ECE: {cal_ece:.4f}  Brier: {brier_score_loss(y_cal, cal_probs):.4f}")

pos_probs = cal_probs[y_cal == 1]
neg_probs = cal_probs[y_cal == 0]
print(f"  Positive probs: min={pos_probs.min():.4f} median={np.median(pos_probs):.4f} "
      f"mean={pos_probs.mean():.4f} max={pos_probs.max():.4f}")
print(f"    >0.5: {(pos_probs>0.5).sum()}/{len(pos_probs)}  "
      f">0.7: {(pos_probs>0.7).sum()}/{len(pos_probs)}  "
      f">0.9: {(pos_probs>0.9).sum()}/{len(pos_probs)}")


# ══════════════════════════════════════════════════════════════
# PHASE 3: Evaluate on held-out test_pos
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"FINAL EVALUATION ON HELD-OUT test_pos (n={len(test_pos)})")
print(f"Pipeline: Retrieve (RRF k=50) -> 13 signals -> LogReg(C={best_C})")
print(f"{'='*70}")

top1 = top3 = top5 = top10 = 0
not_retrieved = 0
total = len(test_pos)

all_test_labels = []
all_test_probs = []
test_e2e_results = []

for qi, (_, row) in enumerate(test_pos.iterrows()):
    vccs_text = get_vccs_text(row)
    vccs_course = str(row["VCCS Course"])
    target_code = str(row["W&M Course Code"]).strip()

    vccs_emb = ft_model.encode([f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True)[0]
    candidates = retrieve_three_signal_rrf(vccs_text, vccs_course, vccs_embedding=vccs_emb, k=50)
    candidate_codes = [c[0] for c in candidates]

    if target_code not in candidate_codes:
        not_retrieved += 1
        test_e2e_results.append({"target": target_code, "retrieved": False, "rank": -1, "prob": 0.0})
        continue

    # Score all candidates
    feat_vecs = []
    cand_codes_list = []
    for cand_code, rrf_score in candidates:
        sigs = extract_signals(vccs_emb, vccs_text, vccs_course, cand_code, rrf_score)
        feat_vec = [sigs[fn] for fn in FEATURE_NAMES]
        feat_vecs.append(feat_vec)
        cand_codes_list.append(cand_code)

    X_test_q = scaler.transform(np.array(feat_vecs, dtype=np.float64))
    probs = best_model.predict_proba(X_test_q)[:, 1]

    scored = list(zip(cand_codes_list, probs))
    scored.sort(key=lambda x: -x[1])

    ranked_codes = [s[0] for s in scored]
    target_idx = ranked_codes.index(target_code)
    target_prob = float(scored[target_idx][1])

    if target_idx < 1: top1 += 1
    if target_idx < 3: top3 += 1
    if target_idx < 5: top5 += 1
    if target_idx < 10: top10 += 1

    test_e2e_results.append({"target": target_code, "retrieved": True,
                             "rank": target_idx + 1, "prob": target_prob})

    for code, prob in scored:
        is_pos = int(code == target_code)
        all_test_labels.append(is_pos)
        all_test_probs.append(float(prob))

    if (qi + 1) % 20 == 0:
        print(f"  Evaluated {qi+1}/{total}...")

print(f"  Evaluated {total}/{total}.")

# Compute metrics
test_labels = np.array(all_test_labels)
test_probs = np.array(all_test_probs)

preds_05 = (test_probs >= 0.5).astype(int)
preds_07 = (test_probs >= 0.7).astype(int)
precision_05 = precision_score(test_labels, preds_05, zero_division=0)
precision_07 = precision_score(test_labels, preds_07, zero_division=0) if preds_07.sum() > 0 else 0
top3_recall = top3 / total

neg_test_probs = test_probs[test_labels == 0]
fc_rate = (neg_test_probs > 0.5).mean() if len(neg_test_probs) > 0 else 0

ece, cal_details = expected_calibration_error(test_labels, test_probs)
brier = brier_score_loss(test_labels, test_probs)

# ── Scorecard ───────────────────────────────────────────────────────────
scorecard = {
    "Precision (t=0.5)":     (precision_05,  "> 0.85", precision_05 >= 0.85),
    "Precision (t=0.7)":     (precision_07,  "> 0.90", precision_07 >= 0.90),
    "Top-3 Recall (E2E)":    (top3_recall,   "> 0.70", top3_recall >= 0.70),
    "False Confidence Rate":  (fc_rate,       "< 0.30", fc_rate <= 0.30),
    "Calibration Error":      (ece,           "< 0.05", ece <= 0.05),
    "Brier Score":            (brier,         "< 0.10", brier <= 0.10),
}

print(f"\n  {'Metric':<28} {'Value':>8} {'Target':>10} {'Status':>8}")
print(f"  {'─'*56}")
for name, (val, tgt, passed) in scorecard.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {name:<28} {val:>8.3f} {tgt:>10} {status:>8}")

passing = sum(1 for _, (_, _, p) in scorecard.items() if p)
print(f"\n  Result: {passing}/{len(scorecard)} metrics passing")

print(f"\n  End-to-end recall:")
print(f"    Top-1: {top1/total:.3f}  Top-3: {top3/total:.3f}  Top-5: {top5/total:.3f}  Top-10: {top10/total:.3f}")
print(f"    Not retrieved: {not_retrieved}/{total}")

# ── v1 vs v2 vs v3 ─────────────────────────────────────────────────────
v1 = {"Precision (t=0.5)": 0.762, "Precision (t=0.7)": 0.884,
      "Top-3 Recall (E2E)": 0.455, "False Confidence": 0.052,
      "ECE": 0.015, "Brier": 0.100}
v2 = {"Precision (t=0.5)": 0.923, "Precision (t=0.7)": 0.970,
      "Top-3 Recall (E2E)": 0.856, "False Confidence": 0.007,
      "ECE": 0.027, "Brier": 0.043}
v3 = {"Precision (t=0.5)": precision_05, "Precision (t=0.7)": precision_07,
      "Top-3 Recall (E2E)": top3_recall, "False Confidence": fc_rate,
      "ECE": ece, "Brier": brier}

print(f"\n  {'Metric':<28} {'v1':>8} {'v2*':>8} {'v3':>8}")
print(f"  {'─'*56}")
print(f"  * v2 on full data (no held-out), not directly comparable")
for name in v1:
    print(f"  {name:<28} {v1[name]:>8.3f} {v2[name]:>8.3f} {v3[name]:>8.3f}")

# ── Calibration table ───────────────────────────────────────────────────
print(f"\n  Calibration bins:")
print(f"  {'Bin':<14} {'n':>5} {'Predicted':>10} {'Actual':>8} {'Gap':>6}")
for d in cal_details:
    print(f"  {d['bin']:<14} {d['n']:>5} {d['predicted']:>10.3f} {d['actual']:>8.3f} {d['gap']:>6.3f}")

# ── Target prob distribution ────────────────────────────────────────────
target_probs = [r["prob"] for r in test_e2e_results if r["retrieved"]]
print(f"\n  Target (correct match) probability distribution:")
print(f"    min={min(target_probs):.4f}  median={np.median(target_probs):.4f}  "
      f"mean={np.mean(target_probs):.4f}  max={max(target_probs):.4f}")
print(f"    >0.5: {sum(1 for p in target_probs if p > 0.5)}/{len(target_probs)}  "
      f">0.7: {sum(1 for p in target_probs if p > 0.7)}/{len(target_probs)}  "
      f">0.9: {sum(1 for p in target_probs if p > 0.9)}/{len(target_probs)}")

# ── False confidence detail ─────────────────────────────────────────────
print(f"\n  False Confidence detail:")
print(f"    P>0.5: {(neg_test_probs > 0.5).sum()} / {len(neg_test_probs)} = {(neg_test_probs > 0.5).mean():.4f}")
print(f"    P>0.7: {(neg_test_probs > 0.7).sum()}")
print(f"    P>0.9: {(neg_test_probs > 0.9).sum()}")

# ══════════════════════════════════════════════════════════════
# SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════
os.makedirs("artifacts", exist_ok=True)

scorecard_data = {
    "scorecard": scorecard,
    "v1": v1, "v2": v2, "v3": v3,
    "model_type": "logistic_regression",
    "best_C": best_C,
    "feature_names": FEATURE_NAMES,
    "top1_recall": top1/total, "top3_recall": top3/total,
    "top5_recall": top5/total, "top10_recall": top10/total,
    "not_retrieved": not_retrieved, "total_test": total,
    "cal_details": cal_details,
}

with open("artifacts/scorecard.pkl", "wb") as f: pickle.dump(scorecard_data, f)
with open("artifacts/classifier.pkl", "wb") as f: pickle.dump(best_model, f)
with open("artifacts/scaler.pkl", "wb") as f: pickle.dump(scaler, f)
with open("artifacts/tfidf.pkl", "wb") as f: pickle.dump(tfidf, f)
np.save("artifacts/wm_embeddings.npy", wm_embs_ft)
with open("artifacts/wm_lookup.pkl", "wb") as f: pickle.dump(wm_lookup, f)
with open("artifacts/dept_map.pkl", "wb") as f: pickle.dump(dict(dept_map), f)
with open("artifacts/dept_prior.pkl", "wb") as f: pickle.dump(dept_prior, f)
with open("artifacts/feature_names.pkl", "wb") as f: pickle.dump(FEATURE_NAMES, f)
with open("artifacts/wm_codes.pkl", "wb") as f: pickle.dump(wm_codes, f)

print(f"\nAll artifacts saved to ./artifacts/")
print("Done.")
