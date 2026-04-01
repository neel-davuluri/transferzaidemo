"""
Step 4 — Three-signal RRF retrieval
Signal 1: Fine-tuned BGE bi-encoder
Signal 2: TF-IDF (1,2)-grams sublinear TF 10K vocab
Signal 3: Department prior P(wm_dept | vccs_dept) from train_pos at 0.5x weight
k=50 retrieval depth. Print retrieval ceiling.
"""

import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

np.random.seed(42)

# ── Replicate data loading + split ──────────────────────────────────────
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
            courses.append({
                "dept": m.group(1), "number": int(m.group(2)),
                "title": m.group(3).strip(), "full": f"{m.group(1)} {m.group(2)}",
            })
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


# Stratified split
pos_df["wm_dept"] = pos_df["W&M Course Code"].apply(
    lambda x: parse_wm_course(x)["dept"] if parse_wm_course(x) else "UNK"
)
dept_counts = pos_df["wm_dept"].value_counts()
rare_depts = dept_counts[dept_counts < 2].index.tolist()
pos_df["strat_dept"] = pos_df["wm_dept"].apply(lambda d: "RARE" if d in rare_depts else d)
train_pos, test_pos = train_test_split(
    pos_df, test_size=0.20, random_state=42, stratify=pos_df["strat_dept"]
)
print(f"train_pos: {len(train_pos)}  |  test_pos: {len(test_pos)}")

# ── Build dept_map from train_pos ONLY ──────────────────────────────────
dept_map = defaultdict(lambda: defaultdict(int))
for _, r in train_pos.iterrows():
    vccs_courses = parse_vccs_course(r["VCCS Course"])
    wm_parsed = parse_wm_course(r["W&M Course Code"])
    if vccs_courses and wm_parsed:
        for vc in vccs_courses:
            dept_map[vc["dept"]][wm_parsed["dept"]] += 1

# ── Department prior: P(wm_dept | vccs_dept) ────────────────────────────
# For each VCCS dept, compute probability distribution over W&M depts
dept_prior = {}
for vccs_dept in dept_map:
    total = sum(dept_map[vccs_dept].values())
    dept_prior[vccs_dept] = {wm_dept: count / total for wm_dept, count in dept_map[vccs_dept].items()}

print(f"\nDepartment prior computed for {len(dept_prior)} VCCS departments")
print("Sample priors:")
for vd in sorted(dept_prior.keys())[:5]:
    targets = sorted(dept_prior[vd].items(), key=lambda x: -x[1])[:3]
    print(f"  {vd} -> {', '.join(f'{d}({p:.2f})' for d, p in targets)}")

# ── Load fine-tuned model + build indexes ───────────────────────────────
print("\nLoading fine-tuned BGE model...")
ft_model = SentenceTransformer("./finetuned_bge", device="cpu")

QUERY_PREFIX = "Represent this course for finding transfer equivalents: "

wm_codes = list(wm_lookup.keys())
wm_texts = [f"{wm_lookup[c]['title']} {clean_text(wm_lookup[c]['description'])}" for c in wm_codes]

# Pre-parse W&M departments for dept prior lookup
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


# ── Three-signal RRF retrieval ──────────────────────────────────────────
def retrieve_three_signal_rrf(vccs_text, vccs_course_str, vccs_embedding=None, k=50):
    """
    Three-signal RRF:
      1. Fine-tuned BGE cosine similarity
      2. TF-IDF (1,2)-gram cosine similarity
      3. Department prior P(wm_dept | vccs_dept) at 0.5x weight
    """
    # Signal 1: Fine-tuned BGE
    if vccs_embedding is None:
        vccs_embedding = ft_model.encode(
            [f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True
        )[0]
    bge_sims = wm_embs_ft @ vccs_embedding
    bge_ranked = np.argsort(bge_sims)[::-1]

    # Signal 2: TF-IDF
    tfidf_vec = tfidf.transform([vccs_text])
    tfidf_sims = cosine_similarity(tfidf_vec, wm_tfidf_matrix).flatten()
    tfidf_ranked = np.argsort(tfidf_sims)[::-1]

    # Signal 3: Department prior ranking
    vccs_parsed = parse_vccs_course(vccs_course_str)
    vccs_dept = vccs_parsed[0]["dept"] if vccs_parsed else None
    prior_probs = dept_prior.get(vccs_dept, {}) if vccs_dept else {}

    # Rank W&M courses by dept prior probability
    dept_scores = []
    for i, code in enumerate(wm_codes):
        wm_dept = wm_code_to_dept[code]
        prob = prior_probs.get(wm_dept, 0.0)
        dept_scores.append((i, prob))
    # Sort by probability descending, then by index for stability
    dept_scores.sort(key=lambda x: (-x[1], x[0]))
    dept_ranked = [idx for idx, _ in dept_scores]

    # RRF fusion
    RRF_K = 60
    DEPT_WEIGHT = 0.5  # half weight for department signal

    rrf_scores = defaultdict(float)

    # Signal 1: BGE (full weight)
    for rank, idx in enumerate(bge_ranked[:200]):
        code = wm_codes[idx]
        rrf_scores[code] += 1.0 / (RRF_K + rank + 1)

    # Signal 2: TF-IDF (full weight)
    for rank, idx in enumerate(tfidf_ranked[:200]):
        code = wm_codes[idx]
        rrf_scores[code] += 1.0 / (RRF_K + rank + 1)

    # Signal 3: Department prior (0.5x weight)
    for rank, idx in enumerate(dept_ranked[:200]):
        code = wm_codes[idx]
        rrf_scores[code] += DEPT_WEIGHT * (1.0 / (RRF_K + rank + 1))

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]

    # Return with individual scores for feature engineering
    code_to_bge = {wm_codes[i]: float(bge_sims[i]) for i in range(len(wm_codes))}
    code_to_tfidf = {wm_codes[i]: float(tfidf_sims[i]) for i in range(len(wm_codes))}

    return [
        {
            "code": code,
            "rrf_score": rrf_score,
            "bge_sim": code_to_bge.get(code, 0),
            "tfidf_sim": code_to_tfidf.get(code, 0),
        }
        for code, rrf_score in ranked
    ]


# ── Also define the old 2-signal RRF for comparison ─────────────────────
def retrieve_two_signal_rrf(vccs_text, vccs_embedding=None, k=50):
    if vccs_embedding is None:
        vccs_embedding = ft_model.encode(
            [f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True
        )[0]
    bge_sims = wm_embs_ft @ vccs_embedding
    bge_ranked = np.argsort(bge_sims)[::-1]
    tfidf_vec = tfidf.transform([vccs_text])
    tfidf_sims = cosine_similarity(tfidf_vec, wm_tfidf_matrix).flatten()
    tfidf_ranked = np.argsort(tfidf_sims)[::-1]

    RRF_K = 60
    rrf_scores = defaultdict(float)
    for rank, idx in enumerate(bge_ranked[:200]):
        rrf_scores[wm_codes[idx]] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(tfidf_ranked[:200]):
        rrf_scores[wm_codes[idx]] += 1.0 / (RRF_K + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]
    return [code for code, _ in ranked]


def retrieve_bge_only(vccs_text, vccs_embedding=None, k=50):
    if vccs_embedding is None:
        vccs_embedding = ft_model.encode(
            [f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True
        )[0]
    bge_sims = wm_embs_ft @ vccs_embedding
    ranked_idx = np.argsort(bge_sims)[::-1][:k]
    return [wm_codes[i] for i in ranked_idx]


# ── Evaluate all conditions on full pos_df ──────────────────────────────
print(f"\n{'='*70}")
print(f"RETRIEVAL RECALL COMPARISON (n={len(pos_df)} positive pairs)")
print(f"{'='*70}")

# Pre-encode all VCCS texts
print("Encoding VCCS queries...")
vccs_texts_all = []
vccs_courses_all = []
for _, row in pos_df.iterrows():
    vccs_texts_all.append(get_vccs_text(row))
    vccs_courses_all.append(str(row["VCCS Course"]))

vccs_embs_all = ft_model.encode(
    [f"{QUERY_PREFIX}{t}" for t in vccs_texts_all],
    batch_size=16, show_progress_bar=True, normalize_embeddings=True,
)

conditions = {}

# Condition 1: Fine-tuned BGE only
print("\nEvaluating Fine-tuned BGE only...")
hits_bge = {k: 0 for k in [1, 3, 5, 10, 20, 50]}
for i, (_, row) in enumerate(pos_df.iterrows()):
    target = str(row["W&M Course Code"]).strip()
    codes = retrieve_bge_only(vccs_texts_all[i], vccs_embs_all[i])
    for k in hits_bge:
        if target in codes[:k]:
            hits_bge[k] += 1
conditions["Fine-tuned BGE"] = {k: hits_bge[k] / len(pos_df) for k in hits_bge}

# Condition 2: Two-signal RRF (FT BGE + TF-IDF)
print("Evaluating Two-signal RRF...")
hits_2rrf = {k: 0 for k in [1, 3, 5, 10, 20, 50]}
for i, (_, row) in enumerate(pos_df.iterrows()):
    target = str(row["W&M Course Code"]).strip()
    codes = retrieve_two_signal_rrf(vccs_texts_all[i], vccs_embs_all[i])
    for k in hits_2rrf:
        if target in codes[:k]:
            hits_2rrf[k] += 1
conditions["Two-signal RRF"] = {k: hits_2rrf[k] / len(pos_df) for k in hits_2rrf}

# Condition 3: Three-signal RRF (FT BGE + TF-IDF + Dept Prior)
print("Evaluating Three-signal RRF...")
hits_3rrf = {k: 0 for k in [1, 3, 5, 10, 20, 50]}
for i, (_, row) in enumerate(pos_df.iterrows()):
    target = str(row["W&M Course Code"]).strip()
    results = retrieve_three_signal_rrf(vccs_texts_all[i], vccs_courses_all[i], vccs_embs_all[i])
    codes = [r["code"] for r in results]
    for k in hits_3rrf:
        if target in codes[:k]:
            hits_3rrf[k] += 1
conditions["Three-signal RRF"] = {k: hits_3rrf[k] / len(pos_df) for k in hits_3rrf}

# ── Print results ───────────────────────────────────────────────────────
print(f"\n{'':>25} {'Top-1':>6} {'Top-3':>6} {'Top-5':>6} {'Top-10':>7} {'Top-20':>7} {'Top-50':>7}")
print(f"  {'─'*68}")
for name in conditions:
    r = conditions[name]
    marker = " <--" if name == "Three-signal RRF" else ""
    print(f"  {name:<23} {r[1]:>6.3f} {r[3]:>6.3f} {r[5]:>6.3f} {r[10]:>7.3f} {r[20]:>7.3f} {r[50]:>7.3f}{marker}")

# Deltas
print(f"\n  Delta (3-signal vs 2-signal):")
for k in [1, 3, 5, 10, 20, 50]:
    d = conditions["Three-signal RRF"][k] - conditions["Two-signal RRF"][k]
    print(f"    Top-{k:>2}: {d:+.3f}")

print(f"\n  Delta (3-signal vs BGE-only):")
for k in [1, 3, 5, 10, 20, 50]:
    d = conditions["Three-signal RRF"][k] - conditions["Fine-tuned BGE"][k]
    print(f"    Top-{k:>2}: {d:+.3f}")

ceiling = conditions["Three-signal RRF"][50]
print(f"\n{'='*70}")
print(f"  RETRIEVAL CEILING (Top-50 recall): {ceiling:.3f}")
print(f"  This is the hard upper bound on end-to-end performance.")
print(f"  {int(ceiling * len(pos_df))}/{len(pos_df)} positive pairs are retrievable.")
print(f"{'='*70}")
