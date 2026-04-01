"""
Step 8 — Prediction interface for TransferZAI v3.

Functions:
  load_artifacts()         -> dict of all model artifacts
  predict_transfer(...)    -> ranked list of W&M matches with calibrated probabilities
  evaluate_transcript(...) -> transcript-level eligibility assessment
"""

import re, pickle
import numpy as np
import os
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from config import (
    BGE_MODEL_PATH, QUERY_PREFIX, RETRIEVAL_K, RRF_K, DEPT_WEIGHT,
    TOP_K_DISPLAY, HIGH_CONFIDENCE_THRESHOLD, TRANSFER_THRESHOLD,
    DEFAULT_CREDITS_PER_COURSE, MIN_CREDITS_REQUIRED, ARTIFACTS_DIR,
)

# Ensure artifacts are available
if not Path(ARTIFACTS_DIR).exists():
    from download_artifacts import download_artifacts
    download_artifacts()


# ── Text parsing helpers ────────────────────────────────────────────────

def parse_wm_course(code_str):
    if not code_str or str(code_str).strip() == "":
        return None
    m = re.match(r"([A-Z]{2,4})\s+(\d{3})", str(code_str).strip())
    if m:
        return {"dept": m.group(1), "number": int(m.group(2)),
                "full": f"{m.group(1)} {m.group(2)}"}
    return None


def parse_vccs_course(raw):
    raw = str(raw).strip()
    parts = re.split(r"\s*TAKEN WITH\s*", raw, flags=re.IGNORECASE)
    courses = []
    for part in parts:
        m = re.match(r"([A-Z]{2,4})\s+(\d{3})\s+(.*)", part.strip())
        if m:
            courses.append({"dept": m.group(1), "number": int(m.group(2)),
                            "title": m.group(3).strip(),
                            "full": f"{m.group(1)} {m.group(2)}"})
    return courses


def clean_text(text):
    if not text or str(text) in ("", "nan", "Description not found"):
        return ""
    text = str(text).lower()
    text = re.sub(r"prerequisite\(s\):.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(CSI|ALV|NQR|Additional)\b", "", text)
    text = re.sub(r"cross-listed with:.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Artifact loading ────────────────────────────────────────────────────

_artifacts = None

def load_artifacts():
    """Load all model artifacts. Cached after first call."""
    global _artifacts
    if _artifacts is not None:
        return _artifacts

    a = {}
    with open(f"{ARTIFACTS_DIR}/classifier.pkl", "rb") as f:
        a["classifier"] = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/scaler.pkl", "rb") as f:
        a["scaler"] = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/tfidf.pkl", "rb") as f:
        a["tfidf"] = pickle.load(f)
    a["wm_embeddings"] = np.load(f"{ARTIFACTS_DIR}/wm_embeddings.npy")
    with open(f"{ARTIFACTS_DIR}/wm_lookup.pkl", "rb") as f:
        a["wm_lookup"] = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/dept_map.pkl", "rb") as f:
        a["dept_map"] = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/dept_prior.pkl", "rb") as f:
        a["dept_prior"] = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/feature_names.pkl", "rb") as f:
        a["feature_names"] = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/wm_codes.pkl", "rb") as f:
        a["wm_codes"] = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/scorecard.pkl", "rb") as f:
        a["scorecard"] = pickle.load(f)

    # Derived indexes
    a["wm_code_to_idx"] = {code: i for i, code in enumerate(a["wm_codes"])}
    a["wm_code_to_dept"] = {
        code: (parse_wm_course(code)["dept"] if parse_wm_course(code) else "UNK")
        for code in a["wm_codes"]
    }
    a["wm_texts"] = [
        f"{a['wm_lookup'][c]['title']} {clean_text(a['wm_lookup'][c]['description'])}"
        for c in a["wm_codes"]
    ]

    # TF-IDF matrix for W&M catalog
    a["wm_tfidf_matrix"] = a["tfidf"].transform(a["wm_texts"])

    # Load BGE model
    print("Loading fine-tuned BGE model...")
    a["bge_model"] = SentenceTransformer(BGE_MODEL_PATH, device="cpu", token=os.environ.get("HF_TOKEN"))
    print("Artifacts loaded.")

    _artifacts = a
    return a


# ── Retrieval ───────────────────────────────────────────────────────────

def retrieve_candidates(vccs_text, vccs_course_str, vccs_embedding, a, k=RETRIEVAL_K):
    """Three-signal RRF retrieval."""
    wm_embs = a["wm_embeddings"]
    wm_codes = a["wm_codes"]
    tfidf_model = a["tfidf"]
    wm_tfidf = a["wm_tfidf_matrix"]
    dept_prior = a["dept_prior"]
    code_to_dept = a["wm_code_to_dept"]

    # Signal 1: BGE
    bge_sims = wm_embs @ vccs_embedding
    bge_ranked = np.argsort(bge_sims)[::-1]

    # Signal 2: TF-IDF
    tfidf_vec = tfidf_model.transform([vccs_text])
    tfidf_sims = cosine_similarity(tfidf_vec, wm_tfidf).flatten()
    tfidf_ranked = np.argsort(tfidf_sims)[::-1]

    # Signal 3: Department prior
    vccs_parsed = parse_vccs_course(vccs_course_str)
    vccs_dept = vccs_parsed[0]["dept"] if vccs_parsed else None
    prior_probs = dept_prior.get(vccs_dept, {}) if vccs_dept else {}
    dept_scores = [(i, prior_probs.get(code_to_dept[wm_codes[i]], 0.0))
                   for i in range(len(wm_codes))]
    dept_scores.sort(key=lambda x: (-x[1], x[0]))
    dept_ranked = [idx for idx, _ in dept_scores]

    # RRF fusion
    rrf_scores = defaultdict(float)
    for rank, idx in enumerate(bge_ranked[:200]):
        rrf_scores[wm_codes[idx]] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(tfidf_ranked[:200]):
        rrf_scores[wm_codes[idx]] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(dept_ranked[:200]):
        rrf_scores[wm_codes[idx]] += DEPT_WEIGHT * (1.0 / (RRF_K + rank + 1))

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]
    return [(code, rrf_sc) for code, rrf_sc in ranked]


# ── Feature extraction ──────────────────────────────────────────────────

def extract_signals(vccs_emb, vccs_text, vccs_course_str, cand_code, rrf_score, a):
    """Extract the 13 features used by the LogReg classifier."""
    wm_embs = a["wm_embeddings"]
    wm_lookup = a["wm_lookup"]
    code_to_idx = a["wm_code_to_idx"]
    code_to_dept = a["wm_code_to_dept"]
    dept_prior = a["dept_prior"]
    dept_map = a["dept_map"]
    tfidf_model = a["tfidf"]
    wm_texts = a["wm_texts"]

    cand_idx = code_to_idx[cand_code]
    wm_emb = wm_embs[cand_idx]

    bge_sim = float(vccs_emb @ wm_emb)

    cand_text = wm_texts[cand_idx]
    vecs = tfidf_model.transform([vccs_text, cand_text])
    tfidf_sim = float(cosine_similarity(vecs[0:1], vecs[1:2])[0, 0])

    vccs_parsed = parse_vccs_course(vccs_course_str)
    vccs_dept = vccs_parsed[0]["dept"] if vccs_parsed else None
    wm_dept = code_to_dept.get(cand_code, "UNK")
    dept_prob = dept_prior.get(vccs_dept, {}).get(wm_dept, 0.0) if vccs_dept else 0.0

    vccs_title = " ".join(c["title"] for c in vccs_parsed) if vccs_parsed else ""
    wm_title = wm_lookup.get(cand_code, {}).get("title", "")
    title_sim = SequenceMatcher(None, clean_text(vccs_title), clean_text(wm_title)).ratio()

    if vccs_title.strip() and wm_title.strip():
        tvecs = tfidf_model.transform([clean_text(vccs_title), clean_text(wm_title)])
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

    return {
        "bge_sim": bge_sim,
        "tfidf_sim": tfidf_sim,
        "tfidf_title_sim": tfidf_title_sim,
        "dept_prob": dept_prob,
        "title_sim": title_sim,
        "num_ratio": num_ratio,
        "same_level": float(same_level),
        "rrf_score": rrf_score,
        "bge_x_dept": bge_sim * dept_prob,
        "bge_x_title": bge_sim * title_sim,
        "bge_x_tfidf": bge_sim * tfidf_sim,
        "dept_x_title": dept_prob * title_sim,
        "dept_x_num": dept_prob * num_ratio,
    }


# ── Prediction ──────────────────────────────────────────────────────────

def predict_transfer(vccs_course, vccs_desc="", top_k=TOP_K_DISPLAY):
    """
    Predict W&M transfer equivalents for a VCCS course.

    Args:
        vccs_course: e.g. "ACC 211 PRINCIPLES OF ACCOUNTING I"
        vccs_desc: course description text
        top_k: number of results to return

    Returns:
        list of dicts with keys: wm_code, wm_title, probability, signals
    """
    a = load_artifacts()
    model = a["bge_model"]
    clf = a["classifier"]
    scaler = a["scaler"]
    feature_names = a["feature_names"]

    vc_parsed = parse_vccs_course(vccs_course)
    vc_title = " ".join(c["title"] for c in vc_parsed) if vc_parsed else ""
    vccs_text = f"{vc_title} {clean_text(vccs_desc)}"

    vccs_emb = model.encode(
        [f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True)[0]

    candidates = retrieve_candidates(vccs_text, vccs_course, vccs_emb, a, k=RETRIEVAL_K)

    feat_vecs = []
    cand_info_list = []
    for cand_code, rrf_score in candidates:
        sigs = extract_signals(vccs_emb, vccs_text, vccs_course, cand_code, rrf_score, a)
        feat_vec = [sigs[fn] for fn in feature_names]
        feat_vecs.append(feat_vec)
        cand_info_list.append((cand_code, rrf_score, sigs))

    X = scaler.transform(np.array(feat_vecs, dtype=np.float64))
    probs = clf.predict_proba(X)[:, 1]

    results = []
    for i, (cand_code, rrf_score, sigs) in enumerate(cand_info_list):
        info = a["wm_lookup"].get(cand_code, {})
        results.append({
            "wm_code": cand_code,
            "wm_title": info.get("title", ""),
            "probability": float(probs[i]),
            "signals": sigs,
        })

    results.sort(key=lambda x: -x["probability"])
    return results[:top_k]


def evaluate_transcript(courses, min_credits_required=MIN_CREDITS_REQUIRED,
                        credit_hours_per_course=DEFAULT_CREDITS_PER_COURSE):
    """
    Evaluate a full transcript for transfer eligibility.

    Args:
        courses: list of dicts with keys: course, description (optional), credits (optional)
        min_credits_required: minimum transferable credits for eligibility
        credit_hours_per_course: default credits if not specified

    Returns:
        dict with eligibility summary and per-course results
    """
    all_results = []
    total_credits = transferable_credits = high_confidence_credits = 0

    for c in courses:
        preds = predict_transfer(c["course"], c.get("description", ""), top_k=1)
        best = preds[0] if preds else None
        cr = c.get("credits", credit_hours_per_course)
        total_credits += cr

        if best and best["probability"] >= TRANSFER_THRESHOLD:
            transferable_credits += cr
        if best and best["probability"] >= HIGH_CONFIDENCE_THRESHOLD:
            high_confidence_credits += cr

        all_results.append({
            "course": c["course"],
            "best_match": best["wm_code"] if best else None,
            "best_title": best["wm_title"] if best else None,
            "probability": best["probability"] if best else 0.0,
            "credits": cr,
        })

    meets_minimum = high_confidence_credits >= min_credits_required
    borderline = not meets_minimum and transferable_credits >= min_credits_required

    if meets_minimum:
        summary = (f"LIKELY ELIGIBLE: {high_confidence_credits} credits transfer with high confidence "
                   f"(>={int(HIGH_CONFIDENCE_THRESHOLD*100)}%), exceeding the "
                   f"{min_credits_required}-credit minimum.")
    elif borderline:
        summary = (f"BORDERLINE: {high_confidence_credits} high-confidence credits "
                   f"(below {min_credits_required} minimum), but {transferable_credits} may transfer. "
                   f"Recommend registrar verification.")
    else:
        summary = (f"UNLIKELY ELIGIBLE: Only {transferable_credits} credits show transfer potential "
                   f"({min_credits_required} required). Do not proceed without registrar confirmation.")

    return {
        "total_courses": len(courses),
        "total_credits": total_credits,
        "transferable_credits_confident": high_confidence_credits,
        "transferable_credits_possible": transferable_credits,
        "min_required": min_credits_required,
        "eligible": meets_minimum,
        "borderline": borderline,
        "summary": summary,
        "course_results": all_results,
    }


# ── Demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DEMO PREDICTIONS (TransferZAI v3)")
    print("=" * 60)

    demo_cases = [
        ("ACC 211 PRINCIPLES OF ACCOUNTING I",
         "Introduces accounting principles with respect to financial reporting.",
         "Known positive: ACC 211 -> BUAD 203"),
        ("ADJ 100 SURVEY OF CRIMINAL JUSTICE",
         "Presents an overview of the United States criminal justice system.",
         "Known non-transfer"),
        ("CSC 221 INTRODUCTION TO PROBLEM SOLVING AND PROGRAMMING",
         "Introduces a modern programming language for problem solving.",
         "Plausible CS transfer"),
    ]

    for course, desc, label in demo_cases:
        print(f"\n{'─'*60}")
        print(f"QUERY: {course}")
        print(f"  ({label})")
        results = predict_transfer(course, desc, top_k=3)
        for i, r in enumerate(results):
            conf = "HIGH" if r["probability"] >= HIGH_CONFIDENCE_THRESHOLD else \
                   "MED" if r["probability"] >= TRANSFER_THRESHOLD else "LOW"
            print(f"  #{i+1}  {r['wm_code']:<12} {r['wm_title']:<40} "
                  f"P={r['probability']:.3f} [{conf}]")

    print(f"\n{'─'*60}")
    print("TRANSCRIPT EVALUATION DEMO")
    print(f"{'─'*60}")

    transcript = [
        {"course": "ACC 211 PRINCIPLES OF ACCOUNTING I",
         "description": "Introduces accounting principles with respect to financial reporting."},
        {"course": "ENG 111 COLLEGE COMPOSITION I",
         "description": "Introduces students to critical thinking and the fundamentals of academic writing."},
        {"course": "MTH 263 CALCULUS I",
         "description": "Presents concepts of limits, derivatives, definite integrals, and indefinite integrals."},
        {"course": "BIO 101 GENERAL BIOLOGY I",
         "description": "Introduces cell structure, genetics, evolution, and ecology."},
        {"course": "CSC 221 INTRODUCTION TO PROBLEM SOLVING AND PROGRAMMING",
         "description": "Introduces a modern programming language for problem solving."},
    ]

    result = evaluate_transcript(transcript)
    print(f"\n  {result['summary']}")
    print(f"\n  Total credits: {result['total_credits']}")
    print(f"  High confidence: {result['transferable_credits_confident']}")
    print(f"  Possible: {result['transferable_credits_possible']}")
    print(f"\n  Per-course results:")
    for cr in result["course_results"]:
        match = f"{cr['best_match']}" if cr['best_match'] else "None"
        conf = "HIGH" if cr["probability"] >= HIGH_CONFIDENCE_THRESHOLD else \
               "MED" if cr["probability"] >= TRANSFER_THRESHOLD else "LOW"
        print(f"    {cr['course'][:45]:<45} -> {match:<12} P={cr['probability']:.3f} [{conf}]")
