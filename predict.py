"""
Prediction interface for TransferZAI.

Functions:
  load_artifacts()         -> dict of all model artifacts
  predict_transfer(...)    -> per-institution ranked matches with calibrated probabilities
  evaluate_transcript(...) -> transcript-level eligibility assessment
"""

import re
import pickle
import numpy as np
import os
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

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


# Institution registry — add new schools here
INSTITUTION_REGISTRY = {
    "wm": {"name": "William & Mary"},
    "northeastern": {"name": "Northeastern University"},
}


# ── Text helpers ────────────────────────────────────────────────────────

def parse_target_course(code_str):
    """Parse a target-institution course code like 'ACCT 1201' or 'BUAD 301'."""
    if not code_str or str(code_str).strip() == "":
        return None
    m = re.match(r"([A-Z]{2,6})\s+(\d{3,4})", str(code_str).strip())
    if m:
        return {"dept": m.group(1), "number": int(m.group(2)),
                "full": f"{m.group(1)} {m.group(2)}"}
    return None


def parse_vccs_course(raw):
    """Parse VCCS course string like 'ACC 211 PRINCIPLES OF ACCOUNTING I'."""
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


def build_course_string(dept, number, title):
    """Reconstruct a VCCS-style course string from separate fields."""
    return f"{dept.strip().upper()} {str(number).strip()} {title.strip().upper()}"


# ── Artifact loading ────────────────────────────────────────────────────

_artifacts = None


def _load_institution(key, a):
    """Load artifacts for one target institution into a['institutions'][key]."""
    prefix = f"{ARTIFACTS_DIR}/{key}"

    lookup_path = f"{prefix}_lookup.pkl"
    codes_path = f"{prefix}_codes.pkl"
    emb_path = f"{prefix}_embeddings.npy"

    missing = [p for p in [lookup_path, codes_path, emb_path] if not Path(p).exists()]
    if missing:
        print(f"  Skipping '{key}': missing {[Path(p).name for p in missing]}")
        return False

    with open(lookup_path, "rb") as f:
        lookup = pickle.load(f)
    with open(codes_path, "rb") as f:
        codes = pickle.load(f)
    embeddings = np.load(emb_path)

    texts = [
        f"{lookup[c]['title']} {clean_text(lookup[c]['description'])}"
        for c in codes
    ]
    tfidf_matrix = a["tfidf"].transform(texts)
    code_to_idx = {code: i for i, code in enumerate(codes)}
    code_to_dept = {
        code: (parse_target_course(code)["dept"] if parse_target_course(code) else "UNK")
        for code in codes
    }

    a["institutions"][key] = {
        "name": INSTITUTION_REGISTRY.get(key, {}).get("name", key),
        "lookup": lookup,
        "codes": codes,
        "embeddings": embeddings,
        "tfidf_matrix": tfidf_matrix,
        "code_to_idx": code_to_idx,
        "code_to_dept": code_to_dept,
        "texts": texts,
    }
    return True


def load_artifacts():
    """Load all model artifacts. Cached after first call."""
    global _artifacts
    if _artifacts is not None:
        return _artifacts

    a = {}

    # Shared model artifacts
    with open(f"{ARTIFACTS_DIR}/classifier.pkl", "rb") as f:
        a["classifier"] = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/scaler.pkl", "rb") as f:
        a["scaler"] = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/tfidf.pkl", "rb") as f:
        a["tfidf"] = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/dept_map.pkl", "rb") as f:
        a["dept_map"] = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/dept_prior.pkl", "rb") as f:
        a["dept_prior"] = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/feature_names.pkl", "rb") as f:
        a["feature_names"] = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/scorecard.pkl", "rb") as f:
        a["scorecard"] = pickle.load(f)

    print("Loading fine-tuned BGE model...")
    a["bge_model"] = SentenceTransformer(
        BGE_MODEL_PATH, device="cpu", token=os.environ.get("HF_TOKEN")
    )
    print("BGE model loaded.")

    # Per-institution artifacts
    a["institutions"] = {}

    # W&M uses legacy artifact names (wm_*)
    with open(f"{ARTIFACTS_DIR}/wm_lookup.pkl", "rb") as f:
        wm_lookup = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/wm_codes.pkl", "rb") as f:
        wm_codes = pickle.load(f)
    wm_embeddings = np.load(f"{ARTIFACTS_DIR}/wm_embeddings.npy")
    wm_texts = [
        f"{wm_lookup[c]['title']} {clean_text(wm_lookup[c]['description'])}"
        for c in wm_codes
    ]
    a["institutions"]["wm"] = {
        "name": "William & Mary",
        "lookup": wm_lookup,
        "codes": wm_codes,
        "embeddings": wm_embeddings,
        "tfidf_matrix": a["tfidf"].transform(wm_texts),
        "code_to_idx": {code: i for i, code in enumerate(wm_codes)},
        "code_to_dept": {
            code: (parse_target_course(code)["dept"] if parse_target_course(code) else "UNK")
            for code in wm_codes
        },
        "texts": wm_texts,
    }
    # Keep legacy top-level keys for backward compat (scorecard display)
    a["wm_codes"] = wm_codes

    # Other institutions
    for key in INSTITUTION_REGISTRY:
        if key == "wm":
            continue
        _load_institution(key, a)

    print(f"Loaded institutions: {list(a['institutions'].keys())}")
    _artifacts = a
    return a


# ── Retrieval ───────────────────────────────────────────────────────────


def retrieve_candidates(vccs_text, vccs_dept, vccs_emb, inst, dept_prior, tfidf, k=RETRIEVAL_K):
    """
    Three-signal RRF retrieval against a single institution catalog.
    dept_prior: P(target_dept | vccs_dept) dict; empty dict if no prior available.
    """
    codes = inst["codes"]
    embs = inst["embeddings"]
    tfidf_mat = inst["tfidf_matrix"]
    code_to_dept = inst["code_to_dept"]

    # Signal 1: BGE
    bge_sims = embs @ vccs_emb
    bge_ranked = np.argsort(bge_sims)[::-1]

    # Signal 2: TF-IDF
    vccs_tfidf_vec = tfidf.transform([vccs_text])
    tfidf_sims = cosine_similarity(vccs_tfidf_vec, tfidf_mat).flatten()
    tfidf_ranked = np.argsort(tfidf_sims)[::-1]

    # Signal 3: Department prior (only meaningful for W&M; others get 0)
    prior_probs = dept_prior.get(vccs_dept, {}) if vccs_dept else {}
    dept_scores = [(i, prior_probs.get(code_to_dept[codes[i]], 0.0))
                   for i in range(len(codes))]
    dept_scores.sort(key=lambda x: (-x[1], x[0]))
    dept_ranked = [idx for idx, _ in dept_scores]

    # RRF fusion
    rrf_scores = defaultdict(float)
    for rank, idx in enumerate(bge_ranked[:200]):
        rrf_scores[codes[idx]] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(tfidf_ranked[:200]):
        rrf_scores[codes[idx]] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(dept_ranked[:200]):
        rrf_scores[codes[idx]] += DEPT_WEIGHT * (1.0 / (RRF_K + rank + 1))

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]
    return ranked


# ── Feature extraction ──────────────────────────────────────────────────

def extract_signals(vccs_emb, vccs_text, vccs_dept, vccs_number, vccs_title,
                    cand_code, rrf_score, inst, dept_prior, tfidf):
    """Extract the 13 features used by the LogReg classifier."""
    code_to_idx = inst["code_to_idx"]
    code_to_dept = inst["code_to_dept"]
    lookup = inst["lookup"]
    texts = inst["texts"]
    embs = inst["embeddings"]

    cand_idx = code_to_idx[cand_code]
    wm_emb = embs[cand_idx]
    bge_sim = float(vccs_emb @ wm_emb)

    cand_text = texts[cand_idx]
    vecs = tfidf.transform([vccs_text, cand_text])
    tfidf_sim = float(cosine_similarity(vecs[0:1], vecs[1:2])[0, 0])

    wm_dept = code_to_dept.get(cand_code, "UNK")
    dept_prob = dept_prior.get(vccs_dept, {}).get(wm_dept, 0.0) if vccs_dept else 0.0

    wm_title = lookup.get(cand_code, {}).get("title", "")
    title_sim = SequenceMatcher(None, clean_text(vccs_title), clean_text(wm_title)).ratio()

    if vccs_title.strip() and wm_title.strip():
        tvecs = tfidf.transform([clean_text(vccs_title), clean_text(wm_title)])
        tfidf_title_sim = float(cosine_similarity(tvecs[0:1], tvecs[1:2])[0, 0])
    else:
        tfidf_title_sim = 0.0

    target_parsed = parse_target_course(cand_code)
    vccs_num = int(vccs_number) if str(vccs_number).isdigit() else 0
    if vccs_num and target_parsed:
        tgt_num = target_parsed["number"]
        num_ratio = min(vccs_num, tgt_num) / max(vccs_num, tgt_num) if max(vccs_num, tgt_num) > 0 else 0.0
        same_level = int(vccs_num // 100 == tgt_num // 100)
    else:
        num_ratio = 0.0
        same_level = 0

    sigs = {
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
    return sigs


# ── Prediction ──────────────────────────────────────────────────────────

def predict_transfer(vccs_dept, vccs_number, vccs_title, vccs_desc="",
                     institutions=None, top_k=TOP_K_DISPLAY):
    """
    Predict transfer equivalents for a course across one or more institutions.

    Args:
        vccs_dept:    Department code, e.g. "ACC"
        vccs_number:  Course number, e.g. "211"
        vccs_title:   Course title, e.g. "PRINCIPLES OF ACCOUNTING I"
        vccs_desc:    Course description (optional)
        institutions: list of institution keys, e.g. ["wm", "northeastern"]
                      Defaults to all loaded institutions.
        top_k:        Number of results per institution

    Returns:
        dict keyed by institution key:
        {
          "wm": [{"code": ..., "title": ..., "probability": ..., "signals": {...}}, ...],
          "northeastern": [...],
        }
    """
    a = load_artifacts()
    clf = a["classifier"]
    scaler = a["scaler"]
    tfidf = a["tfidf"]
    feature_names = a["feature_names"]
    dept_prior = a["dept_prior"]
    bge_model = a["bge_model"]

    if institutions is None:
        institutions = list(a["institutions"].keys())

    vccs_dept = vccs_dept.strip().upper()
    vccs_number = str(vccs_number).strip()
    vccs_title_clean = vccs_title.strip()
    vccs_text = f"{vccs_title_clean} {clean_text(vccs_desc)}"

    vccs_emb = bge_model.encode(
        [f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True
    )[0]

    results = {}
    for inst_key in institutions:
        if inst_key not in a["institutions"]:
            continue
        inst = a["institutions"][inst_key]

        candidates = retrieve_candidates(
            vccs_text, vccs_dept, vccs_emb, inst, dept_prior, tfidf, k=RETRIEVAL_K
        )

        feat_vecs = []
        cand_info_list = []
        for cand_code, rrf_score in candidates:
            sigs = extract_signals(
                vccs_emb, vccs_text, vccs_dept, vccs_number, vccs_title_clean,
                cand_code, rrf_score, inst, dept_prior, tfidf
            )
            feat_vec = [sigs[fn] for fn in feature_names]
            feat_vecs.append(feat_vec)
            cand_info_list.append((cand_code, rrf_score, sigs))

        X = scaler.transform(np.array(feat_vecs, dtype=np.float64))
        probs = clf.predict_proba(X)[:, 1]

        inst_results = []
        for i, (cand_code, rrf_score, sigs) in enumerate(cand_info_list):
            info = inst["lookup"].get(cand_code, {})
            inst_results.append({
                "code": cand_code,
                "title": info.get("title", ""),
                "probability": float(probs[i]),
                "signals": sigs,
            })

        inst_results.sort(key=lambda x: -x["probability"])
        results[inst_key] = inst_results[:top_k]

    return results


def evaluate_transcript(courses, institutions=None,
                        min_credits_required=MIN_CREDITS_REQUIRED,
                        credit_hours_per_course=DEFAULT_CREDITS_PER_COURSE):
    """
    Evaluate a full transcript for transfer eligibility.

    Args:
        courses: list of dicts with keys:
                   dept, number, title, description (optional), credits (optional)
        institutions: list of institution keys (defaults to all loaded)
        min_credits_required: minimum transferable credits for eligibility
        credit_hours_per_course: default credits if not specified

    Returns:
        dict keyed by institution key, each containing eligibility summary
        and per-course results.
    """
    a = load_artifacts()
    if institutions is None:
        institutions = list(a["institutions"].keys())

    # Per-institution accumulation
    inst_totals = {k: {"total_credits": 0, "transferable": 0, "high_conf": 0, "results": []}
                   for k in institutions}
    total_credits_overall = 0

    for c in courses:
        preds_by_inst = predict_transfer(
            c["dept"], c["number"], c["title"],
            c.get("description", ""),
            institutions=institutions,
            top_k=1,
        )
        cr = c.get("credits", credit_hours_per_course)
        total_credits_overall += cr

        for inst_key in institutions:
            best = preds_by_inst.get(inst_key, [{}])[0] if preds_by_inst.get(inst_key) else None
            prob = best["probability"] if best else 0.0
            inst_totals[inst_key]["total_credits"] += cr
            if prob >= TRANSFER_THRESHOLD:
                inst_totals[inst_key]["transferable"] += cr
            if prob >= HIGH_CONFIDENCE_THRESHOLD:
                inst_totals[inst_key]["high_conf"] += cr
            inst_totals[inst_key]["results"].append({
                "dept": c["dept"],
                "number": c["number"],
                "title": c["title"],
                "credits": cr,
                "best_match": best["code"] if best else None,
                "best_title": best["title"] if best else None,
                "probability": prob,
            })

    output = {}
    for inst_key in institutions:
        d = inst_totals[inst_key]
        high_conf = d["high_conf"]
        transferable = d["transferable"]
        total = d["total_credits"]
        meets = high_conf >= min_credits_required
        borderline = not meets and transferable >= min_credits_required

        if meets:
            summary = (f"LIKELY ELIGIBLE: {high_conf} credits transfer with high confidence "
                       f"(>={int(HIGH_CONFIDENCE_THRESHOLD*100)}%), exceeding the "
                       f"{min_credits_required}-credit minimum.")
        elif borderline:
            summary = (f"BORDERLINE: {high_conf} high-confidence credits "
                       f"(below {min_credits_required} minimum), but {transferable} may transfer. "
                       f"Recommend registrar verification.")
        else:
            summary = (f"UNLIKELY ELIGIBLE: Only {transferable} credits show transfer potential "
                       f"({min_credits_required} required). Do not proceed without registrar confirmation.")

        output[inst_key] = {
            "institution_name": a["institutions"][inst_key]["name"],
            "total_courses": len(courses),
            "total_credits": total,
            "transferable_credits_confident": high_conf,
            "transferable_credits_possible": transferable,
            "min_required": min_credits_required,
            "eligible": meets,
            "borderline": borderline,
            "summary": summary,
            "course_results": d["results"],
        }

    return output


# ── Demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DEMO PREDICTIONS (TransferZAI — multi-institution)")
    print("=" * 60)

    demo_cases = [
        ("ACC", "211", "PRINCIPLES OF ACCOUNTING I",
         "Introduces accounting principles with respect to financial reporting.",
         "Known positive: ACC 211 -> BUAD 203 (W&M)"),
        ("CSC", "221", "INTRODUCTION TO PROBLEM SOLVING AND PROGRAMMING",
         "Introduces a modern programming language for problem solving.",
         "Plausible CS transfer"),
    ]

    for dept, num, title, desc, label in demo_cases:
        print(f"\n{'─'*60}")
        print(f"QUERY: {dept} {num} {title}")
        print(f"  ({label})")
        results = predict_transfer(dept, num, title, desc, top_k=3)
        for inst_key, matches in results.items():
            print(f"\n  → {inst_key.upper()}:")
            for i, r in enumerate(matches):
                conf = "HIGH" if r["probability"] >= HIGH_CONFIDENCE_THRESHOLD else \
                       "MED" if r["probability"] >= TRANSFER_THRESHOLD else "LOW"
                print(f"    #{i+1}  {r['code']:<12} {r['title']:<40} "
                      f"P={r['probability']:.3f} [{conf}]")
