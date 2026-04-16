"""
Prediction interface for TransferZAI.

Functions:
  load_artifacts()         -> dict of all model artifacts
  predict_transfer(...)    -> per-institution ranked matches with softmax confidence
  evaluate_transcript(...) -> transcript-level eligibility assessment

Confidence = per-query log-softmax over XGBoost scores.
All fields are optional except course title.
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")  # prevent XGBoost/PyTorch OMP deadlock

import re
import pickle
import numpy as np
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from config import (
    BGE_MODEL_PATH, BGE_HF_REPO, ARTIFACTS_HF_REPO,
    QUERY_PREFIX, RETRIEVAL_K, RRF_K,
    TOP_K_DISPLAY, HIGH_CONFIDENCE_THRESHOLD, TRANSFER_THRESHOLD,
    DEFAULT_CREDITS_PER_COURSE, MIN_CREDITS_REQUIRED, ARTIFACTS_DIR,
    MARGIN_BIAS_CORRECTION,
)


INSTITUTION_REGISTRY = {
    "wm":           {"name": "William & Mary"},
    "vt":           {"name": "Virginia Tech"},
    "ucsc":         {"name": "UC Santa Cruz"},
    # "northeastern": {"name": "Northeastern University"},  # disabled: catalog cleanup needed
}


# ── text helpers ─────────────────────────────────────────────────────────────

def clean_text(text):
    if not text or str(text) in ("", "nan", "Description not found"):
        return ""
    text = str(text).lower()
    text = re.sub(r"prerequisite\(s\):.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(CSI|ALV|NQR|Additional)\b", "", text)
    text = re.sub(r"cross-listed with:.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def parse_target_course(code_str):
    if not code_str or str(code_str).strip() == "":
        return None
    m = re.match(r"([A-Z]{2,6})\s+(\d{2,4})", str(code_str).strip())
    if m:
        return {"dept": m.group(1), "number": int(m.group(2)),
                "full": f"{m.group(1)} {m.group(2)}"}
    return None

def academic_level(num, inst_key):
    if num <= 0: return 0
    if inst_key in ("vccs", "wm"):
        return max(1, min(4, num // 100))
    if inst_key == "vt":
        return max(1, min(4, num // 1000))
    if inst_key == "ucsc":
        if num < 50:  return 1
        if num < 100: return 2
        return 3
    if num >= 1000:
        return max(1, min(4, num // 1000))
    return max(1, min(4, num // 100))


# ── artifact loading ──────────────────────────────────────────────────────────

_artifacts = None


def _load_institution(key, a, adir):
    prefix = adir / key
    lookup_path = Path(f"{prefix}_lookup.pkl")
    codes_path  = Path(f"{prefix}_codes.pkl")
    emb_path    = Path(f"{prefix}_embeddings.npy")
    missing = [p for p in [lookup_path, codes_path, emb_path] if not p.exists()]
    if missing:
        print(f"  Skipping '{key}': missing {[p.name for p in missing]}")
        return False

    with open(lookup_path, "rb") as f: lookup = pickle.load(f)
    with open(codes_path,  "rb") as f: codes  = pickle.load(f)
    embeddings = np.load(emb_path)

    texts = [
        f"{lookup[c]['title']} {clean_text(lookup[c]['description'])}"
        for c in codes
    ]
    tfidf_matrix = a["tfidf"].transform(texts)
    code_to_idx  = {code: i for i, code in enumerate(codes)}
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


def _resolve_artifacts_dir():
    """Return a Path to the artifacts directory, downloading from HF Hub if needed."""
    local = Path(ARTIFACTS_DIR)
    # On Streamlit Cloud (HF_TOKEN set), always sync from HF Hub into ./artifacts
    # to ensure we pick up new artifacts after retraining.
    # Locally, just use the existing directory.
    if local.exists() and not os.environ.get("HF_TOKEN"):
        return local
    from huggingface_hub import snapshot_download
    print(f"Downloading artifacts from {ARTIFACTS_HF_REPO} ...")
    snapshot_download(
        repo_id=ARTIFACTS_HF_REPO,
        repo_type="dataset",
        local_dir=str(local),
        token=os.environ.get("HF_TOKEN"),
        ignore_patterns=["*.git*", ".gitattributes", "README.md"],
    )
    return local


def _resolve_bge_model():
    """Return a model path/ID for SentenceTransformer, falling back to HF Hub."""
    local = Path(BGE_MODEL_PATH)
    if local.exists():
        return str(local.resolve())
    print(f"BGE model not found locally — loading from {BGE_HF_REPO} ...")
    return BGE_HF_REPO


def load_artifacts():
    global _artifacts
    if _artifacts is not None:
        return _artifacts

    adir = _resolve_artifacts_dir()
    print(f"[DIAG] artifacts dir: {adir}")
    import sys
    print(f"[DIAG] artifacts dir: {adir}")
    print(f"[DIAG] files: {sorted(p.name for p in adir.iterdir())}")
    print(f"[DIAG] python: {sys.version}")
    print(f"[DIAG] xgboost: {xgb.__version__}")
    a = {}
    a["_diag"] = {}

    with open(adir / "classifier.pkl", "rb") as f:
        a["classifier"] = pickle.load(f)
    clf = a["classifier"]
    clf_type = type(clf).__name__
    print(f"[DIAG] classifier type: {clf_type}")
    a["_diag"]["classifier_type"] = clf_type
    a["_diag"]["artifacts_dir"] = str(adir)
    a["_diag"]["python"] = sys.version.split()[0]
    a["_diag"]["xgboost"] = xgb.__version__
    if hasattr(clf, "n_estimators"):
        a["_diag"]["n_estimators"] = clf.n_estimators
        print(f"[DIAG] n_estimators: {clf.n_estimators}")
    if hasattr(clf, "get_booster"):
        a["_diag"]["num_features"] = clf.get_booster().num_features()
        print(f"[DIAG] booster num_features: {clf.get_booster().num_features()}")

    if (adir / "iso_cal.pkl").exists():
        with open(adir / "iso_cal.pkl", "rb") as f:
            a["iso_cal"] = pickle.load(f)
    else:
        a["iso_cal"] = None

    with open(adir / "tfidf.pkl", "rb") as f:
        a["tfidf"] = pickle.load(f)

    if (adir / "feature_names.pkl").exists():
        with open(adir / "feature_names.pkl", "rb") as f:
            a["feature_names"] = pickle.load(f)
    else:
        a["feature_names"] = [
            "bge_sim", "tfidf_sim", "tfidf_title_sim", "dept_sim", "title_sim",
            "level_ratio", "same_level", "rrf_score",
            "bge_x_dept", "bge_x_title", "bge_x_tfidf", "dept_x_title", "dept_x_level",
        ]
    a["_diag"]["feature_names"] = a["feature_names"]
    print(f"[DIAG] feature_names ({len(a['feature_names'])}): {a['feature_names']}")

    if (adir / "scorecard.pkl").exists():
        with open(adir / "scorecard.pkl", "rb") as f:
            a["scorecard"] = pickle.load(f)
    else:
        a["scorecard"] = {}

    # Confidence is now calibrated probability (calib_prob), not softmax.
    # Use a single global threshold; scorecard op_tau values were softmax-based.
    print(f"[DIAG] HIGH_CONFIDENCE_THRESHOLD: {HIGH_CONFIDENCE_THRESHOLD}")

    print("Loading fine-tuned BGE model...")
    a["bge_model"] = SentenceTransformer(
        _resolve_bge_model(), device="cpu", token=os.environ.get("HF_TOKEN")
    )
    print("BGE model loaded.")

    a["institutions"] = {}
    for key in INSTITUTION_REGISTRY:
        _load_institution(key, a, adir)

    print(f"Loaded institutions: {list(a['institutions'].keys())}")
    _artifacts = a
    return a


# ── retrieval ─────────────────────────────────────────────────────────────────

def retrieve_candidates(vccs_text, vccs_emb, inst, tfidf, k=RETRIEVAL_K):
    codes     = inst["codes"]
    embs      = inst["embeddings"]
    tfidf_mat = inst["tfidf_matrix"]

    bge_sims     = embs @ vccs_emb
    bge_ranked   = np.argsort(bge_sims)[::-1]
    vccs_tfidf   = tfidf.transform([vccs_text])
    tfidf_sims   = cosine_similarity(vccs_tfidf, tfidf_mat).flatten()
    tfidf_ranked = np.argsort(tfidf_sims)[::-1]

    # Two-signal RRF: BGE + TF-IDF only.
    # Dept is used as a reranking feature by XGBoost (dept_sim), not here.
    # Boosting dept in retrieval floods the top-50 with same-dept courses,
    # diluting softmax confidence even when all signals are strong.
    rrf_scores = defaultdict(float)
    for rank, idx in enumerate(bge_ranked[:200]):
        rrf_scores[codes[idx]] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(tfidf_ranked[:200]):
        rrf_scores[codes[idx]] += 1.0 / (RRF_K + rank + 1)

    return sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]


# ── feature extraction ────────────────────────────────────────────────────────

def extract_signals(vccs_emb, vccs_text, vccs_dept, vccs_number, vccs_title,
                    cand_code, rrf_score, inst, inst_key, tfidf, vccs_level=0):
    code_to_idx  = inst["code_to_idx"]
    code_to_dept = inst["code_to_dept"]
    lookup       = inst["lookup"]
    embs         = inst["embeddings"]

    cand_idx = code_to_idx[cand_code]
    bge_sim  = float(vccs_emb @ embs[cand_idx])

    cand_text = f"{lookup[cand_code]['title']} {clean_text(lookup[cand_code].get('description', ''))}"
    vecs      = tfidf.transform([vccs_text, cand_text])
    tfidf_sim = float(cosine_similarity(vecs[0:1], vecs[1:2])[0, 0])

    cand_dept = code_to_dept.get(cand_code, "UNK")
    dept_sim  = SequenceMatcher(None, vccs_dept.lower(), cand_dept.lower()).ratio() \
                if vccs_dept else 0.0

    cand_title = lookup.get(cand_code, {}).get("title", "")
    title_sim  = SequenceMatcher(None, clean_text(vccs_title), clean_text(cand_title)).ratio()

    if vccs_title.strip() and cand_title.strip():
        tv = tfidf.transform([clean_text(vccs_title), clean_text(cand_title)])
        tfidf_title_sim = float(cosine_similarity(tv[0:1], tv[1:2])[0, 0])
    else:
        tfidf_title_sim = 0.0

    parsed   = parse_target_course(cand_code)
    vccs_num = int(vccs_number) if str(vccs_number).isdigit() else 0
    # vccs_level=1-4 takes precedence over deriving from course number
    if vccs_level > 0:
        q_lvl = vccs_level
    elif vccs_num:
        q_lvl = academic_level(vccs_num, "vccs")
    else:
        q_lvl = 0
    if q_lvl > 0 and parsed:
        tgt_num    = parsed["number"]
        tgt_lvl    = academic_level(tgt_num, inst_key)
        diff       = abs(q_lvl - tgt_lvl)
        level_ratio = 1.0 - diff / 3.0
        same_level  = float(diff == 0)
    else:
        level_ratio = 0.5  # neutral: level unknown
        same_level  = 0.0

    return {
        "bge_sim":         bge_sim,
        "tfidf_sim":       tfidf_sim,
        "tfidf_title_sim": tfidf_title_sim,
        "dept_sim":        dept_sim,
        "title_sim":       title_sim,
        "level_ratio":     level_ratio,
        "same_level":      same_level,
        "rrf_score":       rrf_score,
        "bge_x_dept":      bge_sim * dept_sim,
        "bge_x_title":     bge_sim * title_sim,
        "bge_x_tfidf":     bge_sim * tfidf_sim,
        "dept_x_title":    dept_sim * title_sim,
        "dept_x_level":    dept_sim * level_ratio,
    }


# ── prediction ────────────────────────────────────────────────────────────────

def predict_transfer(vccs_dept="", vccs_number="", vccs_title="", vccs_desc="",
                     vccs_level=0, institutions=None, top_k=TOP_K_DISPLAY):
    """
    Predict transfer equivalents for a course across one or more institutions.

    Args:
        vccs_dept:    Department code, e.g. "ACC" (optional — soft signal)
        vccs_number:  Course number, e.g. "211" (optional — soft signal)
        vccs_title:   Course title (required)
        vccs_desc:    Course description (optional)
        institutions: list of institution keys; defaults to all loaded
        top_k:        Number of results per institution

    Returns:
        dict keyed by institution key:
        {
          "wm": [{"code": ..., "title": ..., "confidence": ...,
                  "confidence_label": ..., "signals": {...}}, ...],
          ...
        }
    """
    a            = load_artifacts()
    clf          = a["classifier"]
    tfidf        = a["tfidf"]
    feature_names = a["feature_names"]
    bge_model    = a["bge_model"]

    if institutions is None:
        institutions = list(a["institutions"].keys())

    vccs_dept    = vccs_dept.strip().upper() if vccs_dept else ""
    vccs_number  = str(vccs_number).strip() if vccs_number else ""
    vccs_title_c = vccs_title.strip()
    vccs_text    = f"{vccs_title_c} {clean_text(vccs_desc)}".strip()

    vccs_emb = bge_model.encode(
        [f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True
    )[0]

    results = {}
    for inst_key in institutions:
        if inst_key not in a["institutions"]:
            continue
        inst = a["institutions"][inst_key]

        candidates = retrieve_candidates(
            vccs_text, vccs_emb, inst, tfidf, RETRIEVAL_K
        )

        feat_vecs      = []
        cand_info_list = []
        for cand_code, rrf_score in candidates:
            sigs = extract_signals(
                vccs_emb, vccs_text, vccs_dept, vccs_number, vccs_title_c,
                cand_code, rrf_score, inst, inst_key, tfidf,
                vccs_level=vccs_level,
            )
            feat_vec = [sigs.get(fn, 0.0) for fn in feature_names]
            feat_vecs.append(feat_vec)
            cand_info_list.append((cand_code, rrf_score, sigs))

        if not feat_vecs:
            results[inst_key] = []
            continue

        X = np.array(feat_vecs, dtype=np.float64)

        # Score with whatever classifier was pickled (XGBoost or sklearn)
        if hasattr(clf, "get_booster"):
            # XGBoost: use booster directly — predict_proba() segfaults after PyTorch loads
            dmat    = xgb.DMatrix(X)
            margins = clf.get_booster().predict(dmat, output_margin=True)
        elif hasattr(clf, "decision_function"):
            # LogisticRegression / SVM: decision_function gives log-odds margins
            margins = clf.decision_function(X)
        else:
            # Generic sklearn: fall back to predict_proba, convert to log-odds
            proba   = clf.predict_proba(X)[:, 1]
            proba   = np.clip(proba, 1e-7, 1 - 1e-7)
            margins = np.log(proba / (1 - proba))

        # Bias-corrected sigmoid confidence.
        # XGBoost was trained with scale_pos_weight≈49 (≈2% positive rate), which
        # shifts the effective decision boundary from margin=0 to margin=log(49)≈3.89.
        # Subtracting MARGIN_BIAS_CORRECTION re-centers it at 0.5, producing
        # probabilities that are:
        #   (a) independent of pool size (no softmax dilution when dept is added)
        #   (b) monotone: adding correct signals only raises the winner's score
        #   (c) robust to OOD inputs (no iso_cal overfitting to training distribution)
        conf = 1.0 / (1.0 + np.exp(-(margins - MARGIN_BIAS_CORRECTION)))

        inst_results = []
        for i, (cand_code, rrf_score, sigs) in enumerate(cand_info_list):
            c    = float(conf[i])
            info = inst["lookup"].get(cand_code, {})
            if c >= HIGH_CONFIDENCE_THRESHOLD:
                conf_label = "High Confidence"
            elif c >= TRANSFER_THRESHOLD:
                conf_label = "Possible Match"
            else:
                conf_label = "Low Confidence"

            inst_results.append({
                "code":             cand_code,
                "title":            info.get("title", ""),
                "confidence":       c,
                "probability":      c,
                "confidence_label": conf_label,
                "signals":          sigs,
            })

        # Sort by confidence (monotone in margin, same order as softmax)
        inst_results.sort(key=lambda x: -x["confidence"])
        results[inst_key] = inst_results[:top_k]

    return results


def evaluate_transcript(courses, institutions=None,
                        min_credits_required=MIN_CREDITS_REQUIRED,
                        credit_hours_per_course=DEFAULT_CREDITS_PER_COURSE):
    a = load_artifacts()
    if institutions is None:
        institutions = list(a["institutions"].keys())

    inst_totals = {k: {"total_credits": 0, "transferable": 0, "high_conf": 0, "results": []}
                   for k in institutions}

    for c in courses:
        preds = predict_transfer(
            c["dept"], c["number"], c["title"],
            c.get("description", ""),
            institutions=institutions,
            top_k=1,
        )
        cr = c.get("credits", credit_hours_per_course)

        for inst_key in institutions:
            best = (preds.get(inst_key) or [{}])[0]
            conf  = best.get("confidence", 0.0)
            label = best.get("confidence_label", "Low Confidence")
            inst_totals[inst_key]["total_credits"] += cr
            if conf >= TRANSFER_THRESHOLD:
                inst_totals[inst_key]["transferable"] += cr
            if label == "High Confidence":
                inst_totals[inst_key]["high_conf"] += cr
            inst_totals[inst_key]["results"].append({
                "dept":             c["dept"],
                "number":           c["number"],
                "title":            c["title"],
                "credits":          cr,
                "best_match":       best.get("code"),
                "best_title":       best.get("title"),
                "confidence":       conf,
                "confidence_label": label,
                "probability":      conf,
            })

    output = {}
    for inst_key in institutions:
        d          = inst_totals[inst_key]
        high_conf  = d["high_conf"]
        transferable = d["transferable"]
        meets      = high_conf >= min_credits_required
        borderline = not meets and transferable >= min_credits_required

        if meets:
            summary = (f"LIKELY ELIGIBLE: {high_conf} credits transfer with high confidence, "
                       f"exceeding the {min_credits_required}-credit minimum.")
        elif borderline:
            summary = (f"BORDERLINE: {high_conf} high-confidence credits "
                       f"(need {min_credits_required}), but {transferable} possible. "
                       f"Advisor review recommended.")
        else:
            summary = (f"UNLIKELY ELIGIBLE: Only {transferable} credits show transfer potential "
                       f"({min_credits_required} required). Registrar confirmation required.")

        output[inst_key] = {
            "institution_name": a["institutions"][inst_key]["name"],
            "total_courses":    len(courses),
            "total_credits":    d["total_credits"],
            "transferable_credits_confident": high_conf,
            "transferable_credits_possible":  transferable,
            "min_required": min_credits_required,
            "eligible":     meets,
            "borderline":   borderline,
            "summary":      summary,
            "course_results": d["results"],
        }
    return output


# ── demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DEMO — TransferZAI multi-institution")
    print("=" * 60)
    demo_cases = [
        ("ACC", "211", "PRINCIPLES OF ACCOUNTING I",
         "Introduces accounting principles with respect to financial reporting."),
        ("CSC", "221", "INTRODUCTION TO PROBLEM SOLVING AND PROGRAMMING",
         "Introduces a modern programming language for problem solving."),
        ("",    "",    "Calculus I",
         "Limits, derivatives, integrals of single-variable functions."),
    ]
    for dept, num, title, desc in demo_cases:
        print(f"\n{'─'*60}")
        print(f"QUERY: {dept} {num} {title}".strip())
        for inst_key, matches in predict_transfer(dept, num, title, desc, top_k=3).items():
            print(f"\n  → {inst_key.upper()}:")
            for i, r in enumerate(matches):
                print(f"    #{i+1}  {r['code']:<12} {r['title']:<40} "
                      f"conf={r['confidence']:.3f} [{r['confidence_label']}]")
