"""
TransferzAI
Three tabs: Single Course Lookup, Transcript Evaluator, Model Card
"""

import streamlit as st
import pickle
import numpy as np
from predict import (
    predict_transfer, evaluate_transcript, load_artifacts,
)
from config import (
    HIGH_CONFIDENCE_THRESHOLD, TRANSFER_THRESHOLD,
    MIN_CREDITS_REQUIRED, DEFAULT_CREDITS_PER_COURSE, ARTIFACTS_DIR,
)

st.set_page_config(page_title="TransferzAI", page_icon="🎓", layout="wide")

# ── Load artifacts once ─────────────────────────────────────────────────
@st.cache_resource
def init():
    try:
        return load_artifacts()
    except FileNotFoundError as e:
        st.error(f"Model artifacts not found: {e}")
        st.info("""
        **To fix this:**
        1. Download `artifacts.tar.gz` from: https://github.com/neel-davuluri/transferzaidemo/releases
        2. Extract: `tar -xzf artifacts.tar.gz`
        3. Refresh this page
        """)
        st.stop()

with st.spinner("Loading model artifacts..."):
    artifacts = init()

# Available institutions (only those successfully loaded)
loaded_institutions = {
    k: v["name"] for k, v in artifacts["institutions"].items()
}

st.title("TransferzAI")
st.caption("Course transfer credit evaluation — compare across institutions")

tab1, tab2, tab3 = st.tabs(["Single Course Lookup", "Transcript Evaluator", "Model Card"])


# ════════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════════

def confidence_label(prob):
    if prob >= HIGH_CONFIDENCE_THRESHOLD:
        return "green", "High Confidence"
    elif prob >= TRANSFER_THRESHOLD:
        return "orange", "Possible Match"
    else:
        return "red", "Low Confidence"

def confidence_icon(prob):
    if prob >= HIGH_CONFIDENCE_THRESHOLD:
        return "🟢"
    elif prob >= TRANSFER_THRESHOLD:
        return "🟡"
    return "🔴"


# ════════════════════════════════════════════════════════════════════════
# TAB 1: Single Course Lookup
# ════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Single Course Lookup")
    st.write("Enter a course to find potential transfer equivalents at each institution.")

    # ── Course input ─────────────────────────────────────────────────────
    col_dept, col_num, col_title = st.columns([1, 1, 3])
    with col_dept:
        vccs_dept = st.text_input("Department", value="ACC",
                                   help="e.g. ACC, CSC, MTH, BIO")
    with col_num:
        vccs_number = st.text_input("Course Number", value="211",
                                     help="e.g. 211, 101, 263")
    with col_title:
        vccs_title = st.text_input("Course Title", value="PRINCIPLES OF ACCOUNTING I",
                                    help="e.g. PRINCIPLES OF ACCOUNTING I")

    vccs_desc = st.text_area("Course Description (optional)",
                              value="Introduces accounting principles with respect to financial reporting.",
                              height=80)

    # ── Institution selector ──────────────────────────────────────────────
    st.markdown("**Target institutions**")
    selected_institutions = st.multiselect(
        "Show results for",
        options=list(loaded_institutions.keys()),
        default=list(loaded_institutions.keys()),
        format_func=lambda k: loaded_institutions[k],
        label_visibility="collapsed",
    )

    top_k = st.slider("Results per institution", 1, 10, 5)

    if st.button("Find Matches", type="primary", key="single"):
        if not vccs_dept.strip() or not vccs_title.strip():
            st.warning("Please enter at least a department and course title.")
        elif not selected_institutions:
            st.warning("Please select at least one institution.")
        else:
            with st.spinner("Searching..."):
                results = predict_transfer(
                    vccs_dept, vccs_number, vccs_title, vccs_desc,
                    institutions=selected_institutions, top_k=top_k,
                )

            for inst_key in selected_institutions:
                inst_name = loaded_institutions.get(inst_key, inst_key)
                st.subheader(inst_name)
                inst_results = results.get(inst_key, [])

                if not inst_results:
                    st.info("No matches found.")
                    continue

                for i, r in enumerate(inst_results):
                    color, label = confidence_label(r["probability"])
                    with st.container():
                        c1, c2, c3 = st.columns([2, 4, 2])
                        with c1:
                            st.markdown(f"**#{i+1}** `{r['code']}`")
                        with c2:
                            st.write(r["title"])
                        with c3:
                            st.markdown(f":{color}[**{r['probability']:.1%}**] — {label}")

                        with st.expander("Signal details"):
                            sigs = r["signals"]
                            sig_cols = st.columns(4)
                            with sig_cols[0]:
                                st.metric("BGE Similarity", f"{sigs['bge_sim']:.3f}")
                            with sig_cols[1]:
                                st.metric("TF-IDF Similarity", f"{sigs['tfidf_sim']:.3f}")
                            with sig_cols[2]:
                                st.metric("Dept Prior", f"{sigs['dept_prob']:.3f}")
                            with sig_cols[3]:
                                st.metric("Title Similarity", f"{sigs['title_sim']:.3f}")

                        st.divider()


# ════════════════════════════════════════════════════════════════════════
# TAB 2: Transcript Evaluator
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Transcript Evaluator")
    st.write("Enter multiple courses to evaluate transfer eligibility across institutions.")

    # ── Institution selector ──────────────────────────────────────────────
    st.markdown("**Target institutions**")
    selected_institutions_t2 = st.multiselect(
        "Show results for",
        options=list(loaded_institutions.keys()),
        default=list(loaded_institutions.keys()),
        format_func=lambda k: loaded_institutions[k],
        label_visibility="collapsed",
        key="t2_institutions",
    )

    min_credits = st.number_input("Minimum credits required",
                                   value=MIN_CREDITS_REQUIRED, min_value=1, max_value=120)

    st.subheader("Courses")
    st.caption("Fill in department, number, and title for each course.")

    default_courses = [
        ("ACC", "211", "PRINCIPLES OF ACCOUNTING I"),
        ("ENG", "111", "COLLEGE COMPOSITION I"),
        ("MTH", "263", "CALCULUS I"),
        ("BIO", "101", "GENERAL BIOLOGY I"),
        ("CSC", "221", "INTRODUCTION TO PROBLEM SOLVING AND PROGRAMMING"),
    ]

    num_courses = st.number_input("Number of courses", 1, 30, len(default_courses))

    course_inputs = []
    for i in range(num_courses):
        cols = st.columns([1, 1, 3, 3, 1])
        defaults = default_courses[i] if i < len(default_courses) else ("", "", "")
        with cols[0]:
            dept = st.text_input("Dept", value=defaults[0], key=f"td_dept_{i}",
                                  label_visibility="collapsed" if i > 0 else "visible")
        with cols[1]:
            num = st.text_input("Num", value=defaults[1], key=f"td_num_{i}",
                                 label_visibility="collapsed" if i > 0 else "visible")
        with cols[2]:
            title = st.text_input("Title", value=defaults[2], key=f"td_title_{i}",
                                   label_visibility="collapsed" if i > 0 else "visible")
        with cols[3]:
            desc = st.text_input("Description (optional)", value="", key=f"td_desc_{i}",
                                  label_visibility="collapsed" if i > 0 else "visible")
        with cols[4]:
            credits = st.number_input("Cr", value=DEFAULT_CREDITS_PER_COURSE,
                                       min_value=1, max_value=6, key=f"td_cr_{i}",
                                       label_visibility="collapsed" if i > 0 else "visible")
        if dept.strip() and title.strip():
            course_inputs.append({
                "dept": dept, "number": num, "title": title,
                "description": desc, "credits": credits,
            })

    if st.button("Evaluate Transcript", type="primary", key="transcript"):
        if not course_inputs:
            st.warning("Please enter at least one course.")
        elif not selected_institutions_t2:
            st.warning("Please select at least one institution.")
        else:
            with st.spinner(f"Evaluating {len(course_inputs)} courses..."):
                results = evaluate_transcript(
                    course_inputs,
                    institutions=selected_institutions_t2,
                    min_credits_required=min_credits,
                )

            for inst_key in selected_institutions_t2:
                r = results.get(inst_key)
                if not r:
                    continue

                st.subheader(r["institution_name"])

                if r["eligible"]:
                    st.success(r["summary"])
                elif r["borderline"]:
                    st.warning(r["summary"])
                else:
                    st.error(r["summary"])

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Courses", r["total_courses"])
                col2.metric("Total Credits", r["total_credits"])
                col3.metric("High Confidence Credits", r["transferable_credits_confident"])
                col4.metric("Possible Credits", r["transferable_credits_possible"])

                st.markdown("**Per-Course Results**")
                for cr in r["course_results"]:
                    icon = confidence_icon(cr["probability"])
                    course_label = f"{cr['dept']} {cr['number']} {cr['title']}"
                    match_str = f"`{cr['best_match']}` — {cr['best_title']}" if cr["best_match"] else "No match"
                    st.markdown(
                        f"{icon} **{course_label}** ({cr['credits']} cr) → {match_str} — **{cr['probability']:.1%}**"
                    )

                st.divider()


# ════════════════════════════════════════════════════════════════════════
# TAB 3: Model Card
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Model Card")
    st.caption("TransferzAI — VCCS transfer credit evaluation across multiple institutions")

    # ── Overview ──────────────────────────────────────────────────────────
    st.subheader("What This System Does")
    st.markdown("""
    TransferzAI predicts whether a course from Virginia Community College System (VCCS)
    will transfer for credit at a target institution. Given a course's department, number,
    title, and optional description, it returns a calibrated probability for each candidate
    equivalent at each selected institution.

    **Key design constraint:** False positives are costly — a coach who pursues a transfer
    recruit based on a wrong "yes" loses weeks and a recruiting slot. The system is tuned
    for precision on positive predictions, not recall.
    """)

    # ── Architecture ──────────────────────────────────────────────────────
    st.subheader("Architecture")
    st.markdown("""
    **Two-stage retrieve-then-classify pipeline, run independently per institution:**

    **Stage 1 — Retrieval (3-signal RRF, top-50 candidates)**
    1. Fine-tuned BGE-small-en-v1.5 bi-encoder cosine similarity
    2. TF-IDF (unigrams + bigrams, 10K vocab) cosine similarity
    3. Department prior — P(target dept | VCCS dept), learned from W&M training data (0.5× weight; zero for non-W&M institutions)

    **Stage 2 — Classification (13 features → LogisticRegression → calibrated probability)**

    | Feature group | Features |
    |:---|:---|
    | Semantic | BGE cosine similarity |
    | Lexical | TF-IDF sim (full), TF-IDF sim (title only), SequenceMatcher title ratio |
    | Structural | Course number ratio, same course level (100/200/300), dept prior score, RRF rank score |
    | Interactions | BGE×dept, BGE×title, BGE×TF-IDF, dept×title, dept×number |

    **Calibration:** LogisticRegression trained at the true inference ratio (1 positive : 50 negatives),
    so output probabilities are honest estimates of P(transfers), not just rankings.

    **Multi-institution generalization:** The model is trained exclusively on VCCS→W&M pairs.
    At inference time, the same retriever and classifier are applied to any institution's course
    catalog — no retraining required. Confidence will be lower for non-W&M institutions because
    the department prior feature is unavailable.
    """)

    # ── Performance ───────────────────────────────────────────────────────
    st.subheader("Held-Out Test Performance")
    st.caption("Evaluated on 67 VCCS→W&M pairs held out from training (stratified by W&M department)")

    scorecard_data = artifacts["scorecard"]
    sc = scorecard_data["scorecard"]

    perf_data = []
    for name, (val, tgt, passed) in sc.items():
        perf_data.append({
            "Metric": name,
            "Value": f"{val:.3f}",
            "Target": tgt,
            "Status": "✓ PASS" if passed else "✗ FAIL",
        })
    st.table(perf_data)

    st.markdown("""
    **What these metrics mean:**
    - **Precision@0.5 / @0.7** — Of courses the model predicts will transfer, how often is it right. The primary metric.
    - **Top-3 Recall** — For a course that does transfer, is the correct W&M equivalent in the top 3 results?
    - **False Confidence Rate** — How often the model assigns P > 0.5 to a pair that does not transfer.
    - **ECE / Brier Score** — Calibration quality. ECE < 0.05 means stated probabilities are accurate within 5%.
    """)

    # ── Recall breakdown ──────────────────────────────────────────────────
    st.subheader("End-to-End Retrieval Recall")
    st.caption("How often the correct W&M equivalent appears in the top-K retrieved candidates")
    r = scorecard_data
    recall_cols = st.columns(5)
    recall_cols[0].metric("Top-1", f"{r['top1_recall']:.3f}")
    recall_cols[1].metric("Top-3", f"{r['top3_recall']:.3f}")
    recall_cols[2].metric("Top-5", f"{r['top5_recall']:.3f}")
    recall_cols[3].metric("Top-10", f"{r['top10_recall']:.3f}")
    recall_cols[4].metric("Not Retrieved", f"{r['not_retrieved']}/{r['total_test']}")

    # ── Iteration history ─────────────────────────────────────────────────
    st.subheader("Development Iterations")
    st.markdown("""
    The final system is the result of three major iterations — two of which failed and revealed
    the core insight: **semantic similarity ≠ transfer eligibility.**
    """)

    v1 = scorecard_data["v1"]
    v2 = scorecard_data["v2"]
    v3 = scorecard_data["v3"]

    comparison = []
    for metric in v1:
        comparison.append({
            "Metric": metric,
            "v1 — Off-the-shelf BGE": f"{v1[metric]:.3f}",
            "v2 — Cross-encoder reranker": f"{v2[metric]:.3f}",
            "v3 — Fine-tuned BGE (current)": f"{v3[metric]:.3f}",
        })
    st.table(comparison)

    st.markdown("""
    - **v1:** Baseline with off-the-shelf BGE + TF-IDF retrieval. Good calibration, weak retrieval ceiling.
    - **v2:** Added cross-encoder reranker (BAAI/bge-reranker-v2-m3). Top-3 recall collapsed from 0.455 → 0.281.
      The cross-encoder promoted semantically similar but non-transferring courses. More semantic power made things worse.
    - **v3 (current):** Fine-tuned BGE with contrastive learning on transfer pairs teaches "transfers as" rather than
      "is similar to." Cross-encoder removed. Department prior added as third retrieval signal.
    """)

    # ── Training data ─────────────────────────────────────────────────────
    st.subheader("Training Data")
    st.markdown(f"""
    | Split | Positives | Negatives | Total |
    |:---|:---:|:---:|:---:|
    | Train | 267 | 1,494 | 1,761 |
    | Test (held-out) | 67 | — | 67 |

    **Negative breakdown (training):**
    - 267 synthetic hard negatives — generated via LLM: same department, semantically similar, but non-transferable
    - 801 retriever-mined hard negatives — courses ranked highly by the retriever but confirmed non-transfers
    - 426 no-transfer negatives — VCCS courses with no W&M equivalent in the articulation agreement

    **Source:** Official VCCS→W&M articulation agreement (760 course entries, 334 verified positive pairs).
    No student PII — FERPA compliant.

    **Loaded institution catalogs:**
    {chr(10).join(f"    - **{v}** — {len(artifacts['institutions'][k]['codes']):,} courses" for k, v in loaded_institutions.items())}
    """)

    # ── Limitations ───────────────────────────────────────────────────────
    st.subheader("Limitations")
    st.markdown("""
    - **Training scope:** Trained on VCCS→W&M only. Predictions for other institutions generalize
      from W&M patterns — confidence scores will be lower and less calibrated for non-W&M targets.
    - **Department prior:** Only available for W&M. Non-W&M retrieval relies on BGE + TF-IDF alone.
    - **Missing descriptions:** Courses without catalog descriptions score zero on TF-IDF features,
      degrading prediction quality.
    - **No registrar feedback loop:** Probabilities are model estimates, not registrar decisions.
      Always verify high-stakes decisions with the registrar.
    - **Single source institution:** Currently supports VCCS as the source only.
      Other community college systems may have different course coding conventions.
    """)
