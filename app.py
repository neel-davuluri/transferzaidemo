"""
TransferzAI
Three tabs: Single Course Lookup, Transcript Evaluator, Model Card
"""

import streamlit as st
import pickle
import numpy as np
from predict import (
    predict_transfer, evaluate_transcript, load_artifacts,
    parse_vccs_course,
)
from config import (
    HIGH_CONFIDENCE_THRESHOLD, TRANSFER_THRESHOLD,
    MIN_CREDITS_REQUIRED, DEFAULT_CREDITS_PER_COURSE, ARTIFACTS_DIR,
)

st.set_page_config(page_title="TransferzAI", page_icon="🎓", layout="wide")

# ── Load artifacts once ─────────────────────────────────────────────────
@st.cache_resource
def init():
    return load_artifacts()

with st.spinner("Loading model artifacts..."):
    artifacts = init()

st.title("TransferzAI")
st.caption("VCCS to William & Mary transfer credit evaluation")

tab1, tab2, tab3 = st.tabs(["Single Course Lookup", "Transcript Evaluator", "Model Card"])

# ════════════════════════════════════════════════════════════════════════
# TAB 1: Single Course Lookup
# ════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Single Course Lookup")
    st.write("Enter a VCCS course to find potential W&M equivalents.")

    col1, col2 = st.columns([1, 2])
    with col1:
        vccs_course = st.text_input(
            "VCCS Course",
            value="ACC 211 PRINCIPLES OF ACCOUNTING I",
            help="Format: DEPT NUM TITLE (e.g. ACC 211 PRINCIPLES OF ACCOUNTING I)")
    with col2:
        vccs_desc = st.text_area(
            "Course Description (optional)",
            value="Introduces accounting principles with respect to financial reporting.",
            height=80)

    top_k = st.slider("Results to show", 1, 10, 5)

    if st.button("Find Matches", type="primary", key="single"):
        if not vccs_course.strip():
            st.warning("Please enter a VCCS course.")
        else:
            with st.spinner("Searching..."):
                results = predict_transfer(vccs_course, vccs_desc, top_k=top_k)

            if not results:
                st.info("No matches found.")
            else:
                for i, r in enumerate(results):
                    prob = r["probability"]
                    if prob >= HIGH_CONFIDENCE_THRESHOLD:
                        color = "green"
                        label = "High Confidence"
                    elif prob >= TRANSFER_THRESHOLD:
                        color = "orange"
                        label = "Possible Match"
                    else:
                        color = "red"
                        label = "Low Confidence"

                    with st.container():
                        c1, c2, c3 = st.columns([2, 4, 2])
                        with c1:
                            st.markdown(f"**#{i+1}** `{r['wm_code']}`")
                        with c2:
                            st.write(r["wm_title"])
                        with c3:
                            st.markdown(f":{color}[**{prob:.1%}**] — {label}")

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
    st.write("Enter multiple courses to evaluate transfer eligibility.")

    min_credits = st.number_input("Minimum credits required",
                                   value=MIN_CREDITS_REQUIRED, min_value=1, max_value=120)

    st.subheader("Courses")
    st.caption("Add courses one per row. Format: DEPT NUM TITLE")

    default_courses = [
        "ACC 211 PRINCIPLES OF ACCOUNTING I",
        "ENG 111 COLLEGE COMPOSITION I",
        "MTH 263 CALCULUS I",
        "BIO 101 GENERAL BIOLOGY I",
        "CSC 221 INTRODUCTION TO PROBLEM SOLVING AND PROGRAMMING",
    ]

    num_courses = st.number_input("Number of courses", 1, 30, len(default_courses))

    course_inputs = []
    for i in range(num_courses):
        cols = st.columns([3, 3, 1])
        with cols[0]:
            default_val = default_courses[i] if i < len(default_courses) else ""
            course = st.text_input(f"Course {i+1}", value=default_val, key=f"tc_{i}")
        with cols[1]:
            desc = st.text_input(f"Description {i+1} (optional)", value="", key=f"td_{i}")
        with cols[2]:
            credits = st.number_input(f"Credits", value=DEFAULT_CREDITS_PER_COURSE,
                                       min_value=1, max_value=6, key=f"tcr_{i}")
        if course.strip():
            course_inputs.append({"course": course, "description": desc, "credits": credits})

    if st.button("Evaluate Transcript", type="primary", key="transcript"):
        if not course_inputs:
            st.warning("Please enter at least one course.")
        else:
            with st.spinner(f"Evaluating {len(course_inputs)} courses..."):
                result = evaluate_transcript(course_inputs, min_credits_required=min_credits)

            # Summary
            if result["eligible"]:
                st.success(result["summary"])
            elif result["borderline"]:
                st.warning(result["summary"])
            else:
                st.error(result["summary"])

            # Stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Courses", result["total_courses"])
            col2.metric("Total Credits", result["total_credits"])
            col3.metric("High Confidence Credits", result["transferable_credits_confident"])
            col4.metric("Possible Credits", result["transferable_credits_possible"])

            # Per-course table
            st.subheader("Per-Course Results")
            for cr in result["course_results"]:
                prob = cr["probability"]
                if prob >= HIGH_CONFIDENCE_THRESHOLD:
                    icon = "🟢"
                elif prob >= TRANSFER_THRESHOLD:
                    icon = "🟡"
                else:
                    icon = "🔴"

                parsed = parse_vccs_course(cr["course"])
                short_name = parsed[0]["full"] if parsed else cr["course"][:10]

                match_str = f"`{cr['best_match']}` — {cr['best_title']}" if cr["best_match"] else "No match"
                st.markdown(
                    f"{icon} **{short_name}** ({cr['credits']} cr) → {match_str} — **{prob:.1%}**"
                )


# ════════════════════════════════════════════════════════════════════════
# TAB 3: Model Card
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Model Card")

    scorecard_data = artifacts["scorecard"]
    sc = scorecard_data["scorecard"]

    st.subheader("Architecture")
    st.markdown("""
    **Pipeline:** 3-signal RRF retrieval (top-50) → 13-feature LogisticRegression → calibrated probability

    **Retrieval signals:**
    1. Fine-tuned BGE-small-en-v1.5 cosine similarity
    2. TF-IDF (unigrams + bigrams) cosine similarity
    3. Department prior from training data (0.5x weight)

    **Scoring features (13):**
    - Base: BGE sim, TF-IDF sim, TF-IDF title sim, dept prior, title similarity, number ratio, same level, RRF score
    - Interactions: BGE x dept, BGE x title, BGE x TF-IDF, dept x title, dept x number

    **Training:** Fine-tuned BGE on 253 positive pairs (3 epochs, MNRL loss).
    LogReg trained on ~13,350 candidates from 267 train queries at true 1:50 ratio.
    """)

    st.subheader("Held-Out Test Performance (n=67)")

    perf_data = []
    for name, (val, tgt, passed) in sc.items():
        perf_data.append({
            "Metric": name,
            "Value": f"{val:.3f}",
            "Target": tgt,
            "Status": "PASS" if passed else "FAIL",
        })
    st.table(perf_data)

    st.subheader("Version Comparison")
    v1 = scorecard_data["v1"]
    v2 = scorecard_data["v2"]
    v3 = scorecard_data["v3"]

    comparison = []
    for metric in v1:
        comparison.append({
            "Metric": metric,
            "v1": f"{v1[metric]:.3f}",
            "v2 (full data)": f"{v2[metric]:.3f}",
            "v3 (held-out)": f"{v3[metric]:.3f}",
        })
    st.table(comparison)

    st.subheader("End-to-End Recall")
    r = scorecard_data
    recall_cols = st.columns(5)
    recall_cols[0].metric("Top-1", f"{r['top1_recall']:.3f}")
    recall_cols[1].metric("Top-3", f"{r['top3_recall']:.3f}")
    recall_cols[2].metric("Top-5", f"{r['top5_recall']:.3f}")
    recall_cols[3].metric("Top-10", f"{r['top10_recall']:.3f}")
    recall_cols[4].metric("Not Retrieved", f"{r['not_retrieved']}/{r['total_test']}")

    st.subheader("Data")
    st.markdown(f"""
    - **Training:** 267 positive pairs (VCCS → W&M), 1494 negatives (synthetic + retriever-mined + no-transfer)
    - **Test:** 67 positive pairs (held-out, stratified by department)
    - **W&M catalog:** {len(artifacts['wm_codes'])} courses
    - **Calibration:** ~13,350 candidates at true 1:50 inference ratio
    """)

    st.subheader("Limitations")
    st.markdown("""
    - Training data covers only VCCS transfers; other Virginia community colleges may differ
    - Courses with no description rely solely on title/code matching
    - Department prior is sparse for rare VCCS departments (<2 training examples)
    - Probabilities reflect model confidence, not registrar decisions — always verify
    """)
