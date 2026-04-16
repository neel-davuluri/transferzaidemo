"""
TransferzAI — AI-powered transfer credit evaluation
"""

import streamlit as st
from predict import predict_transfer, evaluate_transcript, load_artifacts
from config import (
    HIGH_CONFIDENCE_THRESHOLD, TRANSFER_THRESHOLD,
    MIN_CREDITS_REQUIRED, DEFAULT_CREDITS_PER_COURSE,
)

st.set_page_config(
    page_title="TransferzAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ---- global ---- */
  [data-testid="stAppViewContainer"] { background: #f8f9fb; }
  [data-testid="stHeader"] { background: transparent; }
  .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1100px; }

  /* ---- hero ---- */
  .hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
  .hero h1 { font-size: 2.6rem; font-weight: 800; letter-spacing: -0.5px;
              color: #1a1a2e; margin-bottom: 0.3rem; }
  .hero p  { font-size: 1.1rem; color: #555; margin: 0; }

  /* ---- result card ---- */
  .result-card {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e8eaf0;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    position: relative;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .result-card.high   { border-left: 4px solid #22c55e; }
  .result-card.mid    { border-left: 4px solid #f59e0b; }
  .result-card.low    { border-left: 4px solid #ef4444; }
  .card-rank   { font-size: 0.78rem; color: #888; font-weight: 600;
                 text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.2rem; }
  .card-code   { font-size: 1.15rem; font-weight: 700; color: #1a1a2e; display:inline; }
  .card-title  { font-size: 0.95rem; color: #444; margin-top: 0.15rem; }
  .card-verdict { font-size: 0.85rem; margin-top: 0.35rem; }

  /* ---- badges ---- */
  .badge {
    display: inline-block; border-radius: 9999px;
    padding: 0.2em 0.75em; font-size: 0.78rem;
    font-weight: 700; letter-spacing: 0.02em;
    vertical-align: middle; margin-left: 0.5rem;
  }
  .badge-green  { background: #dcfce7; color: #15803d; }
  .badge-yellow { background: #fef9c3; color: #a16207; }
  .badge-red    { background: #fee2e2; color: #b91c1c; }

  /* ---- institution header ---- */
  .inst-header {
    font-size: 1.1rem; font-weight: 700; color: #1a1a2e;
    border-bottom: 2px solid #e8eaf0; padding-bottom: 0.4rem;
    margin: 1.5rem 0 0.75rem;
  }

  /* ---- summary bar ---- */
  .summary-eligible   { background:#dcfce7; border-radius:10px; padding:1rem 1.25rem;
                         border-left:4px solid #22c55e; margin-bottom:1rem; }
  .summary-borderline { background:#fef9c3; border-radius:10px; padding:1rem 1.25rem;
                         border-left:4px solid #f59e0b; margin-bottom:1rem; }
  .summary-ineligible { background:#fee2e2; border-radius:10px; padding:1rem 1.25rem;
                         border-left:4px solid #ef4444; margin-bottom:1rem; }
  .summary-text { font-size: 1rem; font-weight: 600; color: #1a1a2e; margin: 0; }

  /* ---- empty state ---- */
  .empty-state { text-align:center; padding:3rem 1rem; color:#999; }
  .empty-state .icon { font-size: 2.5rem; }
  .empty-state p { margin-top: 0.5rem; font-size: 0.95rem; }

  /* ---- form tweaks ---- */
  label { font-weight: 600 !important; }
  .stTextInput input, .stTextArea textarea {
    border-radius: 8px !important; border: 1px solid #dde1ea !important;
    background: #fff !important;
  }
  .stButton > button[kind="primary"] {
    border-radius: 8px; font-weight: 700; font-size: 1rem;
    padding: 0.5rem 2rem; background: #2563eb; border: none;
  }
  .stButton > button[kind="primary"]:hover { background: #1d4ed8; }

  /* ---- tab styling ---- */
  [data-testid="stTabs"] [data-baseweb="tab"] {
    font-weight: 600; font-size: 0.95rem;
  }
  [data-testid="stTabs"] [aria-selected="true"] {
    color: #2563eb;
  }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts once ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init():
    try:
        return load_artifacts()
    except FileNotFoundError as e:
        st.error(f"Model artifacts not found: {e}")
        st.info("Run `python scripts/build_artifacts.py` to generate artifacts, then refresh.")
        st.stop()

with st.spinner("Initializing TransferzAI…"):
    artifacts = init()

loaded_institutions = {k: v["name"] for k, v in artifacts["institutions"].items()}


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎓 TransferzAI</h1>
  <p>AI-powered transfer credit evaluation — instantly see how your courses transfer</p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  Course Lookup  ", "  Transcript Evaluator  ", "  Model Card  "])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _badge(conf):
    if conf >= HIGH_CONFIDENCE_THRESHOLD:
        return f'<span class="badge badge-green">High Confidence · {conf:.0%}</span>'
    if conf >= TRANSFER_THRESHOLD:
        return f'<span class="badge badge-yellow">Possible Match · {conf:.0%}</span>'
    return f'<span class="badge badge-red">Low Confidence · {conf:.0%}</span>'

def _card_class(conf):
    if conf >= HIGH_CONFIDENCE_THRESHOLD: return "high"
    if conf >= TRANSFER_THRESHOLD:        return "mid"
    return "low"

def _verdict_text(conf, code):
    if conf >= HIGH_CONFIDENCE_THRESHOLD:
        return f"<span style='color:#15803d;font-weight:600;'>Transfers as <b>{code}</b></span>"
    if conf >= TRANSFER_THRESHOLD:
        return "<span style='color:#a16207;'>Likely transfers — advisor review recommended</span>"
    return "<span style='color:#b91c1c;'>Low confidence — refer to registrar</span>"

def render_result(r, idx):
    conf = r["confidence"]
    card_cls = _card_class(conf)
    html = f"""
    <div class="result-card {card_cls}">
      <div class="card-rank">Match #{idx+1}</div>
      <div>
        <span class="card-code">{r['code']}</span>
        {_badge(conf)}
      </div>
      <div class="card-title">{r['title']}</div>
      <div class="card-verdict">{_verdict_text(conf, r['code'])}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    with st.expander("Signal details", expanded=False):
        sigs = r["signals"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BGE Similarity",    f"{sigs['bge_sim']:.3f}")
        c2.metric("TF-IDF Similarity", f"{sigs['tfidf_sim']:.3f}")
        c3.metric("Dept Prior",        f"{sigs['dept_prob']:.3f}")
        c4.metric("Title Similarity",  f"{sigs['title_sim']:.3f}")


def confidence_icon(conf):
    if conf >= HIGH_CONFIDENCE_THRESHOLD: return "🟢"
    if conf >= TRANSFER_THRESHOLD:        return "🟡"
    return "🔴"


# ════════════════════════════════════════════════════════════════════════
# TAB 1 — Course Lookup
# ════════════════════════════════════════════════════════════════════════
with tab1:
    with st.container():
        left, right = st.columns([5, 3], gap="large")

        with left:
            st.markdown("#### Enter your course")
            vccs_title = st.text_input(
                "Course Title",
                value="",
                placeholder="e.g. Calculus I, General Chemistry, Intro to Psychology",
                help="Required. The name of your course.",
            )
            vccs_desc = st.text_area(
                "Course Description",
                value="",
                placeholder="Paste the catalog description for best results (optional)",
                height=90,
            )
            col_dept, col_num = st.columns(2)
            with col_dept:
                vccs_dept = st.text_input(
                    "Department Code",
                    value="",
                    placeholder="ACC, CSC, MTH… (optional)",
                    help="Improves department-prior signal when provided.",
                )
            with col_num:
                vccs_number = st.text_input(
                    "Course Number",
                    value="",
                    placeholder="101, 211… (optional)",
                    help="Used for academic-level matching.",
                )

        with right:
            st.markdown("#### Target institutions")
            selected = st.multiselect(
                "Institutions",
                options=list(loaded_institutions.keys()),
                default=list(loaded_institutions.keys()),
                format_func=lambda k: loaded_institutions[k],
                label_visibility="collapsed",
            )
            top_k = st.slider("Results per institution", min_value=1, max_value=10, value=5)
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button(
                "Find Transfer Matches", type="primary", key="single", use_container_width=True
            )

    st.markdown("---")

    if search_clicked:
        if not vccs_title.strip():
            st.warning("Please enter a course title to search.")
        elif not selected:
            st.warning("Please select at least one institution.")
        else:
            with st.spinner("Searching for matches…"):
                results = predict_transfer(
                    vccs_dept, vccs_number, vccs_title, vccs_desc,
                    institutions=selected, top_k=top_k,
                )

            any_results = any(results.get(k) for k in selected)
            if not any_results:
                st.info("No matches found for the given query.")
            else:
                cols = st.columns(len(selected)) if len(selected) <= 3 else None
                for col_idx, inst_key in enumerate(selected):
                    inst_name    = loaded_institutions.get(inst_key, inst_key)
                    inst_results = results.get(inst_key, [])
                    if cols:
                        container = cols[col_idx]
                    else:
                        container = st
                    with container:
                        st.markdown(f'<div class="inst-header">{inst_name}</div>',
                                    unsafe_allow_html=True)
                        if not inst_results:
                            st.info("No matches found.")
                        else:
                            for i, r in enumerate(inst_results):
                                render_result(r, i)
    else:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">🔍</div>
          <p>Enter a course above and click <strong>Find Transfer Matches</strong> to see results.</p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — Transcript Evaluator
# ════════════════════════════════════════════════════════════════════════
with tab2:
    config_col, _ = st.columns([4, 3])
    with config_col:
        st.markdown("#### Settings")
        col_inst, col_cred = st.columns([3, 2])
        with col_inst:
            selected_t2 = st.multiselect(
                "Target institutions",
                options=list(loaded_institutions.keys()),
                default=list(loaded_institutions.keys()),
                format_func=lambda k: loaded_institutions[k],
                key="t2_inst",
            )
        with col_cred:
            min_credits = st.number_input(
                "Min credits required",
                value=MIN_CREDITS_REQUIRED, min_value=1, max_value=120,
            )

    st.markdown("#### Your courses")

    default_courses = [
        ("ACC", "211", "PRINCIPLES OF ACCOUNTING I"),
        ("ENG", "111", "COLLEGE COMPOSITION I"),
        ("MTH", "263", "CALCULUS I"),
        ("BIO", "101", "GENERAL BIOLOGY I"),
        ("CSC", "221", "INTRODUCTION TO PROBLEM SOLVING AND PROGRAMMING"),
    ]

    num_courses = st.number_input("Number of courses", 1, 30, len(default_courses), key="n_courses")

    hcols = st.columns([1, 1, 3, 3, 1])
    for col, label in zip(hcols, ["Dept", "Number", "Title", "Description", "Credits"]):
        col.markdown(f"<small style='color:#888;font-weight:600;'>{label}</small>",
                     unsafe_allow_html=True)

    course_inputs = []
    for i in range(int(num_courses)):
        defaults = default_courses[i] if i < len(default_courses) else ("", "", "")
        cols = st.columns([1, 1, 3, 3, 1])
        dept  = cols[0].text_input("d", value=defaults[0], key=f"td_dept_{i}",  label_visibility="collapsed")
        num   = cols[1].text_input("n", value=defaults[1], key=f"td_num_{i}",   label_visibility="collapsed")
        title = cols[2].text_input("t", value=defaults[2], key=f"td_title_{i}", label_visibility="collapsed")
        desc  = cols[3].text_input("x", value="",          key=f"td_desc_{i}",  label_visibility="collapsed")
        creds = cols[4].number_input("c", value=DEFAULT_CREDITS_PER_COURSE,
                                     min_value=1, max_value=6, key=f"td_cr_{i}",
                                     label_visibility="collapsed")
        if title.strip():
            course_inputs.append({
                "dept": dept, "number": num, "title": title,
                "description": desc, "credits": creds,
            })

    eval_clicked = st.button("Evaluate Transcript", type="primary", key="transcript",
                             use_container_width=False)

    st.markdown("---")

    if eval_clicked:
        if not course_inputs:
            st.warning("Please enter at least one course.")
        elif not selected_t2:
            st.warning("Please select at least one institution.")
        else:
            with st.spinner(f"Evaluating {len(course_inputs)} courses across {len(selected_t2)} institution(s)…"):
                results = evaluate_transcript(
                    course_inputs, institutions=selected_t2,
                    min_credits_required=min_credits,
                )

            for inst_key in selected_t2:
                r = results.get(inst_key)
                if not r:
                    continue

                # Summary banner
                if r["eligible"]:
                    banner_cls = "summary-eligible"
                elif r["borderline"]:
                    banner_cls = "summary-borderline"
                else:
                    banner_cls = "summary-ineligible"

                st.markdown(
                    f'<div class="{banner_cls}"><p class="summary-text">'
                    f'{r["institution_name"]} — {r["summary"]}'
                    f'</p></div>',
                    unsafe_allow_html=True,
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Courses",           r["total_courses"])
                m2.metric("Total Credits",           r["total_credits"])
                m3.metric("High-Confidence Credits", r["transferable_credits_confident"])
                m4.metric("Possible Credits",        r["transferable_credits_possible"])

                st.markdown("**Per-course breakdown**")
                for cr in r["course_results"]:
                    icon  = confidence_icon(cr["confidence"])
                    label = " ".join(filter(None, [cr["dept"], cr["number"], cr["title"]])).strip()
                    if cr["best_match"]:
                        match_str = f"`{cr['best_match']}` — {cr['best_title']}"
                    else:
                        match_str = "_No match found_"
                    st.markdown(
                        f"{icon} &nbsp; **{label}** &ensp;·&ensp; {cr['credits']} cr "
                        f"&ensp;→&ensp; {match_str} &ensp; **{cr['confidence']:.0%}**"
                    )
                st.markdown("---")
    else:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">📋</div>
          <p>Fill in your courses above and click <strong>Evaluate Transcript</strong>.</p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Card
# ════════════════════════════════════════════════════════════════════════
with tab3:
    c_left, c_right = st.columns([3, 2], gap="large")

    with c_left:
        st.markdown("## How it works")
        st.markdown("""
TransferzAI uses a **two-stage retrieve-then-rank pipeline** to predict whether a course
will transfer for credit at a target institution.

**Stage 1 — Retrieval (top-50 candidates)**

Three signals are fused with Reciprocal Rank Fusion (RRF):

| Signal | What it captures |
|:---|:---|
| Fine-tuned BGE bi-encoder | Semantic similarity (course meaning) |
| TF-IDF (1-2 gram, 15k features) | Keyword and title overlap |
| Department prior | Historical P(target dept given source dept) |

**Stage 2 — XGBoost Reranker**

13 features across four groups are fed to an XGBoost classifier:

| Group | Features |
|:---|:---|
| Semantic | BGE cosine similarity |
| Lexical | TF-IDF (full text), TF-IDF (title), SequenceMatcher ratio |
| Structural | Level ratio, same-level flag, dept prior, RRF score |
| Interactions | BGE×dept, BGE×title, BGE×TF-IDF, dept×title, dept×level |

**Confidence scoring**

Per-query softmax over XGBoost margins answers *"which of these 50 candidates best fits
this specific query?"*, producing a 0–1 score that is meaningful across queries.
Isotonic regression calibrates the displayed probabilities.
        """)

        st.markdown("## Confidence thresholds")
        st.markdown(f"""
| Confidence | Label | Action |
|:---|:---|:---|
| ≥ {HIGH_CONFIDENCE_THRESHOLD:.0%} | 🟢 High Confidence | Transfers as X — show to student |
| {TRANSFER_THRESHOLD:.0%}–{HIGH_CONFIDENCE_THRESHOLD:.0%} | 🟡 Possible Match | Likely transfers — advisor review |
| < {TRANSFER_THRESHOLD:.0%} | 🔴 Low Confidence | Show top-3 only, refer to registrar |

High-Confidence predictions achieve **≥ 90% precision** on held-out data.
        """)

        st.markdown("## Training data")
        st.markdown("""
| Institution pair | Train pairs | Test pairs |
|:---|:---:|:---:|
| VCCS → W&M | ~267 | 67 |
| VCCS → VT | ~242 | 60 |
| CCC → UCSC | ~723 | 181 |

Sources: VCCS→W&M and VCCS→VT articulation agreements; CCC→UCSC articulation data.
No student PII — FERPA compliant.
        """)

        catalog_lines = "\n".join(
            f"- **{v}** — {len(artifacts['institutions'][k]['codes']):,} courses"
            for k, v in loaded_institutions.items()
        )
        st.markdown(f"**Loaded catalogs:**\n\n{catalog_lines}")

    with c_right:
        st.markdown("## Performance")
        scorecard = artifacts.get("scorecard", {})

        if scorecard and any(k in scorecard for k in ("wm", "vt", "ucsc")):
            for inst_key, inst_label in [("wm", "W&M"), ("vt", "Virginia Tech"), ("ucsc", "CCC → UCSC")]:
                sc = scorecard.get(inst_key, {})
                if not sc:
                    continue
                t1  = sc.get("top1_lr", 0)
                t3  = sc.get("top3_lr", 0)
                prec = sc.get("precision_tau", 0)
                cov  = sc.get("coverage_tau", 0)

                def bar(v, good, warn=0.6):
                    color = "#22c55e" if v >= good else ("#f59e0b" if v >= warn else "#ef4444")
                    return (f'<div style="background:#e5e7eb;border-radius:9999px;height:7px;margin-top:3px;">'
                            f'<div style="width:{v*100:.0f}%;background:{color};height:7px;'
                            f'border-radius:9999px;"></div></div>')

                st.markdown(f"**{inst_label}**")
                st.markdown(
                    f'Top-1 Recall &nbsp; <b>{t1:.1%}</b>'
                    + bar(t1, 0.55, 0.45)
                    + f'<br>Top-3 Recall &nbsp; <b>{t3:.1%}</b>'
                    + bar(t3, 0.75, 0.60)
                    + f'<br>Precision@τ &nbsp; <b>{prec:.1%}</b>'
                    + bar(prec, 0.90, 0.75)
                    + f'<br>Coverage@τ &nbsp; <b>{cov:.1%}</b>'
                    + bar(cov, 0.20, 0.10),
                    unsafe_allow_html=True,
                )
                brier = sc.get("brier", 0)
                ece   = sc.get("ece", 0)
                st.caption(
                    f"Brier: {brier:.4f} · ECE: {ece:.4f} · "
                    f"τ = {sc.get('op_tau', 0):.2f}"
                )
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("Run `python scripts/build_artifacts.py` to populate metrics.")

        st.markdown("## Limitations")
        st.markdown(f"""
- **Coverage at high precision is ~15–25%.** The system is reliable when it commits;
  below τ it shows top-3 for advisor review rather than guessing.
- **CCC→UCSC** has weaker Top-3 performance (~68%) — smaller training set and
  115+ California community colleges with varying course formats.
- **Dept/number are soft signals.** Courses with identical titles but different
  numbering schemes may have slightly lower confidence.
- **Not a registrar decision.** Predictions are model estimates. Always confirm
  high-stakes transfers with the registrar.
        """)
