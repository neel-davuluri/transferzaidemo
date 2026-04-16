"""TransferzAI — AI-powered transfer credit evaluation"""

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

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── tokens ── */
  :root {
    --bg:          #0d1117;
    --surface:     #161b22;
    --surface-2:   #1c2128;
    --border:      #30363d;
    --border-sub:  #21262d;
    --accent:      #2f81f7;
    --text-1:      #e6edf3;
    --text-2:      #8b949e;
    --text-3:      #484f58;
    --green:       #3fb950;
    --green-dim:   rgba(63,185,80,0.12);
    --amber:       #d29922;
    --amber-dim:   rgba(210,153,34,0.12);
    --red:         #f85149;
    --red-dim:     rgba(248,81,73,0.12);
    --radius:      10px;
    --radius-sm:   6px;
  }

  /* ── global reset ── */
  [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
  [data-testid="stHeader"]           { background: transparent !important; border-bottom: 1px solid var(--border-sub); }
  .block-container { padding-top: 1.5rem !important; padding-bottom: 4rem !important; max-width: 1200px !important; }
  section[data-testid="stSidebar"]   { background: var(--surface) !important; }

  /* ── hide streamlit branding ── */
  #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

  /* ── typography ── */
  html, body, [class*="css"] { color: var(--text-1); }
  h1,h2,h3,h4 { color: var(--text-1) !important; letter-spacing: -0.3px; }

  /* ── nav / tabs ── */
  [data-testid="stTabs"] { border-bottom: 1px solid var(--border); }
  [data-baseweb="tab-list"] { background: transparent !important; gap: 0; }
  [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-2) !important;
    font-size: 0.9rem; font-weight: 600;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.15s, border-color 0.15s;
  }
  [data-baseweb="tab"]:hover { color: var(--text-1) !important; }
  [aria-selected="true"][data-baseweb="tab"] {
    color: var(--text-1) !important;
    border-bottom: 2px solid var(--accent) !important;
  }
  [data-testid="stTabPanel"] { padding-top: 1.5rem !important; }

  /* ── inputs ── */
  .stTextInput input, .stTextArea textarea, .stNumberInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-1) !important;
    font-size: 0.9rem !important;
  }
  .stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(47,129,247,0.15) !important;
  }
  label, .stTextInput label, .stTextArea label, .stNumberInput label,
  .stSlider label, .stMultiSelect label {
    color: var(--text-2) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  /* ── multiselect ── */
  [data-baseweb="select"] > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-sm) !important;
  }
  [data-baseweb="tag"] {
    background: var(--surface-2) !important;
    border-color: var(--border) !important;
  }

  /* ── slider ── */
  [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
  }

  /* ── primary button ── */
  .stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.5rem !important;
    transition: background 0.15s, transform 0.1s;
  }
  .stButton > button[kind="primary"]:hover {
    background: #388bfd !important;
    transform: translateY(-1px);
  }
  .stButton > button[kind="primary"]:active { transform: translateY(0); }

  /* ── secondary button ── */
  .stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-1) !important;
    font-weight: 600 !important;
  }

  /* ── expander ── */
  [data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-sub) !important;
    border-radius: var(--radius-sm) !important;
  }
  [data-testid="stExpander"] summary { color: var(--text-2) !important; font-size: 0.82rem !important; }

  /* ── metric ── */
  [data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-sub) !important;
    border-radius: var(--radius) !important;
    padding: 0.9rem 1rem !important;
  }
  [data-testid="stMetricLabel"] { color: var(--text-2) !important; font-size: 0.78rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em; }
  [data-testid="stMetricValue"] { color: var(--text-1) !important; font-size: 1.4rem !important; font-weight: 700 !important; }

  /* ── divider ── */
  hr { border-color: var(--border-sub) !important; margin: 1.5rem 0 !important; }

  /* ── alerts ── */
  [data-testid="stAlert"] { border-radius: var(--radius-sm) !important; }

  /* ── number_input buttons ── */
  [data-testid="stNumberInput"] button {
    background: var(--surface-2) !important;
    border-color: var(--border) !important;
    color: var(--text-1) !important;
  }

  /* ─────────────────────────────────────────────────────────
     CUSTOM COMPONENTS
  ───────────────────────────────────────────────────────── */

  /* ── hero ── */
  .tzai-hero {
    padding: 2rem 0 2.5rem;
    display: flex; align-items: center; gap: 1rem;
  }
  .tzai-hero-icon { font-size: 2.2rem; line-height: 1; }
  .tzai-hero-title {
    font-size: 1.9rem; font-weight: 800;
    color: var(--text-1); letter-spacing: -0.5px; margin: 0;
    line-height: 1.1;
  }
  .tzai-hero-sub {
    font-size: 0.9rem; color: var(--text-2); margin: 0.2rem 0 0;
  }

  /* ── section label ── */
  .tzai-section-label {
    font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-3); margin-bottom: 0.75rem;
  }

  /* ── result card ── */
  .tzai-card {
    background: var(--surface);
    border: 1px solid var(--border-sub);
    border-left: 3px solid var(--border);
    border-radius: var(--radius);
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.15s;
  }
  .tzai-card.high { border-left-color: var(--green); }
  .tzai-card.mid  { border-left-color: var(--amber); }
  .tzai-card.low  { border-left-color: var(--red);   }

  .tzai-card-top {
    display: flex; align-items: center;
    justify-content: space-between; gap: 0.5rem;
  }
  .tzai-card-code {
    font-size: 1rem; font-weight: 700;
    color: var(--text-1); font-family: ui-monospace, monospace;
    letter-spacing: 0.02em;
  }
  .tzai-card-title {
    font-size: 0.85rem; color: var(--text-2);
    margin: 0.25rem 0 0.4rem;
    line-height: 1.4;
  }
  .tzai-card-verdict {
    font-size: 0.82rem; font-weight: 600;
    display: flex; align-items: center; gap: 0.4rem;
  }
  .tzai-card-verdict.high { color: var(--green); }
  .tzai-card-verdict.mid  { color: var(--amber); }
  .tzai-card-verdict.low  { color: var(--text-3); }

  /* ── badge ── */
  .tzai-badge {
    display: inline-flex; align-items: center; gap: 0.3em;
    border-radius: 9999px; padding: 0.2em 0.65em;
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.03em; white-space: nowrap; flex-shrink: 0;
  }
  .tzai-badge.high { background: var(--green-dim); color: var(--green); }
  .tzai-badge.mid  { background: var(--amber-dim); color: var(--amber); }
  .tzai-badge.low  { background: var(--red-dim);   color: var(--red);   }

  /* ── institution heading ── */
  .tzai-inst-head {
    font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-3);
    padding-bottom: 0.6rem;
    margin-bottom: 0.6rem;
    border-bottom: 1px solid var(--border-sub);
  }

  /* ── empty state ── */
  .tzai-empty {
    text-align: center; padding: 4rem 1rem;
    color: var(--text-3);
  }
  .tzai-empty .ico { font-size: 2rem; margin-bottom: 0.75rem; }
  .tzai-empty p { font-size: 0.88rem; margin: 0; color: var(--text-3); }

  /* ── summary banner ── */
  .tzai-banner {
    border-radius: var(--radius);
    padding: 1rem 1.25rem; margin-bottom: 1rem;
    border: 1px solid;
  }
  .tzai-banner.eligible   { background: var(--green-dim); border-color: var(--green); }
  .tzai-banner.borderline { background: var(--amber-dim); border-color: var(--amber); }
  .tzai-banner.ineligible { background: var(--red-dim);   border-color: var(--red);   }
  .tzai-banner-text { font-size: 0.92rem; font-weight: 600; color: var(--text-1); margin: 0; }

  /* ── transcript row ── */
  .tzai-tr-row {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border-sub);
    font-size: 0.85rem;
  }
  .tzai-tr-row:last-child { border-bottom: none; }
  .tzai-tr-label { color: var(--text-1); font-weight: 600; flex: 1 1 220px; min-width: 0; }
  .tzai-tr-arrow { color: var(--text-3); flex-shrink: 0; }
  .tzai-tr-match { color: var(--text-2); flex: 1 1 180px; font-family: ui-monospace, monospace; font-size: 0.82rem; }
  .tzai-tr-conf  { flex-shrink: 0; font-weight: 700; font-size: 0.82rem; }
  .tzai-tr-conf.high { color: var(--green); }
  .tzai-tr-conf.mid  { color: var(--amber); }
  .tzai-tr-conf.low  { color: var(--text-3); }

  /* ── stat pill ── */
  .tzai-stat-row { display:flex; gap:0.6rem; margin-bottom:1rem; flex-wrap:wrap; }
  .tzai-stat {
    background: var(--surface-2);
    border: 1px solid var(--border-sub);
    border-radius: var(--radius);
    padding: 0.65rem 1rem; flex: 1 1 100px; min-width: 90px;
  }
  .tzai-stat-val { font-size: 1.5rem; font-weight: 700; color: var(--text-1); line-height:1; }
  .tzai-stat-lbl { font-size: 0.72rem; font-weight: 600; color: var(--text-3);
                   text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.3rem; }

  /* ── perf bar ── */
  .tzai-perf-row { margin-bottom: 0.7rem; }
  .tzai-perf-label { font-size: 0.78rem; color: var(--text-2); margin-bottom: 0.25rem;
                     display: flex; justify-content: space-between; }
  .tzai-perf-label b { color: var(--text-1); }
  .tzai-perf-track { background: var(--surface-2); border-radius: 9999px; height: 6px; }
  .tzai-perf-fill  { height: 6px; border-radius: 9999px; }

  /* ── model card section ── */
  .tzai-mc-section { margin-bottom: 2rem; }
  .tzai-mc-title {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--accent);
    margin-bottom: 0.75rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-sub);
  }

  /* ── table override ── */
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th {
    background: var(--surface-2) !important; color: var(--text-2) !important;
    font-size: 0.72rem !important; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; padding: 0.5rem 0.75rem !important;
    border-bottom: 1px solid var(--border) !important;
    text-align: left !important;
  }
  td {
    padding: 0.55rem 0.75rem !important; color: var(--text-1) !important;
    border-bottom: 1px solid var(--border-sub) !important;
  }
  tr:last-child td { border-bottom: none !important; }
  tr:hover td { background: var(--surface-2) !important; }

  /* ── course input header row ── */
  .tzai-col-header {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; color: var(--text-3);
    padding: 0 0.15rem 0.4rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init():
    try:
        return load_artifacts()
    except FileNotFoundError as e:
        st.error(f"Model artifacts not found: {e}")
        st.info("Run `python scripts/build_artifacts.py` to generate artifacts, then refresh.")
        st.stop()

with st.spinner("Loading model…"):
    artifacts = init()

loaded_institutions = {k: v["name"] for k, v in artifacts["institutions"].items()}


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tzai-hero">
  <div class="tzai-hero-icon">🎓</div>
  <div>
    <p class="tzai-hero-title">TransferzAI</p>
    <p class="tzai-hero-sub">AI-powered transfer credit evaluation across institutions</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Course Lookup", "Transcript Evaluator", "Model Card"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cls(conf):
    if conf >= HIGH_CONFIDENCE_THRESHOLD: return "high"
    if conf >= TRANSFER_THRESHOLD:        return "mid"
    return "low"

def _badge_html(conf):
    cls   = _cls(conf)
    dots  = {"high": "●", "mid": "●", "low": "●"}
    label = {"high": "High Confidence", "mid": "Possible Match", "low": "Low Confidence"}
    return (f'<span class="tzai-badge {cls}">'
            f'{dots[cls]}&nbsp;{label[cls]}&nbsp;·&nbsp;{conf:.0%}</span>')

def _verdict_html(conf, code):
    if conf >= HIGH_CONFIDENCE_THRESHOLD:
        return f'<span class="tzai-card-verdict high">✓ Transfers as {code}</span>'
    if conf >= TRANSFER_THRESHOLD:
        return '<span class="tzai-card-verdict mid">⚠ Advisor review recommended</span>'
    return '<span class="tzai-card-verdict low">– Low confidence</span>'

def render_result(r, idx):
    cls  = _cls(r["confidence"])
    html = f"""
    <div class="tzai-card {cls}">
      <div class="tzai-card-top">
        <span class="tzai-card-code">{r['code']}</span>
        {_badge_html(r['confidence'])}
      </div>
      <div class="tzai-card-title">{r['title']}</div>
      {_verdict_html(r['confidence'], r['code'])}
    </div>"""
    st.markdown(html, unsafe_allow_html=True)
    with st.expander("Signal breakdown", expanded=False):
        sigs = r["signals"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Semantic (BGE)",  f"{sigs['bge_sim']:.3f}")
        c2.metric("Lexical (TF-IDF)",f"{sigs['tfidf_sim']:.3f}")
        c3.metric("Dept Prior",      f"{sigs['dept_prob']:.3f}")
        c4.metric("Title Match",     f"{sigs['title_sim']:.3f}")

def _conf_cls(conf):
    if conf >= HIGH_CONFIDENCE_THRESHOLD: return "high"
    if conf >= TRANSFER_THRESHOLD:        return "mid"
    return "low"

def _perf_bar(v, good, warn=0.55):
    color = "#3fb950" if v >= good else ("#d29922" if v >= warn else "#f85149")
    return (f'<div class="tzai-perf-track">'
            f'<div class="tzai-perf-fill" style="width:{v*100:.1f}%;background:{color};"></div>'
            f'</div>')


# ════════════════════════════════════════════════════════════════════════
# TAB 1 — Course Lookup
# ════════════════════════════════════════════════════════════════════════
with tab1:
    form_col, settings_col = st.columns([5, 3], gap="large")

    with form_col:
        st.markdown('<div class="tzai-section-label">Course details</div>', unsafe_allow_html=True)
        vccs_title = st.text_input(
            "Course Title",
            value="",
            placeholder="e.g. Calculus I, General Chemistry, Intro to Psychology",
        )
        vccs_desc = st.text_area(
            "Course Description",
            value="",
            placeholder="Paste the catalog description for higher accuracy (optional)",
            height=80,
        )
        dc, nc = st.columns(2)
        with dc:
            vccs_dept = st.text_input("Department Code",
                placeholder="ACC, CSC, MTH… (optional)")
        with nc:
            vccs_number = st.text_input("Course Number",
                placeholder="101, 211… (optional)")

    with settings_col:
        st.markdown('<div class="tzai-section-label">Target institutions</div>', unsafe_allow_html=True)
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

    st.markdown("<hr>", unsafe_allow_html=True)

    if search_clicked:
        if not vccs_title.strip():
            st.warning("Course title is required.")
        elif not selected:
            st.warning("Select at least one institution.")
        else:
            with st.spinner("Searching…"):
                results = predict_transfer(
                    vccs_dept, vccs_number, vccs_title, vccs_desc,
                    institutions=selected, top_k=top_k,
                )
            n_inst = len(selected)
            cols = st.columns(n_inst) if n_inst <= 3 else [st] * n_inst
            use_cols = n_inst <= 3

            for i, inst_key in enumerate(selected):
                inst_name    = loaded_institutions.get(inst_key, inst_key)
                inst_results = results.get(inst_key, [])
                container    = cols[i] if use_cols else st
                with container:
                    st.markdown(f'<div class="tzai-inst-head">{inst_name}</div>',
                                unsafe_allow_html=True)
                    if not inst_results:
                        st.caption("No matches found.")
                    else:
                        for j, r in enumerate(inst_results):
                            render_result(r, j)
    else:
        st.markdown("""
        <div class="tzai-empty">
          <div class="ico">🔍</div>
          <p>Enter a course and click <strong>Find Transfer Matches</strong></p>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — Transcript Evaluator
# ════════════════════════════════════════════════════════════════════════
with tab2:
    cfg_col, _ = st.columns([5, 2])
    with cfg_col:
        rc, cc = st.columns([3, 1])
        with rc:
            selected_t2 = st.multiselect(
                "Target institutions",
                options=list(loaded_institutions.keys()),
                default=list(loaded_institutions.keys()),
                format_func=lambda k: loaded_institutions[k],
                key="t2_inst",
            )
        with cc:
            min_credits = st.number_input(
                "Min credits", value=MIN_CREDITS_REQUIRED, min_value=1, max_value=120,
            )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="tzai-section-label">Your courses</div>', unsafe_allow_html=True)

    default_courses = [
        ("ACC", "211", "PRINCIPLES OF ACCOUNTING I"),
        ("ENG", "111", "COLLEGE COMPOSITION I"),
        ("MTH", "263", "CALCULUS I"),
        ("BIO", "101", "GENERAL BIOLOGY I"),
        ("CSC", "221", "INTRODUCTION TO PROBLEM SOLVING AND PROGRAMMING"),
    ]
    num_courses = st.number_input("Number of courses", 1, 30, len(default_courses), key="n_courses")

    # Column headers
    hc = st.columns([1, 1, 3, 3, 1])
    for col, lbl in zip(hc, ["Dept", "Number", "Title", "Description", "Cr"]):
        col.markdown(f'<div class="tzai-col-header">{lbl}</div>', unsafe_allow_html=True)

    course_inputs = []
    for i in range(int(num_courses)):
        defs = default_courses[i] if i < len(default_courses) else ("", "", "")
        row  = st.columns([1, 1, 3, 3, 1])
        dept  = row[0].text_input("d", value=defs[0],  key=f"d_{i}", label_visibility="collapsed")
        num   = row[1].text_input("n", value=defs[1],  key=f"n_{i}", label_visibility="collapsed")
        title = row[2].text_input("t", value=defs[2],  key=f"t_{i}", label_visibility="collapsed")
        desc  = row[3].text_input("x", value="",       key=f"x_{i}", label_visibility="collapsed")
        creds = row[4].number_input("c", value=DEFAULT_CREDITS_PER_COURSE,
                                    min_value=1, max_value=6, key=f"c_{i}",
                                    label_visibility="collapsed")
        if title.strip():
            course_inputs.append({"dept": dept, "number": num, "title": title,
                                   "description": desc, "credits": creds})

    st.markdown("<br>", unsafe_allow_html=True)
    eval_clicked = st.button("Evaluate Transcript", type="primary", key="transcript")

    st.markdown("<hr>", unsafe_allow_html=True)

    if eval_clicked:
        if not course_inputs:
            st.warning("Enter at least one course title.")
        elif not selected_t2:
            st.warning("Select at least one institution.")
        else:
            with st.spinner(f"Evaluating {len(course_inputs)} courses…"):
                results = evaluate_transcript(
                    course_inputs, institutions=selected_t2,
                    min_credits_required=min_credits,
                )

            for inst_key in selected_t2:
                r = results.get(inst_key)
                if not r: continue

                status = "eligible" if r["eligible"] else ("borderline" if r["borderline"] else "ineligible")
                st.markdown(
                    f'<div class="tzai-banner {status}">'
                    f'<p class="tzai-banner-text">{r["institution_name"]} — {r["summary"]}</p>'
                    f'</div>', unsafe_allow_html=True)

                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Courses",              r["total_courses"])
                sc2.metric("Total Credits",        r["total_credits"])
                sc3.metric("Confirmed Credits",    r["transferable_credits_confident"])
                sc4.metric("Possible Credits",     r["transferable_credits_possible"])

                st.markdown('<div class="tzai-section-label" style="margin-top:1rem;">Per-course results</div>',
                            unsafe_allow_html=True)

                rows_html = ""
                for cr in r["course_results"]:
                    conf = cr["confidence"]
                    cls  = _conf_cls(conf)
                    label = " ".join(filter(None, [cr["dept"], cr["number"], cr["title"]])).strip()
                    match = cr["best_match"] or "—"
                    rows_html += f"""
                    <div class="tzai-tr-row">
                      <span class="tzai-tr-label">{label}</span>
                      <span class="tzai-tr-arrow">→</span>
                      <span class="tzai-tr-match">{match}</span>
                      <span class="tzai-tr-conf {cls}">{conf:.0%}</span>
                    </div>"""
                st.markdown(rows_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="tzai-empty">
          <div class="ico">📋</div>
          <p>Fill in your courses above and click <strong>Evaluate Transcript</strong></p>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Card
# ════════════════════════════════════════════════════════════════════════
with tab3:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="tzai-mc-title">Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
**Two-stage retrieve-then-rank pipeline, run per institution.**

**Stage 1 — 3-signal RRF Retrieval** returns top-50 candidates:

| Signal | What it captures |
|:---|:---|
| Fine-tuned BGE bi-encoder | Semantic similarity via TripletLoss fine-tuning |
| TF-IDF (1-2 gram, 15k features) | Lexical keyword overlap |
| Department prior | P(target dept given source dept) |

**Stage 2 — XGBoost Reranker** scores each candidate on 13 features:

| Group | Features |
|:---|:---|
| Semantic | BGE cosine sim |
| Lexical | TF-IDF (full), TF-IDF (title only), SequenceMatcher ratio |
| Structural | Level ratio, same-level flag, dept prior, RRF score |
| Interactions | BGE×dept, BGE×title, BGE×TF-IDF, dept×title, dept×level |

Per-query softmax over XGBoost margins produces calibrated confidence scores.
Isotonic regression calibrates displayed probabilities.
        """)

        st.markdown('<div class="tzai-mc-title" style="margin-top:1.5rem;">Confidence thresholds</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
| Confidence | Threshold | Action |
|:---|:---|:---|
| 🟢 High | ≥ {HIGH_CONFIDENCE_THRESHOLD:.0%} | Show "Transfers as X" — ≥ 90% precision |
| 🟡 Possible | ≥ {TRANSFER_THRESHOLD:.0%} | Advisor review recommended |
| 🔴 Low | < {TRANSFER_THRESHOLD:.0%} | Show top-3 only, refer to registrar |
        """)

        st.markdown('<div class="tzai-mc-title" style="margin-top:1.5rem;">Training data</div>',
                    unsafe_allow_html=True)
        st.markdown("""
| Institution pair | Train pairs | Test pairs |
|:---|:---:|:---:|
| VCCS → W&M | ~267 | 67 |
| VCCS → VT | ~242 | 60 |
| CCC → UCSC | ~723 | 181 |

Sources: VCCS→W&M and VCCS→VT articulation agreements; CCC→UCSC articulation data.
FERPA compliant — no student PII.
        """)
        catalog_lines = "  \n".join(
            f"**{v}** — {len(artifacts['institutions'][k]['codes']):,} courses"
            for k, v in loaded_institutions.items()
        )
        st.markdown(f"\n{catalog_lines}")

    with right:
        st.markdown('<div class="tzai-mc-title">Performance</div>', unsafe_allow_html=True)
        scorecard = artifacts.get("scorecard", {})

        if scorecard and any(k in scorecard for k in ("wm", "vt", "ucsc")):
            for inst_key, inst_label in [("wm", "William & Mary"), ("vt", "Virginia Tech"),
                                          ("ucsc", "CCC → UCSC")]:
                sc = scorecard.get(inst_key, {})
                if not sc: continue

                t1   = sc.get("top1_lr", 0)
                t3   = sc.get("top3_lr", 0)
                prec = sc.get("precision_tau", 0)
                cov  = sc.get("coverage_tau", 0)

                st.markdown(f"**{inst_label}**")
                metrics = [
                    ("Top-1 Recall",  t1,   0.55, 0.45),
                    ("Top-3 Recall",  t3,   0.75, 0.60),
                    ("Precision @ τ", prec, 0.90, 0.75),
                    ("Coverage @ τ",  cov,  0.20, 0.10),
                ]
                bars = ""
                for label, v, good, warn in metrics:
                    color = "#3fb950" if v >= good else ("#d29922" if v >= warn else "#f85149")
                    bars += f"""
                    <div class="tzai-perf-row">
                      <div class="tzai-perf-label">
                        <span>{label}</span>
                        <b style="color:{color}">{v:.1%}</b>
                      </div>
                      <div class="tzai-perf-track">
                        <div class="tzai-perf-fill" style="width:{v*100:.1f}%;background:{color};"></div>
                      </div>
                    </div>"""
                st.markdown(bars, unsafe_allow_html=True)
                st.caption(
                    f"Brier: {sc.get('brier',0):.4f}  ·  "
                    f"ECE: {sc.get('ece',0):.4f}  ·  "
                    f"τ = {sc.get('op_tau',0):.2f}"
                )
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("Run `python scripts/build_artifacts.py` to populate metrics.")

        st.markdown('<div class="tzai-mc-title" style="margin-top:0.5rem;">Limitations</div>',
                    unsafe_allow_html=True)
        st.markdown("""
- **Coverage at high precision is ~15–25%.** The system says "low confidence" rather than guessing wrong.
- **CCC→UCSC** top-3 ~68% — smaller training set and 115+ colleges with varying formats.
- **Dept/number are soft signals.** Missing or non-standard codes reduce confidence slightly.
- **Not a registrar decision.** Always confirm with the registrar before acting on results.
        """)
