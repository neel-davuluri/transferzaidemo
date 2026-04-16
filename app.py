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

st.markdown("""
<style>
/* ── tokens ── */
:root {
  --bg:         #09090b;
  --surface:    #18181b;
  --surface-2:  #27272a;
  --surface-3:  #3f3f46;
  --border:     #3f3f46;
  --border-2:   #52525b;
  --accent:     #60a5fa;
  --accent-dim: rgba(96,165,250,0.15);
  --text-1:     #fafafa;
  --text-2:     #d4d4d8;
  --text-3:     #a1a1aa;
  --text-4:     #71717a;
  --green:      #4ade80;
  --green-bg:   rgba(74,222,128,0.10);
  --amber:      #fbbf24;
  --amber-bg:   rgba(251,191,36,0.10);
  --red:        #f87171;
  --red-bg:     rgba(248,113,113,0.10);
  --radius:     8px;
  --radius-lg:  12px;
}

/* ── reset ── */
[data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"]           { display: none !important; }
.block-container {
  padding: 2.5rem 2rem 6rem !important;
  max-width: 920px !important;
  margin: 0 auto !important;
}
#MainMenu, footer { display: none !important; }

/* ── global type ── */
html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif !important;
  color: var(--text-1);
}
p, li { color: var(--text-2); font-size: 0.88rem; line-height: 1.65; }
h1,h2,h3,h4,h5 { color: var(--text-1) !important; letter-spacing: -0.4px; }

/* ── tabs ── */
[data-testid="stTabs"] { border-bottom: 1px solid var(--border) !important; margin-bottom: 0 !important; }
[data-baseweb="tab-list"] { background: transparent !important; gap: 0 !important; padding: 0 !important; }
[data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-3) !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  padding: 0.65rem 1.1rem !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  transition: color .15s;
}
[data-baseweb="tab"]:hover { color: var(--text-1) !important; }
[aria-selected="true"][data-baseweb="tab"] {
  color: var(--text-1) !important;
  border-bottom-color: var(--accent) !important;
}
[data-testid="stTabPanel"] { padding-top: 2rem !important; }

/* ── inputs ── */
.stTextInput input, .stTextArea textarea, .stNumberInput input {
  background: var(--surface) !important;
  border: 1px solid var(--border-2) !important;
  border-radius: var(--radius) !important;
  color: var(--text-1) !important;
  font-size: 0.9rem !important;
  transition: border-color .15s, box-shadow .15s;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-dim) !important;
}
.stTextInput label, .stTextArea label, .stNumberInput label,
.stSlider label, .stMultiSelect label, .stSelectbox label, label {
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  color: var(--text-3) !important;
}

/* ── selectbox ── */
[data-baseweb="select"] > div {
  background: var(--surface) !important;
  border: 1px solid var(--border-2) !important;
  border-radius: var(--radius) !important;
  color: var(--text-1) !important;
}
[data-baseweb="popover"] { background: var(--surface-2) !important; border: 1px solid var(--border-2) !important; border-radius: var(--radius) !important; }
[role="option"] { background: transparent !important; color: var(--text-2) !important; font-size: 0.88rem !important; }
[role="option"]:hover, [aria-selected="true"][role="option"] { background: var(--surface-3) !important; color: var(--text-1) !important; }

/* ── multiselect ── */
[data-baseweb="tag"] {
  background: var(--surface-3) !important;
  border: 1px solid var(--border-2) !important;
  border-radius: 9999px !important;
}
[data-baseweb="tag"] span { color: var(--text-1) !important; font-size: 0.78rem !important; }

/* ── buttons ── */
.stButton > button {
  border-radius: var(--radius) !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  transition: all .15s !important;
  border: 1px solid var(--border-2) !important;
  background: var(--surface-2) !important;
  color: var(--text-2) !important;
}
.stButton > button:hover {
  color: var(--text-1) !important;
  background: var(--surface-3) !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #09090b !important;
  font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover {
  background: #93c5fd !important;
  border-color: #93c5fd !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 16px rgba(96,165,250,0.3) !important;
}

/* ── expander ── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: transparent !important;
}
[data-testid="stExpander"] summary {
  color: var(--text-4) !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em;
}

/* ── metric ── */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 0.85rem 1rem !important;
}
[data-testid="stMetricLabel"] {
  font-size: 0.68rem !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: 0.07em !important;
  color: var(--text-3) !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.5rem !important; font-weight: 800 !important;
  color: var(--text-1) !important; letter-spacing: -0.5px;
}

/* ── divider ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.75rem 0 !important; }

/* ── number input buttons ── */
[data-testid="stNumberInput"] button {
  background: var(--surface-2) !important; border-color: var(--border-2) !important; color: var(--text-2) !important;
}

/* ── select slider ── */
[data-testid="stSlider"] { padding: 0 !important; }

/* ── tables ── */
table { width:100%; border-collapse:collapse; font-size:0.85rem; }
th {
  background: var(--surface-2) !important; color: var(--text-3) !important;
  font-size: 0.68rem !important; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.07em; padding: 0.55rem 0.75rem !important;
  border-bottom: 1px solid var(--border-2) !important; text-align: left !important;
}
td { padding: 0.55rem 0.75rem !important; color: var(--text-2) !important; border-bottom: 1px solid var(--border) !important; }
tr:last-child td { border-bottom: none !important; }
tr:hover td { background: var(--surface-2) !important; }

/* ══════════════════════════════════════════════════
   CUSTOM COMPONENTS
══════════════════════════════════════════════════ */

/* ── hero wordmark ── */
.tzai-hero { padding: 1rem 0 2rem; }
.tzai-hero-row { display:flex; align-items:baseline; gap:0.65rem; margin-bottom:0.4rem; }
.tzai-hero-name {
  font-size: 3rem; font-weight: 900;
  color: var(--text-1); letter-spacing: -1.5px; line-height: 1;
}
.tzai-hero-sub { font-size: 0.9rem; color: var(--text-3); }

/* ── how-to card ── */
.tzai-howto {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: var(--radius-lg);
  padding: 1.1rem 1.25rem;
  margin-bottom: 1.75rem;
}
.tzai-howto-head {
  font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--accent); margin-bottom: 0.8rem;
}
.tzai-howto-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0.65rem 2rem;
}
.tzai-howto-item { font-size: 0.83rem; color: var(--text-2); line-height: 1.5; }
.tzai-howto-item b { color: var(--text-1); display: block; margin-bottom: 0.15rem; }

/* ── inst dropdown trigger ── */
.tzai-inst-select-label {
  font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--text-3); margin-bottom: 0.4rem;
}

/* ── result: featured ── */
.tzai-card-feat {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: var(--radius-lg);
  padding: 1.1rem 1.25rem;
  margin-bottom: 0.5rem;
}
.tzai-card-feat.high { border-left: 3px solid var(--green); }
.tzai-card-feat.mid  { border-left: 3px solid var(--amber); }
.tzai-card-feat.low  { border-left: 3px solid var(--text-4); }
.tzai-card-top { display:flex; align-items:center; justify-content:space-between; gap:0.75rem; margin-bottom:0.3rem; }
.tzai-card-code { font-family:ui-monospace,"SFMono-Regular","Fira Code",monospace; font-size:1.05rem; font-weight:700; color:var(--text-1); letter-spacing:0.03em; }
.tzai-card-pct  { font-size:1.15rem; font-weight:800; letter-spacing:-0.5px; flex-shrink:0; }
.tzai-card-pct.high { color:var(--green); }
.tzai-card-pct.mid  { color:var(--amber); }
.tzai-card-pct.low  { color:var(--text-4); }
.tzai-card-title  { font-size:0.85rem; color:var(--text-2); margin-bottom:0.4rem; }
.tzai-card-verdict { font-size:0.78rem; font-weight:600; }
.tzai-card-verdict.high { color:var(--green); }
.tzai-card-verdict.mid  { color:var(--amber); }
.tzai-card-verdict.low  { color:var(--text-4); }

/* ── result: secondary ── */
.tzai-card-sub {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.6rem 1rem;
  margin-bottom: 0.3rem;
  display:flex; align-items:center; gap:0.75rem;
  opacity:0.7; transition: opacity .15s;
}
.tzai-card-sub:hover { opacity:1; }
.tzai-sub-code  { font-family:ui-monospace,monospace; font-size:0.83rem; font-weight:700; color:var(--text-1); flex-shrink:0; width:88px; }
.tzai-sub-title { font-size:0.82rem; color:var(--text-2); flex:1; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.tzai-sub-pct   { font-size:0.78rem; font-weight:700; flex-shrink:0; }
.tzai-sub-pct.high { color:var(--green); }
.tzai-sub-pct.mid  { color:var(--amber); }
.tzai-sub-pct.low  { color:var(--text-4); }

/* ── inst heading ── */
.tzai-inst-head {
  font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em;
  color:var(--text-2);
  margin:1.5rem 0 0.75rem;
  display:flex; align-items:center; gap:0.5rem;
}
.tzai-inst-head::after { content:""; flex:1; height:1px; background:var(--border); }

/* ── empty state ── */
.tzai-empty { text-align:center; padding:3rem 1rem 2rem; }
.tzai-empty-ico   { font-size:2rem; margin-bottom:0.75rem; }
.tzai-empty-title { font-size:0.95rem; font-weight:600; color:var(--text-2); margin-bottom:0.35rem; }
.tzai-empty-sub   { font-size:0.82rem; color:var(--text-3); }

/* ── section label ── */
.tzai-lbl {
  font-size:0.72rem; font-weight:700; text-transform:uppercase;
  letter-spacing:0.09em; color:var(--text-2); margin-bottom:0.65rem;
}

/* ── transcript course card ── */
.tzai-course-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.85rem 1rem;
  margin-bottom: 0.5rem;
}
.tzai-course-card-header {
  display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.35rem;
}
.tzai-course-num  { font-size:0.72rem; color:var(--text-4); font-weight:600; flex-shrink:0; }
.tzai-course-code { font-family:ui-monospace,monospace; font-size:0.8rem; color:var(--text-3); }
.tzai-course-title-text { font-size:0.9rem; font-weight:600; color:var(--text-1); flex:1; }
.tzai-course-desc-input { width:100%; }
.tzai-course-row2 { display:flex; gap:0.6rem; margin-top:0.5rem; align-items:center; }
.tzai-course-tag {
  background:var(--surface-2); border:1px solid var(--border-2);
  border-radius:9999px; padding:0.15em 0.6em;
  font-size:0.72rem; color:var(--text-3); font-weight:600;
}

/* ── transcript result row ── */
.tzai-tr {
  display:flex; align-items:center; gap:0.75rem;
  padding:0.6rem 0; border-bottom:1px solid var(--border);
  font-size:0.83rem;
}
.tzai-tr:last-child { border-bottom:none; }
.tzai-tr-src  { color:var(--text-1); font-weight:600; flex:1 1 200px; min-width:0; }
.tzai-tr-arr  { color:var(--text-4); flex-shrink:0; font-size:0.75rem; }
.tzai-tr-tgt  { font-family:ui-monospace,monospace; font-size:0.8rem; color:var(--text-2); flex:0 0 110px; }
.tzai-tr-pct  { font-weight:700; font-size:0.8rem; flex-shrink:0; min-width:36px; text-align:right; }
.tzai-tr-pct.high { color:var(--green); }
.tzai-tr-pct.mid  { color:var(--amber); }
.tzai-tr-pct.low  { color:var(--text-4); }

/* ── banner ── */
.tzai-banner {
  border-radius:var(--radius); padding:0.85rem 1.1rem; margin-bottom:1rem;
  font-size:0.88rem; font-weight:600; display:flex; align-items:center; gap:0.6rem;
}
.tzai-banner.eligible   { background:var(--green-bg); color:var(--green); border:1px solid rgba(74,222,128,0.25); }
.tzai-banner.borderline { background:var(--amber-bg); color:var(--amber); border:1px solid rgba(251,191,36,0.25); }
.tzai-banner.ineligible { background:var(--red-bg);   color:var(--red);   border:1px solid rgba(248,113,113,0.25); }

/* ── col header ── */
.tzai-col-hdr {
  font-size:0.65rem; font-weight:700; text-transform:uppercase;
  letter-spacing:0.08em; color:var(--text-4); padding-bottom:0.3rem;
}

/* ── required star ── */
.req { color: var(--red); }

/* ── perf bar ── */
.tzai-perf-inst { margin-bottom:1.5rem; }
.tzai-perf-inst-name { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:var(--text-2); margin-bottom:0.7rem; }
.tzai-perf-row { margin-bottom:0.6rem; }
.tzai-perf-top { display:flex; justify-content:space-between; font-size:0.78rem; margin-bottom:0.2rem; }
.tzai-perf-top span { color:var(--text-2); }
.tzai-perf-track { background:var(--surface-3); border-radius:9999px; height:5px; }
.tzai-perf-fill  { height:5px; border-radius:9999px; }

/* ── model card ── */
.tzai-mc-pill {
  display:inline-block; background:var(--accent-dim); border:1px solid rgba(96,165,250,0.2);
  border-radius:9999px; padding:0.2em 0.7em; font-size:0.72rem; font-weight:700;
  color:var(--accent); letter-spacing:0.04em; margin-bottom:1rem;
}
.tzai-mc-h {
  font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.09em;
  color:var(--accent); padding-bottom:0.5rem; border-bottom:1px solid var(--border-2); margin-bottom:0.9rem;
}

/* ── skeleton ── */
@keyframes shimmer {
  0%   { background-position:-600px 0; }
  100% { background-position: 600px 0; }
}
.tzai-skel {
  background: linear-gradient(90deg, var(--surface) 25%, var(--surface-2) 50%, var(--surface) 75%);
  background-size:600px 100%; animation:shimmer 1.4s infinite;
  border-radius:var(--radius); height:80px; margin-bottom:0.5rem;
}
.tzai-skel-sm { height:44px; }
</style>
""", unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────────────────────
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

INSTS = {k: v["name"] for k, v in artifacts["institutions"].items()}

INST_LOGOS = {
    "wm":           "🏛️",
    "vt":           "🦃",
    "ucsc":         "🐌",
    "northeastern": "🐾",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _cls(conf):
    if conf >= HIGH_CONFIDENCE_THRESHOLD: return "high"
    if conf >= TRANSFER_THRESHOLD:        return "mid"
    return "low"

def _verdict(conf, code):
    if conf >= HIGH_CONFIDENCE_THRESHOLD:
        return f'<span class="tzai-card-verdict high">✓ Transfers as {code}</span>'
    if conf >= TRANSFER_THRESHOLD:
        return '<span class="tzai-card-verdict mid">~ Advisor review recommended</span>'
    return '<span class="tzai-card-verdict low">– Low confidence</span>'

def render_results(results, selected):
    for inst_key in selected:
        name = INSTS.get(inst_key, inst_key)
        logo = INST_LOGOS.get(inst_key, "🏫")
        rows = results.get(inst_key, [])
        st.markdown(f'<div class="tzai-inst-head">{logo} {name}</div>', unsafe_allow_html=True)
        if not rows:
            st.markdown('<p style="color:var(--text-4);font-size:0.82rem;">No matches found.</p>', unsafe_allow_html=True)
            continue
        r0  = rows[0]; c0 = r0["confidence"]; cls = _cls(c0)
        st.markdown(f"""
        <div class="tzai-card-feat {cls}">
          <div class="tzai-card-top">
            <span class="tzai-card-code">{r0['code']}</span>
            <span class="tzai-card-pct {cls}">{c0:.0%}</span>
          </div>
          <div class="tzai-card-title">{r0['title']}</div>
          {_verdict(c0, r0['code'])}
        </div>""", unsafe_allow_html=True)
        with st.expander("Signal breakdown", expanded=False):
            sigs = r0["signals"]
            a, b, c, d = st.columns(4)
            a.metric("Semantic",   f"{sigs['bge_sim']:.3f}")
            b.metric("Lexical",    f"{sigs['tfidf_sim']:.3f}")
            c.metric("Dept Prior", f"{sigs['dept_prob']:.3f}")
            d.metric("Title",      f"{sigs['title_sim']:.3f}")
        if len(rows) > 1:
            sub = "".join(f"""
            <div class="tzai-card-sub">
              <span class="tzai-sub-code">{r['code']}</span>
              <span class="tzai-sub-title">{r['title']}</span>
              <span class="tzai-sub-pct {_cls(r['confidence'])}">{r['confidence']:.0%}</span>
            </div>""" for r in rows[1:])
            st.markdown(sub, unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tzai-hero">
  <div class="tzai-hero-row">
    <span style="font-size:2.6rem;line-height:1;">🎓</span>
    <span class="tzai-hero-name">TransferzAI</span>
  </div>
  <div class="tzai-hero-sub">AI-powered transfer credit evaluation across institutions</div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Single Class Lookup", "Transcript Evaluator", "Model Card"])


# ════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Class Lookup
# ════════════════════════════════════════════════════════════════════════
with tab1:

    # Session state init (each key independently)
    for _k, _v in [("sel_insts", list(INSTS.keys())),
                   ("example_title",""), ("example_dept",""),
                   ("example_num",""), ("example_desc","")]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── How to use ────────────────────────────────────────────────────────
    st.markdown("""
<div class="tzai-howto">
  <div class="tzai-howto-head">How to use</div>
  <div class="tzai-howto-grid">
    <div class="tzai-howto-item"><b>1. Choose your target schools</b>Select which institutions to check your course against using the dropdown.</div>
    <div class="tzai-howto-item"><b>2. Enter your course</b>Fill in the title and description (<span style="color:#f87171;">required</span>) — description significantly improves accuracy.</div>
    <div class="tzai-howto-item"><b>3. Add dept &amp; number (optional)</b>Including your department code and course number improves the department-prior signal.</div>
    <div class="tzai-howto-item"><b>4. Read the verdict</b><span style="color:#4ade80;font-weight:600;">Green</span> = confirmed transfer &nbsp;·&nbsp; <span style="color:#fbbf24;font-weight:600;">Yellow</span> = likely, ask advisor &nbsp;·&nbsp; <span style="color:#71717a;">Gray</span> = low confidence</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Institution dropdown ──────────────────────────────────────────────
    inst_options = list(INSTS.keys())
    inst_labels  = [f"{INST_LOGOS.get(k,'🏫')}  {INSTS[k]}" for k in inst_options]
    label_to_key = dict(zip(inst_labels, inst_options))

    selected_labels = st.multiselect(
        "Target institutions",
        options=inst_labels,
        default=inst_labels,
        key="inst_dropdown",
    )
    selected = [label_to_key[l] for l in selected_labels]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Course inputs — 2-col layout ──────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        vccs_title = st.text_input(
            "Course Title ✦",
            value=st.session_state.example_title,
            placeholder="e.g. Calculus I",
            key="title_input",
        )
        vccs_desc = st.text_area(
            "Course Description ✦",
            value=st.session_state.example_desc,
            placeholder="Paste your course catalog description here — required for accurate results",
            height=110,
            key="desc_input",
        )

    with right_col:
        vccs_dept = st.text_input(
            "Department Code",
            value=st.session_state.example_dept,
            placeholder="ACC, MTH, CSC… (optional)",
        )
        vccs_number = st.text_input(
            "Course Number",
            value=st.session_state.example_num,
            placeholder="101, 211, 263… (optional)",
        )
        top_k = st.select_slider(
            "Results per institution",
            options=[3, 5, 7, 10], value=5,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("Find Matches →", type="primary", key="search", use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    if search_clicked:
        if not vccs_title.strip():
            st.warning("Course title is required (✦).")
        elif not vccs_desc.strip():
            st.warning("Course description is required (✦) — paste your catalog description for accurate results.")
        elif not selected:
            st.warning("Select at least one institution.")
        else:
            with st.spinner(""):
                skel = st.empty()
                skel.markdown(
                    ''.join(['<div class="tzai-skel"></div>',
                             '<div class="tzai-skel tzai-skel-sm"></div>'] * 2),
                    unsafe_allow_html=True)
                results = predict_transfer(
                    vccs_dept, vccs_number, vccs_title, vccs_desc,
                    institutions=selected, top_k=top_k,
                )
                skel.empty()
            render_results(results, selected)
    else:
        st.markdown("""
        <div class="tzai-empty">
          <div class="tzai-empty-ico">🔍</div>
          <div class="tzai-empty-title">Enter a course to get started</div>
          <div class="tzai-empty-sub">Fill in the title and description, then click Find Matches</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — Transcript Evaluator
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
<div class="tzai-howto" style="margin-bottom:1.5rem;">
  <div class="tzai-howto-head">How to use</div>
  <div class="tzai-howto-grid">
    <div class="tzai-howto-item"><b>1. Add your courses</b>Enter each course below. Title <span style="color:#f87171;">✦</span> and description <span style="color:#f87171;">✦</span> are required for accurate results.</div>
    <div class="tzai-howto-item"><b>2. Choose target schools &amp; credits</b>Select institutions and set the minimum transfer credit threshold.</div>
    <div class="tzai-howto-item"><b>3. Evaluate</b>Click Evaluate Transcript to see a full breakdown of which courses transfer and your total confirmed credits.</div>
    <div class="tzai-howto-item"><b>4. Interpret results</b><span style="color:#4ade80;font-weight:600;">Green</span> = confirmed &nbsp;·&nbsp; <span style="color:#fbbf24;font-weight:600;">Yellow</span> = possible &nbsp;·&nbsp; <span style="color:#71717a;">Gray</span> = no match</div>
  </div>
</div>""", unsafe_allow_html=True)

    top_row = st.columns([4, 1])
    with top_row[0]:
        sel_t2 = st.multiselect(
            "Target institutions",
            options=[f"{INST_LOGOS.get(k,'🏫')}  {INSTS[k]}" for k in INSTS],
            default=[f"{INST_LOGOS.get(k,'🏫')}  {INSTS[k]}" for k in INSTS],
            key="t2_inst",
        )
        sel_t2_keys = [list(INSTS.keys())[i]
                       for i, lbl in enumerate([f"{INST_LOGOS.get(k,'🏫')}  {INSTS[k]}" for k in INSTS])
                       if lbl in sel_t2]
    with top_row[1]:
        min_credits = st.number_input("Min credits", value=MIN_CREDITS_REQUIRED, min_value=1, max_value=120)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="tzai-lbl">Your courses <span style="color:var(--red);font-size:0.65rem;">✦ Title &amp; Description required per course</span></div>', unsafe_allow_html=True)

    DEFAULT_COURSES = [
        ("ACC", "211", "PRINCIPLES OF ACCOUNTING I"),
        ("ENG", "111", "COLLEGE COMPOSITION I"),
        ("MTH", "263", "CALCULUS I"),
        ("BIO", "101", "GENERAL BIOLOGY I"),
        ("CSC", "221", "INTRODUCTION TO PROBLEM SOLVING AND PROGRAMMING"),
    ]
    n = st.number_input("Number of courses", 1, 30, len(DEFAULT_COURSES), key="n_courses")

    # Column headers
    hdr = st.columns([2, 1, 1, 1])
    for col, lbl in zip(hdr, ["Title ✦  /  Description ✦", "Dept", "No.", "Credits"]):
        col.markdown(f'<div class="tzai-col-hdr">{lbl}</div>', unsafe_allow_html=True)

    course_inputs = []
    for i in range(int(n)):
        defs  = DEFAULT_COURSES[i] if i < len(DEFAULT_COURSES) else ("", "", "")
        with st.container():
            row = st.columns([2, 1, 1, 1])
            with row[0]:
                title = st.text_input(f"Title {i+1} ✦", value=defs[2], key=f"t_{i}",
                                      placeholder="Course title (required)",
                                      label_visibility="collapsed")
                desc  = st.text_input(f"Desc {i+1} ✦", value="", key=f"x_{i}",
                                      placeholder="Catalog description (required)",
                                      label_visibility="collapsed")
            with row[1]:
                dept  = st.text_input(f"Dept {i+1}", value=defs[0], key=f"d_{i}",
                                      label_visibility="collapsed")
            with row[2]:
                num   = st.text_input(f"Num {i+1}", value=defs[1], key=f"n_{i}",
                                      label_visibility="collapsed")
            with row[3]:
                creds = st.number_input(f"Cr {i+1}", value=DEFAULT_CREDITS_PER_COURSE,
                                        min_value=1, max_value=6, key=f"c_{i}",
                                        label_visibility="collapsed")
            if title.strip():
                course_inputs.append({"dept": dept, "number": num, "title": title,
                                       "description": desc, "credits": creds})

    st.markdown("<br>", unsafe_allow_html=True)
    eval_clicked = st.button("Evaluate Transcript →", type="primary", key="eval")

    st.markdown("<hr>", unsafe_allow_html=True)

    if eval_clicked:
        missing_desc = [c["title"] for c in course_inputs if not c["description"].strip()]
        if not course_inputs:
            st.warning("Enter at least one course title.")
        elif missing_desc:
            st.warning(f"Description required (✦) for: {', '.join(missing_desc[:3])}{'…' if len(missing_desc) > 3 else ''}")
        elif not sel_t2_keys:
            st.warning("Select at least one institution.")
        else:
            with st.spinner(f"Evaluating {len(course_inputs)} courses…"):
                results = evaluate_transcript(course_inputs, institutions=sel_t2_keys,
                                              min_credits_required=min_credits)
            for inst_key in sel_t2_keys:
                r = results.get(inst_key)
                if not r: continue
                status = "eligible" if r["eligible"] else ("borderline" if r["borderline"] else "ineligible")
                icon   = "✓" if r["eligible"] else ("~" if r["borderline"] else "✕")
                logo   = INST_LOGOS.get(inst_key, "🏫")
                st.markdown(
                    f'<div class="tzai-banner {status}">'
                    f'<span style="font-size:1.1rem;">{icon}</span>'
                    f'<span>{logo} {r["institution_name"]} — {r["summary"]}</span>'
                    f'</div>', unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Courses",        r["total_courses"])
                m2.metric("Total Credits",  r["total_credits"])
                m3.metric("Confirmed",      r["transferable_credits_confident"])
                m4.metric("Possible",       r["transferable_credits_possible"])

                st.markdown('<div class="tzai-lbl" style="margin-top:1rem;">Per-course breakdown</div>',
                            unsafe_allow_html=True)
                rows_html = ""
                for cr in r["course_results"]:
                    conf = cr["confidence"]; cls = _cls(conf)
                    src  = " ".join(filter(None,[cr["dept"],cr["number"],cr["title"]])).strip()
                    tgt  = cr["best_match"] or "—"
                    rows_html += f"""<div class="tzai-tr">
                      <span class="tzai-tr-src">{src}</span>
                      <span class="tzai-tr-arr">→</span>
                      <span class="tzai-tr-tgt">{tgt}</span>
                      <span class="tzai-tr-pct {cls}">{conf:.0%}</span>
                    </div>"""
                st.markdown(rows_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="tzai-empty">
          <div class="tzai-empty-ico">📋</div>
          <div class="tzai-empty-title">Fill in your courses above</div>
          <div class="tzai-empty-sub">then click Evaluate Transcript</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Card
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<span class="tzai-mc-pill">v3 · BGE + XGBoost</span>', unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="tzai-mc-h">Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
**Two-stage retrieve-then-rank pipeline, run per institution.**

**Stage 1 — 3-signal RRF** retrieves top-50 candidates:

| Signal | What it captures |
|:---|:---|
| Fine-tuned BGE bi-encoder | Semantic similarity (TripletLoss) |
| TF-IDF 1–2 gram, 15k features | Lexical keyword overlap |
| Department prior | P(target dept given source dept) |

**Stage 2 — XGBoost Reranker** scores candidates on 13 features:

| Group | Features |
|:---|:---|
| Semantic | BGE cosine sim |
| Lexical | TF-IDF (full), TF-IDF (title), SequenceMatcher |
| Structural | Level ratio, same-level flag, dept prior, RRF score |
| Interactions | BGE×dept, BGE×title, BGE×TF-IDF, dept×title, dept×level |

Per-query softmax over XGBoost margins for calibrated confidence.
        """)
        st.markdown('<div class="tzai-mc-h" style="margin-top:1.5rem;">Thresholds</div>', unsafe_allow_html=True)
        st.markdown(f"""
| | Threshold | Action |
|:---|:---|:---|
| 🟢 High Confidence | ≥ {HIGH_CONFIDENCE_THRESHOLD:.0%} | Transfers as X — ≥ 90% precision |
| 🟡 Possible Match  | ≥ {TRANSFER_THRESHOLD:.0%} | Advisor review recommended |
| 🔴 Low Confidence  | < {TRANSFER_THRESHOLD:.0%} | Refer to registrar |
        """)
        st.markdown('<div class="tzai-mc-h" style="margin-top:1.5rem;">Training data</div>', unsafe_allow_html=True)
        st.markdown("""
| Pair | Train | Test |
|:---|:---:|:---:|
| VCCS → W&M | ~267 | 67 |
| VCCS → VT | ~242 | 60 |
| CCC → UCSC | ~723 | 181 |

FERPA compliant — no student PII.
        """)
        st.markdown("\n".join(
            f"**{v}** — {len(artifacts['institutions'][k]['codes']):,} courses"
            for k, v in INSTS.items()
        ))

    with right:
        st.markdown('<div class="tzai-mc-h">Performance</div>', unsafe_allow_html=True)
        sc = artifacts.get("scorecard", {})
        if sc and any(k in sc for k in ("wm","vt","ucsc")):
            for inst_key, label in [("wm","William & Mary"),("vt","Virginia Tech"),("ucsc","CCC → UCSC")]:
                s = sc.get(inst_key, {})
                if not s: continue
                t1=s.get("top1_lr",0); t3=s.get("top3_lr",0)
                pr=s.get("precision_tau",0); cov=s.get("coverage_tau",0)
                html = f'<div class="tzai-perf-inst"><div class="tzai-perf-inst-name">{label}</div>'
                for lbl_, v_, good_, warn_ in [
                    ("Top-1 Recall",t1,0.55,0.45),("Top-3 Recall",t3,0.75,0.60),
                    ("Precision @ τ",pr,0.90,0.75),("Coverage @ τ",cov,0.20,0.10)]:
                    col_ = "#4ade80" if v_>=good_ else ("#fbbf24" if v_>=warn_ else "#f87171")
                    html += (f'<div class="tzai-perf-row">'
                             f'<div class="tzai-perf-top"><span>{lbl_}</span>'
                             f'<b style="color:{col_}">{v_:.1%}</b></div>'
                             f'<div class="tzai-perf-track">'
                             f'<div class="tzai-perf-fill" style="width:{v_*100:.1f}%;background:{col_}"></div>'
                             f'</div></div>')
                html += '</div>'
                st.markdown(html, unsafe_allow_html=True)
                st.caption(f"Brier {s.get('brier',0):.4f} · ECE {s.get('ece',0):.4f} · τ={s.get('op_tau',0):.2f}")
        else:
            st.info("Run `python scripts/build_artifacts.py` to populate metrics.")

        st.markdown('<div class="tzai-mc-h" style="margin-top:1rem;">Limitations</div>', unsafe_allow_html=True)
        st.markdown("""
- Coverage at high precision ~15–25% — system abstains rather than guessing wrong
- CCC→UCSC top-3 ~68% — smaller training set, 115+ colleges
- Dept/number are soft signals — optional but improve results
- Not a registrar decision — always confirm before acting
        """)

    # ── Diagnostics (temporary) ──
    with st.expander("🔧 System Diagnostics", expanded=True):
        diag = artifacts.get("_diag", {})
        if diag:
            st.code(
                f"BUILD: 2026-04-16-v2\n"
                f"Python: {diag.get('python', '?')}\n"
                f"XGBoost: {diag.get('xgboost', '?')}\n"
                f"Classifier: {diag.get('classifier_type', '?')}\n"
                f"n_estimators: {diag.get('n_estimators', '?')}\n"
                f"num_features: {diag.get('num_features', '?')}\n"
                f"feature_names: {diag.get('feature_names', '?')}\n"
                f"artifacts_dir: {diag.get('artifacts_dir', '?')}\n"
                f"institutions: {list(artifacts.get('institutions', {}).keys())}",
                language="yaml",
            )
        else:
            st.warning("No diagnostic info available")
