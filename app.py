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
/* ── tokens ─────────────────────────────────────────────────────────── */
:root {
  --bg:         #090c10;
  --surface:    #0d1117;
  --surface-2:  #161b22;
  --surface-3:  #1c2128;
  --border:     #21262d;
  --border-2:   #30363d;
  --accent:     #58a6ff;
  --accent-dim: rgba(88,166,255,0.12);
  --text-1:     #f0f6fc;
  --text-2:     #8b949e;
  --text-3:     #3d444d;
  --green:      #3fb950;
  --green-bg:   rgba(63,185,80,0.10);
  --amber:      #d29922;
  --amber-bg:   rgba(210,153,34,0.10);
  --red:        #f85149;
  --red-bg:     rgba(248,81,73,0.10);
  --radius:     8px;
  --radius-lg:  12px;
}

/* ── reset ───────────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"]           { display: none !important; }
.block-container {
  padding: 2rem 2rem 6rem !important;
  max-width: 860px !important;
  margin: 0 auto !important;
}
#MainMenu, footer { display: none !important; }
* { box-sizing: border-box; }

/* ── global type ─────────────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif !important;
  color: var(--text-1);
}
p, li { color: var(--text-2); font-size: 0.88rem; line-height: 1.65; }
h1,h2,h3,h4,h5 { color: var(--text-1) !important; letter-spacing: -0.4px; }

/* ── tabs ────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] { border-bottom: 1px solid var(--border) !important; margin-bottom: 0 !important; }
[data-baseweb="tab-list"] { background: transparent !important; gap: 0 !important; padding: 0 !important; }
[data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-2) !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em;
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

/* ── inputs ──────────────────────────────────────────────────────────── */
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
  outline: none !important;
}
.stTextInput label, .stTextArea label, .stNumberInput label,
.stSlider label, .stMultiSelect label, label {
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  color: var(--text-3) !important;
}

/* ── hero search input (override default sizing) ─────────────────────── */
[data-testid="tzai-hero-input"] input {
  font-size: 1.05rem !important;
  padding: 0.8rem 1rem !important;
  height: auto !important;
}

/* ── buttons ─────────────────────────────────────────────────────────── */
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
  border-color: var(--border-2) !important;
  color: var(--text-1) !important;
  background: var(--surface-3) !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #fff !important;
  font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover {
  background: #79b8ff !important;
  border-color: #79b8ff !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(88,166,255,0.25) !important;
}

/* ── multiselect ─────────────────────────────────────────────────────── */
[data-baseweb="select"] > div {
  background: var(--surface) !important;
  border: 1px solid var(--border-2) !important;
  border-radius: var(--radius) !important;
}
[data-baseweb="tag"] {
  background: var(--surface-3) !important;
  border: 1px solid var(--border-2) !important;
  border-radius: 9999px !important;
}
[data-baseweb="tag"] span { color: var(--text-1) !important; font-size: 0.78rem !important; }

/* ── expander ────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: transparent !important;
}
[data-testid="stExpander"] summary {
  color: var(--text-3) !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em;
}
[data-testid="stExpander"] summary:hover { color: var(--text-2) !important; }

/* ── metric ──────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 0.85rem 1rem !important;
}
[data-testid="stMetricLabel"] {
  font-size: 0.68rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
  color: var(--text-3) !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.5rem !important;
  font-weight: 800 !important;
  color: var(--text-1) !important;
  letter-spacing: -0.5px;
}

/* ── divider ─────────────────────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.75rem 0 !important; }

/* ── number input buttons ────────────────────────────────────────────── */
[data-testid="stNumberInput"] button {
  background: var(--surface-2) !important;
  border-color: var(--border-2) !important;
  color: var(--text-2) !important;
}

/* ── tables ──────────────────────────────────────────────────────────── */
table { width:100%; border-collapse:collapse; font-size:0.85rem; }
th {
  background: var(--surface-2) !important;
  color: var(--text-3) !important;
  font-size: 0.68rem !important;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  padding: 0.55rem 0.75rem !important;
  border-bottom: 1px solid var(--border-2) !important;
  text-align: left !important;
}
td {
  padding: 0.55rem 0.75rem !important;
  color: var(--text-2) !important;
  border-bottom: 1px solid var(--border) !important;
  font-size: 0.85rem;
}
tr:last-child td { border-bottom: none !important; }
tr:hover td { background: var(--surface-2) !important; }

/* ─────────────────────────────────────────────────────────────────────
   CUSTOM COMPONENTS
──────────────────────────────────────────────────────────────────────── */

/* ── wordmark ────────────────────────────────────────────────────────── */
.tzai-wordmark {
  display: flex; align-items: baseline; gap: 0.5rem;
  padding: 2.5rem 0 0.4rem;
}
.tzai-wordmark-logo {
  font-size: 1.5rem; line-height: 1;
}
.tzai-wordmark-name {
  font-size: 1.35rem;
  font-weight: 800;
  color: var(--text-1);
  letter-spacing: -0.5px;
}
.tzai-wordmark-tag {
  font-size: 0.8rem;
  color: var(--text-3);
  font-weight: 400;
  letter-spacing: 0;
}

/* ── example chips ───────────────────────────────────────────────────── */
.tzai-examples { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1.25rem; }
.tzai-chip {
  display: inline-block;
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: 9999px;
  padding: 0.3em 0.85em;
  font-size: 0.78rem;
  color: var(--text-2);
  cursor: pointer;
  transition: all .15s;
  user-select: none;
}
.tzai-chip:hover {
  background: var(--surface-2);
  border-color: var(--accent);
  color: var(--text-1);
}
.tzai-chip-label {
  font-size: 0.68rem;
  color: var(--text-3);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  margin-bottom: 0.5rem;
}

/* ── pill toggles for institutions ──────────────────────────────────── */
.tzai-pill-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1.25rem; }
.tzai-pill {
  display: inline-block;
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: 9999px;
  padding: 0.28em 0.85em;
  font-size: 0.78rem; font-weight: 600;
  color: var(--text-2); cursor: pointer;
  transition: all .15s;
}
.tzai-pill.on {
  background: var(--accent-dim);
  border-color: var(--accent);
  color: var(--accent);
}

/* ── search card ─────────────────────────────────────────────────────── */
.tzai-search-card {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: var(--radius-lg);
  padding: 1.25rem 1.25rem 1rem;
  margin-bottom: 1.5rem;
}

/* ── result: featured (top match) ───────────────────────────────────── */
.tzai-result-featured {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: var(--radius-lg);
  padding: 1.1rem 1.25rem;
  margin-bottom: 0.5rem;
  position: relative;
}
.tzai-result-featured.high { border-left: 3px solid var(--green); }
.tzai-result-featured.mid  { border-left: 3px solid var(--amber); }
.tzai-result-featured.low  { border-left: 3px solid var(--red);   }

.tzai-result-meta {
  display: flex; align-items: center;
  justify-content: space-between; gap: 0.75rem;
  margin-bottom: 0.35rem;
}
.tzai-result-code {
  font-family: ui-monospace, "SFMono-Regular", "Fira Code", monospace;
  font-size: 1.05rem; font-weight: 700;
  color: var(--text-1); letter-spacing: 0.03em;
}
.tzai-result-conf {
  font-size: 1.1rem; font-weight: 800;
  letter-spacing: -0.5px; flex-shrink: 0;
}
.tzai-result-conf.high { color: var(--green); }
.tzai-result-conf.mid  { color: var(--amber); }
.tzai-result-conf.low  { color: var(--text-3); }

.tzai-result-title {
  font-size: 0.85rem; color: var(--text-2);
  margin-bottom: 0.45rem; line-height: 1.4;
}
.tzai-result-verdict {
  font-size: 0.78rem; font-weight: 600;
  display: flex; align-items: center; gap: 0.35rem;
}
.tzai-result-verdict.high { color: var(--green); }
.tzai-result-verdict.mid  { color: var(--amber); }
.tzai-result-verdict.low  { color: var(--text-3); }

/* ── result: secondary (rank 2+) ─────────────────────────────────────── */
.tzai-result-sub {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.65rem 1rem;
  margin-bottom: 0.35rem;
  display: flex; align-items: center; gap: 0.75rem;
  opacity: 0.75;
  transition: opacity .15s;
}
.tzai-result-sub:hover { opacity: 1; }
.tzai-result-sub-code {
  font-family: ui-monospace, "SFMono-Regular", monospace;
  font-size: 0.85rem; font-weight: 700;
  color: var(--text-1); flex-shrink: 0; width: 90px;
}
.tzai-result-sub-title {
  font-size: 0.82rem; color: var(--text-2); flex: 1; min-width: 0;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.tzai-result-sub-conf {
  font-size: 0.78rem; font-weight: 700;
  flex-shrink: 0;
}
.tzai-result-sub-conf.high { color: var(--green); }
.tzai-result-sub-conf.mid  { color: var(--amber); }
.tzai-result-sub-conf.low  { color: var(--text-3); }

/* ── inst section heading ────────────────────────────────────────────── */
.tzai-inst-head {
  font-size: 0.68rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--text-3);
  margin: 1.5rem 0 0.75rem;
  display: flex; align-items: center; gap: 0.5rem;
}
.tzai-inst-head::after {
  content: ""; flex: 1;
  height: 1px; background: var(--border);
}

/* ── empty state ─────────────────────────────────────────────────────── */
.tzai-empty {
  text-align: center;
  padding: 3rem 1rem 2rem;
}
.tzai-empty-ico { font-size: 2rem; margin-bottom: 0.75rem; }
.tzai-empty-title {
  font-size: 0.95rem; font-weight: 600;
  color: var(--text-2); margin-bottom: 0.35rem;
}
.tzai-empty-sub {
  font-size: 0.82rem; color: var(--text-3);
}

/* ── summary banner ──────────────────────────────────────────────────── */
.tzai-banner {
  border-radius: var(--radius);
  padding: 0.85rem 1.1rem;
  margin-bottom: 1rem;
  font-size: 0.88rem; font-weight: 600;
  display: flex; align-items: center; gap: 0.6rem;
}
.tzai-banner.eligible   { background: var(--green-bg); color: var(--green); border: 1px solid rgba(63,185,80,0.25); }
.tzai-banner.borderline { background: var(--amber-bg); color: var(--amber); border: 1px solid rgba(210,153,34,0.25); }
.tzai-banner.ineligible { background: var(--red-bg);   color: var(--red);   border: 1px solid rgba(248,81,73,0.25); }

/* ── transcript row ──────────────────────────────────────────────────── */
.tzai-tr {
  display: flex; align-items: center; gap: 0.75rem;
  padding: 0.6rem 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.83rem;
}
.tzai-tr:last-child { border-bottom: none; }
.tzai-tr-source { color: var(--text-1); font-weight: 600; flex: 1 1 200px; min-width: 0; }
.tzai-tr-arrow  { color: var(--text-3); flex-shrink: 0; font-size: 0.75rem; }
.tzai-tr-target {
  font-family: ui-monospace, monospace;
  font-size: 0.8rem; color: var(--text-2);
  flex: 0 0 120px;
}
.tzai-tr-pct { font-weight: 700; font-size: 0.8rem; flex-shrink: 0; min-width: 36px; text-align: right; }
.tzai-tr-pct.high { color: var(--green); }
.tzai-tr-pct.mid  { color: var(--amber); }
.tzai-tr-pct.low  { color: var(--text-3); }

/* ── perf block ──────────────────────────────────────────────────────── */
.tzai-perf-inst { margin-bottom: 1.5rem; }
.tzai-perf-inst-name {
  font-size: 0.72rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--text-3); margin-bottom: 0.7rem;
}
.tzai-perf-row { margin-bottom: 0.6rem; }
.tzai-perf-top {
  display: flex; justify-content: space-between;
  font-size: 0.78rem; margin-bottom: 0.2rem;
}
.tzai-perf-top span { color: var(--text-2); }
.tzai-perf-top b    { font-weight: 700; }
.tzai-perf-track { background: var(--surface-3); border-radius: 9999px; height: 5px; }
.tzai-perf-fill  { height: 5px; border-radius: 9999px; transition: width .4s; }

/* ── section label ───────────────────────────────────────────────────── */
.tzai-label {
  font-size: 0.68rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--text-3); margin-bottom: 0.65rem;
}

/* ── col header (transcript table) ──────────────────────────────────── */
.tzai-col-hdr {
  font-size: 0.65rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--text-3); padding-bottom: 0.3rem;
}

/* ── model card ──────────────────────────────────────────────────────── */
.tzai-mc-pill {
  display: inline-block;
  background: var(--accent-dim);
  border: 1px solid rgba(88,166,255,0.2);
  border-radius: 9999px;
  padding: 0.2em 0.7em;
  font-size: 0.72rem; font-weight: 700;
  color: var(--accent); letter-spacing: 0.04em;
  margin-bottom: 1rem;
}
.tzai-mc-h {
  font-size: 0.68rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--accent); padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0.9rem;
}

/* ── skeleton loader ─────────────────────────────────────────────────── */
@keyframes shimmer {
  0%   { background-position: -600px 0; }
  100% { background-position:  600px 0; }
}
.tzai-skeleton {
  background: linear-gradient(90deg, var(--surface) 25%, var(--surface-2) 50%, var(--surface) 75%);
  background-size: 600px 100%;
  animation: shimmer 1.4s infinite;
  border-radius: var(--radius);
  height: 80px; margin-bottom: 0.5rem;
}
.tzai-skeleton-sm { height: 48px; }
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

# ── Helpers ───────────────────────────────────────────────────────────────────
def _cls(conf):
    if conf >= HIGH_CONFIDENCE_THRESHOLD: return "high"
    if conf >= TRANSFER_THRESHOLD:        return "mid"
    return "low"

def _verdict(conf, code):
    if conf >= HIGH_CONFIDENCE_THRESHOLD:
        return f'<span class="tzai-result-verdict high">✓ Transfers as {code}</span>'
    if conf >= TRANSFER_THRESHOLD:
        return '<span class="tzai-result-verdict mid">~ Advisor review recommended</span>'
    return '<span class="tzai-result-verdict low">– Low confidence</span>'

def render_results(results, selected):
    for inst_key in selected:
        name = INSTS.get(inst_key, inst_key)
        rows = results.get(inst_key, [])
        st.markdown(f'<div class="tzai-inst-head">{name}</div>', unsafe_allow_html=True)
        if not rows:
            st.markdown('<p style="color:var(--text-3);font-size:0.82rem;">No matches found.</p>',
                        unsafe_allow_html=True)
            continue

        # Top result — featured card
        r0  = rows[0]
        c0  = r0["confidence"]
        cls = _cls(c0)
        st.markdown(f"""
        <div class="tzai-result-featured {cls}">
          <div class="tzai-result-meta">
            <span class="tzai-result-code">{r0['code']}</span>
            <span class="tzai-result-conf {cls}">{c0:.0%}</span>
          </div>
          <div class="tzai-result-title">{r0['title']}</div>
          {_verdict(c0, r0['code'])}
        </div>""", unsafe_allow_html=True)
        with st.expander("Signal breakdown", expanded=False):
            sigs = r0["signals"]
            a, b, c, d = st.columns(4)
            a.metric("Semantic",  f"{sigs['bge_sim']:.3f}")
            b.metric("Lexical",   f"{sigs['tfidf_sim']:.3f}")
            c.metric("Dept Prior",f"{sigs['dept_prob']:.3f}")
            d.metric("Title",     f"{sigs['title_sim']:.3f}")

        # Subsequent results — compact rows
        if len(rows) > 1:
            sub_html = ""
            for r in rows[1:]:
                c = r["confidence"]
                sub_html += f"""
                <div class="tzai-result-sub">
                  <span class="tzai-result-sub-code">{r['code']}</span>
                  <span class="tzai-result-sub-title">{r['title']}</span>
                  <span class="tzai-result-sub-conf {_cls(c)}">{c:.0%}</span>
                </div>"""
            st.markdown(sub_html, unsafe_allow_html=True)


# ── Wordmark ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tzai-wordmark">
  <span class="tzai-wordmark-logo">🎓</span>
  <span class="tzai-wordmark-name">TransferzAI</span>
  <span class="tzai-wordmark-tag">transfer credit intelligence</span>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Lookup", "Transcript", "Model Card"])

EXAMPLES = [
    ("Calculus I", "MTH", "263"),
    ("General Chemistry", "CHM", "101"),
    ("Introduction to Programming", "CSC", "221"),
    ("College Composition", "ENG", "111"),
    ("Microeconomics", "ECO", "201"),
]

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — Lookup
# ════════════════════════════════════════════════════════════════════════
with tab1:

    # Institution pill toggles (stored in session state)
    if "sel_insts" not in st.session_state:
        st.session_state.sel_insts = list(INSTS.keys())

    st.markdown('<div class="tzai-label">Target institutions</div>', unsafe_allow_html=True)
    inst_cols = st.columns(len(INSTS))
    for i, (k, name) in enumerate(INSTS.items()):
        with inst_cols[i]:
            active = k in st.session_state.sel_insts
            if st.button(name, key=f"pill_{k}",
                         type="primary" if active else "secondary",
                         use_container_width=True):
                if active and len(st.session_state.sel_insts) > 1:
                    st.session_state.sel_insts.remove(k)
                elif not active:
                    st.session_state.sel_insts.append(k)
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Main search input
    # Pre-fill from example click
    if "example_title" not in st.session_state:
        st.session_state.example_title = ""
        st.session_state.example_dept  = ""
        st.session_state.example_num   = ""

    vccs_title = st.text_input(
        "Course Title",
        value=st.session_state.example_title,
        placeholder="What course are you looking up?  e.g. Calculus I",
        key="title_input",
    )

    # Advanced toggle
    show_adv = st.toggle("Advanced — department code & number", value=False)
    if show_adv:
        ac, nc = st.columns(2)
        with ac:
            vccs_dept = st.text_input("Department Code",
                value=st.session_state.example_dept,
                placeholder="ACC, MTH, CSC…")
        with nc:
            vccs_number = st.text_input("Course Number",
                value=st.session_state.example_num,
                placeholder="101, 211, 263…")
        vccs_desc = st.text_area("Description (optional)",
            placeholder="Paste catalog description for higher accuracy",
            height=72)
    else:
        vccs_dept = st.session_state.example_dept
        vccs_number = st.session_state.example_num
        vccs_desc = ""

    top_k = st.select_slider("Results per institution",
        options=[3, 5, 7, 10], value=5)

    search_clicked = st.button("Find Matches →", type="primary", key="search")

    st.markdown("<hr>", unsafe_allow_html=True)

    if search_clicked:
        if not vccs_title.strip():
            st.warning("Enter a course title to search.")
        else:
            with st.spinner(""):
                # Show skeleton while loading
                skeletons = st.empty()
                skeletons.markdown(
                    ''.join(['<div class="tzai-skeleton"></div>',
                             '<div class="tzai-skeleton tzai-skeleton-sm"></div>'] * 2),
                    unsafe_allow_html=True)
                results = predict_transfer(
                    vccs_dept, vccs_number, vccs_title, vccs_desc,
                    institutions=st.session_state.sel_insts, top_k=top_k,
                )
                skeletons.empty()
            render_results(results, st.session_state.sel_insts)
    else:
        st.markdown("""
        <div class="tzai-empty">
          <div class="tzai-empty-ico">⟳</div>
          <div class="tzai-empty-title">Enter a course title above</div>
          <div class="tzai-empty-sub">or try one of these examples</div>
        </div>""", unsafe_allow_html=True)

        # Clickable example chips — use columns as buttons
        st.markdown('<div class="tzai-label" style="text-align:center;">Examples</div>',
                    unsafe_allow_html=True)
        ex_cols = st.columns(len(EXAMPLES))
        for i, (title, dept, num) in enumerate(EXAMPLES):
            with ex_cols[i]:
                if st.button(title, key=f"ex_{i}", use_container_width=True):
                    st.session_state.example_title = title
                    st.session_state.example_dept  = dept
                    st.session_state.example_num   = num
                    st.rerun()


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — Transcript
# ════════════════════════════════════════════════════════════════════════
with tab2:
    rc, cc = st.columns([4, 1])
    with rc:
        sel_t2 = st.multiselect("Target institutions",
            options=list(INSTS.keys()),
            default=list(INSTS.keys()),
            format_func=lambda k: INSTS[k],
            key="t2_inst")
    with cc:
        min_credits = st.number_input("Min credits",
            value=MIN_CREDITS_REQUIRED, min_value=1, max_value=120)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="tzai-label">Your courses</div>', unsafe_allow_html=True)

    DEFAULT_COURSES = [
        ("ACC", "211", "PRINCIPLES OF ACCOUNTING I"),
        ("ENG", "111", "COLLEGE COMPOSITION I"),
        ("MTH", "263", "CALCULUS I"),
        ("BIO", "101", "GENERAL BIOLOGY I"),
        ("CSC", "221", "INTRODUCTION TO PROBLEM SOLVING AND PROGRAMMING"),
    ]
    n = st.number_input("Number of courses", 1, 30, len(DEFAULT_COURSES), key="n_courses")

    hc = st.columns([1, 1, 3, 3, 1])
    for col, lbl in zip(hc, ["Dept", "No.", "Title", "Description", "Cr"]):
        col.markdown(f'<div class="tzai-col-hdr">{lbl}</div>', unsafe_allow_html=True)

    course_inputs = []
    for i in range(int(n)):
        defs = DEFAULT_COURSES[i] if i < len(DEFAULT_COURSES) else ("", "", "")
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

    eval_clicked = st.button("Evaluate Transcript →", type="primary", key="eval")

    st.markdown("<hr>", unsafe_allow_html=True)

    if eval_clicked:
        if not course_inputs:
            st.warning("Enter at least one course title.")
        elif not sel_t2:
            st.warning("Select at least one institution.")
        else:
            with st.spinner(f"Evaluating {len(course_inputs)} courses…"):
                results = evaluate_transcript(course_inputs, institutions=sel_t2,
                                              min_credits_required=min_credits)
            for inst_key in sel_t2:
                r = results.get(inst_key)
                if not r: continue
                status = "eligible" if r["eligible"] else ("borderline" if r["borderline"] else "ineligible")
                icon   = "✓" if r["eligible"] else ("~" if r["borderline"] else "✕")
                st.markdown(
                    f'<div class="tzai-banner {status}">'
                    f'<span>{icon}</span>'
                    f'<span>{r["institution_name"]} — {r["summary"]}</span>'
                    f'</div>', unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Courses",           r["total_courses"])
                m2.metric("Total Credits",     r["total_credits"])
                m3.metric("Confirmed",         r["transferable_credits_confident"])
                m4.metric("Possible",          r["transferable_credits_possible"])

                st.markdown('<div class="tzai-label" style="margin-top:1rem;">Per-course</div>',
                            unsafe_allow_html=True)
                rows_html = ""
                for cr in r["course_results"]:
                    conf = cr["confidence"]
                    cls  = _cls(conf)
                    src  = " ".join(filter(None, [cr["dept"], cr["number"], cr["title"]])).strip()
                    tgt  = cr["best_match"] or "—"
                    rows_html += f"""<div class="tzai-tr">
                      <span class="tzai-tr-source">{src}</span>
                      <span class="tzai-tr-arrow">→</span>
                      <span class="tzai-tr-target">{tgt}</span>
                      <span class="tzai-tr-pct {cls}">{conf:.0%}</span>
                    </div>"""
                st.markdown(rows_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="tzai-empty">
          <div class="tzai-empty-ico">📋</div>
          <div class="tzai-empty-title">Fill in your courses</div>
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

        st.markdown('<div class="tzai-mc-h" style="margin-top:1.5rem;">Thresholds</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
| | Threshold | Action |
|:---|:---|:---|
| 🟢 High Confidence | ≥ {HIGH_CONFIDENCE_THRESHOLD:.0%} | Transfers as X — ≥ 90% precision |
| 🟡 Possible Match  | ≥ {TRANSFER_THRESHOLD:.0%} | Advisor review recommended |
| 🔴 Low Confidence  | < {TRANSFER_THRESHOLD:.0%} | Refer to registrar |
        """)

        st.markdown('<div class="tzai-mc-h" style="margin-top:1.5rem;">Training data</div>',
                    unsafe_allow_html=True)
        st.markdown("""
| Pair | Train | Test |
|:---|:---:|:---:|
| VCCS → W&M | ~267 | 67 |
| VCCS → VT | ~242 | 60 |
| CCC → UCSC | ~723 | 181 |

FERPA compliant — no student PII.
        """)
        catalog_md = "  \n".join(
            f"**{v}** — {len(artifacts['institutions'][k]['codes']):,} courses"
            for k, v in INSTS.items()
        )
        st.markdown(catalog_md)

    with right:
        st.markdown('<div class="tzai-mc-h">Performance</div>', unsafe_allow_html=True)
        sc = artifacts.get("scorecard", {})

        if sc and any(k in sc for k in ("wm", "vt", "ucsc")):
            for inst_key, label in [("wm", "William & Mary"),
                                     ("vt", "Virginia Tech"),
                                     ("ucsc", "CCC → UCSC")]:
                s = sc.get(inst_key, {})
                if not s: continue
                t1 = s.get("top1_lr", 0); t3 = s.get("top3_lr", 0)
                pr = s.get("precision_tau", 0); cov = s.get("coverage_tau", 0)

                def bar(v, good, warn=0.5):
                    col = "#3fb950" if v >= good else ("#d29922" if v >= warn else "#f85149")
                    return (f'<div class="tzai-perf-row">'
                            f'<div class="tzai-perf-top">'
                            f'<span>{label_}</span>'
                            f'<b style="color:{col}">{v:.1%}</b></div>'
                            f'<div class="tzai-perf-track">'
                            f'<div class="tzai-perf-fill" style="width:{v*100:.1f}%;background:{col}"></div>'
                            f'</div></div>')

                html = f'<div class="tzai-perf-inst"><div class="tzai-perf-inst-name">{label}</div>'
                for label_, v_, good_, warn_ in [
                    ("Top-1 Recall", t1, 0.55, 0.45),
                    ("Top-3 Recall", t3, 0.75, 0.60),
                    ("Precision @ τ", pr, 0.90, 0.75),
                    ("Coverage @ τ", cov, 0.20, 0.10),
                ]:
                    col_ = "#3fb950" if v_ >= good_ else ("#d29922" if v_ >= warn_ else "#f85149")
                    html += (f'<div class="tzai-perf-row">'
                             f'<div class="tzai-perf-top">'
                             f'<span>{label_}</span>'
                             f'<b style="color:{col_}">{v_:.1%}</b></div>'
                             f'<div class="tzai-perf-track">'
                             f'<div class="tzai-perf-fill" style="width:{v_*100:.1f}%;background:{col_}"></div>'
                             f'</div></div>')
                html += f'</div>'
                st.markdown(html, unsafe_allow_html=True)
                st.caption(f"Brier {s.get('brier',0):.4f} · ECE {s.get('ece',0):.4f} · τ={s.get('op_tau',0):.2f}")
        else:
            st.info("Run `python scripts/build_artifacts.py` to populate metrics.")

        st.markdown('<div class="tzai-mc-h" style="margin-top:1rem;">Limitations</div>',
                    unsafe_allow_html=True)
        st.markdown("""
- Coverage at high precision ~15–25% — the system says "low confidence" rather than guessing wrong
- CCC→UCSC top-3 ~68% — smaller training set, 115+ colleges
- Dept/number optional — missing codes reduce confidence slightly
- Not a registrar decision — always confirm before acting
        """)
