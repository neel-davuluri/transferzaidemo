"""
Compare three training conditions on W&M and VT held-out test sets:
  A: Fine-tuned on W&M only (267 pairs, existing checkpoint)
  B: Fine-tuned on W&M + VT (existing combined checkpoint)
  C: Fine-tuned on W&M + VT + CCC→UCSC (new)
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import WM_MERGED, WM_CATALOG, VT_MERGED, VT_CATALOG, CCC_UCSC_CLEAN, UCSC_CATALOG

import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

np.random.seed(42)

QUERY_PREFIX = "Represent this course for finding transfer equivalents: "

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

_CCC_BOILERPLATE = re.compile(
    r"may be offered in a distance.learning format"
    r"|transfer information:"
    r"|transferable to (?:both )?uc"
    r"|limitations on enrollment:"
    r"|requisites:"
    r"|minimum credit units"
    r"|maximum credit units"
    r"|toggle (?:additional|general|learning)"
    r"|grade options:"
    r"|see open sections"
    r"|connect with an academic counselor"
    r"|some of the class hours for this course",
    re.IGNORECASE,
)

def clean_text(text):
    if pd.isna(text) or str(text) in ("Description not found", "nan", ""): return ""
    text = str(text)
    # Strip CCC boilerplate before lowercasing (markers are mixed case)
    m = _CCC_BOILERPLATE.search(text)
    if m:
        text = text[:m.start()]
    text = text.lower()
    text = re.sub(r"prerequisite\(s\):.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(csi|alv|nqr|additional)\b", "", text)
    text = re.sub(r"cross-listed with:.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ── W&M ─────────────────────────────────────────────────────────
df_wm = pd.read_csv(WM_MERGED)
wm_catalog = pd.read_csv(WM_CATALOG, encoding="latin-1")
df_wm.columns = df_wm.columns.str.strip()
df_wm = df_wm.rename(columns={"Unnamed: 0": "idx"})

has_match = df_wm["W&M Course Code"].notna() & (df_wm["W&M Course Code"].str.strip() != "")
pos_df = df_wm[has_match].copy()

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

def parse_vccs_course(raw):
    raw = str(raw).strip()
    parts = re.split(r"\s*TAKEN WITH\s*", raw, flags=re.IGNORECASE)
    courses = []
    for part in parts:
        m = re.match(r"([A-Z]{2,4})\s+(\d{3})\s+(.*)", part.strip())
        if m:
            courses.append({"dept": m.group(1), "number": int(m.group(2)), "title": m.group(3).strip()})
    return courses

def get_vccs_text(row, course_col="VCCS Course", desc_col="VCCS Description"):
    courses = parse_vccs_course(row[course_col])
    titles = " ".join(c["title"] for c in courses)
    desc = clean_text(row.get(desc_col, ""))
    return f"{titles} {desc}"

def parse_wm_course(code_str):
    if pd.isna(code_str): return None
    m = re.match(r"([A-Z]{2,4})\s+(\d{3})", str(code_str).strip())
    return {"dept": m.group(1), "number": int(m.group(2))} if m else None

pos_df["wm_dept"] = pos_df["W&M Course Code"].apply(
    lambda x: parse_wm_course(x)["dept"] if parse_wm_course(x) else "UNK")
dept_counts = pos_df["wm_dept"].value_counts()
rare_depts = dept_counts[dept_counts < 2].index.tolist()
pos_df["strat_dept"] = pos_df["wm_dept"].apply(lambda d: "RARE" if d in rare_depts else d)
wm_train, wm_test = train_test_split(pos_df, test_size=0.20, random_state=42, stratify=pos_df["strat_dept"])

# ── VT ──────────────────────────────────────────────────────────
df_vt = pd.read_csv(VT_MERGED)

def vt_dept(code_str):
    m = re.match(r"([A-Z]+)\s+\d+", str(code_str).strip())
    return m.group(1) if m else "UNK"

df_vt["vt_dept"] = df_vt["VT Course Code"].apply(lambda x: vt_dept(x.split("+")[0].strip()))
vt_dept_counts = df_vt["vt_dept"].value_counts()
rare_vt = vt_dept_counts[vt_dept_counts < 2].index.tolist()
df_vt["strat_vt"] = df_vt["vt_dept"].apply(lambda d: "RARE" if d in rare_vt else d)
vt_train, vt_test = train_test_split(df_vt, test_size=0.20, random_state=42, stratify=df_vt["strat_vt"])

# ── CCC→UCSC ────────────────────────────────────────────────────
df_ccc = pd.read_csv(CCC_UCSC_CLEAN)
df_ccc.columns = df_ccc.columns.str.strip()

# Split CCC course field into code+title (format: "CHEM 1A General Chemistry")
def parse_ccc_title(course_str):
    m = re.match(r"^([A-Z][A-Z0-9 ]*?\s+\S+)\s+(.*)", str(course_str).strip())
    return m.group(2).strip() if m else str(course_str).strip()

def get_ccc_text(row):
    title = parse_ccc_title(row["CCC Course"])
    desc = clean_text(row.get("CCC Description", ""))
    return f"{title} {desc}".strip()

def get_ucsc_text(row):
    title = str(row.get("UCSC Course Title", ""))
    desc = clean_text(row.get("UCSC Description", ""))
    return f"{title} {desc}".strip()

# 80/20 split stratified by UCSC dept prefix
def ucsc_dept(code_str):
    m = re.match(r"([A-Z]+)\s+", str(code_str).strip())
    return m.group(1) if m else "UNK"

df_ccc["ucsc_dept"] = df_ccc["UCSC Course Code"].apply(ucsc_dept)
ucsc_dept_counts = df_ccc["ucsc_dept"].value_counts()
rare_ucsc = ucsc_dept_counts[ucsc_dept_counts < 2].index.tolist()
df_ccc["strat_ucsc"] = df_ccc["ucsc_dept"].apply(lambda d: "RARE" if d in rare_ucsc else d)
ccc_train, ccc_test = train_test_split(df_ccc, test_size=0.20, random_state=42, stratify=df_ccc["strat_ucsc"])

print(f"W&M   train: {len(wm_train)}   test: {len(wm_test)}")
print(f"VT    train: {len(vt_train)}   test: {len(vt_test)}")
print(f"CCC   train: {len(ccc_train)}  test: {len(ccc_test)}")
print(f"Total training positives (W&M+VT+CCC): {len(wm_train)+len(vt_train)+len(ccc_train)}")


# ══════════════════════════════════════════════════════════════
# CATALOG INDEXES
# ══════════════════════════════════════════════════════════════

wm_codes = list(wm_lookup.keys())
wm_texts = [f"{wm_lookup[c]['title']} {clean_text(wm_lookup[c]['description'])}" for c in wm_codes]
print(f"\nW&M catalog: {len(wm_codes)} courses")

# TF-IDF fit on all catalog texts (W&M + VT + CCC/UCSC) for better vocabulary coverage
# Full VT catalog — same approach as W&M uses wm_courses_2025.csv
vt_full = pd.read_csv(VT_CATALOG)
vt_full = vt_full.dropna(subset=["course_code"])
vt_catalog = {}
for _, row in vt_full.iterrows():
    code = str(row["course_code"]).strip()
    vt_catalog[code] = {
        "title": str(row.get("course_title", "")),
        "description": str(row.get("course_description", "")) if pd.notna(row.get("course_description")) else "",
    }
vt_codes_list = list(vt_catalog.keys())
vt_texts_list = [f"{vt_catalog[c]['title']} {clean_text(vt_catalog[c]['description'])}" for c in vt_codes_list]

# Full UCSC catalog — same approach as W&M uses wm_courses_2025.csv
ucsc_full = pd.read_csv(UCSC_CATALOG)
ucsc_full = ucsc_full.dropna(subset=["course_code"])
ucsc_catalog = {}
for _, row in ucsc_full.iterrows():
    code = str(row["course_code"]).strip()
    ucsc_catalog[code] = {
        "title": str(row.get("course_title", "")),
        "description": str(row.get("course_description", "")) if pd.notna(row.get("course_description")) else "",
    }
ucsc_codes_list = list(ucsc_catalog.keys())
ucsc_texts_list = [f"{ucsc_catalog[c]['title']} {clean_text(ucsc_catalog[c]['description'])}" for c in ucsc_codes_list]

tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=1, max_df=0.95,
                        stop_words="english", sublinear_tf=True)
tfidf.fit(wm_texts + vt_texts_list + ucsc_texts_list)
print(f"TF-IDF vocab: {len(tfidf.vocabulary_)} terms")


# ══════════════════════════════════════════════════════════════
# RETRIEVAL EVAL FUNCTIONS
# ══════════════════════════════════════════════════════════════

def eval_wm(model, label, eval_df=None):
    if eval_df is None: eval_df = wm_test
    embs = model.encode(wm_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    tfidf_mat = tfidf.transform(wm_texts)

    hits = {k: 0 for k in [1, 3, 5, 10, 20, 50]}
    for _, row in eval_df.iterrows():
        vccs_text = get_vccs_text(row)
        target = str(row["W&M Course Code"]).strip()
        q_emb = model.encode([f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True)[0]
        bge_sims = embs @ q_emb
        bge_ranked = np.argsort(bge_sims)[::-1]
        tfidf_sims = cosine_similarity(tfidf.transform([vccs_text]), tfidf_mat).flatten()
        tfidf_ranked = np.argsort(tfidf_sims)[::-1]
        rrf = defaultdict(float)
        for rank, idx in enumerate(bge_ranked[:200]):
            rrf[wm_codes[idx]] += 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(tfidf_ranked[:200]):
            rrf[wm_codes[idx]] += 1.0 / (60 + rank + 1)
        ranked = [c for c, _ in sorted(rrf.items(), key=lambda x: -x[1])]
        for k in hits:
            if target in ranked[:k]: hits[k] += 1

    n = len(eval_df)
    r = {k: hits[k] / n for k in hits}
    print(f"\n  {label} (n={n}):")
    print(f"    Top-1: {r[1]:.3f}  Top-3: {r[3]:.3f}  Top-5: {r[5]:.3f}  Top-10: {r[10]:.3f}  Top-50: {r[50]:.3f}")
    return r


def eval_vt(model, label, eval_df=None):
    if eval_df is None: eval_df = vt_test
    vt_embs = model.encode(vt_texts_list, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    tfidf_mat = tfidf.transform(vt_texts_list)

    hits = {k: 0 for k in [1, 3, 5, 10, 20]}
    found = 0
    for _, row in eval_df.iterrows():
        vccs_text = get_vccs_text(row)
        target = row["VT Course Code"].split("+")[0].strip()
        if target not in vt_codes_list: continue
        found += 1
        q_emb = model.encode([f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True)[0]
        bge_sims = vt_embs @ q_emb
        bge_ranked = np.argsort(bge_sims)[::-1]
        tfidf_sims = cosine_similarity(tfidf.transform([vccs_text]), tfidf_mat).flatten()
        tfidf_ranked = np.argsort(tfidf_sims)[::-1]
        rrf = defaultdict(float)
        for rank, idx in enumerate(bge_ranked[:200]):
            rrf[vt_codes_list[idx]] += 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(tfidf_ranked[:200]):
            rrf[vt_codes_list[idx]] += 1.0 / (60 + rank + 1)
        ranked = [c for c, _ in sorted(rrf.items(), key=lambda x: -x[1])]
        for k in hits:
            if target in ranked[:k]: hits[k] += 1

    if found == 0:
        print(f"\n  {label}: no evaluable rows"); return {}
    r = {k: hits[k] / found for k in hits}
    print(f"\n  {label} (n={found}):")
    print(f"    Top-1: {r[1]:.3f}  Top-3: {r[3]:.3f}  Top-5: {r[5]:.3f}  Top-10: {r[10]:.3f}  Top-20: {r[20]:.3f}")
    return r


def eval_ucsc(model, label, eval_df=None):
    """Evaluate CCC→UCSC retrieval on the held-out CCC test set."""
    if eval_df is None: eval_df = ccc_test
    ucsc_embs = model.encode(ucsc_texts_list, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    tfidf_mat = tfidf.transform(ucsc_texts_list)

    hits = {k: 0 for k in [1, 3, 5, 10, 20]}
    found = 0
    for _, row in eval_df.iterrows():
        query_text = get_ccc_text(row)
        target = str(row["UCSC Course Code"]).strip()
        if target not in ucsc_codes_list: continue
        found += 1
        q_emb = model.encode([f"{QUERY_PREFIX}{query_text}"], normalize_embeddings=True)[0]
        bge_sims = ucsc_embs @ q_emb
        bge_ranked = np.argsort(bge_sims)[::-1]
        tfidf_sims = cosine_similarity(tfidf.transform([query_text]), tfidf_mat).flatten()
        tfidf_ranked = np.argsort(tfidf_sims)[::-1]
        rrf = defaultdict(float)
        for rank, idx in enumerate(bge_ranked[:200]):
            rrf[ucsc_codes_list[idx]] += 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(tfidf_ranked[:200]):
            rrf[ucsc_codes_list[idx]] += 1.0 / (60 + rank + 1)
        ranked = [c for c, _ in sorted(rrf.items(), key=lambda x: -x[1])]
        for k in hits:
            if target in ranked[:k]: hits[k] += 1

    if found == 0:
        print(f"\n  {label}: no evaluable rows"); return {}
    r = {k: hits[k] / found for k in hits}
    print(f"\n  {label} (n={found}):")
    print(f"    Top-1: {r[1]:.3f}  Top-3: {r[3]:.3f}  Top-5: {r[5]:.3f}  Top-10: {r[10]:.3f}  Top-20: {r[20]:.3f}")
    return r


# ══════════════════════════════════════════════════════════════
# CONDITION A — W&M-only baseline
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CONDITION A: W&M-only fine-tuned BGE (267 pairs)")
print("=" * 60)
model_a = SentenceTransformer("./finetuned_bge", device="cpu")
r_a_wm   = eval_wm(model_a,   "A: W&M  test")
r_a_vt   = eval_vt(model_a,   "A: VT   test")
r_a_ucsc = eval_ucsc(model_a, "A: UCSC test")


# ══════════════════════════════════════════════════════════════
# CONDITION B — W&M + VT (previously trained)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"CONDITION B: W&M ({len(wm_train)}) + VT ({len(vt_train)}) = {len(wm_train)+len(vt_train)} pairs")
print("=" * 60)
model_b = SentenceTransformer("./finetuned_bge_combined", device="cpu")
r_b_wm   = eval_wm(model_b,   "B: W&M  test")
r_b_vt   = eval_vt(model_b,   "B: VT   test")
r_b_ucsc = eval_ucsc(model_b, "B: UCSC test")


# ══════════════════════════════════════════════════════════════
# CONDITION C — W&M + VT + CCC (new)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
n_total = len(wm_train) + len(vt_train) + len(ccc_train)
print(f"CONDITION C: W&M ({len(wm_train)}) + VT ({len(vt_train)}) + CCC ({len(ccc_train)}) = {n_total} pairs")
print("=" * 60)

wm_train_ex, vt_train_ex, ccc_train_ex = [], [], []

for _, row in wm_train.iterrows():
    q = get_vccs_text(row)
    wm_info = wm_lookup.get(str(row["W&M Course Code"]).strip(), {})
    t = f"{wm_info.get('title', '')} {clean_text(wm_info.get('description', ''))}"
    if q.strip() and t.strip():
        wm_train_ex.append(InputExample(texts=[f"{QUERY_PREFIX}{q}", t]))

for _, row in vt_train.iterrows():
    q = get_vccs_text(row)
    t = f"{row.get('VT Course Title', '')} {clean_text(str(row.get('VT Description', '')).split('|')[0])}"
    if q.strip() and t.strip():
        vt_train_ex.append(InputExample(texts=[f"{QUERY_PREFIX}{q}", t]))

for _, row in ccc_train.iterrows():
    q = get_ccc_text(row)
    t = get_ucsc_text(row)
    if q.strip() and t.strip():
        ccc_train_ex.append(InputExample(texts=[f"{QUERY_PREFIX}{q}", t]))

train_examples = wm_train_ex + vt_train_ex + ccc_train_ex

print(f"Training examples: {len(train_examples)}")

# Free memory from condition A/B models before training
import gc
try:
    del model_a, model_b
except NameError:
    pass
gc.collect()
try:
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
except Exception:
    pass

# Train from base BGE on ALL three datasets (W&M + VT + CCC) jointly.
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import random, math

model_c = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
model_c.train()

# Raw training loop — bypasses .fit() widget pre-computation entirely
# Uses in-batch negatives (same as MNRL)
all_examples = wm_train_ex + vt_train_ex + ccc_train_ex  # all 1218 pairs

epochs    = 3
bs        = 16
optimizer = AdamW(model_c.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = math.ceil(len(all_examples) / bs) * epochs
warmup_steps = max(1, int(total_steps * 0.1))
scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
scale = torch.tensor(20.0)   # temperature scaling for cosine similarity

print(f"Raw training loop: {epochs} epochs, bs={bs}, {total_steps} steps total ({len(all_examples)} examples)")

step = 0
for epoch in range(epochs):
    random.shuffle(all_examples)
    for i in range(0, len(all_examples), bs):
        batch = all_examples[i:i+bs]
        if len(batch) < 2:
            continue

        # Tokenize anchors and positives separately
        anchors_text  = [ex.texts[0] for ex in batch]
        positives_text = [ex.texts[1] for ex in batch]

        def encode_texts(texts):
            feats = model_c.tokenize(texts)
            feats = {k: v.to("cpu") for k, v in feats.items()}
            out = model_c(feats)
            embs = out["sentence_embedding"]
            return torch.nn.functional.normalize(embs, p=2, dim=1)

        a_embs = encode_texts(anchors_text)
        p_embs = encode_texts(positives_text)

        # In-batch negatives cross-entropy
        scores = torch.mm(a_embs, p_embs.T) * scale
        labels = torch.arange(len(batch))
        loss = torch.nn.CrossEntropyLoss()(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_c.parameters(), 1.0)
        optimizer.step()
        if step < warmup_steps:
            scheduler.step()
        step += 1

    print(f"  Epoch {epoch+1}/{epochs} done, last_loss={loss.item():.4f}")

model_c.save("./finetuned_bge_three")
print("Training complete.")
print("Done fine-tuning.")

r_c_wm   = eval_wm(model_c,   "C: W&M  test")
r_c_vt   = eval_vt(model_c,   "C: VT   test")
r_c_ucsc = eval_ucsc(model_c, "C: UCSC test")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("RESULTS SUMMARY — W&M held-out test set")
print("=" * 72)
print(f"  {'Condition':<44} {'Top-1':>6} {'Top-3':>6} {'Top-5':>6} {'Top-10':>7} {'Top-50':>7}")
print(f"  {'─'*70}")
for label, r in [
    (f"A: W&M only  ({len(wm_train)} pairs)",                    r_a_wm),
    (f"B: W&M+VT   ({len(wm_train)+len(vt_train)} pairs)",       r_b_wm),
    (f"C: W&M+VT+CCC ({n_total} pairs)",                         r_c_wm),
]:
    print(f"  {label:<44} {r[1]:>6.3f} {r[3]:>6.3f} {r[5]:>6.3f} {r[10]:>7.3f} {r[50]:>7.3f}")

print(f"\n  Deltas vs A:")
for label, r in [("B - A", r_b_wm), ("C - A", r_c_wm)]:
    deltas = {k: r[k] - r_a_wm[k] for k in [1, 3, 5, 10, 50]}
    print(f"    {label}: " + "  ".join(f"Top-{k}: {deltas[k]:+.3f}" for k in [1, 3, 5, 10, 50]))

print("\n" + "=" * 72)
print("RESULTS SUMMARY — VT held-out test set")
print("=" * 72)
print(f"  {'Condition':<44} {'Top-1':>6} {'Top-3':>6} {'Top-5':>6} {'Top-10':>7}")
print(f"  {'─'*64}")
for label, r in [
    (f"A: W&M only",    r_a_vt),
    (f"B: W&M+VT",      r_b_vt),
    (f"C: W&M+VT+CCC",  r_c_vt),
]:
    if r:
        print(f"  {label:<44} {r[1]:>6.3f} {r[3]:>6.3f} {r[5]:>6.3f} {r[10]:>7.3f}")

print("\n" + "=" * 72)
print("RESULTS SUMMARY — UCSC held-out test set (new, CCC→UCSC)")
print("=" * 72)
print(f"  {'Condition':<44} {'Top-1':>6} {'Top-3':>6} {'Top-5':>6} {'Top-10':>7}")
print(f"  {'─'*64}")
for label, r in [
    (f"A: W&M only (never saw CCC data)",  r_a_ucsc),
    (f"B: W&M+VT   (never saw CCC data)",  r_b_ucsc),
    (f"C: W&M+VT+CCC (trained on CCC)",    r_c_ucsc),
]:
    if r:
        print(f"  {label:<44} {r[1]:>6.3f} {r[3]:>6.3f} {r[5]:>6.3f} {r[10]:>7.3f}")
