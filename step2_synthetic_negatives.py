"""
Step 2 — Hard synthetic negative generation via Claude API (claude-haiku-4-5)
For each positive pair in train_pos, generate 1 synthetic VCCS course description
that is in the same department, sounds similar, but should NOT transfer.
"""

import re, json, time
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split
import anthropic

np.random.seed(42)

# ── Replicate Step 1 split ──────────────────────────────────────────────
df = pd.read_csv("vccs_wm_merged.csv")
wm_catalog = pd.read_csv("wm_courses_2025.csv", encoding="latin-1")
df.columns = df.columns.str.strip()
df = df.rename(columns={"Unnamed: 0": "idx"})

has_match = df["W&M Course Code"].notna() & (df["W&M Course Code"].str.strip() != "")
pos_df = df[has_match].copy()
neg_df = df[~has_match].copy()

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
            if pd.isna(row.get("W&M Course Title")) or row.get("W&M Course Title") == "":
                pos_df.at[idx, "W&M Course Title"] = wm_lookup[wm_code]["title"]


def parse_wm_course(code_str):
    if pd.isna(code_str):
        return None
    m = re.match(r"([A-Z]{2,4})\s+(\d{3})", str(code_str).strip())
    if m:
        return {"dept": m.group(1), "number": int(m.group(2)), "full": f"{m.group(1)} {m.group(2)}"}
    return None


def parse_vccs_course(raw):
    raw = str(raw).strip()
    parts = re.split(r"\s*TAKEN WITH\s*", raw, flags=re.IGNORECASE)
    courses = []
    for part in parts:
        m = re.match(r"([A-Z]{2,4})\s+(\d{3})\s+(.*)", part.strip())
        if m:
            courses.append({
                "dept": m.group(1), "number": int(m.group(2)),
                "title": m.group(3).strip(), "full": f"{m.group(1)} {m.group(2)}",
            })
    return courses


pos_df["wm_dept"] = pos_df["W&M Course Code"].apply(
    lambda x: parse_wm_course(x)["dept"] if parse_wm_course(x) else "UNK"
)
dept_counts = pos_df["wm_dept"].value_counts()
rare_depts = dept_counts[dept_counts < 2].index.tolist()
pos_df["strat_dept"] = pos_df["wm_dept"].apply(lambda d: "RARE" if d in rare_depts else d)

train_pos, test_pos = train_test_split(
    pos_df, test_size=0.20, random_state=42, stratify=pos_df["strat_dept"]
)

print(f"train_pos: {len(train_pos)}  |  test_pos: {len(test_pos)} (untouched)")

# ── Build dept_map from train_pos only ──────────────────────────────────
dept_map = defaultdict(lambda: defaultdict(int))
for _, r in train_pos.iterrows():
    vccs_courses = parse_vccs_course(r["VCCS Course"])
    wm_parsed = parse_wm_course(r["W&M Course Code"])
    if vccs_courses and wm_parsed:
        for vc in vccs_courses:
            dept_map[vc["dept"]][wm_parsed["dept"]] += 1


# ── Generate synthetic negatives via Claude API ─────────────────────────
client = anthropic.Anthropic()

BATCH_SIZE = 10  # pairs per API call to reduce total calls

# Prepare batches
train_rows = list(train_pos.iterrows())
batches = [train_rows[i:i + BATCH_SIZE] for i in range(0, len(train_rows), BATCH_SIZE)]

synthetic_negatives = []
cache_file = Path("_cache_synthetic_negatives.json")

# Resume from cache if partial run
if cache_file.exists():
    with open(cache_file) as f:
        synthetic_negatives = json.load(f)
    print(f"Loaded {len(synthetic_negatives)} cached synthetic negatives")
    start_batch = len(synthetic_negatives) // BATCH_SIZE
else:
    start_batch = 0

print(f"\nGenerating synthetic negatives: {len(batches)} batches of {BATCH_SIZE}")
print(f"Starting from batch {start_batch}")

for batch_idx in range(start_batch, len(batches)):
    batch = batches[batch_idx]

    # Build prompt with batch of pairs
    pair_descriptions = []
    for i, (_, row) in enumerate(batch):
        vccs_course = str(row["VCCS Course"])
        vccs_desc = str(row.get("VCCS Description", ""))
        wm_code = str(row["W&M Course Code"]).strip()
        wm_title = str(row.get("W&M Course Title", ""))

        vccs_parsed = parse_vccs_course(vccs_course)
        vccs_dept = vccs_parsed[0]["dept"] if vccs_parsed else "GEN"

        pair_descriptions.append(
            f"Pair {i+1}:\n"
            f"  VCCS dept: {vccs_dept}\n"
            f"  VCCS course: {vccs_course}\n"
            f"  VCCS description: {vccs_desc[:300]}\n"
            f"  W&M match: {wm_code} {wm_title}"
        )

    prompt = f"""You are generating hard synthetic negatives for a college transfer credit ML model.

For each pair below, generate exactly 1 synthetic VCCS course description that:
1. Is in the SAME VCCS department as the real course
2. Sounds semantically similar to the real course (uses related terminology)
3. Should NOT transfer to the given W&M course — because it covers a different level, narrower scope, is lab-only, lacks core theoretical content, or focuses on applied/vocational skills not equivalent to the university course

Return ONLY a JSON array with one object per pair. Each object must have:
- "pair_num": integer (1-indexed)
- "synthetic_course_name": string (format: "DEPT NUM TITLE", e.g. "BIO 145 ANATOMY LAB TECHNIQUES")
- "synthetic_description": string (2-3 sentences, realistic community college catalog style)
- "reason_no_transfer": string (brief reason why this should NOT transfer)

{chr(10).join(pair_descriptions)}

Return ONLY the JSON array, no other text."""

    for attempt in range(3):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Extract JSON from response
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            results = json.loads(text)

            for item in results:
                synthetic_negatives.append({
                    "synthetic_course_name": item["synthetic_course_name"],
                    "synthetic_description": item["synthetic_description"],
                    "reason_no_transfer": item.get("reason_no_transfer", ""),
                })
            break
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  Batch {batch_idx+1}: parse error (attempt {attempt+1}): {e}")
            if attempt == 2:
                print(f"  SKIPPING batch {batch_idx+1} after 3 failures")
        except anthropic.RateLimitError:
            print(f"  Rate limited, waiting 30s...")
            time.sleep(30)

    # Save checkpoint
    with open(cache_file, "w") as f:
        json.dump(synthetic_negatives, f)

    if (batch_idx + 1) % 5 == 0:
        print(f"  Completed {batch_idx+1}/{len(batches)} batches ({len(synthetic_negatives)} negatives)")

print(f"\nTotal synthetic negatives generated: {len(synthetic_negatives)}")

# ── Also generate retriever-mined hard negatives (from v2 pipeline) ─────
# We need embeddings for this — import sentence_transformers
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    if pd.isna(text) or text == "Description not found" or text == "nan":
        return ""
    text = str(text).lower()
    text = re.sub(r"prerequisite\(s\):.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(CSI|ALV|NQR|Additional)\b", "", text)
    text = re.sub(r"cross-listed with:.*?(?=\.\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_vccs_text(row):
    courses = parse_vccs_course(row["VCCS Course"])
    titles = " ".join(c["title"] for c in courses)
    desc = clean_text(row.get("VCCS Description", ""))
    return f"{titles} {desc}"


print("\nLoading BGE-small for retriever-mined negatives...")
bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

# Embed W&M catalog
wm_codes = list(wm_lookup.keys())
wm_texts = [f"{wm_lookup[c]['title']} {clean_text(wm_lookup[c]['description'])}" for c in wm_codes]
print(f"Embedding {len(wm_texts)} W&M courses...")
WM_EMBEDDINGS = bge_model.encode(wm_texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)

# TF-IDF
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95,
                         stop_words="english", sublinear_tf=True)
wm_tfidf_matrix = tfidf.fit_transform(wm_texts)

QUERY_PREFIX = "Represent this course for finding transfer equivalents: "


def retrieve_top_k_simple(vccs_text, vccs_embedding, k=20, exclude_codes=None):
    """Simple 2-signal RRF retrieval for negative mining."""
    exclude_codes = exclude_codes or set()
    bge_sims = WM_EMBEDDINGS @ vccs_embedding
    bge_ranked = np.argsort(bge_sims)[::-1]
    tfidf_vec = tfidf.transform([vccs_text])
    tfidf_sims = cosine_similarity(tfidf_vec, wm_tfidf_matrix).flatten()
    tfidf_ranked = np.argsort(tfidf_sims)[::-1]

    RRF_K = 60
    rrf_scores = defaultdict(float)
    for rank, idx in enumerate(bge_ranked[:200]):
        code = wm_codes[idx]
        if code not in exclude_codes:
            rrf_scores[code] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(tfidf_ranked[:200]):
        code = wm_codes[idx]
        if code not in exclude_codes:
            rrf_scores[code] += 1.0 / (RRF_K + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]
    return [code for code, _ in ranked]


# Mine hard negatives from retriever for train_pos
print("\nMining retriever hard negatives for train_pos...")
retriever_negatives = []

for _, row in train_pos.iterrows():
    vccs_text = get_vccs_text(row)
    target_code = str(row["W&M Course Code"]).strip()
    vccs_emb = bge_model.encode([f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True)[0]

    candidates = retrieve_top_k_simple(vccs_text, vccs_emb, k=20, exclude_codes={target_code})
    for neg_code in candidates[:3]:  # 3 hard negatives per positive
        neg_info = wm_lookup.get(neg_code, {})
        retriever_negatives.append({
            "vccs_course": str(row["VCCS Course"]),
            "vccs_desc": str(row.get("VCCS Description", "")),
            "wm_code": neg_code,
            "wm_title": neg_info.get("title", ""),
            "wm_desc": neg_info.get("description", ""),
            "label": 0,
            "source": "retriever_hard",
        })

# No-transfer negatives from neg_df
print("Mining no-transfer negatives from neg_df...")
for _, row in neg_df.iterrows():
    vccs_text = get_vccs_text(row)
    vccs_emb = bge_model.encode([f"{QUERY_PREFIX}{vccs_text}"], normalize_embeddings=True)[0]
    candidates = retrieve_top_k_simple(vccs_text, vccs_emb, k=3)
    if candidates:
        best_code = candidates[0]
        best_info = wm_lookup.get(best_code, {})
        retriever_negatives.append({
            "vccs_course": str(row["VCCS Course"]),
            "vccs_desc": str(row.get("VCCS Description", "")),
            "wm_code": best_code,
            "wm_title": best_info.get("title", ""),
            "wm_desc": best_info.get("description", ""),
            "label": 0,
            "source": "no_transfer",
        })

print(f"Retriever-mined negatives: {len(retriever_negatives)}")

# ── Combine into synthetic negative pairs ───────────────────────────────
# Synthetic negatives: the VCCS course is fake, the W&M course is the original target
synthetic_pairs = []
for i, (_, row) in enumerate(train_pos.iterrows()):
    if i >= len(synthetic_negatives):
        break
    syn = synthetic_negatives[i]
    synthetic_pairs.append({
        "vccs_course": syn["synthetic_course_name"],
        "vccs_desc": syn["synthetic_description"],
        "wm_code": str(row["W&M Course Code"]).strip(),
        "wm_title": str(row.get("W&M Course Title", "")),
        "wm_desc": str(row.get("W&M Description", "")),
        "label": 0,
        "source": "synthetic_claude",
    })

# ── Build training positive pairs ──────────────────────────────────────
positive_pairs = []
for _, row in train_pos.iterrows():
    positive_pairs.append({
        "vccs_course": str(row["VCCS Course"]),
        "vccs_desc": str(row.get("VCCS Description", "")),
        "wm_code": str(row["W&M Course Code"]).strip(),
        "wm_title": str(row.get("W&M Course Title", "")),
        "wm_desc": str(row.get("W&M Description", "")),
        "label": 1,
        "source": "ground_truth",
    })

# ── Merge all training data ────────────────────────────────────────────
all_train_pairs = positive_pairs + retriever_negatives + synthetic_pairs
np.random.shuffle(all_train_pairs)
train_pairs_df = pd.DataFrame(all_train_pairs)

print(f"\n{'='*60}")
print(f"TRAINING DATA SUMMARY")
print(f"{'='*60}")
print(f"  Total training pairs: {len(train_pairs_df)}")
print(f"\n  Class balance:")
print(f"    Label 1 (positive): {(train_pairs_df['label'] == 1).sum()}")
print(f"    Label 0 (negative): {(train_pairs_df['label'] == 0).sum()}")
print(f"    Ratio (neg/pos):    {(train_pairs_df['label'] == 0).sum() / max((train_pairs_df['label'] == 1).sum(), 1):.1f}:1")

print(f"\n  Negative sources:")
for src, cnt in train_pairs_df[train_pairs_df["label"] == 0]["source"].value_counts().items():
    print(f"    {src}: {cnt}")

print(f"\n  Sample synthetic negatives:")
for i, syn in enumerate(synthetic_negatives[:3]):
    print(f"    [{i+1}] {syn['synthetic_course_name']}")
    print(f"        {syn['synthetic_description'][:120]}...")
    print(f"        Reason: {syn['reason_no_transfer'][:100]}")

# Save training pairs for later steps
train_pairs_df.to_csv("_train_pairs.csv", index=False)
print(f"\nSaved training pairs to _train_pairs.csv")

# Also save test_pos indices for reproducibility
test_pos.to_csv("_test_pos.csv", index=False)
print(f"Saved test_pos ({len(test_pos)} rows) to _test_pos.csv")
