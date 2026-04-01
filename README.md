<h1 align="center">TransfersAI</h1>
<p align="center"><strong>Course transfer eligibility prediction for college athletic recruiting</strong></p>
<p align="center">
  Fine-tuned retrieval · Calibrated classification · Built for W&amp;M Men's Basketball
</p>

<p align="center">
  <a href="#architecture">Architecture</a> · 
  <a href="#the-key-insight">Key Insight</a> · 
  <a href="#results">Results</a> · 
  <a href="#iteration-story">Iteration Story</a> · 
  <a href="#quickstart">Quickstart</a>
</p>

---

## The Problem

When a college basketball coach identifies a transfer portal player, the first question is: **will their credits transfer?** If they don't, the player can't compete immediately — and the coach just burned 3 weeks of recruiting effort on an ineligible candidate.

Today, coaches email the registrar, wait days for a response, and manually cross-reference course-by-course. TransfersAI compresses that into minutes: paste a player's coursework, get back a per-course transfer assessment with calibrated probabilities and plain-English explanations.

**The critical constraint: false positives are catastrophic.** A system that says "yes, this transfers" when it doesn't costs a coaching staff weeks and a recruiting slot. Precision on positive predictions isn't just a metric — it's the entire design target.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRANSFERS AI PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐                                                  │
│   │  Input:       │    ┌──────────────────────────────────────────┐  │
│   │  VCCS Course  │───▶│  RETRIEVER (3-signal fusion via RRF)    │  │
│   │  code + desc  │    │                                          │  │
│   └──────────────┘    │  ┌────────────┐  ┌────────┐  ┌────────┐ │  │
│                        │  │ Fine-tuned │  │ TF-IDF │  │  Dept  │ │  │
│                        │  │ BGE bi-enc │  │ (1,2)- │  │ Prior  │ │  │
│                        │  │ (contrastv │  │ gram   │  │ (from  │ │  │
│                        │  │  learning) │  │        │  │ train) │ │  │
│                        │  └─────┬──────┘  └───┬────┘  └───┬────┘ │  │
│                        │        └──────┬──────┘      ┌────┘      │  │
│                        │               ▼             ▼           │  │
│                        │        Reciprocal Rank Fusion            │  │
│                        │         (weighted merge)                 │  │
│                        └──────────────┬───────────────────────────┘  │
│                                       │ top-50 candidates            │
│                                       ▼                              │
│                        ┌──────────────────────────────────────────┐  │
│                        │  CLASSIFIER (probability, NOT ranking)   │  │
│                        │                                          │  │
│                        │  52 features ──▶ XGBoost ──▶ Isotonic   │  │
│                        │  (structural,     (GBT)     Calibration │  │
│                        │   lexical,                               │  │
│                        │   semantic)                              │  │
│                        └──────────────┬───────────────────────────┘  │
│                                       │ calibrated P(transfer)       │
│                                       ▼                              │
│                        ┌──────────────────────────────────────────┐  │
│                        │  OUTPUT                                  │  │
│                        │  Per-course: P(transfer), confidence     │  │
│                        │  tier (HIGH/MOD/LOW), W&M equivalent,   │  │
│                        │  plain-English explanation               │  │
│                        │                                          │  │
│                        │  Transcript-level: total transferable    │  │
│                        │  credits, eligibility assessment         │  │
│                        └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**Two-stage retrieve-then-classify.** The retriever finds the 50 most plausible W&M matches for a given course. The classifier assigns a calibrated probability to each pair. The retriever ranks; the classifier scores. This separation is a deliberate architectural choice, explained [below](#the-key-insight).

---

## The Key Insight

**Semantic similarity ≠ transfer eligibility.**

This is the core finding of the project, discovered empirically across three major iterations. "Introduction to Chemistry" and "Advanced Organic Chemistry" are highly semantically similar — any embedding model will place them close together. But they don't transfer. They're different courses at different levels covering different depths of material.

I initially assumed that a stronger semantic model would improve predictions. I was wrong, and proving *why* I was wrong became the technical foundation of the system.

When I added a cross-encoder reranker (BAAI/bge-reranker-v2-m3) — a model specifically designed to score passage relevance — **top-3 recall dropped from 0.455 to 0.281.** The cross-encoder was confidently promoting semantically similar but non-transferring courses above the correct match. More semantic power made things worse.

The fix: **fine-tune the retriever to learn "transfers as" rather than "is similar to."** Using contrastive learning (MultipleNegativesRankingLoss) on 267 labeled transfer pairs, the bi-encoder learns that "Intro to Accounting" at VCCS should be closer to "Principles of Accounting" at W&M than to "Accounting Information Systems" — not because of word overlap, but because that's the transfer relationship in the articulation agreement.

This reframing — from similarity to institutional equivalence — is what the entire architecture is built around.

---

## Results

<!-- 
UPDATE: Replace these with your actual held-out test set numbers
after running pipeline.py. These are from v1/v3 development iterations.
-->

### Scorecard

| Metric | Value | Target | Status |
|:---|:---:|:---:|:---:|
| Precision (P ≥ 0.5) | — | > 0.85 | — |
| Precision (P ≥ 0.7) | — | > 0.90 | — |
| Top-3 Recall (end-to-end) | — | > 0.70 | — |
| False Confidence Rate | — | < 0.30 | — |
| Calibration Error (ECE) | — | < 0.05 | — |
| Brier Score | — | < 0.10 | — |

### What the metrics mean

**Precision** is the priority metric. When the system says "this course transfers," it needs to be right. A false positive means a coach wastes weeks recruiting a player whose credits won't transfer. The 0.85 target at P ≥ 0.5 means that at least 85% of positive predictions are correct.

**Calibration** means the probabilities are honest. When TransfersAI says 70% confident, it should be right about 70% of the time. This is verified via Expected Calibration Error and reliability diagrams. Isotonic regression calibration achieves ECE < 0.02 consistently.

**Top-3 Recall** measures the full pipeline: for a course that *does* transfer, is the correct W&M equivalent in the top 3 candidates? This is bounded by the retrieval ceiling — if the retriever doesn't surface the right course in its top-50 candidates, the classifier never gets a chance.

**False Confidence Rate** measures how often the system is confidently wrong — assigning P > 0.5 to a pair that doesn't actually transfer. This is kept under 5% across all iterations.

---

## Iteration Story

The final system is the result of three major iterations. Each one tested a hypothesis, and two of the three were wrong.

### v1: Off-the-shelf BGE + XGBoost

**Hypothesis:** BGE embeddings capture enough course semantics for retrieval; XGBoost learns the decision boundary from structural and lexical features.

| Metric | Result |
|:---|:---:|
| TF-IDF retrieval top-3 | 0.374 |
| BGE+TF-IDF retrieval top-3 | 0.476 |
| Precision (t=0.5) | 0.762 |
| ECE | 0.015 |
| False confidence rate | 0.052 |

**Finding:** Calibration and false confidence were excellent out of the gate. The bottleneck was retrieval — 25% of true matches weren't in the top-20 candidates. The classifier's reranking actually *hurt* top-3 recall (0.476 → 0.455), foreshadowing the v2 failure.

**Feature importance revealed the problem:** `tfidf_full_sim` dominated at 0.233, while `bge_cosine_sim` contributed only 0.067. The BGE embeddings weren't learning anything TF-IDF didn't already know.

### v2: Cross-encoder reranker

**Hypothesis:** A cross-encoder (BAAI/bge-reranker-v2-m3) jointly attending to both courses would make finer-grained distinctions than the bi-encoder's independent embeddings.

| Metric | v1 | v2 | Delta |
|:---|:---:|:---:|:---:|
| Top-3 Recall (E2E) | 0.455 | **0.281** | ↓ 0.174 |
| Top-1 Recall (E2E) | 0.222 | **0.093** | ↓ 0.129 |
| Precision (t=0.5) | 0.762 | 0.771 | → |
| False confidence | 0.052 | 0.034 | ↓ |

**Finding:** Catastrophic regression. The cross-encoder promoted semantically similar but non-transferring courses above correct matches. With 50 candidates (expanded from 20 to raise the retrieval ceiling), there were more high-similarity distractors to mislead the reranker. The cross-encoder scored *general relevance*, not *institutional equivalence*. Top-1 collapsed from 0.222 to 0.093 — the model was confidently wrong.

**This is the experiment that revealed the core insight.** Semantic similarity and transfer eligibility are different relationships. More powerful semantic models don't help — they hurt — because they amplify the wrong signal.

### v3: Domain-adapted retrieval + structural priors

**Hypothesis:** Instead of adding semantic power, add *structural knowledge* to retrieval (department mapping), and stop using the classifier for ranking entirely.

Changes:
- Department prior added as a third retrieval signal in RRF (0.5x weight)
- Classifier produces probabilities only — retrieval order is final
- Cross-encoder removed entirely
- Fine-tuned bi-encoder with contrastive learning teaches "transfers as" rather than "is similar to"

This is the production architecture.

---

## Technical Details

### Data

- **Source:** VCCS→W&M official articulation agreement (760 course entries)
- **Positive pairs:** 334 verified course transfers
- **Negative sampling:** Hard negatives mined from the retriever's top candidates (the confusing near-misses) plus 426 VCCS courses with no W&M equivalent
- **Train/test split:** 80/20 stratified by W&M department, with dept_map rebuilt exclusively from training data to prevent leakage
- **No student PII.** FERPA compliant — only course-level data (codes, titles, descriptions)

### Retrieval: Three-Signal RRF

Reciprocal Rank Fusion merges three ranked lists:

1. **Fine-tuned BGE bi-encoder** (contrastive learning on transfer pairs)
2. **TF-IDF** with (1,2)-grams, sublinear TF, 10K vocabulary
3. **Department prior** — P(W&M_dept | VCCS_dept) learned from training data

Each signal contributes `1/(k + rank)` to each candidate's score. The department prior receives 0.5x weight (softer signal — allows novel cross-department transfers to surface while boosting established pathways).

### Classification: 52 Features → XGBoost → Isotonic Calibration

**Structural features** (11): Course number proximity, level matching, department mapping strength, multi-course indicators.

**Lexical features** (9): TF-IDF cosine similarity (descriptions, titles, combined), SequenceMatcher fuzzy title match, word overlap metrics (Jaccard, shared ratio).

**Semantic features** (1): BGE cosine similarity from fine-tuned encoder.

**Description features** (6): Length, length ratio, presence indicators.

**Domain features** (28): Subject keyword detectors for 7 academic domains (math, science, computing, humanities, language, arts, business), each producing 4 features (VCCS count, W&M count, overlap, difference).

The classifier is calibrated post-hoc with isotonic regression (3-fold inner CV). Isotonic was chosen over Platt scaling because the relationship between raw scores and true probabilities is non-linear with this feature set.

### Evaluation Protocol

All reported metrics are from a **held-out test set** (20% of positive pairs, stratified by W&M department). The department mapping, negative sampling, and feature engineering pipeline are built exclusively from training data. Cross-validation metrics are reported during development but the scorecard uses only held-out numbers.

---

## Quickstart

### Requirements

```
Python ≥ 3.9
xgboost
sentence-transformers
scikit-learn
pandas
numpy
```

### Run the Pipeline

```python
# In Google Colab:
!pip install xgboost sentence-transformers -q

import sys
sys.path.insert(0, "/content/transfers_ai")
exec(open("/content/transfers_ai/pipeline.py").read())
```

The pipeline runs 13 cells sequentially:
1. Load data + train/test split
2. Fine-tune bi-encoder (~10 min on T4 GPU)
3. Build retriever + evaluate retrieval recall
4. Generate hard negative training pairs
5. Extract 52 features per pair
6. Train XGBoost + isotonic calibration (5-fold CV)
7. Compute feature importance
8. End-to-end evaluation on held-out test set
9. Print final scorecard

### Predict on New Courses

```python
from pipeline import predict_transfer

results = predict_transfer(
    vccs_course="CSC 221 INTRODUCTION TO PROBLEM SOLVING AND PROGRAMMING",
    vccs_desc="Introduces a modern programming language for problem solving.",
    top_k=3,
)

for r in results:
    print(f"{r['wm_code']} — {r['wm_title']}")
    print(f"  P(transfer) = {r['probability']:.3f} [{r['confidence']}]")
```

### Evaluate a Full Transcript

```python
from pipeline import evaluate_transcript

assessment = evaluate_transcript([
    {"course": "CSC 221 INTRO TO PROGRAMMING", "description": "...", "credits": 3},
    {"course": "ENG 111 COLLEGE COMPOSITION I", "description": "...", "credits": 3},
    {"course": "MTH 263 CALCULUS I", "description": "...", "credits": 4},
])

print(assessment["summary"])
# → "LIKELY ELIGIBLE: 10 credits at ≥70% confidence, exceeding 9-credit minimum."
```

---

## Project Structure

```
transfers_ai/
├── config.py        # All hyperparameters (one source of truth)
├── data.py          # Loading, parsing, leakage-free splitting
├── retriever.py     # BGE + TF-IDF + dept prior + fine-tuning
├── features.py      # 52-feature pair engineering
├── classifier.py    # XGBoost + isotonic calibration + evaluation
├── evaluate.py      # End-to-end pipeline eval + scorecard
└── pipeline.py      # Main entry point (run this)
```

---

## Limitations and Future Work

**Single institution pair.** Currently trained and validated on VCCS→W&M only. The architecture is institution-agnostic (any source courses, any target catalog), but generalization to other school pairs is untested. Scraping additional Virginia articulation agreements (VCCS→VCU, VCCS→JMU) is the highest-priority next step.

**Missing descriptions.** 170 of 760 VCCS courses (22%) have no catalog description. The strongest feature (`tfidf_full_sim`, importance 0.29) goes to zero for these courses. Scraping VCCS catalog descriptions would likely recover 10-15 positive pairs currently classifying blind.

**Small dataset.** 334 positive pairs is enough for the current approach (XGBoost + calibration) but limits the complexity of models that can be trained. Multi-institution data would enable more expressive retrieval models.

**No live registrar feedback loop yet.** The system's real-world precision is estimated from held-out data, not confirmed by registrar decisions. Building a feedback mechanism where registrar confirmations become new training examples would continuously improve the model.

---

## Quickstart

### Local (Development)

```bash
# Clone
git clone https://github.com/neel-davuluri/transferzaidemo.git
cd transferzaidemo

# Install dependencies
pip install -r requirements.txt

# Download pre-trained artifacts
python download_artifacts.py

# Run Streamlit app
streamlit run app.py
```

### Streamlit Cloud (Production)

1. Fork the repo on GitHub
2. Deploy via [Streamlit Cloud](https://streamlit.io/cloud)
3. If artifacts are missing, download from [releases](https://github.com/neel-davuluri/transferzaidemo/releases), extract, and commit to a separate branch

### Manual artifact setup

If the download script fails:
1. Download `artifacts.tar.gz` from: https://github.com/neel-davuluri/transferzaidemo/releases
2. Extract: `tar -xzf artifacts.tar.gz`
3. Restart the app

---

## Built For

W&M Men's Basketball — to evaluate transfer portal candidates' academic eligibility before committing recruiting resources.

---

## Contact

Neel Davuluri 
Garrett Bellin
