# TransferzAI — Engineering Guide

## What this project does

TransferzAI predicts whether a community college course will transfer for credit at a 4-year university, and which course it maps to. Given a VCCS (Virginia Community College System) or CCC (California Community College) course, it retrieves and ranks candidates from the target institution's catalog, then returns a ranked list with calibrated confidence scores.

Three institutions are supported: William & Mary (wm), Virginia Tech (vt), UC Santa Cruz (ucsc).

The product surface is a Streamlit app (`app.py`) with a Transcript Evaluator — a student pastes their full course list and gets back a transfer eligibility assessment with per-course matches and an estimated transferable credit count.

---

## Architecture

### Two-stage retrieve-then-rerank pipeline

```
Query (VCCS/CCC course title + description)
    │
    ▼
Stage 1 — RRF Retrieval (predict.py:retrieve_candidates)
    BGE cosine sim (top-200) + TF-IDF cosine sim (top-200)
    fused with Reciprocal Rank Fusion (RRF_K=60)
    → top-100 candidates per institution
    │
    ▼
Stage 2 — XGBoost Reranker (predict.py:rerank)
    13 features extracted per candidate (extract_signals)
    XGBClassifier → raw margins → softmax over top-10 → confidence
    │
    ▼
Abstention gate
    confidence < HIGH_CONFIDENCE_THRESHOLD → no answer returned
    confidence ≥ HIGH_CONFIDENCE_THRESHOLD → answer returned with label
```

### Key files

| File | Role |
|---|---|
| `app.py` | Streamlit UI — Transcript Evaluator only |
| `predict.py` | Inference: load_artifacts, retrieve_candidates, rerank, evaluate_transcript |
| `config.py` | All hyperparameters and thresholds |
| `paths.py` | Centralized data path definitions |
| `scripts/build_artifacts.py` | Full training pipeline — run this to retrain |
| `artifacts/` | All serialized model artifacts (pkl + npy) |
| `data/equivalency/` | Ground-truth transfer equivalency tables |
| `data/catalogs/` | Full course catalogs per institution |
| `eval/` | Evaluation scripts and error analysis |
| `pipeline/` | Step-by-step training pipeline (BGE finetuning, hard negatives, etc.) |

### Artifacts produced by build_artifacts.py

```
artifacts/
  classifier.pkl       — XGBoost reranker (XGBClassifier)
  iso_cal.pkl          — IsotonicRegression calibrator for Brier/ECE display
  tfidf.pkl            — TfidfVectorizer (15k features, fitted on all 3 catalogs)
  feature_names.pkl    — ordered list of 13 feature names (must match extract_signals)
  scorecard.pkl        — per-institution metrics dict including full threshold sweep
  dept_prior_map.pkl   — P(target_dept | source_dept) per institution (built but NOT yet used as a feature)
  {wm,vt,ucsc}_lookup.pkl     — dict: code → {title, description}
  {wm,vt,ucsc}_codes.pkl      — ordered list of course codes
  {wm,vt,ucsc}_embeddings.npy — BGE embeddings matrix (n_courses × 384)
```

---

## Data

### Equivalency tables (ground truth)

| File | Rows | Positives | Notes |
|---|---|---|---|
| `vccs_wm_merged.csv` | 760 | 334 | 426 rows are true negatives |
| `vccs_vt_merged.csv` | 303 | 303 | all have VT equivalencies |
| `ccc_ucsc_clean.csv` | 904 | 904 | all have UCSC equivalencies |

Columns: `[source] Course`, `[source] Description`, `[target] Course Code`, `[target] Course Title`, `[target] Description`

### Training data

`data/_train_pairs.csv` — 1,761 rows (267 positives, 1,494 negatives), W&M only. VT and CCC training pairs are generated inline in `build_artifacts.py` via `collect()`.

`data/_cache_synthetic_negatives.json` — cached Claude-generated synthetic negative examples. Do not delete; regenerating is expensive.

### Class distribution issues

- W&M: 204 unique target courses, mean ~1.6 training examples per target
- VT: 219 unique targets, 173/219 (79%) appear only once — primary driver of low VT Top-1
- UCSC: 141 unique targets but heavily skewed (THEA 30 = 48 examples, THEA 20 = 35)
- Catalog coverage: 7.4% (W&M), 4.2% (VT), 3.4% (UCSC) of catalog courses have any training labels

---

## Current model performance

Metrics from latest training run (`artifacts/scorecard.pkl`):

| Institution | n | Top-1 | Top-3 | Prec@τ | Cov@τ | op_τ | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| W&M | 67 | 56.7% | 77.6% | 90.9% | 32.8% | 0.80 | 0.0125 | 0.0089 |
| VT | 60 | 53.3% | 85.0% | 94.1% | 28.3% | 0.85 | 0.0143 | 0.0089 |
| CCC→UCSC | 181 | 43.6% | 65.7% | 91.7% | 19.9% | 0.90 | 0.0148 | 0.0098 |

**op_τ** = first threshold where precision ≥ 90%. Thresholds differ per institution — do not use a single global threshold. Coverage of 20-33% means the system abstains on ~70% of queries; this is the primary product problem.

### Metric priorities

1. **Precision@τ** — when the system answers, it must be right. Wrong answers cost students a semester.
2. **Coverage** — currently too low; 70% abstention rate means most students get no answer.
3. **Top-3 recall** — show 3 ranked options. Top-1 being ~50% is acceptable if Top-3 is 65-85%.
4. **Calibration** — Brier and ECE are excellent; do not degrade them.

---

## Configuration (config.py)

```python
RETRIEVAL_K = 100           # candidates returned from RRF per institution
RRF_K = 60                  # RRF fusion constant
SOFTMAX_K = 10              # softmax normalization window over top-K margins
HIGH_CONFIDENCE_THRESHOLD = 0.65   # minimum confidence to return an answer
TRANSFER_THRESHOLD = 0.35          # minimum score to show any result
TOP_K_DISPLAY = 5           # results shown in UI
```

### Per-institution optimal thresholds (from sweep)

To achieve ~87.5% precision equivalently across institutions:
- W&M: τ = 0.75
- VT: τ = 0.85
- UCSC: τ = 0.85

These are not yet implemented as per-institution values — currently a single global threshold is used.

---

## Features (13 total)

```
bge_sim          — BGE cosine similarity between query and candidate embeddings
tfidf_sim        — TF-IDF cosine similarity (full text)
tfidf_title_sim  — TF-IDF cosine similarity (titles only)
dept_sim         — SequenceMatcher ratio: source dept code vs target dept code
title_sim        — SequenceMatcher ratio: source title vs target title
level_ratio      — course number level ratio (100/200/300/400-level)
same_level       — binary: same academic level
rrf_score        — RRF fusion score from retrieval stage
bge_x_dept       — interaction: bge_sim × dept_sim
bge_x_title      — interaction: bge_sim × title_sim
bge_x_tfidf      — interaction: bge_sim × tfidf_sim
dept_x_title     — interaction: dept_sim × title_sim
dept_x_level     — interaction: dept_sim × level_ratio
```

Feature order is fixed — `feature_names.pkl` must always match `extract_signals()` in `predict.py`. If you add a feature, update both.

---

## Known issues and next improvements (ranked by impact)

### Immediate (1-2 days each)

**1. dept_prior_map is built but never used**
`artifacts/dept_prior_map.pkl` encodes P(target_dept | source_dept) from all training data (e.g., P(BUAD | ACC) ≈ 0.9). It is saved but never loaded in `predict.py`. Adding it as a 14th feature would give the model a strong prior signal for cross-semantic dept pairs. Requires rebuilding artifacts after adding to `extract_signals()`.

**2. Per-institution confidence thresholds**
Replace the global `HIGH_CONFIDENCE_THRESHOLD` with a dict `{"wm": 0.75, "vt": 0.85, "ucsc": 0.85}` in `config.py` and look up the right value in `predict.py:rerank()`.

**3. Fix isotonic calibration to use held-out data**
`build_artifacts.py` currently fits `IsotonicRegression` on the training set itself (in-sample). This overfits the calibrator. Fix: hold out 10% as a calibration split, fit calibrator on that, evaluate on the remaining test set.

### Medium-term (1-2 weeks)

**4. Switch XGBoost to rank:pairwise**
Currently uses `XGBClassifier` with binary logloss (pointwise). Switch to `xgb.train()` with `rank:pairwise` objective and `qid` query grouping. This directly optimizes Top-1 ranking rather than binary classification. Expected +3-6pp Top-1. Requires refactoring `collect()` to emit group boundaries.

**5. Enable cross-encoder at inference**
A trained cross-encoder exists at `./cross_encoder` and is evaluated in `eval/test_cross_encoder.py` and `eval/ucsc_error_analysis.py`. It is blocked at inference by a comment in `config.py:32`. Enable it on the top-10 XGBoost outputs only (not all 100). Expected +5-12pp Top-1. Add `CROSS_ENCODER_ENABLED = True` to config.

**6. Synthetic positives for rare VT classes**
79% of VT target courses appear only once in training. Use the same LLM paraphrasing approach from `pipeline/step2_synthetic_negatives.py` to generate additional positive pairs for single-example targets.

---

## How to run

```bash
# Activate environment
source .venv/bin/activate

# Run the app
streamlit run app.py

# Retrain all artifacts (takes ~8 min on MPS, ~20 min on CPU)
python scripts/build_artifacts.py

# MLflow UI (view experiment runs)
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Training logs metrics to MLflow experiment `transferzai-reranker` and uploads all artifacts to S3 bucket `transferzai-artifacts` automatically after training.

---

## MLOps infrastructure

- **MLflow** — experiment tracking. Each `build_artifacts.py` run logs hyperparams, per-institution metrics (Top-1, Top-3, Precision@τ, Coverage@τ, Brier, ECE), and the XGBoost model artifact.
- **AWS S3** (`transferzai-artifacts`) — artifact storage. Training auto-uploads all pkl/npy files after every run. `predict.py` downloads from S3 when `AWS_ACCESS_KEY_ID` is set, falls back to HuggingFace Hub otherwise.
- **HuggingFace Hub** — `hyperalpha/transferzai-artifacts` (artifacts), `hyperalpha/transferzai-bge` (BGE model). Used for Streamlit Cloud deployment.
- **Docker** — `Dockerfile` containerizes the Streamlit app. Build: `docker build -t transferzai:latest .`

---

## Invariants — do not break these

- **Feature order is fixed.** `extract_signals()` in `predict.py` and `extract_features()` in `build_artifacts.py` must produce features in the exact order listed in `feature_names.pkl`. XGBoost uses positional indexing.
- **dept_sim uses source dept codes at inference.** The user enters a VCCS/CCC source dept code (e.g., ACC, HIS). The feature compares this against the target catalog dept code using SequenceMatcher. Do not compare target dept against target dept.
- **Training retrieval includes a 0.5-weighted dept boost.** `build_artifacts.py:rrf_retrieve()` adds a dept similarity signal. `predict.py:retrieve_candidates()` does not. This is intentional — the dept boost helps training recall but was found to hurt inference distribution when both used it. Do not add the dept boost to inference retrieval.
- **SOFTMAX_K is independent of TOP_K_DISPLAY.** Confidence is computed over the top-SOFTMAX_K XGBoost margins; the UI shows TOP_K_DISPLAY results. These are separate concerns.
- **Calibrator is fit on training margins, not raw XGBoost probabilities.** `iso_cal` takes `raw_probs` (sigmoid of margins) as input. It is used for Brier/ECE display only — confidence for abstention uses raw softmax over margins, not calibrated probs.
- **TOKENIZERS_PARALLELISM and OMP_NUM_THREADS must be set before any imports.** They are set at the top of `predict.py`. Do not move them.

---

## Institution keys

Always use these exact string keys: `"wm"`, `"vt"`, `"ucsc"`. They index into `INSTITUTION_REGISTRY`, artifact filenames (`wm_lookup.pkl`), and `dept_prior_map`. Northeastern is in the registry but disabled (catalog cleanup needed).
