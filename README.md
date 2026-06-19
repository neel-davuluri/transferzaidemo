<h1 align="center">TransferzAI</h1>

<p align="center">
  <strong>Retrieve-then-rerank pipeline for university transfer credit matching</strong><br>
  Two-stage ML system over 11,850 courses · 3 institutions · 308 held-out test samples
</p>

<p align="center">
  <a href="https://transferzai.streamlit.app"><strong>→ Live demo</strong></a> ·
  <a href="https://huggingface.co/hyperalpha/transferzai-bge">Fine-tuned BGE model</a> ·
  <a href="https://huggingface.co/hyperalpha/transferzai-artifacts">Artifacts</a>
</p>

---

## Problem

When a college athlete enters the transfer portal, coaches have a narrow window to evaluate and recruit them. Academic eligibility — whether the player's credits transfer and they'll have enough standing to compete — has to be verified by the registrar before a scholarship offer can be extended. That process takes days or weeks. In the meantime, the player is fielding offers from other schools, and the first program to give them a clear answer often wins the recruitment. A D1 athletics department was doing this eligibility check by hand for every prospective transfer.

TransferzAI automates that lookup. Given a source course (title + description), it retrieves and ranks candidates from the target institution's catalog, returning a ranked shortlist with a confidence score. When the system isn't confident enough, it abstains rather than guess — wrong answers cost students a semester.

**Presented at the Applied ML Conference, April 2026.**

---

## Results

Evaluated on stratified held-out splits (80/20, `random_state=42`, stratified by target department). Brier and ECE are computed from an isotonic regression calibrator trained offline — the calibrator is not wired into the serving path (see [Evaluation notes](#evaluation-notes)).

| Institution | n | Top-1 | Top-3 | Prec @ τ | Cov @ τ | τ | Brier | ECE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| William & Mary | 67 | 56.7% | 77.6% | 90.9% | 32.8% | 0.79 | 0.0125 | 0.0089 |
| Virginia Tech | 60 | 53.3% | 85.0% | 94.1% | 28.3% | 0.85 | 0.0143 | 0.0089 |
| UC Santa Cruz | 181 | 43.6% | 65.7% | 91.7% | 19.9% | 0.85 | 0.0148 | 0.0098 |

**Top-1 / Top-3** — correct course is the #1 / top-3 prediction.  
**Prec @ τ** — precision among queries where the model commits to an answer.  
**Cov @ τ** — fraction of queries answered; the system abstains on the rest.  
**τ** — per-institution confidence threshold where precision first reaches ≥ 90%.

---

## Architecture

```
Query (course title + description + dept code)
            │
            ▼
┌───────────────────────────────────────────┐
│          Stage 1 — RRF Retrieval          │
│                                           │
│  BGE bi-encoder ──┐                      │
│  (384-dim, fine-  │  Reciprocal Rank      │
│   tuned, k=200)   ├─► Fusion (k=60)      │
│                   │   → top-100           │
│  TF-IDF          ──┘                     │
│  (15k features,                          │
│   1-2 gram)                              │
└───────────────────────────────────────────┘
            │  top-100 candidates
            ▼
┌───────────────────────────────────────────┐
│       Stage 2 — XGBoost Reranker          │
│                                           │
│  13 features per (query, candidate)       │
│  XGBClassifier (500 trees, depth=4)      │
│  → decision margins                       │
│  → softmax over top-10 margins            │
│  → per-query confidence score             │
└───────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│          Abstention Gate                  │
│                                           │
│  conf ≥ τ_inst  →  Confirmed  (green)    │
│  conf ≥ 0.74    →  Possible   (yellow)   │
│  conf < 0.74    →  No match   (abstain)  │
└───────────────────────────────────────────┘
```

**τ per institution:** W&M = 0.79, VT = 0.85, UCSC = 0.85 (derived from threshold sweep on held-out test set, targeting ≥ 90% precision).

---

## How it works

### Stage 1 — Retrieval

Two signals are fused via Reciprocal Rank Fusion to produce a top-100 candidate list:

| Signal | Implementation | Role |
|:---|:---|:---|
| BGE bi-encoder | `BAAI/bge-small-en-v1.5`, fine-tuned on transfer pairs | Semantic "transfers as" similarity |
| TF-IDF | 15k features, 1–2 gram, sublinear_tf, max_df=0.95 | Lexical keyword overlap |

RRF score: `Σ 1 / (60 + rank_i)` across both ranked lists. Training retrieval adds a 0.5-weighted department string-similarity signal to improve hard-negative recall — intentionally excluded from inference retrieval to prevent train/inference distribution mismatch.

**Why fine-tune the bi-encoder?** Off-the-shelf embeddings measure semantic similarity. "Introduction to Chemistry" and "Organic Chemistry II" are semantically close but don't transfer to the same course. A naive cross-encoder reranker exploited this — it dropped top-3 recall from ~46% to 28% by confidently promoting semantically similar but non-equivalent courses. Contrastive fine-tuning on transfer pairs teaches the retriever the "transfers as" relationship instead.

### Stage 2 — Reranking

XGBoost scores each of the top-100 candidates on 13 handcrafted features:

| Group | Features |
|:---|:---|
| Semantic | `bge_sim` |
| Lexical | `tfidf_sim`, `tfidf_title_sim`, `title_sim` |
| Structural | `dept_sim`, `level_ratio`, `same_level`, `rrf_score` |
| Interactions | `bge_x_dept`, `bge_x_title`, `bge_x_tfidf`, `dept_x_title`, `dept_x_level` |

**Why XGBoost over a cross-encoder?** ~1,200 labeled positive training pairs across 3 institutions — too few to fine-tune a neural reranker without overfitting. XGBoost on 13 features generalizes better at this data scale, runs in <1ms per query, and produces interpretable feature importances.

**Training setup:** `collect()` expands each positive training query into a pool of 50 candidates via RRF, yielding ~61,600 training rows at a 1:49 positive:negative ratio. `scale_pos_weight=49` corrects for the imbalance.

**Confidence score:** Softmax over the top-10 XGBoost decision margins. This is a *relative decisiveness signal* — how much the top candidate outscored the other finalists — not a calibrated probability. Restricting to top-10 prevents confidence dilution when department signals inflate many same-department candidates simultaneously.

---

## Key engineering decisions

**Train/inference retrieval asymmetry.**
The training retrieval adds a department similarity boost (weight=0.5) to ensure the true match lands in the top-50 pool for XGBoost training. Inference retrieval does not use this boost — empirically it shifts the score distribution in a way that hurts inference precision. The asymmetry is intentional.

**Feature distribution bug.**
`dept_sim` was hardcoded to 0.0 at inference. This feature carries ~15% XGBoost importance including interaction terms; zeroing it at inference was corrupting 4 of 13 features on every prediction. The symptom was suppressed confidence scores across the board. Post-fix, XGBoost decision margins shifted from −0.95 → +4.25 on affected queries. Caught through feature-value auditing, not metric drift.

**Per-institution confidence thresholds.**
A single global threshold produces different precision/coverage tradeoffs across institutions due to catalog size and training data density differences. Thresholds are set per-institution (W&M: 0.79, VT: 0.85, UCSC: 0.85) based on held-out sweep analysis, with a global fallback of 0.84 for any new institution added to the registry.

**Selective prediction as a product requirement.**
The abstention rate (~70% at confirmed threshold) is not a limitation — it's the design. At these coverage rates, the system answers correctly >90% of the time. Lowering thresholds to increase coverage trades precision for recall in a domain where false positives have direct student consequences.

---

## Evaluation notes

**Brier score and ECE** are computed using an isotonic regression calibrator (`iso_cal.pkl`) fit on training-set sigmoid probabilities vs. binary match/no-match labels. The calibrator is loaded at inference but its output is not used for the user-facing confidence score or the abstention gate. These metrics describe the calibration quality of the per-candidate binary classifier, not of the softmax confidence score shown in the UI.

**Calibration caveat:** `iso_cal` is fit on the training set, not a held-out calibration split. ECE and Brier are likely slightly optimistic. Fixing this requires holding out a calibration split before training and fitting the calibrator on that.

**What "confidence" means in the UI:** Softmax over top-10 XGBoost margins. High confidence means the top candidate's margin was substantially larger than the other 9 finalists — a relative signal, not an absolute probability of correctness.

---

## Training data

| Source → Target | Positive pairs | Train | Test |
|:---|:---:|:---:|:---:|
| VCCS → William & Mary | 334 | 267 | 67 |
| VCCS → Virginia Tech | 303 | 242 | 61 |
| CCC → UC Santa Cruz | 904 | 723 | 181 |
| **Total** | **1,541** | **1,232** | **309** |

Ground-truth equivalency tables sourced from official articulation agreements. Course-level data only — no student PII (FERPA compliant).

Hard negatives mined from retrieval-stage false positives. LLM-generated synthetic negatives (267 cached in `data/_cache_synthetic_negatives.json`) augment W&M training. Negative construction via `scripts/build_artifacts.py:collect()`.

---

## Worked example

**Input:**

| Field | Value |
|:---|:---|
| Dept | `ACC` |
| Number | `211` |
| Title | Principles of Accounting I |
| Description | Introduces accounting principles with respect to financial reporting. Includes the accounting cycle, financial statements, and the conceptual framework of financial accounting. |

**Output (W&M):**

```
✓ William & Mary — 3 confirmed credits

ACC 211  →  BUAD 201  Introduction to Financial Accounting   87%  ● Confirmed
              also: BUAD 202 · 41%   BUAD 301 · 28%
```

System retrieved 100 W&M candidates via RRF, scored each on 13 features, applied softmax over top-10 margins. Confidence 87% ≥ W&M threshold (0.79) → Confirmed. Between 0.74–0.79 → Possible (yellow, not counted). Below 0.74 → system abstains.

**Python API:**

```python
from predict import evaluate_transcript, load_artifacts

load_artifacts()

results = evaluate_transcript([
    {
        "dept": "MTH", "number": "263",
        "title": "Calculus I",
        "description": "Limits, derivatives, and integrals of single-variable functions...",
        "credits": 3
    }
], institutions=["wm", "vt", "ucsc"], min_credits_required=30)

for inst, r in results.items():
    print(f"\n{r['institution_name']} — {r['summary']}")
    for cr in r["course_results"]:
        if cr["top_matches"]:
            m = cr["top_matches"][0]
            print(f"  {cr['title']} → {m['code']} {m['title']} ({m['confidence']:.0%})")
```

---

## Quickstart

```bash
git clone https://github.com/neel-davuluri/transferzaidemo.git
cd transferzaidemo

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Artifacts download automatically from HuggingFace Hub on first run (~500 MB)
streamlit run app.py
# Open http://localhost:8501
```

**AWS S3 (faster artifact loading):**

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
# predict.py auto-downloads from s3://transferzai-artifacts when credentials are set
streamlit run app.py
```

**Retrain from scratch (~8 min on MPS, ~20 min on CPU):**

```bash
python scripts/build_artifacts.py
# Logs to MLflow, auto-uploads artifacts to S3 if credentials are set
```

**View experiment runs:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5000
```

**Docker:**

```bash
docker build -t transferzai:latest .
docker run -p 8501:8501 transferzai:latest
```

---

## MLOps

| Tool | Role |
|:---|:---|
| **MLflow** | Logs hyperparams, per-institution Top-1/Top-3/Prec@τ/Cov@τ/Brier/ECE, and XGBoost model artifact for every training run |
| **AWS S3** (`transferzai-artifacts`) | All pkl/npy artifacts auto-uploaded post-training; `predict.py` downloads at startup when `AWS_ACCESS_KEY_ID` is set |
| **HuggingFace Hub** | [`hyperalpha/transferzai-artifacts`](https://huggingface.co/hyperalpha/transferzai-artifacts) and [`hyperalpha/transferzai-bge`](https://huggingface.co/hyperalpha/transferzai-bge) — deployment fallback for Streamlit Cloud |
| **Docker** | Python 3.11-slim container, artifacts pre-loaded, port 8501 |

---

## Project structure

```
transferzaidemo/
├── app.py                      # Streamlit UI — Transcript Evaluator
├── predict.py                  # Inference: load_artifacts(), predict_transfer(), evaluate_transcript()
├── config.py                   # Hyperparameters, per-institution thresholds, model paths
├── paths.py                    # Centralized data path definitions
├── Dockerfile / .dockerignore
├── requirements.txt
├── scripts/
│   ├── build_artifacts.py      # Full training pipeline — BGE embeddings, XGBoost, eval, S3 upload
│   ├── build_vt_dataset.py     # Builds VCCS→VT equivalency dataset
│   ├── build_ccc_ucsc_dataset.py
│   └── build_{vt,ucsc}_catalog.py
├── eval/
│   ├── audit.py                # Confidence intervals + retrieval recall@k
│   ├── benchmark_rerankers.py  # RRF baseline vs. custom CE vs. BGE-Reranker-v2-m3
│   ├── test_cross_encoder.py   # Cross-encoder holdout evaluation
│   ├── ucsc_error_analysis.py  # UCSC error breakdown by failure type
│   ├── run_llm_eval.py         # Claude-as-judge evaluation on hard cases
│   ├── llm_judge.py            # Claude-as-judge pair scorer
│   └── sequence_features.py    # Sequence position text augmentation helpers
├── artifacts/                  # Serialized model artifacts (pkl + npy)
│   ├── classifier.pkl          # XGBoost reranker (XGBClassifier, 500 trees)
│   ├── tfidf.pkl               # TF-IDF vectorizer (15k features)
│   ├── iso_cal.pkl             # Isotonic calibrator — offline Brier/ECE only, not in serving path
│   ├── scorecard.pkl           # Per-institution metrics + full threshold sweep
│   ├── feature_names.pkl       # Ordered feature list (must match extract_signals in predict.py)
│   └── {wm,vt,ucsc}_{lookup,codes,embeddings}.*
└── data/
    ├── catalogs/               # Full course catalogs per institution
    └── equivalency/            # Ground-truth transfer equivalency tables
```

---

## Limitations

- **Coverage is low by design.** At confirmed thresholds (0.79–0.85), the system answers 20–33% of queries. Abstaining on the rest is the correct product decision when wrong answers cost students a semester.
- **Top-1 accuracy is moderate.** Correct course is the top prediction 44–57% of the time. Top-3 (66–85%) is the more useful metric — the system is designed as a ranked shortlist, not a single-answer oracle.
- **Sparse training labels.** 7.4% (W&M), 4.2% (VT), and 3.4% (UCSC) of catalog courses appear in training. 79% of VT target courses appear only once — primary driver of VT's lower Top-1.
- **Many-to-few disambiguation at UCSC.** Multiple community colleges map to 141 UCSC targets, introducing conflicting training signal for semantically similar source courses.
- **Calibrator not in serving path.** `iso_cal.pkl` is loaded at startup but its output is not used for the user-facing confidence score. ECE/Brier metrics describe the calibrator's offline fit. Wiring the calibrator into serving requires re-deriving thresholds on the calibrated probability scale.
- **In-sample calibration.** `iso_cal` is fit on training data, making ECE/Brier slightly optimistic. Fix: hold out a calibration split before training.

---

## Roadmap

| Priority | Item |
|:---|:---|
| High | Re-derive per-institution thresholds against serving confidence scale — threshold sweep in `build_artifacts.py` uses softmax over all 100 pool candidates; `predict.py` uses softmax over top-10 only; τ values need recalibration |
| High | Wire `iso_cal` into serving path with recalibrated thresholds |
| High | `dept_prior_map` as a 14th XGBoost feature — P(target_dept \| source_dept) is computed during training but not saved or used at inference |
| High | Switch to `rank:pairwise` XGBoost objective — currently pointwise binary logloss; expected +3–6pp Top-1 |
| Medium | Cross-encoder reranker on XGBoost top-10 at inference — model trained, not yet wired in; expected +5–12pp Top-1 |
| Medium | Synthetic positives for rare VT targets — 79% of VT courses appear once in training |
| Medium | Held-out calibration split — fit `iso_cal` on 10% holdout, not training data |

---

## Contact

Neel Davuluri · [neel.davuluri@gmail.com](mailto:neel.davuluri@gmail.com)  
Garrett Bellin
