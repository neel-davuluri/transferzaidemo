# TransferzAI — Technical Reference

Results, evaluation methodology, training data, and implementation details.

---

## Results

Evaluated on stratified held-out splits (80/20, `random_state=42`, stratified by target department).

| Institution | n | Top-1 | Top-3 | Prec @ τ | Cov @ τ | τ | Brier | ECE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| William & Mary | 67 | 56.7% | 77.6% | 90.9% | 32.8% | 0.79 | 0.0125 | 0.0089 |
| Virginia Tech | 60 | 53.3% | 85.0% | 94.1% | 28.3% | 0.85 | 0.0143 | 0.0089 |
| UC Santa Cruz | 181 | 43.6% | 65.7% | 91.7% | 19.9% | 0.85 | 0.0148 | 0.0098 |

**Top-1 / Top-3** — correct course is the #1 / top-3 prediction.  
**Prec @ τ** — precision among queries where the model commits to an answer.  
**Cov @ τ** — fraction of queries answered; the system abstains on the rest.  
**τ** — per-institution confidence threshold where precision first reaches ≥ 90%.  
**Brier / ECE** — calibration metrics from an isotonic regression calibrator fit offline; see caveat below.

---

## Evaluation Notes

**What "confidence" means.** Softmax over the top-10 XGBoost decision margins — how decisively the top candidate outscored the other 9 finalists. A relative signal, not a calibrated probability.

**Brier and ECE.** Computed from an isotonic regression calibrator (`iso_cal.pkl`) fit on training-set sigmoid probabilities vs. binary match/no-match labels. The calibrator is loaded at startup but its output is **not** used for the user-facing confidence score or the abstention gate. These metrics describe the calibration of the per-candidate binary classifier, not of the softmax confidence score shown in the UI.

**Calibration caveat.** `iso_cal` is fit on the training set, not a held-out calibration split, making ECE and Brier slightly optimistic. Fix: hold out 10% as a calibration split before training, fit the calibrator on that.

**Threshold caveat.** The threshold sweep in `build_artifacts.py` runs softmax over all 100 pool candidates. `predict.py` runs softmax over the top-10 only. The reported τ values are not directly comparable to the serving confidence scale — they need recalibration against the top-10 softmax distribution.

---

## Training Data

| Source → Target | Positive pairs | Train | Test |
|:---|:---:|:---:|:---:|
| VCCS → William & Mary | 334 | 267 | 67 |
| VCCS → Virginia Tech | 303 | 242 | 61 |
| CCC → UC Santa Cruz | 904 | 723 | 181 |
| **Total** | **1,541** | **1,232** | **309** |

Ground-truth equivalency tables sourced from official articulation agreements. Course-level data only — no student PII (FERPA compliant).

Hard negatives mined from retrieval-stage false positives. 267 LLM-generated synthetic negatives (cached in `data/_cache_synthetic_negatives.json`) augment W&M training.

**Class distribution issues worth noting:**
- W&M: 204 unique target courses, mean ~1.6 training examples per target
- VT: 219 unique targets, 79% appear only once — primary driver of lower VT Top-1
- UCSC: 141 unique targets, heavily skewed (THEA 30 = 48 examples, THEA 20 = 35)

---

## Worked Example

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

The system retrieved 100 W&M candidates via RRF, scored each on 13 features, applied softmax over the top-10 margins. Confidence 87% ≥ W&M threshold (0.79) → Confirmed.

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

## Feature Set (13 features)

| Group | Feature | Description |
|:---|:---|:---|
| Semantic | `bge_sim` | BGE cosine similarity |
| Lexical | `tfidf_sim` | TF-IDF cosine sim (full text) |
| Lexical | `tfidf_title_sim` | TF-IDF cosine sim (titles only) |
| Lexical | `title_sim` | SequenceMatcher ratio on titles |
| Structural | `dept_sim` | SequenceMatcher ratio on dept codes |
| Structural | `level_ratio` | Course level alignment (1.0 = same level) |
| Structural | `same_level` | Binary: same academic level |
| Structural | `rrf_score` | RRF fusion score from Stage 1 |
| Interactions | `bge_x_dept` | BGE sim × dept sim |
| Interactions | `bge_x_title` | BGE sim × title sim |
| Interactions | `bge_x_tfidf` | BGE sim × TF-IDF sim |
| Interactions | `dept_x_title` | dept sim × title sim |
| Interactions | `dept_x_level` | dept sim × level ratio |

Feature order is fixed — `feature_names.pkl` must match `extract_signals()` in `predict.py`. XGBoost uses positional indexing.

---

## Limitations

- **Coverage is low by design.** At confirmed thresholds (0.79–0.85), the system answers 20–33% of queries. Abstaining on the rest is the correct product decision when wrong answers cost students a semester.
- **Top-1 accuracy is moderate.** Correct course is the top prediction 44–57% of the time. Top-3 (66–85%) is the more useful metric — the system is designed as a ranked shortlist, not a single-answer oracle.
- **Sparse training labels.** 7.4% (W&M), 4.2% (VT), and 3.4% (UCSC) of catalog courses appear in training. 79% of VT target courses appear only once.
- **Many-to-few disambiguation at UCSC.** Multiple community colleges map to 141 UCSC targets, introducing conflicting training signal for semantically similar source courses.
- **Calibrator not in serving path.** `iso_cal.pkl` is loaded at startup but not used for the confidence score or abstention gate. Wiring it in requires re-deriving thresholds on the calibrated probability scale.

---

## Roadmap

| Priority | Item |
|:---|:---|
| High | Recalibrate τ values against serving confidence scale (top-10 softmax, not top-100) |
| High | Wire `iso_cal` into serving path with recalibrated thresholds |
| High | `dept_prior_map` as a 14th XGBoost feature — P(target_dept \| source_dept) built during training, not yet used at inference |
| High | Switch to `rank:pairwise` XGBoost objective — expected +3–6pp Top-1 |
| Medium | Cross-encoder reranker on XGBoost top-10 at inference — model trained, not yet wired in; expected +5–12pp Top-1 |
| Medium | Synthetic positives for rare VT targets — 79% appear once in training |
| Medium | Held-out calibration split for `iso_cal` |

---

## Project Structure

```
transferzaidemo/
├── app.py                      # Streamlit UI
├── predict.py                  # Inference: load_artifacts(), predict_transfer(), evaluate_transcript()
├── config.py                   # Hyperparameters, per-institution thresholds, model paths
├── inference_service/main.py   # FastAPI service wrapping predict.py
├── scripts/
│   ├── build_artifacts.py      # Full training pipeline — BGE, XGBoost, eval, S3 upload
│   └── fetch_artifacts.py      # CI step: download artifacts before Docker build
├── eval/
│   ├── audit.py                # Confidence intervals + retrieval recall@k
│   ├── benchmark_rerankers.py  # RRF baseline vs. cross-encoder vs. BGE-Reranker-v2-m3
│   ├── ucsc_error_analysis.py  # UCSC error breakdown by failure type
│   └── run_llm_eval.py         # Claude-as-judge evaluation on hard cases
├── artifacts/                  # Serialized model artifacts (pkl + npy)
│   ├── classifier.pkl          # XGBoost reranker (500 trees)
│   ├── tfidf.pkl               # TF-IDF vectorizer (15k features)
│   ├── iso_cal.pkl             # Isotonic calibrator — offline Brier/ECE only
│   ├── scorecard.pkl           # Per-institution metrics + threshold sweep
│   └── {wm,vt,ucsc}_{lookup,codes,embeddings}.*
└── data/
    ├── catalogs/               # Full course catalogs per institution
    └── equivalency/            # Ground-truth transfer equivalency tables
```
