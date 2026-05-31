<h1 align="center">TransferzAI</h1>
<p align="center"><strong>AI-powered transfer credit evaluation across institutions</strong></p>
<p align="center">
  Paste your community college courses — get back calibrated transfer probabilities for partner universities.
</p>

---

## What it does

TransferzAI tells you whether your community college courses transfer to William & Mary, Virginia Tech, or UC Santa Cruz — with a confidence score and a plain-English verdict. Paste a full transcript of course titles and descriptions and get back a per-course breakdown plus estimated transferable credit count.

**Live demo:** [transferzai.streamlit.app](https://transferzai.streamlit.app)

Designed for students and college advisors where false positives are costly. The system is built to abstain rather than guess: at the high-confidence threshold, precision exceeds 90% on held-out test data.

---

## Results

| Institution | Top-1 | Top-3 | Precision @ τ | Coverage @ τ | τ | Brier | ECE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| William & Mary | 56.7% | 77.6% | 90.9% | 32.8% | 0.80 | 0.0125 | 0.0089 |
| Virginia Tech | 53.3% | 85.0% | 94.1% | 28.3% | 0.85 | 0.0143 | 0.0089 |
| UC Santa Cruz | 43.6% | 65.7% | 91.7% | 19.9% | 0.90 | 0.0148 | 0.0098 |

**Top-1** = correct course is the #1 prediction. **Top-3** = correct course is in the top 3 predictions.  
**τ** = first confidence threshold where precision ≥ 90% on held-out test data. **Coverage** = fraction of queries where the model is confident enough to return an answer — the system abstains on the rest.

---

## Architecture

Two-stage retrieve-then-rerank pipeline, run per institution.

**Stage 1 — RRF Retrieval**

Fuses two signals to find the top-100 candidates:

| Signal | Role |
|:---|:---|
| Fine-tuned BGE bi-encoder | Semantic similarity (contrastive learning on transfer pairs) |
| TF-IDF (1–2 gram, 15k features) | Lexical keyword overlap |

Reciprocal Rank Fusion (k=60) combines the ranked lists. Training retrieval also adds a 0.5-weighted department string-similarity signal to improve hard-negative recall — this is intentionally excluded from inference retrieval.

**Stage 2 — XGBoost Reranker**

Scores each of the top-100 candidates on 13 features:

| Group | Features |
|:---|:---|
| Semantic | `bge_sim` — BGE cosine similarity |
| Lexical | `tfidf_sim` (full text), `tfidf_title_sim` (titles only), `title_sim` (SequenceMatcher ratio) |
| Structural | `dept_sim` (source→target dept code similarity), `level_ratio`, `same_level`, `rrf_score` |
| Interactions | `bge_x_dept`, `bge_x_title`, `bge_x_tfidf`, `dept_x_title`, `dept_x_level` |

Per-query softmax over the top-10 XGBoost margins gives the confidence score. Restricting softmax to the top-10 candidates prevents dilution across the full 100-candidate pool.

**Abstention gate**

| Confidence | Label | Display |
|:---|:---|:---|
| ≥ 0.84 | Confirmed | Green — counts toward transfer eligibility |
| ≥ 0.74 | Possible | Yellow — shown but not counted as confirmed |
| < 0.74 | No match | Gray — system abstains |

### Key design decisions

**Semantic similarity ≠ transfer eligibility.**

"Introduction to Chemistry" and "Organic Chemistry II" score highly similar in any embedding space — but they don't transfer to the same course. Naively using a cross-encoder reranker dropped top-3 recall from ~46% to 28% by confidently promoting semantically similar but non-transferring courses.

The fix: fine-tune the retriever on transfer pairs using contrastive learning, so it learns the "transfers as" relationship rather than generic similarity. Structural features (level, title match, dept alignment) in the reranker provide additional signal that pure semantic models miss.

**Department similarity is live at inference.**

Source dept codes (e.g. VCCS `HIS`, `MTH`, `BIO`) are compared against target institution dept codes using SequenceMatcher. Cross-dept matches like `HIS→HIST` score ~0.86, while cross-semantic mappings like `ACC→BUAD` score ~0.29 — matching the distribution seen during training. All 13 features including dept interactions are active.

---

## Training data

| Source → Target | Train pairs | Test pairs |
|:---|:---:|:---:|
| VCCS → William & Mary | ~214 | 67 |
| VCCS → Virginia Tech | ~193 | 60 |
| CCC → UC Santa Cruz | ~578 | 181 |

FERPA compliant — course-level data only, no student PII. Equivalency tables sourced from official articulation agreements.

---

## MLOps

| Tool | Role |
|:---|:---|
| **MLflow** | Experiment tracking — logs hyperparams, Top-1/Top-3/Precision@τ/Brier/ECE per institution, and XGBoost model artifact for every training run |
| **AWS S3** | Artifact storage — all pkl/npy files auto-uploaded to `transferzai-artifacts` after training; `predict.py` downloads from S3 when `AWS_ACCESS_KEY_ID` is set |
| **HuggingFace Hub** | Deployment fallback — `hyperalpha/transferzai-artifacts` (artifacts), `hyperalpha/transferzai-bge` (BGE model) used for Streamlit Cloud |
| **Docker** | Container — `Dockerfile` packages the Streamlit app for deployment |

---

## Quickstart

```bash
git clone https://github.com/neel-davuluri/transferzaidemo.git
cd transferzaidemo
pip install -r requirements.txt
python download_artifacts.py   # pulls pre-trained model from HuggingFace
streamlit run app.py
```

**Retrain from scratch:**

```bash
python scripts/build_artifacts.py
# ~8 min on Apple MPS, ~20 min on CPU
# Auto-logs to MLflow and uploads artifacts to S3
```

**View training runs:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5000
```

**Docker:**

```bash
docker build -t transferzai:latest .
docker run -p 8501:8501 transferzai:latest
```

**Inference API:**

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

## Project structure

```
transferzaidemo/
├── app.py                      # Streamlit app — Transcript Evaluator
├── predict.py                  # Inference: load_artifacts(), evaluate_transcript()
├── config.py                   # Hyperparameters, thresholds, model paths
├── paths.py                    # Centralized data file paths
├── download_artifacts.py       # Pull pre-trained artifacts from HuggingFace Hub
├── Dockerfile                  # Container for Streamlit deployment
├── requirements.txt            # Python dependencies
├── scripts/
│   ├── build_artifacts.py      # Full retrain pipeline (BGE embeddings + XGBoost + eval + S3 upload)
│   ├── build_vt_dataset.py     # Build VCCS→VT equivalency dataset
│   ├── build_ccc_ucsc_dataset.py  # Build CCC→UCSC equivalency dataset
│   └── ...                     # Catalog builders, data utilities
├── pipeline/                   # Step-by-step training modules
│   ├── step1_split.py          # Train/test split
│   ├── step2_hard_negatives.py # Mine hard negatives from retrieval
│   ├── step2_synthetic_negatives.py  # Claude-generated synthetic negatives
│   ├── step3_finetune_bge.py   # Fine-tune BGE bi-encoder (contrastive learning)
│   ├── step4_three_signal_rrf.py    # Build RRF retrieval
│   ├── step5_xgboost.py        # Train XGBoost reranker
│   ├── step6_final_eval.py     # Final evaluation
│   └── step7_train_cross_encoder.py # Train cross-encoder (eval only, not used at inference)
├── eval/
│   ├── benchmark_rerankers.py  # Compare reranking strategies side by side
│   ├── test_cross_encoder.py   # Cross-encoder evaluation
│   ├── eval_product_metrics.py # Core metric suite (Top-1/3, Prec@τ, Cov@τ, Brier, ECE)
│   ├── sequence_features.py    # Shared text augmentation helpers
│   └── ...                     # Error analysis, institution-specific audits
├── artifacts/                  # Serialized model artifacts (pkl + npy)
│   ├── classifier.pkl          # XGBoost reranker
│   ├── tfidf.pkl               # TF-IDF vectorizer (15k features)
│   ├── iso_cal.pkl             # Isotonic calibrator
│   ├── scorecard.pkl           # Per-institution metrics + full threshold sweep
│   ├── feature_names.pkl       # Ordered feature names (must match extract_signals)
│   ├── dept_prior_map.pkl      # P(target_dept | source_dept) — built, not yet used
│   └── {wm,vt,ucsc}_{lookup,codes,embeddings}.*  # Per-institution catalog artifacts
└── data/
    ├── catalogs/               # Full course catalogs (W&M, VT, UCSC, Northeastern)
    └── equivalency/            # Ground-truth transfer equivalency tables
```

---

## Limitations

- **Coverage is intentionally low.** At the high-confidence threshold (0.84), the system answers ~20-33% of queries. This is by design — better to say "check with an advisor" than predict incorrectly.
- **Top-1 recall is moderate.** The correct course is the top prediction ~44-57% of the time; top-3 is 66-85%. The system is most useful when showing ranked options rather than a single answer.
- **Sparse training data.** Only 7.4% (W&M), 4.2% (VT), and 3.4% (UCSC) of catalog courses have any training labels. 79% of VT target courses appear only once in training.
- **Not a registrar decision.** Always confirm transfer credit decisions with your school's registrar before acting on results.

---

## Contact

Neel Davuluri · neel.davuluri@gmail.com  
Garrett Bellin
