<h1 align="center">TransferzAI</h1>
<p align="center"><strong>AI-powered transfer credit evaluation across institutions</strong></p>
<p align="center">
  Enter any course title and description — get back calibrated transfer probabilities for partner institutions.
</p>

---

## What it does

Paste a course title and description from any school. TransferzAI tells you whether it transfers to William & Mary, Virginia Tech, or UC Santa Cruz — with a confidence score and a plain-English verdict.

**Live demo:** [transferzai.streamlit.app](https://transferzai.streamlit.app)

Designed for college advising and athletic recruiting, where false positives are costly. The system is built to abstain rather than guess: at the high-confidence threshold, precision exceeds 90% on held-out test data.

---

## Results

| Institution | Top-1 Recall | Top-3 Recall | Precision @ τ | Coverage @ τ | τ |
|:---|:---:|:---:|:---:|:---:|:---:|
| William & Mary | 55.2% | 76.1% | 90.0% | 29.9% | 0.85 |
| Virginia Tech | 51.7% | 83.3% | 92.3% | 21.7% | 0.90 |
| UC Santa Cruz | 43.6% | 64.1% | 91.2% | 18.8% | 0.90 |

**Brier scores:** W&M 0.013 · VT 0.014 · UCSC 0.015  
**ECE:** W&M 0.010 · VT 0.010 · UCSC 0.009

Thresholds τ are data-derived: the lowest confidence where precision ≥ 90% on the held-out test set. Coverage is the fraction of queries where the model is confident enough to make a prediction — the system abstains on the rest.

---

## Architecture

Two-stage retrieve-then-rank pipeline, run per institution.

**Stage 1 — RRF Retrieval** fuses two signals to find the top-50 candidates:

| Signal | Role |
|:---|:---|
| Fine-tuned BGE bi-encoder | Semantic similarity (contrastive learning on transfer pairs) |
| TF-IDF (1–2 gram, 15k features) | Lexical keyword overlap |

**Stage 2 — XGBoost Reranker** scores each candidate on 13 features:

| Group | Features |
|:---|:---|
| Semantic | BGE cosine similarity |
| Lexical | TF-IDF (full text), TF-IDF (title only), SequenceMatcher title ratio |
| Structural | Level ratio, same-level flag, dept name similarity†, RRF score |
| Interactions | BGE×dept†, BGE×title, BGE×TF-IDF, dept×title†, dept×level† |

† Dept features are zeroed out at inference (see below). The model uses 9 active features.

Per-query softmax over the top-5 XGBoost margins gives calibrated confidence. Restricting softmax to the top-5 candidates prevents dilution across the full 50-candidate pool.

### Key insights

**Semantic similarity ≠ transfer eligibility.**

"Introduction to Chemistry" and "Organic Chemistry II" score highly similar in any embedding space — but they don't transfer. A cross-encoder reranker tested in development dropped top-3 recall from 45.5% to 28.1% by confidently promoting semantically similar but non-transferring courses.

The fix: fine-tune the retriever on transfer pairs using contrastive learning, so it learns the "transfers as" relationship rather than generic similarity. Then use structural features (level, title match) in the reranker — not more semantic power.

**Dept codes cause training/inference distribution mismatch.**

Training data pairs VCCS source dept codes (ACC, MTH, CSC) against target institution codes (BUAD, MATH, CSCI). Accounting positives always had `dept_sim ≈ 0` during training because "ACC" never string-matches "BUAD". If a user provides their dept code at inference (BUAD vs BUAD → `dept_sim = 1.0`), XGBoost sees a feature combination it never encountered for accounting queries and produces incorrect negative margins.

The fix: dept features are computed for display in the UI but zeroed out before being fed to XGBoost, restoring in-distribution behavior. The model relies on semantic and lexical features for subject-area matching — which it was actually trained on.

---

## Training data

| Source → Target | Train pairs | Test pairs |
|:---|:---:|:---:|
| VCCS → William & Mary | ~214 | 67 |
| VCCS → Virginia Tech | ~193 | 60 |
| CCC → UC Santa Cruz | ~578 | 181 |

FERPA compliant — course-level data only, no student PII. Equivalency tables sourced from official articulation agreements.

---

## Quickstart

```bash
git clone https://github.com/neel-davuluri/transferzaidemo.git
cd transferzaidemo
pip install -r requirements.txt
python download_artifacts.py   # pulls pre-trained model from HuggingFace
streamlit run app.py
```

Or use the inference API directly:

```python
from predict import predict_transfer

results = predict_transfer(
    vccs_dept="MTH",
    vccs_title="Calculus I",
    vccs_desc="Limits, derivatives, and integrals of single-variable functions...",
    vccs_level=1,   # 1=introductory, 2=intermediate, 3=upper, 4=advanced
)

for inst, matches in results.items():
    print(f"\n{inst.upper()}")
    for r in matches[:3]:
        print(f"  {r['code']:<12} {r['title']:<40} {r['confidence']:.0%} [{r['confidence_label']}]")
```

---

## Project structure

```
transferzaidemo/
├── app.py                    # Streamlit demo (Single Class Lookup + Transcript Evaluator)
├── predict.py                # Inference: predict_transfer(), evaluate_transcript()
├── config.py                 # Hyperparameters (thresholds, retrieval K, model paths)
├── paths.py                  # Data file paths (used by pipeline and eval scripts)
├── download_artifacts.py     # Pull pre-trained artifacts from HuggingFace Hub
├── scripts/
│   ├── build_artifacts.py    # Full retrain: BGE embeddings + XGBoost + artifacts
│   └── ...                   # Catalog/dataset builders
├── pipeline/                 # Step-by-step training modules (split, negatives, finetune, eval)
├── eval/                     # Evaluation scripts and metric reporting
└── data/
    ├── catalogs/             # Institution course catalogs (W&M, VT, UCSC)
    └── equivalency/          # Ground-truth transfer equivalency tables
```

---

## Limitations

- **Coverage is intentionally low.** At the high-confidence threshold, the system abstains on ~75% of queries. This is by design — better to say "check with an advisor" than predict incorrectly.
- **Top-1 recall is moderate.** The correct course is the top prediction ~50% of the time, top-3 ~65–83%. The reranker improves precision over pure retrieval; the tradeoff is some recall loss.
- **Dept code is display-only.** Due to training/inference distribution mismatch, dept features are zeroed out before the model scores candidates. Department is shown in the signal breakdown for context but does not affect rankings.
- **Course level is an optional soft signal.** Providing a level (1–4) improves level-matching across different numbering schemes (100-level vs 1000-level). The system works without it.
- **Not a registrar decision.** Always confirm with the institution before acting on these results.

---

## Contact

Neel Davuluri (neel.davuluri@gmail.com)
Garrett Bellin
