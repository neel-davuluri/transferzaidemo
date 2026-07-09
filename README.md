<h1 align="center">TransferzAI</h1>

<p align="center">
  <strong>ML pipeline for university transfer credit matching</strong>
</p>

<p align="center">
  <a href="https://transfersai.streamlit.app"><strong>→ Live demo</strong></a> ·
  <a href="https://huggingface.co/hyperalpha/transferzai-bge">Fine-tuned BGE model</a> ·
  <a href="https://huggingface.co/hyperalpha/transferzai-artifacts">Artifacts</a> ·
  <a href="TECHNICAL.md">Results & evaluation</a>
</p>

---

## Problem

When a college athlete enters the transfer portal, coaches have a narrow window to evaluate and recruit them. Academic eligibility — whether the player's credits transfer and they'll have enough standing to compete — has to be verified by the registrar before a scholarship offer can be extended. That process takes days or weeks. In the meantime, the player is fielding offers from other schools, and the first program to give them a clear answer often wins the recruitment. A D1 athletics department was doing this eligibility check by hand for every prospective transfer.

TransferzAI automates that lookup. Give it a course title and description; it returns a ranked shortlist of matches from the target institution's catalog with a confidence score. When it isn't confident enough, it abstains — wrong answers cost students a semester.

---

## Architecture

Two-stage retrieve-then-rerank pipeline over 11,850 catalog courses across three institutions (William & Mary, Virginia Tech, UC Santa Cruz).

```
Course (title + description + dept)
              │
              ▼
┌─────────────────────────────────────────────────┐
│                Stage 1 — Retrieval              │
│                                                 │
│  Fine-tuned BGE bi-encoder                      │
│  (contrastive loss on 1,232 transfer pairs,     │
│   teaches "transfers as" not just "similar to") │
│              │                                  │
│              ├──── cosine sim → top-200         │
│              │                                  │
│  TF-IDF (15k features, 1-2 gram)                │
│              │                                  │
│              └──── cosine sim → top-200         │
│                         │                       │
│               RRF  Σ 1/(60 + rank)              │
│                    → top-100 candidates         │
└─────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│               Stage 2 — Reranking               │
│                                                 │
│  13 features per candidate:                     │
│  BGE sim · TF-IDF sim · title sim ·             │
│  dept code sim · course level ·                 │
│  5 interaction terms                            │
│              │                                  │
│  XGBoost (500 trees, depth=4, pointwise)        │
│              │                                  │
│  softmax over top-10 decision margins           │
│  → confidence score                             │
└─────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│           Per-Institution Confidence Gate        │
│                                                 │
│  conf ≥ τ_inst  →  Confirmed   (green)          │
│  conf ≥ 0.74    →  Possible    (yellow)         │
│  conf < 0.74    →  abstain                      │
│                                                 │
│  τ: W&M = 0.79 · VT = 0.85 · UCSC = 0.85      │
└─────────────────────────────────────────────────┘
```

---

## Key Engineering Decisions

**Why two retrievers instead of one?**
BGE captures semantic equivalence; TF-IDF catches keyword overlap that embedding models miss (e.g., "Organic Chemistry I" ↔ "CHEM 213"). Fusing both via Reciprocal Rank Fusion consistently outperforms either retriever alone on recall@100, which is the ceiling for Stage 2.

**Why fine-tune the bi-encoder rather than use it off-the-shelf?**
Off-the-shelf embeddings measure topical similarity, not transfer equivalence. "Intro to Chemistry" and "Organic Chemistry II" are semantically close but don't map to the same course. A cross-encoder trained on transfer pairs actually made this worse — it dropped Top-3 recall from 46% to 28% by confidently promoting semantically similar but non-equivalent courses. Contrastive fine-tuning on 1,232 labeled pairs teaches the retriever the "transfers as" relationship specifically.

**Why XGBoost over a neural reranker?**
~1,200 labeled pairs — too few to fine-tune a cross-encoder without overfitting. XGBoost on 13 handcrafted features generalizes better at this data scale, runs in <1ms per query, and produces interpretable feature importances. The 13 features cover semantic similarity, lexical overlap, structural signals (dept code similarity, course level), and their pairwise interactions.

**Why train/inference retrieval use different signals?**
Training retrieval adds a department similarity boost (weight=0.5) to ensure the true match lands in the top-100 pool — without it, ~8% of positives fall outside the window XGBoost ever sees. Inference retrieval drops this boost; empirically it shifts the score distribution in a way that hurts precision. The asymmetry is intentional and documented.

**Abstention is the product.**
At the confirmed threshold, the system answers 20–33% of queries and is right >90% of the time. This isn't a coverage problem — it's the design. For a domain where a wrong answer costs a student a semester, selective prediction at high precision is more useful than high recall at moderate precision. Per-institution thresholds (not a single global value) account for catalog size and training density differences across schools.

---

## How It Works

**Stage 1 — Retrieval**

BGE and TF-IDF each produce a ranked list of up to 200 candidates independently. Reciprocal Rank Fusion combines them: each candidate gets a score of `Σ 1/(60 + rank)` across both lists. The 60 constant down-weights rank differences at the top while preserving signal from candidates that appear in both lists. The top-100 by RRF score pass to Stage 2.

**Stage 2 — Reranking**

XGBoost scores each of the 100 candidates on 13 features. The confidence score is softmax over the top-10 decision margins — a relative signal measuring how decisively the top candidate outscored the other finalists, not a calibrated match probability. Using only the top-10 prevents dilution when department signals surface many same-department candidates simultaneously.

---

## Quickstart

```bash
git clone https://github.com/neel-davuluri/transferzaidemo.git
cd transferzaidemo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Artifacts download automatically from HuggingFace Hub on first run (~500 MB)
streamlit run app.py
```

**Retrain from scratch (~8 min on MPS, ~20 min on CPU):**

```bash
python scripts/build_artifacts.py
# Logs to MLflow, auto-uploads artifacts to S3 if credentials are set
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## MLOps

| Tool | Role |
|:---|:---|
| **MLflow** | Logs hyperparams + per-institution metrics for every training run |
| **AWS S3** (`transferzai-artifacts`) | Artifacts auto-uploaded post-training; downloaded at startup when credentials are set |
| **HuggingFace Hub** | [`hyperalpha/transferzai-artifacts`](https://huggingface.co/hyperalpha/transferzai-artifacts) · [`hyperalpha/transferzai-bge`](https://huggingface.co/hyperalpha/transferzai-bge) — deployment fallback |
| **Docker** | Python 3.11-slim, artifacts pre-loaded, port 8501 |
| **FastAPI + Lambda** | Inference service deployable as a container on Lambda with SAM; scale-to-zero with provisioned concurrency on a schedule |

---

## Contact

Neel Davuluri · [neel.davuluri@gmail.com](mailto:neel.davuluri@gmail.com)
