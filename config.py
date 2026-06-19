"""Configuration constants for TransferZAI v3."""

# Model
BGE_MODEL_PATH       = "./finetuned_bge_three"   # local path (takes precedence)
BGE_HF_REPO          = "hyperalpha/transferzai-bge"
ARTIFACTS_HF_REPO    = "hyperalpha/transferzai-artifacts"
QUERY_PREFIX = "Represent this course for finding transfer equivalents: "

# Retrieval
RETRIEVAL_K = 100
RRF_K = 60
TOP_K_DISPLAY = 5

HIGH_CONFIDENCE_THRESHOLD = 0.84   # confirmed transfer (global fallback)
TRANSFER_THRESHOLD = 0.74          # possible transfer

# Per-institution confirmed thresholds — override the global above.
# W&M can use a lower value because its precision-coverage curve is better behaved
# (89% precision at 0.79 vs. VT/UCSC which drop to ~80% at that threshold).
HIGH_CONFIDENCE_THRESHOLDS = {
    "wm":   0.79,
    "vt":   0.85,
    "ucsc": 0.85,
}

# Internal K used for softmax normalization (independent of display top_k)
SOFTMAX_K = 10

# Transcript evaluation
DEFAULT_CREDITS_PER_COURSE = 3
MIN_CREDITS_REQUIRED = 30

# Artifacts directory
ARTIFACTS_DIR = "./artifacts"

# Cross-encoder reranker (training pipeline only — not used at inference)
CROSS_ENCODER_PATH = "./cross_encoder"
