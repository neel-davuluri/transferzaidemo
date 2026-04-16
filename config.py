"""Configuration constants for TransferZAI v3."""

# Model
BGE_MODEL_PATH       = "./finetuned_bge_three"   # local path (takes precedence)
BGE_HF_REPO          = "hyperalpha/transferzai-bge"
ARTIFACTS_HF_REPO    = "hyperalpha/transferzai-artifacts"
QUERY_PREFIX = "Represent this course for finding transfer equivalents: "

# Retrieval
RETRIEVAL_K = 50
RRF_K = 60
DEPT_WEIGHT = 0.5
TOP_K_DISPLAY = 5

# Thresholds — softmax confidence over top-K displayed candidates (not all 50).
# Softmax over K=5 means a winner with margin-gap=3 above competitors shows 85%.
# A gap=1 (uncertain) shows 55%. Calibrated thresholds for K≈5:
HIGH_CONFIDENCE_THRESHOLD = 0.65
TRANSFER_THRESHOLD = 0.35

# Internal K used for softmax normalization (independent of display top_k)
SOFTMAX_K = 5

# Transcript evaluation
DEFAULT_CREDITS_PER_COURSE = 3
MIN_CREDITS_REQUIRED = 30

# Artifacts directory
ARTIFACTS_DIR = "./artifacts"

# Cross-encoder reranker
CROSS_ENCODER_PATH = "./cross_encoder"
CROSS_ENCODER_RERANK_K = 10   # rerank top-K from LogReg with cross-encoder
