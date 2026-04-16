"""Configuration constants for TransferZAI v3."""

# Model
BGE_MODEL_PATH = "./finetuned_bge_three"
QUERY_PREFIX = "Represent this course for finding transfer equivalents: "

# Retrieval
RETRIEVAL_K = 50
RRF_K = 60
DEPT_WEIGHT = 0.5
TOP_K_DISPLAY = 5

# Thresholds — per-query log-softmax confidence (NOT absolute predict_proba)
# At τ=0.60: W&M/VT hit 100% precision (~20% coverage). Show "Transfers as X".
# At τ=0.30: ~70% precision, ~50% coverage. Show "Possible match — advisor review".
HIGH_CONFIDENCE_THRESHOLD = 0.60
TRANSFER_THRESHOLD = 0.30

# Transcript evaluation
DEFAULT_CREDITS_PER_COURSE = 3
MIN_CREDITS_REQUIRED = 30

# Artifacts directory
ARTIFACTS_DIR = "./artifacts"

# Cross-encoder reranker
CROSS_ENCODER_PATH = "./cross_encoder"
CROSS_ENCODER_RERANK_K = 10   # rerank top-K from LogReg with cross-encoder
