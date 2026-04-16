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

# Thresholds — isotonic-calibrated XGBoost probability (absolute, not softmax)
# calib_prob is fitted on sigmoid(margin) vs ground-truth labels, so it measures
# "how likely is this specific course to transfer" independently of pool size.
# At τ=0.65: high precision; "Transfers as X" verdict.
# At τ=0.35: moderate confidence; "Advisor review recommended".
HIGH_CONFIDENCE_THRESHOLD = 0.65
TRANSFER_THRESHOLD = 0.35

# Transcript evaluation
DEFAULT_CREDITS_PER_COURSE = 3
MIN_CREDITS_REQUIRED = 30

# Artifacts directory
ARTIFACTS_DIR = "./artifacts"

# Cross-encoder reranker
CROSS_ENCODER_PATH = "./cross_encoder"
CROSS_ENCODER_RERANK_K = 10   # rerank top-K from LogReg with cross-encoder
