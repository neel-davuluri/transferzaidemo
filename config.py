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

# Thresholds — bias-corrected sigmoid confidence
# XGBoost trained with scale_pos_weight≈49 (2% positive rate).
# Subtracting log(49)≈3.89 from each margin shifts the decision boundary to 0.5
# so probabilities are centered correctly and independent of pool size.
# Monotone in signal quality: adding correct dept/level never decreases confidence.
MARGIN_BIAS_CORRECTION = 3.89   # log(scale_pos_weight) ≈ log(49)
HIGH_CONFIDENCE_THRESHOLD = 0.85
TRANSFER_THRESHOLD = 0.50

# Transcript evaluation
DEFAULT_CREDITS_PER_COURSE = 3
MIN_CREDITS_REQUIRED = 30

# Artifacts directory
ARTIFACTS_DIR = "./artifacts"

# Cross-encoder reranker
CROSS_ENCODER_PATH = "./cross_encoder"
CROSS_ENCODER_RERANK_K = 10   # rerank top-K from LogReg with cross-encoder
