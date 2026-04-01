"""Configuration constants for TransferZAI v3."""

# Model
BGE_MODEL_PATH = "./finetuned_bge"
QUERY_PREFIX = "Represent this course for finding transfer equivalents: "

# Retrieval
RETRIEVAL_K = 50
RRF_K = 60
DEPT_WEIGHT = 0.5
TOP_K_DISPLAY = 5

# Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.7
TRANSFER_THRESHOLD = 0.5

# Transcript evaluation
DEFAULT_CREDITS_PER_COURSE = 3
MIN_CREDITS_REQUIRED = 30

# Artifacts directory
ARTIFACTS_DIR = "./artifacts"
