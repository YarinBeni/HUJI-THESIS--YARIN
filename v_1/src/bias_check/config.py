"""
Configuration for Bias Check Pipeline.
Paths, hyperparameters, and model architecture definitions.
"""
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = Path(__file__).parent.parent.parent   # → v_1/
DATA_DIR = BASE_DIR / "data"
EVAL_DIR = DATA_DIR / "evaluation"

BIAS_DIR = EVAL_DIR / "bias_check"
FEATURES_DIR = BIAS_DIR / "features"
MODELS_DIR = BIAS_DIR / "models"
METRICS_DIR = BIAS_DIR / "metrics"
PLOTS_DIR = BIAS_DIR / "plots"

# Source data (produced by evaluation pipeline step 02_prepare_texts.py)
TEXTS_PARQUET = EVAL_DIR / "corpora" / "texts_for_evaluation.parquet"

# Output files
ALL_METRICS_JSON = METRICS_DIR / "all_metrics.json"
TRAINING_HISTORY_JSON = METRICS_DIR / "training_history.json"
BIAS_REPORT_MD = BIAS_DIR / "bias_check_report.md"

# =============================================================================
# Labels
# =============================================================================
LABELS = ["Old Babylonian", "Neo-Assyrian", "Late Babylonian"]
LABEL2IDX = {label: i for i, label in enumerate(LABELS)}
IDX2LABEL = {i: label for i, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# =============================================================================
# Featurization
# =============================================================================
TFIDF_KWARGS = dict(
    analyzer='char_wb',
    ngram_range=(2, 5),
    max_features=10_000,
    sublinear_tf=True,
)
INPUT_DIM = TFIDF_KWARGS["max_features"]

# Splits (stratified by period)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42

# =============================================================================
# Model architectures
# Each entry: (name, n_attention_blocks, n_mlp_layers)
#   n_attention_blocks=0 → pure MLP
#   n_attention_blocks>0 → Attention+MLP hybrid
# =============================================================================
MODEL_VARIANTS = [
    # MLP sweep (isolate depth effect)
    ("mlp_1layer",    0, 1),
    ("mlp_2layer",    0, 2),
    ("mlp_3layer",    0, 3),
    ("mlp_5layer",    0, 5),
    # Attention sweep (MLP head fixed at 3 layers)
    ("attn1_mlp3",    1, 3),
    ("attn2_mlp3",    2, 3),
    ("attn3_mlp3",    3, 3),
    ("attn5_mlp3",    5, 3),
]

# =============================================================================
# Training hyperparameters
# =============================================================================
HIDDEN_DIM = 256
ATTN_DIM = 100      # projection dimension for attention blocks (must divide INPUT_DIM=10000)
ATTN_HEADS = 4
DROPOUT = 0.3
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10    # on val loss
LR_SCHEDULER_PATIENCE = 5   # ReduceLROnPlateau

# =============================================================================
# Permutation testing (Ojala & Garriga, JMLR 2010)
# =============================================================================
N_PERMUTATIONS = 1000
PERM_SEED = 0

# Significance thresholds
PVALUE_FAIL = 0.01    # p < 0.01 → FAIL (statistically significant bias)
PVALUE_WARN = 0.05    # p < 0.05 → WARN (marginal)
# p ≥ 0.05 → PASS

# Reference baselines
CHANCE_ACCURACY = 1.0 / NUM_CLASSES          # 33.3%
MAJORITY_BASELINE = 0.491                    # Neo-Assyrian dominance
BINOMIAL_CI_HALF_WIDTH = 0.034              # ±3.4% at 95% CI
