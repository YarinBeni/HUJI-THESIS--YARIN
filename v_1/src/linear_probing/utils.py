"""
Shared utilities for the linear probing pipeline.
Data loading, cleaning functions, splits, constants.
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

# =============================================================================
# Constants
# =============================================================================
SEED = 42

# Paths (relative to this file's location)
_THIS_DIR = Path(__file__).resolve().parent
DATA_PATH = _THIS_DIR / '../../data/evaluation/corpora/texts_for_evaluation.parquet'
RESULTS_DIR = _THIS_DIR / 'results'

# Mapping from full period names in parquet -> short labels used throughout
PERIOD_MAP = {
    'Old Babylonian': 'OB',
    'Neo-Assyrian':   'NA',
    'Late Babylonian': 'LB',
}

PERIODS = ['OB', 'NA', 'LB']  # label order for encoding

PERIOD_COLORS = {
    'OB': '#1976D2',
    'NA': '#7B1FA2',
    'LB': '#E53935',
}

# TF-IDF baselines from the bias check (letters, 5-fold CV logistic regression)
TFIDF_BASELINES = {
    'unigram_raw':     0.848,
    'bigram_raw':      0.983,
    '2_5gram_raw':     0.992,
    'unigram_cleaned': 0.691,
    'bigram_cleaned':  0.912,
    '2_5gram_cleaned': 0.967,
}

# Regularization grid for logistic regression
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Split ratios (must match bias check)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42


# =============================================================================
# Data loading
# =============================================================================
def load_letters() -> pd.DataFrame:
    """Load the 4,957 letters with columns: text, period, fragment_id, etc."""
    df = pd.read_parquet(DATA_PATH)
    # Rename full_text -> text for consistent downstream usage
    if 'full_text' in df.columns and 'text' not in df.columns:
        df = df.rename(columns={'full_text': 'text'})
    assert 'text' in df.columns, f"Missing 'text' column. Columns: {df.columns.tolist()}"
    assert 'period' in df.columns, f"Missing 'period' column. Columns: {df.columns.tolist()}"
    assert len(df) == 4957, f"Expected 4,957 rows, got {len(df)}"
    # Normalize period labels to short form (OB/NA/LB)
    if df['period'].iloc[0] in PERIOD_MAP:
        df = df.copy()
        df['period'] = df['period'].map(PERIOD_MAP)
    assert set(df['period'].unique()) <= set(PERIODS), f"Unexpected period values: {df['period'].unique()}"
    return df


# =============================================================================
# Cleaning functions
# =============================================================================
def clean_tier0(t: str) -> str:
    """Minimal: strip ORACC @v markup, non-breaking space, subscript-x."""
    t = re.sub(r'@[a-z0-9]+', '', t)
    t = t.replace('\xa0', ' ')       # non-breaking space
    t = t.replace('\u2093', '')       # subscript x (U+2093)
    return t


# All 11 cleaning filters from the bias check notebook (cell 16).
# Applied in order for maximal cleaning.
FILTERS = {
    'strip ALL digits':
        lambda t: re.sub(r'[0-9]', '', t),
    'truncate 30 tokens':
        lambda t: ' '.join(t.split()[:30]),
    'strip case endings (-am,-im,-um,-tam,-tim,-šum)':
        lambda t: re.sub(r'-(am|im|um|tam|tim|šum)\b', '', t),
    'strip w/y':
        lambda t: t.replace('w', '').replace('y', ''),
    'remove logograms (ALL UPPERCASE tokens)':
        lambda t: re.sub(r'\b[A-ZŠṢṬḪ][A-ZŠṢṬḪ0-9]+-?', '', t),
    'strip determinatives (I-,d-,lu2-,uru-,giš-,tug2-)':
        lambda t: re.sub(r'\b(I|d|lu2|uru|giš|tug2)-', '', t),
    'keep only syllabic tokens':
        lambda t: ' '.join(re.findall(r'[a-zšṣṭḫāīūē][a-zšṣṭḫāīūē0-9-]*', t)),
    'normalize long vowels (ā→a etc)':
        lambda t: t.translate(str.maketrans('āīūēĀĪŪĒ', 'aiueAIUE')),
    'strip subscript digits (sign2→sign)':
        lambda t: re.sub(r'([a-zšṣṭḫ])([2-9])', r'\1', t),
    'lowercase':
        lambda t: t.lower(),
    'strip -meš plural':
        lambda t: re.sub(r'-meš\b', '', t),
}


def clean_maximal(text: str) -> str:
    """Apply tier0 + all 11 filters stacked in order."""
    t = clean_tier0(text)
    for name, fn in FILTERS.items():
        t = fn(t)
    return t


# =============================================================================
# Train / Val / Test splits
# =============================================================================
def get_splits(df: pd.DataFrame, seed: int = SPLIT_SEED):
    """
    70/15/15 stratified split matching the bias check.
    Returns (train_idx, val_idx, test_idx) as numpy arrays of integer indices.
    """
    labels = df['period'].values
    all_idx = np.arange(len(df))

    # First split: separate test (15%) from the rest (85%)
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_RATIO, random_state=seed
    )
    rest_idx, test_idx = next(sss_test.split(all_idx, labels))

    # Second split: separate val (15% of total = 15/85 of remainder) from train
    val_frac = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)   # 0.15 / 0.85 ≈ 0.1765
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=seed
    )
    labels_rest = labels[rest_idx]
    train_local, val_local = next(sss_val.split(rest_idx, labels_rest))

    train_idx = rest_idx[train_local]
    val_idx = rest_idx[val_local]

    # Sanity checks
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df), \
        f"Split sizes don't sum to {len(df)}"
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap"

    return train_idx, val_idx, test_idx


# =============================================================================
# Model helpers
# =============================================================================
def model_short_name(model_id: str) -> str:
    """Convert HuggingFace model ID to a filesystem-safe short name.

    Example: 'meta-llama/Llama-3.1-8B-Instruct' -> 'llama-3.1-8b-instruct'
    """
    name = model_id.split('/')[-1]   # drop org prefix
    return name.lower()


def mean_pool(hidden_states, attention_mask):
    """Mean-pool hidden states over non-padding positions.

    Args:
        hidden_states: (batch, seq_len, hidden_dim) tensor
        attention_mask: (batch, seq_len) tensor of 0/1

    Returns:
        (batch, hidden_dim) tensor
    """
    mask = attention_mask.unsqueeze(-1).float()   # (batch, seq_len, 1)
    summed = (hidden_states * mask).sum(dim=1)    # (batch, hidden_dim)
    counts = mask.sum(dim=1).clamp(min=1)         # (batch, 1)
    return summed / counts


# =============================================================================
# Activation I/O helpers
# =============================================================================
def activations_dir(model_name: str, cleaning: str) -> Path:
    """Return path to activations directory for a model + cleaning condition."""
    return RESULTS_DIR / 'activations' / model_name / cleaning


def load_layer_activations(model_name: str, cleaning: str, layer: int) -> np.ndarray:
    """Load activations for one layer. Returns (n_texts, hidden_dim) array."""
    path = activations_dir(model_name, cleaning) / f'layer_{layer:02d}.npz'
    return np.load(path)['activations']


def load_metadata(model_name: str, cleaning: str) -> dict:
    """Load metadata JSON for a model + cleaning condition."""
    path = activations_dir(model_name, cleaning) / 'metadata.json'
    with open(path) as f:
        return json.load(f)
