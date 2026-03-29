"""
Reproduce bias_dim_reduction.png but for maximal-cleaned data.
Saves to v_1/data/evaluation/bias_check/plots/bias_dim_reduction_maximal.png
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

# ── Paths ────────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
DATA_PATH = THIS_DIR / '../../data/evaluation/corpora/texts_for_evaluation.parquet'
PLOT_DIR  = THIS_DIR / '../../data/evaluation/bias_check/plots'

PERIOD_MAP = {'Old Babylonian': 'OB', 'Neo-Assyrian': 'NA', 'Late Babylonian': 'LB'}
COLORS = {'OB': '#1976D2', 'NA': '#7B1FA2', 'LB': '#E53935'}

# ── Cleaning ─────────────────────────────────────────────────────────────────
def clean_tier0(t):
    t = re.sub(r'@[a-z0-9]+', '', t)
    t = t.replace('\xa0', ' ').replace('\u2093', '')
    return t

FILTERS = [
    lambda t: re.sub(r'[0-9]', '', t),
    lambda t: ' '.join(t.split()[:30]),
    lambda t: re.sub(r'-(am|im|um|tam|tim|šum)\b', '', t),
    lambda t: t.replace('w', '').replace('y', ''),
    lambda t: re.sub(r'\b[A-ZŠṢṬḪ][A-ZŠṢṬḪ0-9]+-?', '', t),
    lambda t: re.sub(r'\b(I|d|lu2|uru|giš|tug2)-', '', t),
    lambda t: ' '.join(re.findall(r'[a-zšṣṭḫāīūē][a-zšṣṭḫāīūē0-9-]*', t)),
    lambda t: t.translate(str.maketrans('āīūēĀĪŪĒ', 'aiueAIUE')),
    lambda t: re.sub(r'([a-zšṣṭḫ])([2-9])', r'\1', t),
    lambda t: t.lower(),
    lambda t: re.sub(r'-meš\b', '', t),
]

def clean_maximal(text):
    t = clean_tier0(text)
    for fn in FILTERS:
        t = fn(t)
    return t

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
if 'full_text' in df.columns:
    df = df.rename(columns={'full_text': 'text'})
df['period'] = df['period'].map(PERIOD_MAP)

print("Applying maximal cleaning...")
texts = df['text'].apply(clean_maximal)
labels = df['period'].values

print(f"  {len(texts)} texts, periods: {dict(pd.Series(labels).value_counts())}")

# ── Plot ──────────────────────────────────────────────────────────────────────
configs = [
    ('Unigrams (1,1)', (1, 1)),
    ('Bigrams (2,2)',   (2, 2)),
    ('2-5 grams (2,5)', (2, 5)),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (label, ngram_range) in zip(axes, configs):
    print(f"  TF-IDF + t-SNE for {label}...")
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range,
                          max_features=10_000, sublinear_tf=True)
    X_sparse = vec.fit_transform(texts)

    n_svd = min(50, X_sparse.shape[1] - 1)
    X_svd = TruncatedSVD(n_components=n_svd, random_state=42).fit_transform(X_sparse)
    X_2d  = TSNE(n_components=2, random_state=42, perplexity=40,
                 max_iter=500, metric='euclidean').fit_transform(X_svd)

    for period, color in COLORS.items():
        mask = labels == period
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, label=f'{period} (n={mask.sum()})',
                   alpha=0.35, s=6, linewidths=0, rasterized=True)

    ax.set_title(f'{label}\n(tSNE, cosine, maximal clean)', fontsize=11)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
    ax.legend(markerscale=4, fontsize=9, loc='best', framealpha=0.8)

fig.suptitle('TF-IDF Period Separation — Char N-gram Feature Space (Maximal Cleaning)',
             fontsize=14, y=1.01)
plt.tight_layout()

out_path = PLOT_DIR / 'bias_dim_reduction_maximal.png'
plt.savefig(out_path, bbox_inches='tight', dpi=200)
print(f"Saved {out_path}")
