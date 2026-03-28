"""
Quick test: does StandardScaler actually speed up LogisticRegression convergence?
Runs one layer, one C value, one 5-fold CV — with and without scaling.
Run on cluster: python v_1/src/linear_probing/test_scaler_speed.py
"""

import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import (
    load_letters, get_splits, load_layer_activations, load_metadata,
    PERIODS, SEED, C_GRID,
)

warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# Setup
df = load_letters()
le = LabelEncoder()
le.fit(PERIODS)
y_all = le.transform(df['period'].values)
train_idx, val_idx, test_idx = get_splits(df)
train_val_idx = np.concatenate([train_idx, val_idx])
y_tv = y_all[train_val_idx]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Pick middle layer (layer 14) and tier0 cleaning, C=1.0
model_name = 'qwen2.5-7b-instruct'
layer = 14
C = 1.0

print(f"Loading layer {layer} activations...")
X = load_layer_activations(model_name, 'tier0', layer)
X_tv = X[train_val_idx]
print(f"X_tv shape: {X_tv.shape}, y_tv shape: {y_tv.shape}")
print(f"X_tv value range: [{X_tv.min():.2f}, {X_tv.max():.2f}], mean={X_tv.mean():.4f}, std={X_tv.std():.4f}")
print()

# --- Without scaling ---
print("=" * 60)
print("WITHOUT StandardScaler")
print("=" * 60)
clf = LogisticRegression(C=C, max_iter=1000, random_state=SEED, solver='lbfgs')

t0 = time.time()
acc_scores = cross_val_score(clf, X_tv, y_tv, cv=skf, scoring='accuracy', n_jobs=-1)
t1 = time.time()

print(f"  Time: {t1 - t0:.1f}s")
print(f"  Accuracy: {acc_scores.mean():.4f} +/- {acc_scores.std():.4f}")
print()

# --- With scaling ---
print("=" * 60)
print("WITH StandardScaler")
print("=" * 60)
scaler = StandardScaler()
X_tv_scaled = scaler.fit_transform(X_tv)
print(f"X_tv_scaled range: [{X_tv_scaled.min():.2f}, {X_tv_scaled.max():.2f}], mean={X_tv_scaled.mean():.4f}, std={X_tv_scaled.std():.4f}")

clf2 = LogisticRegression(C=C, max_iter=1000, random_state=SEED, solver='lbfgs')

t0 = time.time()
acc_scores2 = cross_val_score(clf2, X_tv_scaled, y_tv, cv=skf, scoring='accuracy', n_jobs=-1)
t1 = time.time()

print(f"  Time: {t1 - t0:.1f}s")
print(f"  Accuracy: {acc_scores2.mean():.4f} +/- {acc_scores2.std():.4f}")
print()

# --- Summary ---
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Without scaler: {acc_scores.mean():.4f} acc")
print(f"  With scaler:    {acc_scores2.mean():.4f} acc")
print(f"  Speed difference will show in the times above.")
print(f"  If times are similar, scaling won't help with speed.")
