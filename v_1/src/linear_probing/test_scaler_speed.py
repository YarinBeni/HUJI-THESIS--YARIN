"""
Quick test: does StandardScaler speed up LogisticRegression?
Single fit (no CV), tracks iterations. Safe for interactive use.
Run on cluster: python v_1/src/linear_probing/test_scaler_speed.py
"""

import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from utils import (
    load_letters, get_splits, load_layer_activations,
    PERIODS, SEED,
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

# Load one layer
model_name = 'qwen2.5-7b-instruct'
layer = 14
C = 1.0

print(f"Loading layer {layer} activations...")
X = load_layer_activations(model_name, 'tier0', layer)
X_tv = X[train_val_idx]

# Simple 80/20 split for speed
X_train, X_test, y_train, y_test = train_test_split(
    X_tv, y_tv, test_size=0.2, random_state=SEED, stratify=y_tv
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Value range: [{X_train.min():.1f}, {X_train.max():.1f}], std={X_train.std():.2f}")
print()


def run_test(name, X_tr, X_te, y_tr, y_te, solver, max_iter):
    print(f"--- {name} (solver={solver}, max_iter={max_iter}) ---")
    clf = LogisticRegression(
        C=C, solver=solver, max_iter=max_iter,
        random_state=SEED, verbose=0,
    )
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    acc = clf.score(X_te, y_te)
    n_iter = clf.n_iter_[0]  # actual iterations used
    print(f"  Time: {elapsed:.1f}s | Iterations: {n_iter}/{max_iter} | Acc: {acc:.4f}")
    return elapsed, n_iter, acc


# --- Test 1: lbfgs, no scaling ---
e1, i1, a1 = run_test("lbfgs NO scaler", X_train, X_test, y_train, y_test,
                       solver='lbfgs', max_iter=1000)

# --- Test 2: lbfgs, WITH scaling ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print(f"  (Scaled range: [{X_train_s.min():.1f}, {X_train_s.max():.1f}], std={X_train_s.std():.2f})")
e2, i2, a2 = run_test("lbfgs WITH scaler", X_train_s, X_test_s, y_train, y_test,
                       solver='lbfgs', max_iter=1000)

# --- Test 3: saga, no scaling ---
e3, i3, a3 = run_test("saga NO scaler", X_train, X_test, y_train, y_test,
                       solver='saga', max_iter=1000)

# --- Test 4: saga, WITH scaling ---
e4, i4, a4 = run_test("saga WITH scaler", X_train_s, X_test_s, y_train, y_test,
                       solver='saga', max_iter=1000)

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Config':<30s} {'Time':>6s} {'Iters':>6s} {'Acc':>7s}")
print("-" * 60)
print(f"{'lbfgs, no scaler':<30s} {e1:>5.1f}s {i1:>5d}  {a1:>6.4f}")
print(f"{'lbfgs, with scaler':<30s} {e2:>5.1f}s {i2:>5d}  {a2:>6.4f}")
print(f"{'saga, no scaler':<30s} {e3:>5.1f}s {i3:>5d}  {a3:>6.4f}")
print(f"{'saga, with scaler':<30s} {e4:>5.1f}s {i4:>5d}  {a4:>6.4f}")
