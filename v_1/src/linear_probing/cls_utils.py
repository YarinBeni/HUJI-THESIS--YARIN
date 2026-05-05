"""
Shared utilities for linear classification probing.

Public API
----------
fit_cls_cv(X, y, cv_strategy, groups, n_splits, C, random_state)
    -> dict with accuracy/macro_f1/weighted_f1 mean/std/folds + baselines
"""

import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


def fit_cls_cv(
    X: np.ndarray,
    y,
    cv_strategy: str = 'stratified',  # 'stratified' or 'grouped'
    groups=None,
    n_splits: int = 5,
    C: float = 1.0,
    random_state: int = 42,
) -> dict:
    """
    Logistic regression CV probe.

    cv_strategy='stratified'  — StratifiedKFold; use for ruler task.
    cv_strategy='grouped'     — GroupKFold by groups; use for year task (groups=ruler).

    Returns dict: accuracy / macro_f1 / weighted_f1  mean + std + folds,
                  plus chance baselines.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(np.asarray(y))
    n_classes = len(le.classes_)

    if cv_strategy == 'grouped':
        splits = list(GroupKFold(n_splits=n_splits).split(X, y_enc, groups))
    else:
        splits = list(
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            .split(X, y_enc)
        )

    accs, mac_f1s, wt_f1s = [], [], []
    for tr_idx, val_idx in splits:
        clf = LogisticRegression(
            max_iter=2000, C=C, solver='lbfgs', random_state=random_state,
        )
        clf.fit(X[tr_idx], y_enc[tr_idx])
        yp = clf.predict(X[val_idx])
        accs.append(float(accuracy_score(y_enc[val_idx], yp)))
        mac_f1s.append(float(f1_score(y_enc[val_idx], yp, average='macro', zero_division=0)))
        wt_f1s.append(float(f1_score(y_enc[val_idx], yp, average='weighted', zero_division=0)))

    counts = Counter(y_enc.tolist())
    majority_frac = max(counts.values()) / len(y_enc)

    return {
        'n_classes':           n_classes,
        'n_splits':            n_splits,
        'chance_accuracy':     float(majority_frac),
        'chance_macro_f1':     float(1.0 / n_classes),
        'accuracy_mean':       float(np.mean(accs)),
        'accuracy_std':        float(np.std(accs)),
        'accuracy_folds':      accs,
        'macro_f1_mean':       float(np.mean(mac_f1s)),
        'macro_f1_std':        float(np.std(mac_f1s)),
        'macro_f1_folds':      mac_f1s,
        'weighted_f1_mean':    float(np.mean(wt_f1s)),
        'weighted_f1_std':     float(np.std(wt_f1s)),
        'weighted_f1_folds':   wt_f1s,
    }
