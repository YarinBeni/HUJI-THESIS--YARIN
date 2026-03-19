#!/usr/bin/env python3
"""
Step 3: Evaluate trained models on the held-out test set.

Computes accuracy, F1 scores, confusion matrices, and permutation test p-values.
Saves all_metrics.json with per-model metrics and overall verdict.

Usage:
    python 03_evaluate.py
    python 03_evaluate.py --models mlp_1layer mlp_3layer
    python 03_evaluate.py --n-permutations 100   # faster during debugging
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, SGDClassifier

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FEATURES_DIR,
    MODELS_DIR,
    METRICS_DIR,
    ALL_METRICS_JSON,
    LABELS,
    LABEL2IDX,
    IDX2LABEL,
    NUM_CLASSES,
    MODEL_VARIANTS,
    INPUT_DIM,
    N_PERMUTATIONS,
    PERM_SEED,
    PVALUE_FAIL,
    PVALUE_WARN,
    CHANCE_ACCURACY,
    MAJORITY_BASELINE,
    BINOMIAL_CI_HALF_WIDTH,
)
from models import build_model


def load_test_data():
    """Load test split, return dense numpy arrays."""
    X = load_npz(FEATURES_DIR / "test.npz").toarray()
    y = np.load(FEATURES_DIR / "y_test.npy")
    return X, y


@torch.no_grad()
def predict(model, X_dense: np.ndarray, batch_size: int = 512, device=None) -> np.ndarray:
    """Run inference, return predicted class indices."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    preds = []
    X_tensor = torch.tensor(X_dense, dtype=torch.float32)
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size].to(device)
        logits = model(batch)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def per_class_metrics(y_true, y_pred):
    """Return per-class precision, recall, F1 as dicts keyed by class name."""
    report = classification_report(
        y_true, y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    per_class = {}
    for label in LABELS:
        per_class[label] = {
            "precision": report[label]["precision"],
            "recall":    report[label]["recall"],
            "f1":        report[label]["f1-score"],
            "support":   report[label]["support"],
        }
    return per_class


def compute_confusion_matrix(y_true, y_pred):
    """Return confusion matrix as list-of-lists (JSON-serializable)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    return cm.tolist()


def _single_permutation(X_train, y_train, X_test, y_test, seed):
    """Fit SGD on shuffled labels, return accuracy on test set."""
    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(y_train)
    clf = SGDClassifier(loss="log_loss", max_iter=100, tol=1e-3, random_state=seed)
    clf.fit(X_train, y_shuffled)
    return accuracy_score(y_test, clf.predict(X_test))


def compute_null_distribution(X_train, y_train, X_test, y_test, n_perms, seed,
                              n_jobs=-1):
    """
    Compute the permutation null distribution (model-independent).

    Ojala & Garriga (JMLR 2010): shuffle labels n_perms times, refit a fast
    SGDClassifier, record test accuracy. Parallelised with joblib.
    Returns array of permuted accuracies.
    """
    seeds = [seed + i for i in range(n_perms)]
    perm_accs = Parallel(n_jobs=n_jobs)(
        delayed(_single_permutation)(X_train, y_train, X_test, y_test, s)
        for s in seeds
    )
    return np.array(perm_accs)


def p_value_from_null(null_accs, real_acc):
    """Compute p-value = fraction of null accuracies >= real accuracy."""
    return float((null_accs >= real_acc).mean())


def verdict(p_value: float) -> str:
    if p_value < PVALUE_FAIL:
        return "FAIL"
    elif p_value < PVALUE_WARN:
        return "WARN"
    else:
        return "PASS"


def evaluate_model(name, X_test, y_test, null_accs, n_perms, device):
    """Evaluate one model: metrics + p-value from precomputed null distribution."""
    ckpt = MODELS_DIR / f"{name}.pt"
    if not ckpt.exists():
        print(f"    WARNING: checkpoint not found for {name}, skipping.")
        return None

    model = build_model(name).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    y_pred = predict(model, X_test, device=device)

    acc    = float(accuracy_score(y_test, y_pred))
    f1_mac = float(f1_score(y_test, y_pred, average="macro",    zero_division=0))
    f1_wt  = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    prec   = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec    = float(recall_score(y_test,    y_pred, average="macro", zero_division=0))

    per_class = per_class_metrics(y_test, y_pred)
    cm        = compute_confusion_matrix(y_test, y_pred)

    p_val = p_value_from_null(null_accs, acc)
    v = verdict(p_val)
    print(f"    acc={acc:.3f}  f1_macro={f1_mac:.3f}  p-value={p_val:.4f} → {v}")

    return {
        "name": name,
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "precision_macro": prec,
        "recall_macro": rec,
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": LABELS,
        "permutation_test": {
            "n_permutations": n_perms,
            "p_value": p_val,
            "perm_accs": null_accs.tolist(),
        },
        "verdict": v,
    }


def overall_verdict(model_results):
    """Overall verdict: FAIL if any model FAILs, WARN if any WARNs, else PASS."""
    verdicts = [r["verdict"] for r in model_results if r is not None]
    if "FAIL" in verdicts:
        return "FAIL"
    if "WARN" in verdicts:
        return "WARN"
    return "PASS"


def main():
    parser = argparse.ArgumentParser(description="Evaluate bias check models")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to evaluate (default: all)")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS,
                        help=f"Number of permutations (default: {N_PERMUTATIONS})")
    parser.add_argument("--device", default=None,
                        help="Device: cuda / cpu (default: auto-detect)")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 3: Evaluate Bias Check Models")
    print("=" * 60)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\nDevice: {device}")

    # Load data
    print("\nLoading test split...")
    X_test, y_test = load_test_data()
    print(f"  Test: {X_test.shape}")

    print("Loading train split (for permutation test)...")
    X_train = load_npz(FEATURES_DIR / "train.npz")  # keep sparse for SGDClassifier
    y_train = np.load(FEATURES_DIR / "y_train.npy")
    print(f"  Train: {X_train.shape}")

    # Select variants
    variants = MODEL_VARIANTS
    if args.models:
        variants = [v for v in MODEL_VARIANTS if v[0] in args.models]

    # Baselines
    majority_pred = np.full_like(y_test, fill_value=np.bincount(y_test).argmax())
    majority_acc  = float(accuracy_score(y_test, majority_pred))

    print(f"\nBaselines:")
    print(f"  Chance level:      {CHANCE_ACCURACY:.3f} ({CHANCE_ACCURACY:.1%})")
    print(f"  Majority class:    {majority_acc:.3f} ({majority_acc:.1%})")
    print(f"  Config majority:   {MAJORITY_BASELINE:.3f} ({MAJORITY_BASELINE:.1%})")

    # Compute permutation null distribution once (model-independent)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nComputing permutation null distribution ({args.n_permutations} perms)...")
    null_accs = compute_null_distribution(
        X_train, y_train, X_test, y_test,
        n_perms=args.n_permutations, seed=PERM_SEED,
    )
    print(f"  Null distribution: mean={null_accs.mean():.3f}, std={null_accs.std():.3f}")

    # Evaluate each model against the shared null
    model_results = []

    print(f"\nEvaluating {len(variants)} model(s)...")
    for name, _, _ in variants:
        print(f"\n  {name}:")
        result = evaluate_model(
            name, X_test, y_test, null_accs,
            n_perms=args.n_permutations,
            device=device,
        )
        model_results.append(result)

    valid_results = [r for r in model_results if r is not None]

    # Overall verdict
    overall = overall_verdict(valid_results)
    print(f"\nOverall verdict: {overall}")

    # Build output JSON
    output = {
        "metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "n_test_samples": int(len(y_test)),
            "n_train_samples": int(len(y_train)),
            "n_permutations": args.n_permutations,
            "labels": LABELS,
            "baselines": {
                "chance": CHANCE_ACCURACY,
                "majority_class_observed": majority_acc,
                "majority_class_config": MAJORITY_BASELINE,
                "binomial_ci_half_width": BINOMIAL_CI_HALF_WIDTH,
            },
        },
        "models": {r["name"]: r for r in valid_results},
        "overall_verdict": overall,
        "verdict_explanation": {
            "PASS": "p-value >= 0.05: no statistically significant bias detected → proceed to Track A",
            "WARN": "p-value 0.01–0.05: marginal significance → investigate before proceeding",
            "FAIL": "p-value < 0.01: statistically significant bias detected → investigate corpora",
        },
    }

    print(f"\nSaving metrics to {ALL_METRICS_JSON}...")
    with open(ALL_METRICS_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Summary table
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Model':<18} {'Accuracy':>9} {'F1 Macro':>9} {'p-value':>9} {'Verdict':>8}")
    print("-" * 58)
    for r in valid_results:
        print(f"{r['name']:<18} {r['accuracy']:>9.3f} {r['f1_macro']:>9.3f} "
              f"{r['permutation_test']['p_value']:>9.4f} {r['verdict']:>8}")
    print(f"\nOverall: {overall}")
    print(f"\nMetrics saved to {ALL_METRICS_JSON}")
    print("=" * 60)
    print("Done! Run 04_plot.py next.")
    print("=" * 60)


if __name__ == "__main__":
    main()
