#!/usr/bin/env python3
"""06_bias_check_cv.py — SEAL/DLL/LBPL bias check (TF-IDF + LR + permutation test).

For each of 6 tasks × 2 cleanings:
  - Loads data via seal_tasks.load_task_data()
  - Fits TF-IDF char_wb(2,5) + LogisticRegression with adaptive-k stratified CV
    and C selected by cross-validated grid search
  - Runs permutation test (Ojala & Garriga 2010)
  - Writes per-task outputs: task_summary.json, metrics.json, report.md, plots/

Output layout:
  v_1/data/evaluation/bias_check/seal_round4/<task>/<cleaning>/
      task_summary.json
      metrics.json
      report.md
      plots/confusion.png, perm_null.png, per_class_f1.png  (when --plots)

Usage (all from repo root):
    # Debug mode — domain/tier0, 100 perms, no plots (DEFAULT):
    python3 v_1/src/bias_check/06_bias_check_cv.py --debug

    # Full run — all 6 tasks × tier0 + maximal, 1000 perms, plots:
    python3 v_1/src/bias_check/06_bias_check_cv.py --plots

    # Single task:
    python3 v_1/src/bias_check/06_bias_check_cv.py --tasks domain --cleanings tier0

Plan reference: Section 11 steps 12–15 of
  v_1/justification/seal_round4_pipeline_plan.md

Dependencies: stdlib + pandas + numpy + sklearn only.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Paths + seal_tasks import
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[2]  # v_1/src/bias_check/ → repo root

sys.path.insert(0, str(_THIS_DIR))
from seal_tasks import TASK_NAMES, load_task_data  # noqa: E402

OUTPUT_ROOT = (
    REPO_ROOT / "v_1" / "data" / "evaluation" / "bias_check" / "seal_round4"
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

TFIDF_KWARGS: dict[str, Any] = dict(
    analyzer="char_wb",
    ngram_range=(2, 5),
    max_features=10_000,
    sublinear_tf=True,
)

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
LR_MAX_ITER = 1000
LR_SEED = 42
CV_SEED = 42
PERM_SEED = 0

N_PERMS_FULL = 1000
N_PERMS_DEBUG = 100

PVALUE_FAIL = 0.01   # p < 0.01 → FAIL (significant bias)
PVALUE_WARN = 0.05   # p < 0.05 → WARN (marginal)
# p ≥ 0.05 → PASS


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def _make_pipeline(c: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("lr", LogisticRegression(
            C=c,
            max_iter=LR_MAX_ITER,
            class_weight="balanced",
            solver="lbfgs",
            random_state=LR_SEED,
        )),
    ])


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------


def run_task(
    task_name: str,
    cleaning: str,
    n_perms: int,
    make_plots: bool,
    out_dir: Path,
) -> dict[str, Any]:
    """Run one task × cleaning combination. Returns the metrics dict."""

    # ── Load data ─────────────────────────────────────────────────────────
    df, task_summary = load_task_data(task_name, cleaning=cleaning)
    k = task_summary["k_used"]
    n_classes = task_summary["n_classes_after_drop"]
    n_frags = task_summary["fragments_after_drop"]
    class_names: list[str] = sorted(df["label_raw"].unique())

    print(f"\n{'='*60}")
    print(f"Task: {task_name}  |  cleaning: {cleaning}")
    print(f"  N={n_frags}, classes={n_classes}, k={k}, perms={n_perms}")
    print(f"{'='*60}")

    X: list[str] = df["text"].tolist()
    y: np.ndarray = df["label_idx"].values

    # ── Adaptive-k stratified CV splitter ──────────────────────────────────
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=CV_SEED)

    # ── C selection via cross_val_score ────────────────────────────────────
    best_c = C_GRID[0]
    best_cv_score = -1.0
    c_scores: dict[float, float] = {}

    for c in C_GRID:
        pipe = _make_pipeline(c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
        mean_f1 = float(np.mean(scores))
        c_scores[c] = mean_f1
        if mean_f1 > best_cv_score:
            best_cv_score = mean_f1
            best_c = c

    print(f"  Best C={best_c}  (f1_macro={best_cv_score:.4f})")

    best_pipe = _make_pipeline(best_c)

    # ── OOF predictions (for accuracy/F1/confusion) ────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred: np.ndarray = cross_val_predict(best_pipe, X, y, cv=cv, method="predict")

    acc = float(accuracy_score(y, y_pred))
    macro_f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y, y_pred, labels=list(range(n_classes)))

    report_dict = classification_report(
        y, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    per_class_f1: dict[str, float] = {
        cls: float(report_dict[cls]["f1-score"])
        for cls in class_names
        if cls in report_dict
    }

    print(f"  OOF accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")

    # ── Permutation test ───────────────────────────────────────────────────
    print(f"  Running {n_perms} permutations ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual_score, perm_scores, pvalue = permutation_test_score(
            best_pipe, X, y,
            cv=cv,
            scoring="f1_macro",
            n_permutations=n_perms,
            random_state=PERM_SEED,
            n_jobs=1,
        )

    pvalue = float(pvalue)
    actual_score = float(actual_score)
    print(f"  Perm score={actual_score:.4f}, p={pvalue:.4f}")

    # ── Significance classification ────────────────────────────────────────
    if pvalue < PVALUE_FAIL:
        significance = "FAIL (p < 0.01 — significant bias detected)"
        sig_short = "FAIL"
    elif pvalue < PVALUE_WARN:
        significance = "WARN (0.01 ≤ p < 0.05 — marginal)"
        sig_short = "WARN"
    else:
        significance = "PASS (p ≥ 0.05 — no significant bias)"
        sig_short = "PASS"

    # ── Write outputs ──────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()[:19] + "Z"

    # task_summary.json — augments the registry summary with runtime info
    ts_out: dict[str, Any] = OrderedDict(task_summary)
    ts_out["run_cleaning"] = cleaning
    ts_out["generated_at"] = generated_at
    _write_json(out_dir / "task_summary.json", ts_out)

    # metrics.json
    metrics_dict: dict[str, Any] = OrderedDict([
        ("task_name", task_name),
        ("cleaning", cleaning),
        ("n_fragments", n_frags),
        ("n_classes", n_classes),
        ("k_used", k),
        ("best_C", best_c),
        ("c_grid_scores", {str(c_): round(v, 6) for c_, v in c_scores.items()}),
        ("cv_accuracy", round(acc, 6)),
        ("cv_macro_f1", round(macro_f1, 6)),
        ("cv_weighted_f1", round(weighted_f1, 6)),
        ("per_class_f1", {c_: round(v, 6) for c_, v in per_class_f1.items()}),
        ("perm_test", OrderedDict([
            ("n_permutations", n_perms),
            ("actual_score", round(actual_score, 6)),
            ("perm_mean", round(float(np.mean(perm_scores)), 6)),
            ("perm_std", round(float(np.std(perm_scores)), 6)),
            ("pvalue", round(pvalue, 6)),
            ("significance", sig_short),
        ])),
        ("generated_at", generated_at),
    ])
    _write_json(out_dir / "metrics.json", metrics_dict)

    # report.md
    _write_report(
        out_dir / "report.md",
        task_name=task_name,
        cleaning=cleaning,
        task_summary=task_summary,
        metrics=metrics_dict,
        class_names=class_names,
        per_class_f1=per_class_f1,
        significance=significance,
        sig_short=sig_short,
        cm=cm,
    )

    # plots/ (optional)
    if make_plots:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        _plot_confusion(cm, class_names, task_name, cleaning, plots_dir)
        _plot_perm_null(perm_scores, actual_score, pvalue, task_name, cleaning, plots_dir)
        _plot_per_class_f1(per_class_f1, task_name, cleaning, plots_dir)
        print(f"  Plots written to {plots_dir.relative_to(REPO_ROOT)}")

    print(f"  [{sig_short}] Written to {out_dir.relative_to(REPO_ROOT)}")
    return metrics_dict


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_report(
    path: Path,
    *,
    task_name: str,
    cleaning: str,
    task_summary: dict[str, Any],
    metrics: dict[str, Any],
    class_names: list[str],
    per_class_f1: dict[str, float],
    significance: str,
    sig_short: str,
    cm: np.ndarray,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()[:19] + "Z"
    n = task_summary["fragments_after_drop"]
    k = task_summary["k_used"]
    n_classes = task_summary["n_classes_after_drop"]
    singletons = task_summary["singletons_dropped"]
    perm = metrics["perm_test"]
    singleton_str = (
        ", ".join(repr(s) for s in singletons[:5])
        + (f" (+{len(singletons) - 5} more)" if len(singletons) > 5 else "")
    ) if singletons else "none"

    lines: list[str] = [
        f"# Bias Check — {task_name} ({cleaning})",
        "",
        f"Generated: `{ts}`",
        f"Script: `v_1/src/bias_check/06_bias_check_cv.py`",
        "",
        f"## Result: {sig_short}",
        "",
        f"**{significance}**",
        "",
        "## Data",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Task | `{task_name}` |",
        f"| Cleaning | `{cleaning}` |",
        f"| Corpora pooled | {', '.join(task_summary['corpora_pooled'])} |",
        f"| N fragments | {n} |",
        f"| N classes (surviving) | {n_classes} |",
        f"| Singletons dropped | {len(singletons)} ({singleton_str}) |",
        f"| Effective k | {k} |",
        "",
        "## CV Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Best C | {metrics['best_C']} |",
        f"| CV accuracy | {metrics['cv_accuracy']:.4f} |",
        f"| CV macro-F1 | {metrics['cv_macro_f1']:.4f} |",
        f"| CV weighted-F1 | {metrics['cv_weighted_f1']:.4f} |",
        "",
        "## Permutation Test",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| N permutations | {perm['n_permutations']} |",
        f"| Actual macro-F1 | {perm['actual_score']:.4f} |",
        f"| Null mean | {perm['perm_mean']:.4f} |",
        f"| Null std | {perm['perm_std']:.4f} |",
        f"| p-value | {perm['pvalue']:.4f} |",
        f"| Significance | **{sig_short}** |",
        "",
        "## Per-Class F1",
        "",
        "| Class | F1 |",
        "|-------|----|",
    ]
    for cls in sorted(per_class_f1, key=lambda c: -per_class_f1[c]):
        lines.append(f"| `{cls}` | {per_class_f1[cls]:.4f} |")

    lines += [
        "",
        "## Confusion Matrix",
        "",
        "Rows = true label, columns = predicted label.",
        "",
    ]
    # Header row
    lines.append("| True \\ Pred | " + " | ".join(f"`{c}`" for c in class_names) + " |")
    lines.append("|---|" + "---|" * len(class_names))
    for i, row_cls in enumerate(class_names):
        row_vals = " | ".join(str(cm[i, j]) for j in range(len(class_names)))
        lines.append(f"| `{row_cls}` | {row_vals} |")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_confusion(
    cm: np.ndarray,
    class_names: list[str],
    task_name: str,
    cleaning: str,
    plots_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plots] matplotlib not available — skipping confusion matrix")
        return

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.5)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if n <= 20:
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
    else:
        ax.set_xticklabels([""] * n)
        ax.set_yticklabels([""] * n)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion: {task_name} ({cleaning})")
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_perm_null(
    perm_scores: np.ndarray,
    actual_score: float,
    pvalue: float,
    task_name: str,
    cleaning: str,
    plots_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plots] matplotlib not available — skipping perm_null")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(perm_scores, bins=30, color="steelblue", alpha=0.7, label="Null distribution")
    ax.axvline(
        actual_score, color="red", linestyle="--", linewidth=2,
        label=f"Actual F1={actual_score:.3f}  p={pvalue:.3f}",
    )
    ax.set_xlabel("Macro-F1")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation null: {task_name} ({cleaning})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "perm_null.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_class_f1(
    per_class_f1: dict[str, float],
    task_name: str,
    cleaning: str,
    plots_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plots] matplotlib not available — skipping per_class_f1")
        return

    sorted_items = sorted(per_class_f1.items(), key=lambda x: -x[1])
    labels = [it[0] for it in sorted_items]
    values = [it[1] for it in sorted_items]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 5))
    ax.bar(range(len(labels)), values, color="steelblue", alpha=0.8)
    ax.axhline(
        1.0 / len(labels), color="red", linestyle="--", linewidth=1, label="Chance",
    )
    ax.set_xticks(range(len(labels)))
    if len(labels) <= 25:
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    else:
        ax.set_xticklabels([""] * len(labels))
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-class F1: {task_name} ({cleaning})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "per_class_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SEAL bias check: TF-IDF + LR + permutation test, 6 tasks × 2 cleanings."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode: domain/tier0 only, 100 perms, no plots.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=TASK_NAMES,
        metavar="TASK",
        help=f"Tasks to run (default: all 6). Choices: {TASK_NAMES}",
    )
    parser.add_argument(
        "--cleanings",
        nargs="+",
        default=None,
        choices=["tier0", "maximal"],
        metavar="CLEANING",
        help="Cleanings to run (default: tier0 maximal).",
    )
    parser.add_argument(
        "--perms",
        type=int,
        default=None,
        help="Override number of permutations (default: 100 in debug, 1000 in full).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=False,
        help="Generate plots (always off in --debug mode).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.debug:
        tasks = ["domain"]
        cleanings = ["tier0"]
        n_perms = args.perms if args.perms is not None else N_PERMS_DEBUG
        make_plots = False
        print("[DEBUG MODE] domain/tier0 only, 100 perms, no plots.")
    else:
        tasks = args.tasks if args.tasks is not None else TASK_NAMES
        cleanings = args.cleanings if args.cleanings is not None else ["tier0", "maximal"]
        n_perms = args.perms if args.perms is not None else N_PERMS_FULL
        make_plots = args.plots

    all_results: list[dict[str, Any]] = []
    errors: list[str] = []

    for task_name in tasks:
        for cleaning in cleanings:
            out_dir = OUTPUT_ROOT / task_name / cleaning
            try:
                metrics = run_task(
                    task_name=task_name,
                    cleaning=cleaning,
                    n_perms=n_perms,
                    make_plots=make_plots,
                    out_dir=out_dir,
                )
                all_results.append({
                    "task": task_name,
                    "cleaning": cleaning,
                    "macro_f1": metrics["cv_macro_f1"],
                    "pvalue": metrics["perm_test"]["pvalue"],
                    "significance": metrics["perm_test"]["significance"],
                })
            except Exception as exc:
                msg = f"ERROR in {task_name}/{cleaning}: {exc}"
                print(f"\n{'!'*60}\n{msg}\n{'!'*60}")
                errors.append(msg)
                import traceback
                traceback.print_exc()

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<20} {'Cleaning':<10} {'F1':>8} {'p':>8}  Result")
    print(f"{'-'*20} {'-'*10} {'-'*8} {'-'*8}  {'-'*8}")
    for r in all_results:
        print(
            f"{r['task']:<20} {r['cleaning']:<10} "
            f"{r['macro_f1']:>8.4f} {r['pvalue']:>8.4f}  {r['significance']}"
        )
    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
