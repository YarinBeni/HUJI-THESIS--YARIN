#!/usr/bin/env python3
"""
Step 4: Generate all plots for the bias check.

Reads all_metrics.json and training_history.json.
Produces 5 types of plots in v_1/data/evaluation/bias_check/plots/.

Usage:
    python 04_plot.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ALL_METRICS_JSON,
    TRAINING_HISTORY_JSON,
    PLOTS_DIR,
    LABELS,
    CHANCE_ACCURACY,
    BINOMIAL_CI_HALF_WIDTH,
    MODEL_VARIANTS,
)

# Style
sns.set_style("whitegrid")
DPI = 300
COLORS = {"MLP": "#2196F3", "Attention+MLP": "#FF9800"}
LABEL_COLORS = ["#4CAF50", "#9C27B0", "#F44336"]   # per class


def load_json(path):
    with open(path) as f:
        return json.load(f)


def model_order():
    """Return display order and series labels."""
    mlp_names  = [v[0] for v in MODEL_VARIANTS if v[1] == 0]
    attn_names = [v[0] for v in MODEL_VARIANTS if v[1] > 0]
    return mlp_names, attn_names


def plot_accuracy_vs_complexity(metrics, out_dir):
    """Line plot: accuracy vs model complexity, two series."""
    models_dict = metrics["models"]
    mlp_names, attn_names = model_order()

    mlp_accs  = [models_dict[n]["accuracy"] for n in mlp_names  if n in models_dict]
    attn_accs = [models_dict[n]["accuracy"] for n in attn_names if n in models_dict]

    mlp_xlabels  = [n.replace("mlp_", "MLP-")  for n in mlp_names  if n in models_dict]
    attn_xlabels = [n.replace("attn", "Attn").replace("_mlp", "+MLP-") for n in attn_names if n in models_dict]

    fig, ax = plt.subplots(figsize=(10, 5))

    if mlp_accs:
        ax.plot(range(len(mlp_accs)), mlp_accs,
                marker="o", color=COLORS["MLP"], linewidth=2, label="MLP only")
        ax.set_xticks(range(len(mlp_accs)))
        ax.set_xticklabels(mlp_xlabels, rotation=20, ha="right")

    if attn_accs:
        offset = len(mlp_accs) + 1
        x_attn = range(offset, offset + len(attn_accs))
        ax.plot(x_attn, attn_accs,
                marker="s", color=COLORS["Attention+MLP"], linewidth=2, label="Attention + MLP")
        ax.set_xticks(list(range(len(mlp_accs))) + list(x_attn))
        ax.set_xticklabels(mlp_xlabels + attn_xlabels, rotation=20, ha="right")

    # Chance level + CI band
    ax.axhline(CHANCE_ACCURACY, color="gray", linestyle="--", linewidth=1.5, label="Chance (33.3%)")
    ax.axhspan(
        CHANCE_ACCURACY - BINOMIAL_CI_HALF_WIDTH,
        CHANCE_ACCURACY + BINOMIAL_CI_HALF_WIDTH,
        alpha=0.15, color="gray", label="95% CI band"
    )

    ax.set_ylabel("Test Accuracy")
    ax.set_title("Bias Check: Test Accuracy vs Model Complexity")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out = out_dir / "accuracy_vs_complexity.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_f1_per_class(metrics, out_dir):
    """Grouped bar chart: per-class F1 for each model."""
    models_dict = metrics["models"]
    mlp_names, attn_names = model_order()
    all_names = [n for n in (mlp_names + attn_names) if n in models_dict]

    x = np.arange(len(all_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(all_names) * 1.5), 5))

    for i, label in enumerate(LABELS):
        f1_vals = [models_dict[n]["per_class"][label]["f1"] for n in all_names]
        ax.bar(x + i * width, f1_vals, width, label=label, color=LABEL_COLORS[i], alpha=0.85)

    ax.axhline(CHANCE_ACCURACY, color="gray", linestyle="--", linewidth=1.5, label="Chance (33.3%)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(all_names, rotation=25, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Bias Check: Per-Class F1 Score")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out = out_dir / "f1_per_class.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_confusion_matrices(metrics, out_dir):
    """One heatmap per model."""
    models_dict = metrics["models"]
    mlp_names, attn_names = model_order()
    all_names = [n for n in (mlp_names + attn_names) if n in models_dict]

    short_labels = ["OB", "NA", "LB"]  # abbreviated for heatmap axes

    for name in all_names:
        cm = np.array(models_dict[name]["confusion_matrix"])
        acc = models_dict[name]["accuracy"]
        verdict = models_dict[name]["verdict"]

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=short_labels,
            yticklabels=short_labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}  |  acc={acc:.3f}  |  {verdict}")
        fig.tight_layout()
        out = out_dir / f"confusion_{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out.name}")


def plot_training_curves(history, out_dir):
    """2×4 grid of train/val loss curves."""
    mlp_names, attn_names = model_order()
    all_names = mlp_names + attn_names
    available = [n for n in all_names if n in history]

    n_models = len(available)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(available):
        ax = axes[i]
        h = history[name]["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], label="train", color="#2196F3")
        ax.plot(epochs, h["val_loss"],   label="val",   color="#F44336")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss",  fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Bias Check: Training Curves", y=1.01)
    fig.tight_layout()
    out = out_dir / "training_curves.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_permutation_test(metrics, out_dir):
    """Histogram of permuted accuracies with real accuracy line."""
    models_dict = metrics["models"]
    mlp_names, attn_names = model_order()
    all_names = [n for n in (mlp_names + attn_names) if n in models_dict]

    n_models = len(all_names)
    n_cols = min(4, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(all_names):
        ax = axes[i]
        r = models_dict[name]
        perm_data = r["permutation_test"]
        perm_accs = perm_data.get("perm_accs", perm_data.get("perm_accs_sample", []))
        real_acc  = r["accuracy"]
        p_val     = perm_data["p_value"]
        v         = r["verdict"]

        if perm_accs:
            ax.hist(perm_accs, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(real_acc, color="red", linewidth=2, label=f"Real acc={real_acc:.3f}")
        ax.axvline(CHANCE_ACCURACY, color="gray", linestyle="--", linewidth=1,
                   label=f"Chance={CHANCE_ACCURACY:.3f}")
        ax.set_title(f"{name}\np={p_val:.4f} → {v}", fontsize=9)
        ax.set_xlabel("Accuracy", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Bias Check: Permutation Test Distributions", y=1.01)
    fig.tight_layout()
    out = out_dir / "permutation_test.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def main():
    print("=" * 60)
    print("Step 4: Generate Plots")
    print("=" * 60)

    for path, name in [(ALL_METRICS_JSON, "all_metrics.json"),
                       (TRAINING_HISTORY_JSON, "training_history.json")]:
        if not path.exists():
            print(f"\nError: {path} not found. Run earlier steps first.")
            sys.exit(1)

    metrics = load_json(ALL_METRICS_JSON)
    history = load_json(TRAINING_HISTORY_JSON)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting plots to {PLOTS_DIR}/...\n")

    plot_accuracy_vs_complexity(metrics, PLOTS_DIR)
    plot_f1_per_class(metrics, PLOTS_DIR)
    plot_confusion_matrices(metrics, PLOTS_DIR)
    plot_training_curves(history, PLOTS_DIR)
    plot_permutation_test(metrics, PLOTS_DIR)

    print("\n" + "=" * 60)
    print("Done! Run 05_report.py next.")
    print("=" * 60)


if __name__ == "__main__":
    main()
