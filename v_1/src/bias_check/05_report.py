#!/usr/bin/env python3
"""
Step 5: Generate the bias check markdown report.

Reads all_metrics.json → bias_check_report.md.

Usage:
    python 05_report.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ALL_METRICS_JSON,
    BIAS_REPORT_MD,
    PLOTS_DIR,
    LABELS,
    CHANCE_ACCURACY,
    MAJORITY_BASELINE,
    BINOMIAL_CI_HALF_WIDTH,
    MODEL_VARIANTS,
    PVALUE_FAIL,
    PVALUE_WARN,
)


def load_metrics():
    with open(ALL_METRICS_JSON) as f:
        return json.load(f)


def confusion_matrix_str(cm, labels):
    """Format confusion matrix as a text table."""
    short = ["OB", "NA", "LB"]  # Old Babylonian, Neo-Assyrian, Late Babylonian
    header = " " * 18 + "  ".join(f"{s:>6}" for s in short) + "  (predicted)"
    lines = [header]
    for i, label in enumerate(labels):
        row_vals = "  ".join(f"{cm[i][j]:>6}" for j in range(len(labels)))
        lines.append(f"{label:<18}{row_vals}")
    return "\n".join(lines)


def verdict_badge(v):
    mapping = {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL"}
    return mapping.get(v, v)


def generate_report(metrics):
    md = metrics["metadata"]
    models_dict = metrics["models"]
    overall = metrics["overall_verdict"]

    # Ordered model list
    all_names = [v[0] for v in MODEL_VARIANTS if v[0] in models_dict]

    lines = []
    lines.append("# Bias Check Report: Test Data Temporal Signal")
    lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Overall verdict**: {verdict_badge(overall)}")

    # -------------------------------------------------------------------------
    # Context
    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## Context\n")
    lines.append(
        "Before running LLM evaluation (Track A), we verify that the test set carries "
        "no exploitable surface-level temporal signal. "
        "A simple classifier that can distinguish Old Babylonian / Neo-Assyrian / Late Babylonian "
        "texts from transliteration alone would indicate dataset bias."
    )
    lines.append(
        "\nBias risks flagged by Chungrong: orthography, morphology, geographical/deity names, "
        "and provenance markers."
    )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## Dataset\n")
    lines.append(f"- **Total test samples**: {md['n_test_samples']:,}")
    lines.append(f"- **Total train samples**: {md['n_train_samples']:,}")
    lines.append(f"- **Labels**: {', '.join(LABELS)}")

    # -------------------------------------------------------------------------
    # Methodology
    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## Methodology\n")
    lines.append("**Features**: TF-IDF char n-grams, analyzer=`char_wb`, n-gram range (2,5), "
                 "max 10,000 features, sublinear TF. Fit on train only.\n")
    lines.append("**Models**: 8 variants — MLP depth sweep (1→5 layers) and attention+MLP sweep "
                 "(1→5 attention blocks, fixed 3-layer MLP head).\n")
    lines.append("**Permutation testing** (Ojala & Garriga, JMLR 2010): "
                 f"{md['n_permutations']} permutations of train labels, "
                 "SGDClassifier (log loss) as proxy estimator. "
                 "p-value = fraction of permuted runs ≥ real accuracy.\n")
    lines.append(f"**Significance thresholds**: FAIL p<{PVALUE_FAIL}, WARN p<{PVALUE_WARN}, PASS p≥{PVALUE_WARN}")
    lines.append(
        f"\n**Note on multiple comparisons**: 8 models are tested without Bonferroni correction. "
        f"Under H₀, ~0.4 false positives are expected at α=0.05. "
        f"The conservative overall verdict (any FAIL → FAIL) partially compensates."
    )

    # -------------------------------------------------------------------------
    # Baselines
    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## Baselines\n")
    bl = md["baselines"]
    lines.append(f"| Baseline | Accuracy |")
    lines.append(f"|----------|----------|")
    lines.append(f"| Chance (uniform random) | {bl['chance']:.1%} |")
    lines.append(f"| Majority class (observed) | {bl['majority_class_observed']:.1%} |")
    lines.append(f"| Binomial 95% CI half-width | ±{bl['binomial_ci_half_width']:.1%} |")

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## Results Summary\n")
    lines.append(f"| Model | Accuracy | F1 Macro | p-value | Verdict |")
    lines.append(f"|-------|----------|----------|---------|---------|")
    for name in all_names:
        r = models_dict[name]
        pt = r["permutation_test"]
        lines.append(
            f"| {name} | {r['accuracy']:.3f} | {r['f1_macro']:.3f} | "
            f"{pt['p_value']:.4f} | {verdict_badge(r['verdict'])} |"
        )

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## Plots\n")
    for fname, caption in [
        ("accuracy_vs_complexity.png", "Test accuracy vs model complexity"),
        ("f1_per_class.png",           "Per-class F1 scores"),
        ("permutation_test.png",       "Permutation test distributions"),
        ("training_curves.png",        "Training curves (loss)"),
    ]:
        lines.append(f"![{caption}](plots/{fname})")
        lines.append(f"*{caption}*\n")

    # -------------------------------------------------------------------------
    # Per-model details
    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## Per-Model Details\n")

    for name in all_names:
        r = models_dict[name]
        pt = r["permutation_test"]
        lines.append(f"### {name}  {verdict_badge(r['verdict'])}\n")
        lines.append(f"- Accuracy: **{r['accuracy']:.3f}**")
        lines.append(f"- F1 macro: {r['f1_macro']:.3f}  |  F1 weighted: {r['f1_weighted']:.3f}")
        lines.append(f"- Precision macro: {r['precision_macro']:.3f}  |  Recall macro: {r['recall_macro']:.3f}")
        lines.append(f"- Permutation p-value: {pt['p_value']:.4f} (n={pt['n_permutations']})")

        # Per-class table
        lines.append(f"\n| Class | Precision | Recall | F1 | Support |")
        lines.append(f"|-------|-----------|--------|----|---------|")
        for label in LABELS:
            pc = r["per_class"][label]
            lines.append(
                f"| {label} | {pc['precision']:.3f} | {pc['recall']:.3f} | "
                f"{pc['f1']:.3f} | {int(pc['support'])} |"
            )

        # Confusion matrix
        lines.append(f"\n**Confusion matrix** (rows=true, cols=predicted):\n")
        lines.append("```")
        lines.append(confusion_matrix_str(r["confusion_matrix"], LABELS))
        lines.append("```")
        lines.append(f"\n![confusion_{name}](plots/confusion_{name}.png)\n")

    # -------------------------------------------------------------------------
    # Overall verdict and recommendation
    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## Overall Verdict and Recommendation\n")
    lines.append(f"**Verdict: {verdict_badge(overall)}**\n")

    if overall == "PASS":
        lines.append(
            "No statistically significant temporal signal detected in the transliteration features. "
            "All classifiers perform at or near chance level (p ≥ 0.05 for all models). "
            "\n\n**Recommendation**: Proceed to Track A (LLM evaluation)."
        )
    elif overall == "WARN":
        lines.append(
            "Marginal statistical significance detected (at least one model has p < 0.05). "
            "This may indicate weak surface-level temporal signal or a false positive due to "
            "multiple comparisons. "
            "\n\n**Recommendation**: Investigate the affected model(s) before proceeding. "
            "Check which n-grams drive the prediction and whether they correspond to "
            "orthographic/morphological features vs. thematic content."
        )
    else:  # FAIL
        lines.append(
            "Statistically significant temporal signal detected (at least one model has p < 0.01). "
            "The transliteration features contain exploitable bias that a classifier can leverage. "
            "The benchmark may not reliably measure LLM knowledge vs. surface pattern matching. "
            "\n\n**Recommendation**: Halt Track A evaluation. Investigate the source of bias "
            "(orthography, morphology, names, provenance markers). "
            "Consider text preprocessing to remove known bias signals, or report the bias "
            "as a limitation with controlled experiments."
        )

    return "\n".join(lines) + "\n"


def main():
    print("=" * 60)
    print("Step 5: Generate Bias Check Report")
    print("=" * 60)

    if not ALL_METRICS_JSON.exists():
        print(f"\nError: {ALL_METRICS_JSON} not found. Run 03_evaluate.py first.")
        sys.exit(1)

    metrics = load_metrics()
    report = generate_report(metrics)

    BIAS_REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(BIAS_REPORT_MD, "w") as f:
        f.write(report)

    print(f"\nReport saved to {BIAS_REPORT_MD}")
    print(f"Overall verdict: {metrics['overall_verdict']}")
    print("\n" + "=" * 60)
    print("Done! Bias check pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
