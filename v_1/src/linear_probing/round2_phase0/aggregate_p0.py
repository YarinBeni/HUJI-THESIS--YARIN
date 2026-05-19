"""aggregate_p0.py — Round 2 Phase 0: gate-check aggregator.

Consumes the per-draw / summary JSONs written by `run_mc_probes.py`
(see run_mc_probes.py:33-43 for the output layout, and lines 391-443 for the
summary schema) and decides whether the balanced re-probing experiment passes
its pre-committed gate (handoff doc, locked 2026-05-19).

Pre-committed gate
------------------
PRIMARY  : for each of {TF-IDF, MLM, Random-Qwen, pretrained Qwen}, the mean
           Macro-F1 over 200 balanced MC draws, MINUS 2*std, must EXCEED that
           method's best Round-1 imbalanced Macro-F1 (Round-1 result file:
           orcc__probe_cls/cls_best_layers.json and orcc__probe_pls/pls_best_layers.json).
SECONDARY: TF-IDF k-NN accuracy on the balanced subset >= 0.70 (RINA anchor).
           Currently `run_mc_probes.py` does NOT compute a k-NN probe; this
           secondary gate is reported as 'not_evaluated' if absent.

Alignment between MC summary and Round-1 baselines
--------------------------------------------------
Round-1 CLS baseline keys (cls_best_layers.json):
    {method}__{cleaning}__{pooling}__{task}        → best_layer, best_layer_macro_f1
Round-1 PLS baseline keys (pls_best_layers.json):
    {method}__{cleaning}__{pooling}__year-ruler    → best_layer, best_k, macro_f1_mean
    {method}__{cleaning}__{pooling}__year-{raw,log}→ best_layer, best_k, spearman/r2

MC summary keys (per-config inside {probe}__mc_balanced__summary.json):
    {method}__{cleaning}__{pooling}__L{NN}__{ruler|year|year-raw|year-log}
        → macro_f1_mean / macro_f1_std / accuracy_mean / spearman_mean / r2_mean

Pairing rule (ruler task, the primary gate):
    For each method M:
      1. From Round-1 CLS, pick (cleaning, pooling) maximizing best_layer_macro_f1;
         record (cleaning_cls, pooling_cls, best_layer_cls, r1_f1_cls).
         MC key to compare: '{M}__{cleaning_cls}__{pooling_cls}__L{bl:02d}__ruler'
         in '{M}_cls__mc_balanced__summary.json'.
      2. From Round-1 PLS year-ruler entries, pick (cleaning, pooling) maximizing
         macro_f1_mean; pair with MC key in '{M}_pls__mc_balanced__summary.json'.
      3. Method-level R1 score = max(r1_f1_cls, r1_f1_pls).
         Method-level MC mean ± std = whichever (cls/pls) had the higher R1 score.
      4. Method passes iff (mc_mean - 2*mc_std) > r1_score.

CLI
---
    python aggregate_p0.py \\
        --probes_dir .../orcc_round2_phase0/probes \\
        --round1_cls .../orcc__probe_cls/cls_best_layers.json \\
        --round1_pls .../orcc__probe_pls/pls_best_layers.json \\
        --out_dir   .../orcc_round2_phase0/aggregated

Outputs:
    {out_dir}/phase0_summary.json       — all aggregated numbers + per-method gate verdicts
    {out_dir}/phase0_report.md          — human-readable
    {out_dir}/phase0_macrof1_compare.png — bar chart (Round-1 vs. balanced MC, with 2σ bands)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The 4 methods we evaluate against the pre-committed gate.
# 'random' is Random-Qwen (random-init Qwen activations). It is gated on
# whether random_{pls,cls} summaries actually exist; if missing, reported.
METHODS = ["tfidf", "mlm", "random", "qwen"]
PROBES_PER_METHOD = {  # MC summary file stems
    "tfidf":  ("tfidf_pls",  "tfidf_cls"),
    "mlm":    ("mlm_pls",    "mlm_cls"),
    "qwen":   ("qwen_pls",   "qwen_cls"),
    "random": ("random_pls", "random_cls"),  # may be absent
}
METHOD_DISPLAY = {
    "tfidf":  "TF-IDF",
    "mlm":    "MLM",
    "random": "Random-Qwen",
    "qwen":   "Qwen-7B (pretrained)",
}
METHOD_COLORS = {
    "qwen":   "#1976D2",
    "random": "#7B1FA2",
    "mlm":    "#E53935",
    "tfidf":  "#388E3C",
}

METHOD_TAG = "mc_balanced"

# Secondary gate
SECONDARY_KNN_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probes_dir", "--probes-dir", dest="probes_dir", type=Path, required=True,
                   help="Directory containing {probe}__mc_balanced__{draw*,summary}.json")
    p.add_argument("--round1_cls", "--round1-cls", dest="round1_cls", type=Path, required=True,
                   help="Path to Round-1 cls_best_layers.json")
    p.add_argument("--round1_pls", "--round1-pls", dest="round1_pls", type=Path, required=True,
                   help="Path to Round-1 pls_best_layers.json")
    p.add_argument("--out_dir", "--out-dir", dest="out_dir", type=Path, required=True,
                   help="Where to write phase0_summary.json / phase0_report.md / phase0_macrof1_compare.png")
    p.add_argument("--method_tag", "--method-tag", dest="method_tag", default=METHOD_TAG,
                   help="MC method tag in filenames (default: mc_balanced)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# MC summary loading (with fallback to per-draw glob aggregation)
# ---------------------------------------------------------------------------

def _aggregate_from_per_draw(probes_dir: Path, probe: str, method_tag: str) -> dict:
    """Fallback aggregator: re-implements run_mc_probes.py:_aggregate_summary (lines 391-443).

    Used when {probe}__{tag}__summary.json is absent OR may be stale relative
    to the per-draw files on disk.
    """
    pattern = f"{probe}__{method_tag}__draw*.json"
    files = sorted(probes_dir.glob(pattern))
    if not files:
        return {"probe": probe, "method_tag": method_tag, "n_draws": 0, "per_config": {}}

    per_key: dict[str, dict[str, list[float]]] = {}
    for fp in files:
        with open(fp) as f:
            doc = json.load(f)
        results = doc.get("results", {})
        for cfg_key, rec in results.items():
            slot = per_key.setdefault(cfg_key, {
                "macro_f1": [], "accuracy": [], "spearman": [], "r2": [],
            })
            if "macro_f1_mean" in rec:
                slot["macro_f1"].append(rec["macro_f1_mean"])
                slot["accuracy"].append(rec.get("accuracy_mean", float("nan")))
            if "best_k_by_macro_f1" in rec:
                bk = str(rec["best_k_by_macro_f1"])
                mpk = rec["metrics_per_k"][bk]
                slot["macro_f1"].append(mpk["macro_f1_mean"])
                slot["accuracy"].append(mpk.get("accuracy_mean", float("nan")))
            if "best_k_by_spearman" in rec:
                bk_sp = str(rec["best_k_by_spearman"])
                bk_r2 = str(rec["best_k_by_r2"])
                slot["spearman"].append(rec["metrics_per_k"][bk_sp]["spearman_mean"])
                slot["r2"].append(rec["metrics_per_k"][bk_r2]["r2_mean"])

    summary_per_key: dict[str, Any] = {}
    for cfg_key, slot in per_key.items():
        agg: dict[str, Any] = {"n_draws": len(files)}
        for metric, vals in slot.items():
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if not vals:
                continue
            agg[f"{metric}_mean"]   = float(np.mean(vals))
            agg[f"{metric}_std"]    = float(np.std(vals))
            agg[f"{metric}_median"] = float(np.median(vals))
            agg[f"{metric}_n"]      = len(vals)
        summary_per_key[cfg_key] = agg

    return {
        "probe": probe,
        "method_tag": method_tag,
        "n_draws": len(files),
        "per_config": summary_per_key,
    }


def load_mc_summary(probes_dir: Path, probe: str, method_tag: str) -> dict | None:
    """Load a probe's MC summary, falling back to per-draw aggregation.

    Returns None ONLY if neither a summary nor any per-draw files exist.
    """
    summary_path = probes_dir / f"{probe}__{method_tag}__summary.json"
    per_draw_glob = list(probes_dir.glob(f"{probe}__{method_tag}__draw*.json"))

    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        # Cross-check: if the on-disk draw count exceeds summary's n_draws,
        # rebuild from per-draw (stale summary case).
        if per_draw_glob and summary.get("n_draws", 0) < len(per_draw_glob):
            print(f"  [{probe}] summary n_draws={summary.get('n_draws')} < "
                  f"disk draws={len(per_draw_glob)}; rebuilding from per-draw files")
            return _aggregate_from_per_draw(probes_dir, probe, method_tag)
        return summary

    if per_draw_glob:
        print(f"  [{probe}] no summary.json found; aggregating from {len(per_draw_glob)} per-draw files")
        return _aggregate_from_per_draw(probes_dir, probe, method_tag)

    return None


# ---------------------------------------------------------------------------
# Round-1 baseline parsing
# ---------------------------------------------------------------------------

def best_cls_baseline_for(method: str, round1_cls: dict, task: str = "ruler") -> dict | None:
    """From cls_best_layers.json, pick the best (cleaning, pooling) for a method+task.

    Round-1 CLS key format: '{method}__{cleaning}__{pooling}__{task}'.
    """
    best = None
    for k, v in round1_cls.items():
        parts = k.split("__")
        if len(parts) != 4:
            continue
        m, cleaning, pooling, tk = parts
        if m != method or tk != task:
            continue
        f1 = v.get("best_layer_macro_f1")
        if f1 is None:
            continue
        if best is None or f1 > best["macro_f1"]:
            best = {
                "method": method,
                "cleaning": cleaning,
                "pooling": pooling,
                "best_layer": int(v["best_layer"]),
                "macro_f1": float(f1),
                "accuracy": float(v.get("best_layer_accuracy", float("nan"))),
                "source": "cls",
                "round1_key": k,
            }
    return best


def best_pls_baseline_for(method: str, round1_pls: dict) -> dict | None:
    """From pls_best_layers.json, pick the best (cleaning, pooling) on the
    'year-ruler' (i.e. ruler-as-classification-from-PLS) task.

    Round-1 PLS ruler-task key format: '{method}__{cleaning}__{pooling}__year-ruler'.
    """
    best = None
    for k, v in round1_pls.items():
        parts = k.split("__")
        if len(parts) != 4:
            continue
        m, cleaning, pooling, tk = parts
        if m != method or tk != "year-ruler":
            continue
        f1 = v.get("macro_f1_mean")
        if f1 is None:
            continue
        if best is None or f1 > best["macro_f1"]:
            best = {
                "method": method,
                "cleaning": cleaning,
                "pooling": pooling,
                "best_layer": int(v.get("best_layer", 0)),
                "best_k": v.get("best_k"),
                "macro_f1": float(f1),
                "accuracy": float(v.get("accuracy_mean", float("nan"))),
                "source": "pls",
                "round1_key": k,
            }
    return best


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

def _mc_key(method: str, cleaning: str, pooling: str, layer: int, task: str = "ruler") -> str:
    return f"{method}__{cleaning}__{pooling}__L{layer:02d}__{task}"


def lookup_mc_entry(mc_summary: dict | None, method: str, cleaning: str,
                    pooling: str, layer: int, task: str = "ruler") -> dict | None:
    if mc_summary is None:
        return None
    per_cfg = mc_summary.get("per_config", {})
    key = _mc_key(method, cleaning, pooling, layer, task)
    return per_cfg.get(key)


def evaluate_method(method: str,
                    cls_summary: dict | None, pls_summary: dict | None,
                    round1_cls: dict, round1_pls: dict) -> dict:
    """Compute the gate result for one of the 4 methods.

    Returns:
        {
          method, display_name,
          cls: {round1: {...}, mc: {...}, gate: bool, gap},
          pls: {round1: {...}, mc: {...}, gate: bool, gap},
          method_round1_f1, method_mc_mean, method_mc_std, method_gate: bool,
          notes: [...]
        }
    """
    notes: list[str] = []
    r1_cls = best_cls_baseline_for(method, round1_cls, task="ruler")
    r1_pls = best_pls_baseline_for(method, round1_pls)

    def _pair(r1: dict | None, mc_sum: dict | None, label: str) -> dict:
        if r1 is None:
            return {"available": False, "reason": f"no round1 {label} baseline"}
        mc_entry = lookup_mc_entry(
            mc_sum, method, r1["cleaning"], r1["pooling"], r1["best_layer"], "ruler"
        )
        if mc_entry is None or "macro_f1_mean" not in mc_entry:
            return {
                "available": False,
                "reason": (f"no MC entry for "
                           f"{_mc_key(method, r1['cleaning'], r1['pooling'], r1['best_layer'])}"),
                "round1": r1,
            }
        mc_mean = float(mc_entry["macro_f1_mean"])
        mc_std  = float(mc_entry.get("macro_f1_std", 0.0))
        mc_med  = float(mc_entry.get("macro_f1_median", float("nan")))
        n_draws = int(mc_entry.get("macro_f1_n", mc_sum.get("n_draws", 0)))
        gate_pass = (mc_mean - 2.0 * mc_std) > r1["macro_f1"]
        return {
            "available": True,
            "round1": r1,
            "mc": {
                "macro_f1_mean": mc_mean,
                "macro_f1_std":  mc_std,
                "macro_f1_median": mc_med,
                "n_draws": n_draws,
                "key": _mc_key(method, r1["cleaning"], r1["pooling"], r1["best_layer"]),
            },
            "gap":  mc_mean - r1["macro_f1"],
            "gap_minus_2sigma": (mc_mean - 2.0 * mc_std) - r1["macro_f1"],
            "gate": bool(gate_pass),
        }

    cls_part = _pair(r1_cls, cls_summary, "cls")
    pls_part = _pair(r1_pls, pls_summary, "pls")

    # Method-level: take the pairing whose R1 F1 is higher (the bar that
    # must be beaten in either CLS or PLS regime).
    candidates = [p for p in (cls_part, pls_part) if p.get("available")]
    if not candidates:
        return {
            "method": method,
            "display_name": METHOD_DISPLAY[method],
            "cls": cls_part,
            "pls": pls_part,
            "method_gate": None,
            "notes": notes + ["no available pairings"],
        }

    chosen = max(candidates, key=lambda p: p["round1"]["macro_f1"])
    return {
        "method": method,
        "display_name": METHOD_DISPLAY[method],
        "cls": cls_part,
        "pls": pls_part,
        "chosen_regime": chosen["round1"]["source"],
        "method_round1_f1": chosen["round1"]["macro_f1"],
        "method_mc_mean":   chosen["mc"]["macro_f1_mean"],
        "method_mc_std":    chosen["mc"]["macro_f1_std"],
        "method_mc_median": chosen["mc"]["macro_f1_median"],
        "method_n_draws":   chosen["mc"]["n_draws"],
        "method_gap":       chosen["gap"],
        "method_gap_minus_2sigma": chosen["gap_minus_2sigma"],
        "method_gate":      bool(chosen["gate"]),
        "notes":            notes,
    }


# ---------------------------------------------------------------------------
# Secondary gate: TF-IDF k-NN accuracy >= 0.70
# ---------------------------------------------------------------------------

def evaluate_secondary_gate(tfidf_cls_summary: dict | None) -> dict:
    """Per WASSERMAN_MC.md, RINA anchor is k-NN. `run_mc_probes.py` currently
    runs a logistic-regression CLS probe (see fit_cls_cv in cls_utils), not k-NN.
    We surface the TF-IDF CLS accuracy as a proxy and flag the discrepancy.
    """
    if tfidf_cls_summary is None:
        return {"available": False, "reason": "no tfidf_cls summary"}
    per_cfg = tfidf_cls_summary.get("per_config", {})
    best_acc = None
    best_key = None
    for k, v in per_cfg.items():
        if not k.endswith("__ruler"):
            continue
        if v.get("accuracy_mean") is None:
            continue
        if best_acc is None or v["accuracy_mean"] > best_acc:
            best_acc = v["accuracy_mean"]
            best_key = k
    if best_acc is None:
        return {"available": False, "reason": "no accuracy_mean in tfidf_cls per_config"}
    return {
        "available": True,
        "metric": "tfidf_cls_accuracy_mean (proxy for kNN; logistic-regression CV-mean)",
        "value": float(best_acc),
        "config_key": best_key,
        "threshold": SECONDARY_KNN_THRESHOLD,
        "pass": bool(best_acc >= SECONDARY_KNN_THRESHOLD),
        "note": ("run_mc_probes.py uses fit_cls_cv (logistic regression), "
                 "not k-NN. Treat as proxy only."),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_summary_json(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


def write_markdown_report(out_path: Path, payload: dict) -> None:
    lines: list[str] = []
    lines.append("# Phase 0 — Balanced MC Re-probing Gate Report")
    lines.append("")
    lines.append(f"- Probes dir: `{payload['probes_dir']}`")
    lines.append(f"- Method tag: `{payload['method_tag']}`")
    lines.append(f"- Round-1 CLS baseline: `{payload['round1_cls_path']}`")
    lines.append(f"- Round-1 PLS baseline: `{payload['round1_pls_path']}`")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    v = payload["verdict"]
    lines.append(f"**Overall: {v['overall']}**")
    if v["overall"] == "FAIL":
        lines.append(f"Failing methods: {', '.join(v['failing_methods']) or '(none — see notes)'}")
    if v.get("missing_methods"):
        lines.append(f"Missing methods (no MC data): {', '.join(v['missing_methods'])}")
    lines.append("")
    sec = payload["secondary"]
    if sec.get("available"):
        sec_status = "PASS" if sec["pass"] else "FAIL"
        lines.append(f"Secondary gate (TF-IDF accuracy ≥ {sec['threshold']:.2f}): "
                     f"**{sec_status}** — value={sec['value']:.4f} (`{sec['config_key']}`)")
        lines.append(f"_{sec['note']}_")
    else:
        lines.append(f"Secondary gate: not evaluated — {sec.get('reason')}")
    lines.append("")

    # Per-method tables
    for m_entry in payload["per_method"]:
        m = m_entry["method"]
        lines.append(f"## {m_entry['display_name']} ({m})")
        if m_entry["method_gate"] is None:
            lines.append("Status: **NO DATA** — both CLS and PLS pairings unavailable.")
            for k in ("cls", "pls"):
                part = m_entry[k]
                if not part.get("available"):
                    lines.append(f"- {k}: {part.get('reason')}")
            lines.append("")
            continue
        gate_str = "PASS" if m_entry["method_gate"] else "FAIL"
        lines.append(f"Status: **{gate_str}** "
                     f"(chosen regime: {m_entry['chosen_regime']}, "
                     f"n_draws={m_entry['method_n_draws']})")
        lines.append("")
        lines.append("| Regime | Cleaning | Pooling | Layer | R1 Macro-F1 | "
                     "MC mean | MC std | MC median | (mean - 2σ) - R1 | Gate |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for k in ("cls", "pls"):
            part = m_entry[k]
            if not part.get("available"):
                lines.append(f"| {k} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | "
                             f"_{part.get('reason')}_ |")
                continue
            r1 = part["round1"]
            mc = part["mc"]
            lines.append(
                f"| {k} | {r1['cleaning']} | {r1['pooling']} | {r1['best_layer']} | "
                f"{r1['macro_f1']:.4f} | {mc['macro_f1_mean']:.4f} | {mc['macro_f1_std']:.4f} | "
                f"{mc['macro_f1_median']:.4f} | {part['gap_minus_2sigma']:+.4f} | "
                f"{'PASS' if part['gate'] else 'FAIL'} |"
            )
        lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    if v["overall"] == "PASS":
        lines.append(
            "All four methods exceed their Round-1 imbalanced Macro-F1 with a 2σ buffer "
            "when evaluated on 8-ruler × 21-fragment balanced sub-draws. This is consistent "
            "with the hypothesis that Round-1's poor Qwen Macro-F1 (~0.117) was driven by "
            "class imbalance rather than by the representations themselves. Phase 1 hypotheses "
            "(factual ignorance, prompt framing) are not required to explain Round-1; "
            "Phase 1a/1b can proceed as orthogonal validation."
        )
    elif v["overall"] == "FAIL":
        lines.append(
            "At least one method failed the balanced-subset gate. Class imbalance ALONE "
            "does not explain Round-1's gap, so the Phase 1a (factual ignorance) and "
            "Phase 1b (prompt-framing) hypotheses remain live and should be evaluated."
        )
    else:
        lines.append("Verdict indeterminate — see missing_methods above.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_compare_plot(out_path: Path, payload: dict) -> None:
    methods = []
    r1_vals = []
    mc_means = []
    mc_stds = []
    colors = []
    for m_entry in payload["per_method"]:
        if m_entry["method_gate"] is None:
            continue
        methods.append(m_entry["display_name"])
        r1_vals.append(m_entry["method_round1_f1"])
        mc_means.append(m_entry["method_mc_mean"])
        mc_stds.append(m_entry["method_mc_std"])
        colors.append(METHOD_COLORS.get(m_entry["method"], "#666"))

    if not methods:
        # Plot a placeholder so the file exists
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No method pairings available",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_axis_off()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_r1 = ax.bar(x - width/2, r1_vals, width, label="Round-1 imbalanced",
                     color="#999999", alpha=0.85, edgecolor="black")
    bars_mc = ax.bar(x + width/2, mc_means, width, yerr=[2 * s for s in mc_stds],
                     capsize=6, label="MC balanced (mean ± 2σ)",
                     color=colors, alpha=0.9, edgecolor="black")

    # Per-method horizontal reference line at the R1 score, spanning each pair
    for i, r1 in enumerate(r1_vals):
        ax.hlines(r1, x[i] - 0.5, x[i] + 0.5,
                  colors="black", linestyles="dashed", linewidth=0.8, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0)
    ax.set_ylabel("Macro-F1 (ruler task)")
    ax.set_title("Phase 0 — Round-1 imbalanced vs. balanced MC re-probe")
    ax.legend(loc="best")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Pass/fail annotations
    for i, m_entry in enumerate(
        [m for m in payload["per_method"] if m["method_gate"] is not None]
    ):
        status = "PASS" if m_entry["method_gate"] else "FAIL"
        ymax = mc_means[i] + 2 * mc_stds[i]
        ax.annotate(status, xy=(x[i] + width/2, ymax),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=10,
                    color=("#2E7D32" if m_entry["method_gate"] else "#C62828"),
                    fontweight="bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== aggregate_p0.py ===")
    print(f"  probes_dir : {args.probes_dir}")
    print(f"  round1_cls : {args.round1_cls}")
    print(f"  round1_pls : {args.round1_pls}")
    print(f"  out_dir    : {args.out_dir}")
    print()

    with open(args.round1_cls) as f:
        round1_cls = json.load(f)
    with open(args.round1_pls) as f:
        round1_pls = json.load(f)

    # Load MC summaries (per probe). Some may not exist (e.g. random_*).
    summaries: dict[str, dict | None] = {}
    for method in METHODS:
        pls_probe, cls_probe = PROBES_PER_METHOD[method]
        summaries[pls_probe] = load_mc_summary(args.probes_dir, pls_probe, args.method_tag)
        summaries[cls_probe] = load_mc_summary(args.probes_dir, cls_probe, args.method_tag)

    # Evaluate per-method gates
    per_method: list[dict] = []
    failing_methods: list[str] = []
    missing_methods: list[str] = []
    for method in METHODS:
        pls_probe, cls_probe = PROBES_PER_METHOD[method]
        m_entry = evaluate_method(
            method,
            cls_summary=summaries.get(cls_probe),
            pls_summary=summaries.get(pls_probe),
            round1_cls=round1_cls,
            round1_pls=round1_pls,
        )
        per_method.append(m_entry)
        if m_entry["method_gate"] is None:
            missing_methods.append(method)
        elif not m_entry["method_gate"]:
            failing_methods.append(method)

    # Overall verdict
    if missing_methods and not failing_methods:
        # missing data — be explicit
        overall = "INDETERMINATE"
    elif failing_methods:
        overall = "FAIL"
    elif not per_method or all(m["method_gate"] is None for m in per_method):
        overall = "INDETERMINATE"
    else:
        overall = "PASS"

    # Secondary gate (TF-IDF k-NN proxy)
    secondary = evaluate_secondary_gate(summaries.get("tfidf_cls"))

    # n_draws diagnostic
    draws_by_probe = {
        probe: (s.get("n_draws", 0) if isinstance(s, dict) else 0)
        for probe, s in summaries.items()
    }

    payload = {
        "probes_dir": str(args.probes_dir),
        "round1_cls_path": str(args.round1_cls),
        "round1_pls_path": str(args.round1_pls),
        "method_tag": args.method_tag,
        "verdict": {
            "overall": overall,
            "failing_methods": failing_methods,
            "missing_methods": missing_methods,
        },
        "secondary": secondary,
        "n_draws_per_probe": draws_by_probe,
        "per_method": per_method,
    }

    summary_json = args.out_dir / "phase0_summary.json"
    report_md   = args.out_dir / "phase0_report.md"
    plot_png    = args.out_dir / "phase0_macrof1_compare.png"

    write_summary_json(summary_json, payload)
    write_markdown_report(report_md, payload)
    write_compare_plot(plot_png, payload)

    print(f"Wrote: {summary_json}")
    print(f"Wrote: {report_md}")
    print(f"Wrote: {plot_png}")
    print()
    print(f"Verdict: {overall}")
    if failing_methods:
        print(f"  failing: {failing_methods}")
    if missing_methods:
        print(f"  missing: {missing_methods}")


if __name__ == "__main__":
    main()
