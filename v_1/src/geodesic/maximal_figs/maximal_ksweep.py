#!/usr/bin/env python3
"""Maximal PLS k-sweep: components-tradeoff + per-method panels (balanced/maximal/mean).

Two modes:
  --emit-layers   print the comma-sep union of best maximal layers (for the
                  sbatch to pass to run_mc_probes' --layers). Reads the STANDARD
                  probe summaries (maximal_figs/probes) written by M1/M3.
  (default)       read the WIDE-k per-draw JSONs (maximal_figs/ksweep_probes),
                  build k->Spearman per model at its best maximal layer, and
                  render ksweep_tradeoff_maximal.png + ksweep_per_method_maximal.png.

Per-draw JSON schema (run_mc_probes): doc["results"][cfg_key]["metrics_per_k"][k]
["spearman_mean"], cfg_key = "{model}__maximal__{pool}__L{NN}__year-raw".
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GEO = HERE.parent
sys.path.insert(0, str(GEO))
import plot_round3_story_figures as prs  # noqa: E402  styling registries

CLEANING = "maximal"
STD_TAG = "mc_balanced_maximal"          # standard probes (best-layer source)
KSWEEP_TAG = "mc_balanced_ksweep_maximal"
MODELS = ["tfidf", "mlm", "thalesian_akk300m", "thalesian_cunei400m",
          "qwen3_1b7", "qwen3_8b", "qwen3_32b", "random"]
_LKEY = re.compile(r"__L(\d+)__")


def best_key(std_probes: Path, model: str) -> str | None:
    """Full year-raw cfg_key of the model's best maximal layer (argmax Spearman)."""
    p = std_probes / f"{model}_pls__{STD_TAG}__summary.json"
    if not p.exists():
        return None
    pc = json.load(open(p)).get("per_config", {})
    best = (None, -np.inf)
    for k, rec in pc.items():
        if f"__{CLEANING}__" not in k or "__last__" in k or not k.endswith("year-raw"):
            continue
        sp = rec.get("spearman_mean")
        if sp is not None and sp > best[1]:
            best = (k, sp)
    return best[0]


def resolve(std_probes: Path) -> dict[str, str]:
    return {m: bk for m in MODELS if (bk := best_key(std_probes, m))}


def collect(ksweep_probes: Path, model: str, cfg_key: str) -> dict[int, list]:
    """k -> [spearman_mean per draw] from the wide-k per-draw JSONs."""
    per_k: dict[int, list] = defaultdict(list)
    for fp in sorted(ksweep_probes.glob(f"{model}_pls__{KSWEEP_TAG}__draw*.json")):
        try:
            rec = json.load(open(fp)).get("results", {}).get(cfg_key)
        except (json.JSONDecodeError, OSError):
            continue
        if not rec or "metrics_per_k" not in rec:
            continue
        for ks, m in rec["metrics_per_k"].items():
            v = m.get("spearman_mean")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                per_k[int(ks)].append(float(v))
    return per_k


def ridge_baseline(t2_csv: Path, model: str) -> float | None:
    if not t2_csv.exists():
        return None
    best = None
    for r in csv.DictReader(open(t2_csv)):
        if r.get("model") == model:
            try:
                v = float(r["spearman_mean"])
            except (ValueError, KeyError, TypeError):
                continue
            best = v if best is None else max(best, v)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--std-probes", type=Path, default=HERE / "probes")
    ap.add_argument("--ksweep-probes", type=Path, default=HERE / "ksweep_probes")
    ap.add_argument("--t2-csv", type=Path, default=HERE / "tables" / "T2_year_ridge_maximal.csv")
    ap.add_argument("--fig-out", type=Path, default=HERE / "figures")
    ap.add_argument("--emit-layers", action="store_true",
                    help="print comma-sep union of best maximal layers and exit")
    args = ap.parse_args()

    keys = resolve(args.std_probes)
    if args.emit_layers:
        layers = sorted({int(m.group(1)) for k in keys.values()
                         if (m := _LKEY.search(k))})
        print(",".join(str(l) for l in layers) or "0")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.fig_out.mkdir(parents=True, exist_ok=True)
    curves = {}                          # model -> (ks, means, stds, ndraws, ridge)
    for m in MODELS:
        ck = keys.get(m)
        if not ck:
            print(f"[skip] {m}: no best maximal key")
            continue
        per_k = collect(args.ksweep_probes, m, ck)
        if not per_k:
            print(f"[skip] {m}: no k-sweep draws at {ck}")
            continue
        ks = sorted(per_k)
        means = [float(np.mean(per_k[k])) for k in ks]
        stds = [float(np.std(per_k[k])) for k in ks]
        curves[m] = (ks, means, stds, len(per_k[ks[0]]), ridge_baseline(args.t2_csv, m), ck)

    # --- tradeoff: all models on one axis ---
    fig, ax = plt.subplots(figsize=(9, 6))
    rows = []
    for m, (ks, means, stds, nd, rb, ck) in curves.items():
        c = prs.MODEL_COLOR.get(m, "#333")
        ax.plot(ks, means, "-o", color=c, label=f"{prs.MODEL_SHORT.get(m, m)} (n={nd})")
        ax.fill_between(ks, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                        color=c, alpha=0.12)
        if rb is not None:
            ax.axhline(rb, ls="--", color=c, lw=1, alpha=0.6)
        for k, mu, sd in zip(ks, means, stds):
            rows.append((m, k, mu, sd, rb))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("PLS components (k) [log2]")
    ax.set_ylabel("Year Spearman (balanced, mean ± SD over draws)")
    ax.set_title("PLS components tradeoff — MAXIMAL/mean — dashed = Ridge")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.fig_out / "ksweep_tradeoff_maximal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- per-method panels ---
    ms = list(curves)
    if ms:
        n = len(ms)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.3), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, m in zip(axes, ms):
            ks, means, stds, nd, rb, ck = curves[m]
            c = prs.MODEL_COLOR.get(m, "#333")
            ax.plot(ks, means, "-o", color=c)
            ax.fill_between(ks, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                            color=c, alpha=0.15)
            if rb is not None:
                ax.axhline(rb, ls="--", color="firebrick", lw=1.2, label=f"Ridge {rb:.3f}")
            bi = int(np.argmax(means))
            ax.scatter([ks[bi]], [means[bi]], marker="*", s=130, color=c,
                       edgecolor="black", zorder=4)
            ax.set_title(f"{prs.MODEL_SHORT.get(m, m)}\npeak k={ks[bi]} ({means[bi]:.3f})", fontsize=9)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("k [log2]")
            ax.legend(fontsize=7, loc="lower left")
        axes[0].set_ylabel("Year Spearman (balanced, maximal)")
        fig.suptitle("Per-method PLS k-sweep — MAXIMAL / mean / balanced", fontsize=12)
        fig.tight_layout()
        fig.savefig(args.fig_out / "ksweep_per_method_maximal.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    tdir = HERE / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / "ksweep_tradeoff_maximal.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "k", "spearman_mean", "spearman_std", "ridge_baseline"])
        w.writerows(rows)
    print(f"[done] {len(curves)} models -> {args.fig_out}/ksweep_*_maximal.png")


if __name__ == "__main__":
    main()
