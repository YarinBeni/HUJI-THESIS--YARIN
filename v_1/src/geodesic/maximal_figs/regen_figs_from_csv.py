#!/usr/bin/env python3
"""Regenerate maximal figures from the committed CSVs (no cluster probe JSONs needed).

Reads:
  tables/T1_year_pls_maximal.csv
  tables/T2_year_ridge_maximal.csv
  tables/T3_ruler_maximal.csv
  tables/ksweep_tradeoff_maximal.csv

Writes into figures/:
  fig1_maximal_ACD.png
  fig2_maximal_AB.png
  fig4_maximal_A.png
  ksweep_tradeoff_maximal.png
  ksweep_per_method_maximal.png
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GEO = HERE.parent
sys.path.insert(0, str(GEO))
import plot_round3_story_figures as prs  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import make_maximal_figs as mmf  # noqa: E402

MODELS = ["tfidf", "mlm", "thalesian_akk300m", "thalesian_cunei400m",
          "qwen3_1b7", "qwen3_8b", "qwen3_32b", "random"]


def _float(s: str) -> float | None:
    try:
        return float(s) if s not in ("", "nan") else None
    except (ValueError, TypeError):
        return None


def load_data(tables: Path) -> dict:
    data: dict = {"pls": {}, "ridge": {}, "ruler": {}}
    for r in csv.DictReader(open(tables / "T1_year_pls_maximal.csv")):
        m, L = r["model"], int(r["layer"])
        data["pls"].setdefault(m, {})[L] = (_float(r["spearman_mean"]), _float(r["spearman_std"]))
    for r in csv.DictReader(open(tables / "T2_year_ridge_maximal.csv")):
        m, L = r["model"], int(r["layer"])
        data["ridge"].setdefault(m, {})[L] = (_float(r["spearman_mean"]), _float(r["spearman_std"]))
    for r in csv.DictReader(open(tables / "T3_ruler_maximal.csv")):
        data["ruler"][r["model"]] = (_float(r["macro_f1"]), _float(r["macro_f1_std"]),
                                     r.get("source", ""))
    return data


_LKEY = re.compile(r"__L(\d+)__")
FINETUNE_PROBES = HERE.parent.parent / "finetune" / "results" / "probes"


def _load_probe_layers(probe_json: Path, value_key: str, suffix: str) -> dict:
    """Read {layer -> (mean, std)} from a probe summary JSON for maximal/mean configs."""
    if not probe_json.exists():
        return {}
    pc = json.load(open(probe_json)).get("per_config", {})
    out: dict = {}
    for k, rec in pc.items():
        if "__maximal__" not in k or "__last__" in k or not k.endswith(suffix):
            continue
        if "__mean__" not in k:
            continue
        m = _LKEY.search(k)
        L = int(m.group(1)) if m else 0
        out[L] = (rec.get(f"{value_key}_mean"), rec.get(f"{value_key}_std"))
    return out


def inject_gpt_oss_120b(data: dict) -> None:
    """Load gpt_oss_120b probe data from the finetune results and inject into data."""
    pls = _load_probe_layers(
        FINETUNE_PROBES / "gpt_oss_120b_pls__mc_balanced_maximal__summary.json",
        "spearman", "year-raw",
    )
    ridge = _load_probe_layers(
        FINETUNE_PROBES / "gpt_oss_120b_cls_numeric__mc_balanced_maximal__summary.json",
        "spearman", "year-raw",
    )
    # ruler: best macro_f1 over cls and pls probes
    best_ruler: tuple = (None, None, None)
    for probe, src in [("cls", "cls"), ("pls", "plsda")]:
        p = FINETUNE_PROBES / f"gpt_oss_120b_{probe}__mc_balanced_maximal__summary.json"
        if not p.exists():
            continue
        for k, rec in json.load(open(p)).get("per_config", {}).items():
            if "__maximal__" not in k or "__last__" in k or not k.endswith("ruler"):
                continue
            f1 = rec.get("macro_f1_mean")
            if f1 is not None and (best_ruler[0] is None or f1 > best_ruler[0]):
                best_ruler = (f1, rec.get("macro_f1_std"), src)
    if pls:
        data["pls"]["gpt_oss_120b"] = pls
        print(f"[gpt_oss_120b] {len(pls)} PLS layers loaded")
    if ridge:
        data["ridge"]["gpt_oss_120b"] = ridge
        print(f"[gpt_oss_120b] {len(ridge)} Ridge layers loaded")
    if best_ruler[0] is not None:
        data["ruler"]["gpt_oss_120b"] = best_ruler
        print(f"[gpt_oss_120b] ruler macro_f1={best_ruler[0]:.3f}")


def regen_ksweep(tables: Path, fig_out: Path) -> None:
    ksweep_csv = tables / "ksweep_tradeoff_maximal.csv"
    if not ksweep_csv.exists():
        print(f"[skip] {ksweep_csv} not found")
        return

    curves: dict[str, dict] = {}
    for r in csv.DictReader(open(ksweep_csv)):
        m, k = r["model"], int(r["k"])
        curves.setdefault(m, {})[k] = (
            float(r["spearman_mean"]),
            float(r["spearman_std"]),
            _float(r.get("ridge_baseline", "")),
        )

    present = [m for m in MODELS if m in curves]

    # tradeoff: all models on one axis
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in present:
        dm = curves[m]
        ks = sorted(dm)
        means = [dm[k][0] for k in ks]
        rb = dm[ks[0]][2]
        c = prs.MODEL_COLOR.get(m, "#333")
        ax.plot(ks, means, "-o", color=c, label=prs.MODEL_SHORT.get(m, m))
        if rb is not None:
            ax.axhline(rb, ls="--", color=c, lw=1, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("PLS components (k) [log2]")
    ax.set_ylabel("Year Spearman (balanced, mean over draws)")
    ax.set_title("PLS components tradeoff — MAXIMAL/mean — dashed = Ridge")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_out / "ksweep_tradeoff_maximal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {fig_out}/ksweep_tradeoff_maximal.png")

    # per-method panels
    if not present:
        return
    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.3), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, present):
        dm = curves[m]
        ks = sorted(dm)
        means = [dm[k][0] for k in ks]
        rb = dm[ks[0]][2]
        c = prs.MODEL_COLOR.get(m, "#333")
        ax.plot(ks, means, "-o", color=c)
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
    fig.savefig(fig_out / "ksweep_per_method_maximal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {fig_out}/ksweep_per_method_maximal.png")


def main() -> None:
    tables = HERE / "tables"
    fig_out = HERE / "figures"
    fig_out.mkdir(parents=True, exist_ok=True)

    data = load_data(tables)
    inject_gpt_oss_120b(data)
    mmf.fig1_ACD(data, fig_out)
    mmf.fig2_AB(data, fig_out)
    mmf.fig4_A(data, fig_out)
    regen_ksweep(tables, fig_out)
    print("[done] all figures regenerated from CSVs")


if __name__ == "__main__":
    main()
