"""E-spec — the specification curve for the E1 headline (closes gap G13).

Reads EVERY pairwise-probe result on disk — every arm x variant x site x m —
and draws one figure: the distribution of macro accuracies per arm, so the
reader sees the headline numbers in the context of every specification that was
run, not just the one that was quoted. "You picked the configuration that
works" is answered by showing all of them.

    python spec_curve.py

Writes results/spec_curve.csv + results/figs/spec_curve.png.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def collect():
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "probes", "*.json"))):
        j = json.load(open(p))
        f = j.get("full", {})
        if f.get("skipped") or "macro_acc_mean" not in f:
            continue
        rows.append({"method": j["method"], "variant": j["variant"],
                     "site": j["site"], "m": j["m"],
                     "macro": f["macro_acc_mean"], "sd": f["macro_acc_std"],
                     "spec": f"{j['site']}/m{j['m']}"})
    return pd.DataFrame(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = collect()
    if t.empty:
        raise SystemExit("no probe results found")
    t.to_csv(os.path.join(RESULTS, "spec_curve.csv"), index=False)
    print(t.sort_values(["variant", "macro"], ascending=[True, False])
          .to_string(index=False))

    variants = sorted(t.variant.unique())
    fig, axes = plt.subplots(1, len(variants), figsize=(7 * len(variants), 6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, var in zip(axes, variants):
        d = t[t.variant == var]
        order = (d.groupby("method").macro.median()
                 .sort_values(ascending=False).index.tolist())
        for i, m in enumerate(order):
            g = d[d.method == m]
            ax.scatter([i] * len(g), g.macro, s=42,
                       c=["#2b4a8b" if s == "mean/m21" else "#a25e33"
                          for s in g.spec], zorder=3)
            ax.errorbar([i] * len(g), g.macro, yerr=g.sd, fmt="none",
                        ecolor="#999", lw=1, zorder=2)
        floor = d[d.method == "tfidf_char"].macro.median()
        if np.isfinite(floor):
            ax.axhline(floor, color="#8c3a3a", ls="--", lw=1.2,
                       label=f"tfidf floor (median {floor:.3f})")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{var} — every specification on disk")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("pairwise macro accuracy (both-rulers-held-out)")
    fig.suptitle("E1 specification curve: blue = headline spec (mean/m21), "
                 "copper = every other spec", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.join(RESULTS, "figs"), exist_ok=True)
    out = os.path.join(RESULTS, "figs", "spec_curve.png")
    fig.savefig(out, dpi=200)
    print(f"[done] {len(t)} specs -> {out}")


if __name__ == "__main__":
    main()
