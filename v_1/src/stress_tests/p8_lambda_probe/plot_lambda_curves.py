"""P8 figure — lambda-dial curves per method: x = lambda (1 = pure manifold
geometry, 0 = pure supervised dependence), y = held-out Spearman. Rows =
readout (align1 / pred), cols = cleaning, lines = k neighbors. Gray dashed
reference = the method's P1b PLS best-k MC Spearman (where known).

Usage: python plot_lambda_curves.py            # every results/p8_lambda__*.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve()
RESULTS = _THIS.parent / "results"

# validated categorical slots 1-3 (dataviz reference palette, light mode)
COLORS = ["#2a78d6", "#1baf7a", "#eda100"]
INK, MUT = "#0b0b0b", "#52514e"

# P1b PLS best-k MC reference (mean site) — ADVISOR_WALKTHROUGH table
PLS_REF = {
    "tfidf": {"tier0": 0.376, "maximal": 0.271},
    "qwen3_8b": {"tier0": 0.348, "maximal": 0.339},
    "qwen3_1b7": {"tier0": 0.352, "maximal": 0.334},
    "qwen3_32b": {"tier0": 0.381, "maximal": 0.332},
    "random": {"tier0": 0.351, "maximal": 0.293},
    "thalesian_cunei400m": {"tier0": 0.377, "maximal": 0.391},
    "mlm": {"tier0": 0.399, "maximal": 0.286},
}


def plot_method(fp: Path):
    d = json.loads(fp.read_text())
    method = d["method"]
    cleanings = [c for c, blk in d["cleanings"].items()
                 if blk and not blk.get("missing") and not blk.get("skipped")]
    if not cleanings:
        print(f"{fp.name}: nothing to plot"); return
    readouts = [("align1", "leading-coordinate alignment |ρ(z₁, y)|"),
                ("pred", "ridge-on-Z₃ prediction ρ")]
    fig, axes = plt.subplots(len(readouts), len(cleanings),
                             figsize=(5.2 * len(cleanings), 3.6 * len(readouts)),
                             sharex=True, sharey=True, squeeze=False)
    for ci, cl in enumerate(cleanings):
        blk = d["cleanings"][cl]
        if blk.get("missing") or blk.get("skipped"):
            continue
        if "curves" in blk:      # run_acts.py schema: best-layer curves
            blk = blk["curves"]
            kkeys = list(blk)
            blk = {kk: {"per_lambda": blk[kk]} for kk in kkeys}
        else:                    # run_tfidf_local.py schema: k-sweep blocks
            kkeys = sorted((s for s in blk if s.startswith("k")),
                           key=lambda s: int(s[1:]))
        for ri, (key, title) in enumerate(readouts):
            ax = axes[ri][ci]
            for ki, kk in enumerate(kkeys):
                pl = blk[kk].get("per_lambda")
                if not pl:
                    continue
                lams = sorted(pl, key=float)
                x = [float(l) for l in lams]
                mu = np.array([pl[l][f"{key}_mean"] for l in lams])
                sd = np.array([pl[l][f"{key}_std"] for l in lams])
                c = COLORS[ki % len(COLORS)]
                lab = f"k={kk[1:]}" if kk.startswith("k") and kk[1:].isdigit() else kk
                ax.plot(x, mu, color=c, lw=2, marker="o", ms=4, label=lab)
                ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.12, lw=0)
            ref = PLS_REF.get(method, {}).get(cl)
            if ref is not None and key == "pred":
                ax.axhline(ref, color=MUT, lw=1.2, ls="--")
                ax.text(0.02, ref + 0.012, f"P1b PLS {ref:.3f}", color=MUT,
                        fontsize=8, va="bottom")
            ax.axhline(0, color=MUT, lw=0.8, ls=":", alpha=0.6)
            if ri == 0:
                ax.set_title(cl, color=INK, fontsize=11)
            if ci == 0:
                ax.set_ylabel(title, color=INK, fontsize=9)
            if ri == len(readouts) - 1:
                ax.set_xlabel("λ   (0 = pure supervision · 1 = pure geometry)",
                              fontsize=9, color=INK)
            ax.grid(True, color="#e6e6e3", lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")

    def _ndraws():
        for blk in d["cleanings"].values():
            for v in (blk or {}).values():
                if isinstance(v, dict) and "n_draws_used" in v:
                    return v["n_draws_used"]
        return "?"

    fig.suptitle(f"P8 supervision-dial probe — {method} "
                 f"({_ndraws()} balanced draws, GroupKFold-by-ruler)",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULTS / f"fig_p8_lambda__{method}.png"
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    files = sorted(RESULTS.glob("p8_lambda__*.json"))
    assert files, f"no p8_lambda__*.json under {RESULTS}"
    for fp in files:
        plot_method(fp)
