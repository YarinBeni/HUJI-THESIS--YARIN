"""plot_fragments_by_ruler.py — fragment counts for ALL labelled rulers, ordered by
year (lower bound of the sub-period, as built by corpus/03_build_orcc_corpus.py:
year = min digit of `sub_period`). Blue = all fragments, green = fragments whose
king-name token is present under the maximal_keepking cleaning. Bar label = total
(king-found). Retained-5 rulers (the maxking probe set) get red x-tick labels.

Usage:
    python v_1/src/stress_tests/eda/plot_fragments_by_ruler.py
Output: results/eda/fig_fragments_by_ruler_all.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "shared"))
from cleaning import clean_maximal_keepking          # noqa: E402
from king_token import find_name_word, load_spellings  # noqa: E402

CORPUS = REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
OUT = REPO / "v_1/src/stress_tests/results/eda/fig_fragments_by_ruler_all.png"
RETAINED = {"Ashurbanipal", "Sennacherib", "Esarhaddon", "Sargon II", "Sîn-šarru-iškun"}


def king_found_count(df, spell, ruler):
    sub = df[df["ruler"] == ruler]
    sp = spell.get(ruler)
    if not sp:
        return 0
    return int(sub["text_tier0"].apply(
        lambda x: find_name_word(clean_maximal_keepking(str(x), sp)[0], sp) is not None).sum())


def main():
    df = pd.read_parquet(CORPUS)
    spell = load_spellings()
    rows = []
    for r, sub in df.groupby("ruler"):
        y = sub["year"].dropna()
        if len(y) == 0:      # drop rulers without any year (e.g. 'ribo')
            continue
        rows.append((r, int(y.min()), len(sub), king_found_count(df, spell, r)))
    rows.sort(key=lambda t: -t[1])   # oldest (largest BCE) on the left
    names = [r for r, _, _, _ in rows]
    yrs = [y for _, y, _, _ in rows]
    tot = [n for _, _, n, _ in rows]
    kfs = [k for _, _, _, k in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(17, 7.4))
    ax.bar(x, tot, color="#3498db", zorder=3, width=0.8)
    ax.bar(x, kfs, color="#2ecc71", zorder=4, width=0.8)
    for i, (t, k) in enumerate(zip(tot, kfs)):
        lab = f"{t}({k})" if k > 0 else f"{t}"
        ax.annotate(lab, (i, t), textcoords="offset points", xytext=(0, 2), ha="center",
                    va="bottom", fontsize=6, rotation=90,
                    color="#0b6b2e" if k > 0 else "#555")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}  ({y})" for n, y in zip(names, yrs)], rotation=90, fontsize=7)
    for tick, n in zip(ax.get_xticklabels(), names):
        if n in RETAINED:
            tick.set_color("#c0392b"); tick.set_fontweight("bold")
    ax.set_ylabel("# fragments")
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylim(0, max(tot) * 1.18)   # headroom so tall bar labels clear the title
    ax.set_title("ORCC fragments per ruler — all 40 labelled rulers, ordered by year "
                 "(lower bound of sub-period, BCE, oldest→left)\nblue = all fragments, "
                 "green = king-name token present (maximal-with-kings)   |   label = total(king-found)")
    ax.legend(handles=[Patch(color="#3498db", label="all fragments"),
                       Patch(color="#2ecc71", label="with king-name token"),
                       Patch(color="#c0392b", label="retained in maxking probe (5)")],
              loc="upper right", fontsize=9)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}  ({len(rows)} rulers)")


if __name__ == "__main__":
    main()
