"""class_imbalance_analysis.py — why is king_last "so easy"?

Diagnoses the balanced-MC probe from the *data* side, for the king_last / king_mean
pooling sites vs the whole-text mean pool. Answers the two questions:

  1. How biased/imbalanced are the classes (ruler / period / provenance), and did the
     balanced-MC undersample to the SMALLEST class (k = min per-ruler count)?
  2. For the king_* sites, king-name coverage shrinks the usable set further — so how
     small does the effective per-draw sample actually get, and how many distinct
     year-values / ruler-groups survive (which is what makes the probe trivial)?

Everything here is CPU + data only (no model, no activations). King-name "found" is
computed at the WORD level exactly like shared/king_token.coverage_report (the J1
coverage proxy), so it matches the `found` mask used by the king pools up to
tokenizer edge-cases.

Usage:
    python v_1/src/stress_tests/eda/class_imbalance_analysis.py
Outputs (under v_1/src/stress_tests/results/eda/):
    class_imbalance.md         — the report (tables)
    fig_ruler_counts.png       — full-corpus fragments per ruler (8 balanced ones flagged)
    fig_king_coverage.png      — king-name coverage per balanced ruler (mean vs king pool)
    fig_effective_per_draw.png — effective fragments/ruler per draw: mean(=21) vs king_last
    fig_year_by_ruler.png      — the probe target: year is ~1 value per ruler (8 discrete)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "shared"))
from king_token import find_name_word, load_spellings  # noqa: E402

CORPUS = REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
SUBSET = REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"
OUT = REPO / "v_1/src/stress_tests/results/eda"


def king_found_series(df: pd.DataFrame, spell: dict[str, list[str]], col: str) -> pd.Series:
    """Per-fragment boolean: was the commissioning ruler's own name located in `col`?
    (word-level, tier0 — the king_* sites are tier0-only). Rulers with no spelling
    entry -> always False (matches the extractor: no span -> not found)."""
    def hit(row):
        sp = spell.get(row["ruler"])
        if not sp:
            return False
        return find_name_word(str(row[col]), sp) is not None
    return df.apply(hit, axis=1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CORPUS)
    spell = load_spellings()
    manifest = json.loads((SUBSET / "manifest.json").read_text())
    rulers8 = manifest["rulers"]
    k = manifest["k"]
    n_draws = manifest["n_draws"]

    df["king_found"] = king_found_series(df, spell, "text_tier0")

    lines: list[str] = []
    P = lines.append
    P("# Class-imbalance / sample-size diagnosis of the balanced-MC king probe\n")
    P(f"Corpus: {len(df)} fragments, {df['ruler'].nunique()} rulers, "
      f"{df['period'].nunique(dropna=True)} periods, "
      f"{df['provenance'].nunique(dropna=True)} provenances.\n")
    P(f"Balanced-MC subset: **{len(rulers8)} rulers × k={k} = {k*len(rulers8)} "
      f"fragments/draw**, {n_draws} draws. k is capped by the SMALLEST of the 8 "
      f"classes (undersampling to the min).\n")

    # ---- 1. per-ruler counts + year over the WHOLE corpus -------------------
    g = (df.groupby("ruler")
           .agg(n=("fragment_id", "size"),
                year=("year", lambda s: s.dropna().median() if s.notna().any() else np.nan),
                king_cov=("king_found", "mean"))
           .sort_values("n", ascending=False))
    P("## 1. Full-corpus fragments per ruler (top 15) + king-name coverage\n")
    P("| ruler | n frags | median year (BCE) | king cov (tier0) | in balanced-8 |")
    P("|---|---|---|---|---|")
    for r, row in g.head(15).iterrows():
        yr = "" if pd.isna(row["year"]) else f"{int(-row['year'])}" if row["year"] < 0 else f"{int(row['year'])}"
        P(f"| {r} | {int(row['n'])} | {yr} | {row['king_cov']:.2f} | "
          f"{'✅' if r in rulers8 else ''} |")
    tail_n = g[~g.index.isin(rulers8)]["n"]
    P(f"\n*{(g['n'] < k).sum()} of {len(g)} rulers have < k={k} fragments and are "
      f"excluded from balanced-MC entirely; the 33 excluded rulers hold "
      f"{int(tail_n.sum())} fragments ({tail_n.sum()/len(df):.0%} of the corpus).*\n")

    # ---- 2. the balanced 8: counts, year, king coverage --------------------
    P("## 2. The 8 balanced rulers — the probe only ever sees these\n")
    P("| ruler | full n | median year (BCE) | king cov (tier0) | E[king-found in a "
      f"k={k} draw] |")
    P("|---|---|---|---|---|")
    b = df[df["ruler"].isin(rulers8)]
    eff = {}
    for r in rulers8:
        sub = df[df["ruler"] == r]
        cov = sub["king_found"].mean()
        yr = sub["year"].dropna()
        yv = int(yr.median()) if len(yr) else None
        eff[r] = cov * k
        P(f"| {r} | {len(sub)} | {abs(yv) if yv is not None else '?'} | {cov:.2f} | "
          f"{cov*k:.1f} / {k} |")
    P(f"\nDistinct median-year values among the 8 rulers: "
      f"**{b.groupby('ruler')['year'].median().nunique()}** — i.e. the regression "
      f"target `year` is essentially an **8-level step function of ruler identity**.\n")

    # ---- 3. effective sample: mean vs king_last ----------------------------
    total_mean = k * len(rulers8)
    total_king = sum(eff.values())
    surviving = sum(1 for r in rulers8 if eff[r] >= 1.0)
    weak = [r for r in rulers8 if eff[r] < 3.0]
    P("## 3. Effective per-draw sample: `mean` vs `king_last`/`king_mean`\n")
    P(f"- **mean pool:** all {total_mean} fragments/draw, all 8 ruler-groups, balanced 21/ruler.")
    P(f"- **king pool:** only name-found fragments survive → ~**{total_king:.0f} "
      f"fragments/draw** (≈{total_king/total_mean:.0%} of mean), and only "
      f"~**{surviving}/8 ruler-groups** contribute ≥1 point on average.")
    if weak:
        P(f"- Rulers nearly absent from the king pool (E[found] < 3 per draw): "
          f"{', '.join(weak)} — mostly Neo-Babylonian admin that never name the king.")
    P(f"\nWith GroupKFold-by-ruler (n_splits=5) over ~{surviving} surviving groups and "
      f"only a handful of distinct year values, each test fold holds 1–2 rulers = 1–2 "
      f"distinct years. Spearman on a fold with one year is undefined (the "
      f"`ConstantInputWarning` in J6/J3r logs); with two it collapses to 'are the two "
      f"groups separated?'. That is why king_last is **high AND high-variance** "
      f"(±0.3–0.4) and why an **untrained/random** model scores ~0.64: the name token "
      f"is a near one-hot ruler id, and year is a function of ruler, so any pooling that "
      f"reads the name token trivially recovers year — no learned chronology needed.\n")

    # ---- 4. period / provenance imbalance ----------------------------------
    P("## 4. Period & provenance imbalance (whole corpus)\n")
    P("### period\n| period | n | share |\n|---|---|---|")
    for p, n in df["period"].value_counts(dropna=False).items():
        P(f"| {p} | {n} | {n/len(df):.1%} |")
    P("\n### provenance (top 12)\n| provenance | n | share |\n|---|---|---|")
    for p, n in df["provenance"].value_counts(dropna=False).head(12).items():
        P(f"| {p} | {n} | {n/len(df):.1%} |")
    prov_g = df["provenance"].value_counts(dropna=True)
    P(f"\n*{(prov_g == 1).sum()} provenances have a single fragment; the top-3 sites "
      f"cover {prov_g.head(3).sum()/len(df):.0%} of the corpus.*\n")

    (OUT / "class_imbalance.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- figures -----------------------------------------------------------
    # fig 1: ruler counts
    fig, ax = plt.subplots(figsize=(10, 5))
    gg = g.sort_values("n", ascending=False)
    colors = ["#c0392b" if r in rulers8 else "#bdc3c7" for r in gg.index]
    ax.bar(range(len(gg)), gg["n"], color=colors)
    ax.axhline(k, ls="--", c="k", lw=1, label=f"k={k} (balanced draw size / ruler)")
    ax.set_xticks(range(len(gg))); ax.set_xticklabels(gg.index, rotation=90, fontsize=6)
    ax.set_ylabel("fragments"); ax.set_title("Fragments per ruler (red = in balanced-8)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig_ruler_counts.png", dpi=130); plt.close(fig)

    # fig 2: king coverage per balanced ruler
    fig, ax = plt.subplots(figsize=(9, 4.5))
    covs = [df[df["ruler"] == r]["king_found"].mean() for r in rulers8]
    ax.bar(rulers8, covs, color="#2980b9")
    ax.axhline(np.mean(covs), ls="--", c="k", lw=1, label=f"mean cov={np.mean(covs):.2f}")
    ax.set_ylabel("king-name coverage (tier0)"); ax.set_ylim(0, 1)
    ax.set_title("King-name coverage per balanced ruler (fraction of frags usable by king_*)")
    ax.set_xticklabels(rulers8, rotation=35, ha="right", fontsize=8)
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig_king_coverage.png", dpi=130); plt.close(fig)

    # fig 3: effective per-draw
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(rulers8)); w = 0.4
    ax.bar(x - w/2, [k]*len(rulers8), w, label="mean pool (=k=21)", color="#95a5a6")
    ax.bar(x + w/2, [eff[r] for r in rulers8], w, label="king pool (E[found])", color="#c0392b")
    ax.set_xticks(x); ax.set_xticklabels(rulers8, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("fragments per ruler per draw")
    ax.set_title("Effective per-draw sample per ruler: mean vs king_last/king_mean")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig_effective_per_draw.png", dpi=130); plt.close(fig)

    # fig 4: year by ruler (target structure)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    yrs = [df[df["ruler"] == r]["year"].dropna() for r in rulers8]
    ax.boxplot([(-y).values for y in yrs], tick_labels=rulers8)
    ax.set_ylabel("year (BCE)")
    ax.set_title("Probe target `year` by ruler — ~1 value/ruler → 8 discrete targets")
    ax.set_xticklabels(rulers8, rotation=35, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_year_by_ruler.png", dpi=130); plt.close(fig)

    print("wrote report + 4 figures to", OUT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
