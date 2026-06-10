"""Fine-tune round scoreboard: base vs depth-ablation arms, fig4-style.

Reads:
  v_1/src/finetune/results/probes/*_pls__mc_balanced*__summary.json  (FT arms + gpt-oss base)
  v_1/src/geodesic/maximal_figs/tables/T1_year_pls_maximal.csv       (base curves, maximal)
  v_1/src/geodesic/results/tables/T1_year_pls.csv                    (base curves, tier0)

Writes:
  v_1/src/finetune/results/scoreboard_layers.csv  (long: family, arm, cleaning, layer, spearman)
  v_1/src/finetune/results/scoreboard_best.csv    (best layer per family x arm x cleaning)
  v_1/src/finetune/results/scoreboard.md
  v_1/src/finetune/results/figures/ftcurves_<family>_<cleaning>.png

Safe to run anytime — uses whatever summaries exist so far.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
PROBES_DIR = REPO / "v_1" / "src" / "finetune" / "results" / "probes"
OUT_DIR = REPO / "v_1" / "src" / "finetune" / "results"
FIG_DIR = OUT_DIR / "figures"

BASE_TABLES = {
    "maximal": REPO / "v_1/src/geodesic/maximal_figs/tables/T1_year_pls_maximal.csv",
    "tier0":   REPO / "v_1/src/geodesic/results/tables/T1_year_pls.csv",
}
TAG_TO_CLEANING = {"mc_balanced": "tier0", "mc_balanced_maximal": "maximal"}
FAMILIES = ("qwen3_1b7", "qwen3_8b", "qwen3_32b", "gpt_oss_120b")

KEY_RE = re.compile(r"__L(\d+)__year-raw$")


def split_method(method: str) -> tuple[str, str]:
    """qwen3_8b_ft12 -> (qwen3_8b, ft12); gpt_oss_120b -> (gpt_oss_120b, base)."""
    m = re.match(r"^(.*)_ft(\d+)$", method)
    if m:
        return m.group(1), f"ft{m.group(2)}"
    return method, "base"


def main() -> None:
    rows = []

    for path in sorted(PROBES_DIR.glob("*_pls__mc_balanced*__summary.json")):
        with open(path) as f:
            s = json.load(f)
        probe, tag = s.get("probe", ""), s.get("method_tag", "")
        cleaning = TAG_TO_CLEANING.get(tag)
        if cleaning is None or not probe.endswith("_pls"):
            continue
        method = probe[: -len("_pls")]
        family, arm = split_method(method)
        for key, agg in s.get("per_config", {}).items():
            m = KEY_RE.search(key)
            if not m or "spearman_mean" not in agg:
                continue
            rows.append({"family": family, "arm": arm, "cleaning": cleaning,
                         "layer": int(m.group(1)),
                         "spearman_mean": agg["spearman_mean"],
                         "spearman_std": agg.get("spearman_std")})

    # base curves from the canonical Round-3 tables (skip families we already
    # have a probed 'base' for, e.g. gpt_oss_120b from FT0b)
    have_base = {(r["family"], r["cleaning"]) for r in rows if r["arm"] == "base"}
    for cleaning, table in BASE_TABLES.items():
        if not table.exists():
            continue
        t = pd.read_csv(table)
        for fam in FAMILIES:
            if (fam, cleaning) in have_base:
                continue
            sub = t[t["model"] == fam]
            for _, r in sub.iterrows():
                # maximal table stores ints, tier0 table stores 'L00' strings
                layer = int(str(r["layer"]).lstrip("L"))
                rows.append({"family": fam, "arm": "base", "cleaning": cleaning,
                             "layer": layer,
                             "spearman_mean": float(r["spearman_mean"]),
                             "spearman_std": float(r.get("spearman_std", float("nan")))})

    if not rows:
        print("[scoreboard] no probe summaries found yet — nothing to do")
        return

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["family", "arm", "cleaning", "layer"]).sort_values(
        ["family", "arm", "cleaning", "layer"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "scoreboard_layers.csv", index=False)

    best = (df.loc[df.groupby(["family", "arm", "cleaning"])["spearman_mean"].idxmax()]
            .rename(columns={"layer": "best_layer"})
            .sort_values(["family", "cleaning", "arm"]))
    best.to_csv(OUT_DIR / "scoreboard_best.csv", index=False)

    lines = ["# Fine-tune scoreboard — best-layer year-PLS Spearman (balanced, 200 draws)", ""]
    for cleaning in ("maximal", "tier0"):
        sub = best[best["cleaning"] == cleaning]
        if sub.empty:
            continue
        lines += [f"## {cleaning}", "", "| family | arm | best layer | Spearman |", "|---|---|---|---|"]
        for _, r in sub.iterrows():
            lines.append(f"| {r['family']} | {r['arm']} | {r['best_layer']} | "
                         f"{r['spearman_mean']:.4f} ± {r['spearman_std']:.4f} |")
        lines.append("")
    (OUT_DIR / "scoreboard.md").write_text("\n".join(lines))
    print(f"[scoreboard] {len(df)} rows -> scoreboard_layers.csv / scoreboard_best.csv / scoreboard.md")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        for (fam, cleaning), sub in df.groupby(["family", "cleaning"]):
            if sub["arm"].nunique() < 2:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            for arm, a in sub.groupby("arm"):
                a = a.sort_values("layer")
                ax.plot(a["layer"], a["spearman_mean"], marker="o", ms=3,
                        lw=1.5 if arm == "base" else 1.0,
                        color="black" if arm == "base" else None, label=arm)
            ax.set(xlabel="layer (hidden-state index)", ylabel="year-PLS Spearman",
                   title=f"{fam} — {cleaning} · mean-pool · balanced 200 draws")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"ftcurves_{fam}_{cleaning}.png", dpi=150)
            plt.close(fig)
        print(f"[scoreboard] figures -> {FIG_DIR}")
    except Exception as e:  # plotting is best-effort
        print(f"[scoreboard] WARN figure rendering skipped: {e}")


if __name__ == "__main__":
    main()
