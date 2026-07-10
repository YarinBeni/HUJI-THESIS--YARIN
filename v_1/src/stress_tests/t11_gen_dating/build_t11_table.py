"""build_t11_table.py — the T11 headline table: the model's GENERATED answer
(behavioral, MC Spearman over the same 200 balanced 8x21 draws) next to the
ACTIVATION-probe Spearman (mean site) for the same model x cleaning.

Probe-column sources (all committed result JSONs):
  tier0 / maximal -> p1_gurnee_tegmark/results/mc/p1_year_mc__{m}.json
                     sites.mean_tier0 / mean_maximal (same 200 draws — apples-to-apples)
  engtier0        -> translation/results/trans_mc__{m}.json cleanings.engtier0.year_best
                     (same 200 draws — apples-to-apples)
  maxking         -> p1_gurnee_tegmark/results/maxking/p1_maxking__{m}.json
                     sites.mean.best.year_strat  (NOTE: different protocol — 5 rulers x 9
                     king-found frags, StratifiedKFold — flagged with * in the table)

Also reports parse/decline rates and the named-vs-unnamed conditional Spearman
(name-lookup dating vs style dating).

Usage: python build_t11_table.py           # -> results/t11_vs_probe.md + stdout
Pure stdlib. Safe on a login node.
"""
from __future__ import annotations

import json
from pathlib import Path

ST = Path(__file__).resolve().parents[1]
T11 = ST / "t11_gen_dating" / "results"
MODELS = ["qwen3_1b7", "qwen3_8b", "qwen3_32b", "gpt_oss_120b"]
CLEANINGS = ["tier0", "maximal", "maxking", "engtier0"]


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, pm=None):
    if x is None or x != x:
        return "—"
    return f"{x:.3f}" if pm is None else f"{x:.3f}±{pm:.2f}"


def probe_spearman(model: str, cleaning: str):
    """-> (value, is_same_protocol) for the mean-site activation probe."""
    if cleaning in ("tier0", "maximal"):
        d = jload(ST / f"p1_gurnee_tegmark/results/mc/p1_year_mc__{model}.json")
        if d:
            blk = d.get("sites", {}).get(f"mean_{cleaning}", {})
            b = blk.get("best")
            if b:
                return b.get("spearman_mean"), True
    elif cleaning == "engtier0":
        d = jload(ST / f"translation/results/trans_mc__{model}.json")
        if d:
            b = d.get("cleanings", {}).get("engtier0", {}).get("year_best")
            if b:
                return b.get("spearman_mean"), True
    elif cleaning == "maxking":
        d = jload(ST / f"p1_gurnee_tegmark/results/maxking/p1_maxking__{model}.json")
        if d:
            blk = d.get("sites", {}).get("mean", {})
            b = blk.get("best", {}).get("year_strat") if blk.get("best") else None
            if b:
                return b.get("spearman_mean"), False
    return None, True


def main():
    lines = [
        "# T11 — generated-answer dating vs activation probe (mean site)",
        "",
        "Answer = Spearman of the model's parsed year answers over the SAME 200",
        "balanced 8x21 draws as every probe. Probe = PLS best-k MC Spearman on the",
        "mean-pooled activations (\\*maxking probe uses its own 5x9 StratifiedKFold",
        "protocol — indicative only). named/unnamed = full-corpus Spearman",
        "conditioned on the answer text naming the true ruler.",
        "",
        "| model | cleaning | answer MC | probe MC | scoreable | declined | named rate | ρ named (n) | ρ unnamed (n) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    n_found = 0
    for m in MODELS:
        for cl in CLEANINGS:
            d = jload(T11 / f"t11_gen__{m}__{cl}.json")
            if d is None:
                lines.append(f"| {m} | {cl} | PENDING | | | | | | |")
                continue
            n_found += 1
            f, mc = d["full_corpus"], d["mc_balanced"]
            sc = f.get("parse_status_counts", {})
            n = f.get("n_generated") or 0
            declined = sum(sc.get(k, 0) for k in ("declined", "ce_only", "unparsed"))
            pv, same = probe_spearman(m, cl)
            decline_cell = f"{declined / n:.2f}" if n else "—"
            lines.append(
                f"| {m} | {cl} | {fmt(mc.get('spearman_mean'), mc.get('spearman_std'))} "
                f"| {fmt(pv)}{'' if same else '*'} "
                f"| {f.get('n_scoreable', 0)}/{n} "
                f"| {decline_cell} "
                f"| {fmt(f.get('named_true_ruler_rate'))} "
                f"| {fmt(f.get('spearman_when_named'))} ({f.get('n_named', 0)}) "
                f"| {fmt(f.get('spearman_when_unnamed'))} ({f.get('n_unnamed', 0)}) |"
            )
    out = "\n".join(lines) + "\n"
    T11.mkdir(parents=True, exist_ok=True)
    fp = T11 / "t11_vs_probe.md"
    fp.write_text(out, encoding="utf-8")
    print(out)
    print(f"[{n_found}/{len(MODELS) * len(CLEANINGS)} cells landed]  wrote {fp}")


if __name__ == "__main__":
    main()
