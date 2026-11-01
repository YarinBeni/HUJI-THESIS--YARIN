"""aggregate_head_ladder.py — P1 head ladder: the Chrono-Barlow head trained on
LEACE-erased features (C5) next to the frozen probe on the same erased features
(C4) and the unerased references (E-MIN v2).

    python chrono/scripts/aggregate_head_ladder.py --out chrono/reports/HEAD_LADDER_RESULT.md
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chrono.eval.protocol import mc_balanced_rho, pooled_rho   # noqa: E402
from aggregate_emin import load_oof                            # noqa: E402

ENC = {"cunei400m": 12, "llama2_7b": 16, "qwen3_8b": 18}
CONS = ["provenance", "period", "subgenre", "length"]


def head_stats(files, corpus, mc, gkf, cond="orig"):
    seeds = sorted({re.search(r"-s(\d+)-f", f).group(1) for f in files})
    mcs, pools = [], []
    for sd in seeds:
        oof = load_oof([f for f in files if f"-s{sd}-f" in f])
        o = oof[oof["condition"] == cond]
        s = pd.Series(o["s"].to_numpy(), index=pd.Index(o["doc_id"], name="doc_id"))
        mcs.append(float(np.nanmean(mc_balanced_rho(s, corpus, mc)))); pools.append(float(pooled_rho(s, corpus, gkf)))
    return np.mean(mcs), np.std(mcs), np.mean(pools), len(seeds)


def probe_ladder(enc):
    d = {}
    p = f"chrono/reports/tier0/ladder/emin2_{enc}_t0_akk_ridge.md"
    if os.path.exists(p):
        for line in open(p):
            m = re.match(r"\| (\w+) \| \d+ \| ([+-][\d.]+) \| ([+-][\d.]+) \|", line)
            if m: d[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="chrono/artifacts_tier0")
    ap.add_argument("--head-dir", default="chrono/reports/tier0/ladder/head_scores")
    ap.add_argument("--v2-dir", default="chrono/reports/tier0/scores")
    ap.add_argument("--out", default="chrono/reports/HEAD_LADDER_RESULT.md")
    args = ap.parse_args(argv)
    corpus = pd.read_parquet(os.path.join(args.art, "corpus_chrono.parquet"))
    mc = json.load(open(os.path.join(args.art, "splits", "mc_balanced.json")))
    gkf = json.load(open(os.path.join(args.art, "splits", "gkf_ruler.json")))
    lines = ["# P1 — head ladder (C5): Chrono-Barlow on LEACE-erased features — tier0 Akkadian", "",
             "mc ρ (mean ± sd over seeds; pooled gkf ρ in brackets). Probe = ridge on the SAME erased "
             "features (C4). Δ = head − probe on the same rung.", ""]
    for enc in ENC:
        pl = probe_ladder(enc)
        lines += [f"## {enc} (L{ENC[enc]} mean)", "", "| erased | head | probe (ridge) | Δ head−probe | head retains |", "|---|---|---|---|---|"]
        ref_files = glob.glob(os.path.join(args.v2_dir, f"emin2_{enc}_t0_akk-s*-f*.parquet"))
        h0 = head_stats(ref_files, corpus, mc, gkf) if ref_files else None
        if h0:
            p0 = pl.get("none", (np.nan, np.nan))
            lines.append(f"| none | {h0[0]:+.3f} ± {h0[1]:.3f} [{h0[2]:+.3f}] | {p0[0]:+.3f} [{p0[1]:+.3f}] | {h0[0]-p0[0]:+.3f} | 100 % |")
        for c in CONS:
            files = glob.glob(os.path.join(args.head_dir, f"emin2_{enc}_t0_akk_erase_{c}-s*-f*.parquet"))
            if not files:
                lines.append(f"| {c} | *pending* | | | |"); continue
            h = head_stats(files, corpus, mc, gkf); pc = pl.get(c, (np.nan, np.nan))
            keep = h[0] / h0[0] if h0 else np.nan
            lines.append(f"| {c} | {h[0]:+.3f} ± {h[1]:.3f} [{h[2]:+.3f}] (n={h[3]}) | {pc[0]:+.3f} [{pc[1]:+.3f}] | {h[0]-pc[0]:+.3f} | {keep:.0%} |")
        lines.append("")
    txt = "\n".join(lines) + "\n"
    open(args.out, "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
