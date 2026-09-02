"""aggregate_emin2.py — E-MIN v2 summary: every run in a scores dir, read out
through aggregate_emin.load_oof + battery, laid side by side per
(encoder, language arm): Chrono-Barlow head (seeds) vs ridge vs PLS, for each
condition INCLUDING the per-language '<cond>@<lang>' rows of the mixed arm.

    python chrono/scripts/aggregate_emin2.py \
        --scores-dir chrono/reports/tier0/scores --art chrono/artifacts_tier0 \
        --out chrono/reports/EMIN2_RESULT.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chrono.eval.robustness import battery                  # noqa: E402
from chrono.eval.protocol import block_placebo_rho          # noqa: E402
from aggregate_emin import load_oof                         # noqa: E402

ENC_OF_MODEL = {"NousResearch/Llama-2-7b-hf": "llama2_7b", "Qwen/Qwen3-8B": "qwen3_8b",
                "Thalesian/cuneiformBase-400m": "cunei400m", "Thalesian/AKK_300m": "akk300m"}


def runs_in(scores_dir):
    """{run_prefix: [files]} where run_prefix drops '-s<seed>-f<fold>'."""
    out = {}
    for f in glob.glob(os.path.join(scores_dir, "*.parquet")):
        m = re.match(r"(.+)-s(\d+)-f(\d+)\.parquet$", os.path.basename(f))
        if m:
            out.setdefault(m.group(1), []).append(f)
    return out


def summarise(files, corpus, splits, with_null=True):
    """mean/sd over seeds of battery rho per (condition, split); and the
    ruler-block null of the mean-over-draws statistic on 'orig'."""
    seeds = sorted({re.search(r"-s(\d+)-f", f).group(1) for f in files})
    rows, null = [], None
    for sd in seeds:
        oof = load_oof([f for f in files if f"-s{sd}-f" in f])
        b = battery(oof[["doc_id", "condition", "s"]], corpus, splits)
        b["seed"] = int(sd); rows.append(b)
        if with_null and null is None:
            o = oof[oof["condition"] == "orig"]
            s = pd.Series(o["s"].to_numpy(), index=pd.Index(o["doc_id"], name="doc_id"))
            null = np.array([np.nanmean(block_placebo_rho(s, corpus, splits["mc_balanced"],
                                                          seed=10_000 + b_))
                             for b_ in range(30)])
    B = pd.concat(rows, ignore_index=True)
    agg = (B.groupby(["condition", "split"], sort=False)["rho_mean"]
             .agg(["mean", "std", "count"]).reset_index())
    return agg, len(seeds), null


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores-dir", default="chrono/reports/tier0/scores")
    ap.add_argument("--art", default="chrono/artifacts_tier0")
    ap.add_argument("--out", default="chrono/reports/EMIN2_RESULT.md")
    args = ap.parse_args(argv)

    corpus = pd.read_parquet(os.path.join(args.art, "corpus_chrono.parquet"))
    splits = {}
    for name in ("gkf_ruler", "mc_balanced"):
        with open(os.path.join(args.art, "splits", f"{name}.json")) as f:
            splits[name] = json.load(f)

    runs = runs_in(args.scores_dir)
    # arm = (encoder, langs) parsed from run names:
    #   head:     emin2_<enc>_t0_<akk|eng|mix>
    #   baseline: baseline_<probe>_L<layer><site>[_<lang>[+<lang>]][_allviews]
    table = {}   # (enc, arm) -> {"head": agg, "ridge": agg, "pls": agg}
    meta = {}
    for run, files in sorted(runs.items()):
        m = re.match(r"emin2_(\w+?)_t0_(akk|eng|mix)$", run)
        if m:
            enc, arm = m.group(1), m.group(2); key = "head"
        else:
            m = re.match(r"baseline_(ridge|pls)_L(\d+)mean(?:_([a-z+]+))?$", run)
            if not m:
                continue
            key, layer, langs = m.group(1), int(m.group(2)), m.group(3)
            arm = {None: "mix", "akk": "akk", "eng": "eng", "akk+eng": "mix"}.get(langs, langs)
            enc = {16: "llama2_7b", 18: "qwen3_8b", 12: "cunei400m", 8: "akk300m"}.get(layer, f"L{layer}")
        agg, n_seeds, null = summarise(files, corpus, splits, with_null=(key == "head"))
        table.setdefault((enc, arm), {})[key] = agg
        meta[(enc, arm, key)] = (n_seeds, null)

    lines = ["# E-MIN v2 result — tier0 Akkadian, three encoders, three language arms", "",
             "Read-out: SLA §7 (gkf pooled over centred OOF scores; mc = mean Spearman over the frozen "
             "draws). Head = Chrono-Barlow adapter (mean ± sd over seeds); ridge / PLS = frozen probes fit "
             "on orig views of the train docs, same folds. Conditions with `@lang` are the per-language "
             "read-out inside the mixed arm.", ""]
    for (enc, arm), d in sorted(table.items()):
        lines += [f"## {enc} — arm `{arm}`", ""]
        conds = list(dict.fromkeys(pd.concat([a for a in d.values()])["condition"]))
        cols = [k for k in ("head", "ridge", "pls") if k in d]
        lines.append("| condition | " + " | ".join(f"{c} mc ρ" for c in cols) + " | " +
                     " | ".join(f"{c} gkf ρ" for c in cols) + " |")
        lines.append("|---|" + "---|" * (2 * len(cols)))
        for cond in conds:
            cells = []
            for split in ("mc_balanced", "gkf_ruler"):
                for c in cols:
                    a = d[c]; r = a[(a["condition"] == cond) & (a["split"] == split)]
                    if len(r) == 0:
                        cells.append("—")
                    elif c == "head" and r["count"].iloc[0] > 1:
                        cells.append(f"{r['mean'].iloc[0]:+.3f} ± {r['std'].iloc[0]:.3f}")
                    else:
                        cells.append(f"{r['mean'].iloc[0]:+.3f}")
            lines.append(f"| `{cond}` | " + " | ".join(cells) + " |")
        n_seeds, null = meta.get((enc, arm, "head"), (0, None))
        if null is not None:
            lines.append(f"\nhead seeds: {n_seeds} · ruler-block null of the mc statistic (orig): "
                         f"{null.mean():+.3f} ± {null.std():.3f}")
        lines.append("")
    txt = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
