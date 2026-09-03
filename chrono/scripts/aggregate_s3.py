"""S3 fine-tune sweep read-out: does SSL pretraining help the dated task?

One row per (encoder, init, label fraction): the SLA §7 pooled gkf rho and the
mc_balanced mean rho over the centred out-of-fold scores, mean +- sd over
seeds. `init = none` is the control -- the same head trained from a random
init on the same folds -- so the SSL claim is the barlow/jepa row MINUS the
none row, not the absolute number.
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from chrono.eval.robustness import battery                     # noqa: E402
from chrono.scripts.aggregate_emin import load_oof             # noqa: E402

RUN_RE = re.compile(r"^s3_(?P<enc>.+?)_init-(?P<init>none|barlow|byol|jepa|infonce)"
                    r"_frac(?P<frac>\d+)(?:_h(?P<hid>\d+))?-s(?P<seed>\d+)-f(?P<fold>\d+)$")


def cells(scores_dir):
    """{(enc, init, frac): {seed: [fold files]}}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(scores_dir, "s3_*.parquet"))):
        m = RUN_RE.match(os.path.basename(f)[:-len(".parquet")])
        if m:
            key = (m["enc"], m["init"], int(m["frac"]), int(m["hid"] or 512))
            out.setdefault(key, {}).setdefault(int(m["seed"]), []).append(f)
    return out


def rho_of(files, corpus, splits):
    """(mc mean rho, gkf pooled rho) for one seed's folds, orig condition."""
    b = battery(load_oof(files)[["doc_id", "condition", "s"]], corpus, splits)
    b = b[b["condition"] == "orig"]
    g = lambda s: (b.loc[b["split"] == s, "rho_mean"].iloc[0] if (b["split"] == s).any() else np.nan)
    return g("mc_balanced"), g("gkf_ruler")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores-dir", default="chrono/reports/tier0/s3_scores")
    ap.add_argument("--art", default="chrono/artifacts_tier0")
    ap.add_argument("--out", default="chrono/reports/S3_RESULT.md")
    args = ap.parse_args(argv)

    corpus = pd.read_parquet(os.path.join(args.art, "corpus_chrono.parquet"))
    splits = {n: json.load(open(os.path.join(args.art, "splits", f"{n}.json")))
              for n in ("gkf_ruler", "mc_balanced")}

    rows = []
    for (enc, init, frac, hid), by_seed in sorted(cells(args.scores_dir).items()):
        for seed, files in sorted(by_seed.items()):
            try:
                mc, gkf = rho_of(files, corpus, splits)
            except Exception as e:                       # a half-written cell must not sink the table
                print(f"[skip] {enc} {init} frac{frac} s{seed}: {type(e).__name__}: {e}")
                continue
            rows.append(dict(enc=enc, init=init, frac=frac, hid=hid, seed=seed,
                             folds=len(files), mc=mc, gkf=gkf))
    if not rows:
        raise SystemExit(f"no S3 score files in {args.scores_dir}")
    R = pd.DataFrame(rows)
    agg = (R.groupby(["enc", "init", "frac", "hid"])
             .agg(seeds=("seed", "nunique"), folds=("folds", "max"),
                  mc_mean=("mc", "mean"), mc_sd=("mc", "std"),
                  gkf_mean=("gkf", "mean"), gkf_sd=("gkf", "std"))
             .reset_index())

    lines = ["# S3 — does SSL pretraining help dating the 1,193 dated inscriptions?", "",
             "Read-out: SLA §7 on centred out-of-fold scores, `orig` condition. `gkf` is the POOLED "
             "Spearman over the held-out docs of the ruler-grouped folds (per-fold rho is undefined "
             "there: 39 of 40 rulers carry a single year); `mc` is the mean over the frozen balanced "
             "draws. Mean +- sd over seeds.", "",
             "**The SSL claim is a difference.** Compare each `barlow`/`jepa` row with the `none` row "
             "of the same encoder and label fraction; the absolute rho is dominated by the frozen "
             "encoder underneath.", "",
             "| encoder | init | label frac | head width | seeds | folds | mc rho | gkf rho (pooled) |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        sd = lambda v: "" if pd.isna(v) else f" ± {v:.3f}"
        lines.append(f"| `{r.enc}` | {r.init} | {r.frac}% | {r.hid} | {r.seeds} | {r.folds} | "
                     f"{r.mc_mean:.3f}{sd(r.mc_sd)} | {r.gkf_mean:.3f}{sd(r.gkf_sd)} |")

    base = agg[agg.init == "none"].set_index(["enc", "frac", "hid"])
    lines += ["", "## SSL init minus the `none` control (same encoder, same label fraction)", "",
              "| encoder | init | label frac | Δ mc rho | Δ gkf rho |", "|---|---|---|---|---|"]
    for _, r in agg[agg.init != "none"].iterrows():
        if (r.enc, r.frac, r.hid) not in base.index:
            continue
        b = base.loc[(r.enc, r.frac, r.hid)]
        lines.append(f"| `{r.enc}` | {r.init} | {r.frac}% | {r.mc_mean - b.mc_mean:+.3f} | "
                     f"{r.gkf_mean - b.gkf_mean:+.3f} |")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(agg)} cells, {R.seed.nunique()} seeds)")


if __name__ == "__main__":
    main()
