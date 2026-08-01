"""Build TIDY_all_year_results.csv — the single source every figure reads.

One row per (level, language, salience, cleaning, pooling, probe, protocol, metric,
target, arm) so nothing about a number is implicit. Everything is read from committed
result JSONs; no cluster access needed.

PROTOCOL NOTE (this is the important knob)
------------------------------------------
For the FRAGMENT cells the probe JSONs carry several protocol blocks, and they are NOT
interchangeable, because in r8 `year` takes only 17 distinct values over 8 rulers — the
label is very nearly the ruler's identity:

  mc        StratifiedKFold-by-ruler  — a ruler appears in train AND test. The probe can
                                        re-identify a ruler's scribal style and read the
                                        year off it. TF-IDF reaches .707 here on
                                        name-stripped text, which is the leak made visible.
  mc_group  GroupKFold-by-ruler       — a ruler is wholly in train or wholly in test. This
                                        mirrors shared/mc_probe.py, the engine behind the
                                        thesis's headline table (p1_year_mc.csv, slide 4:
                                        cuneiform-400M .391, TF-IDF .266) and behind the
                                        deck's stated protocol on slide 2.
  loro      leave-ONE-ruler-out       — pooled single pass, ridge only. Same family as
                                        mc_group but not the deck's protocol.

`--mode mc_group` is the correct setting for anything compared against the thesis deck.
`mc` is retained only for reproducing the earlier (leaky) figures.

    python build_tidy.py --mode mc_group
"""
from __future__ import annotations

import argparse
import csv
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_WM = os.path.dirname(_HERE)

ENC = {"thalesian_akk300m", "thalesian_cunei400m", "umt5_base"}
ARMS = ["llama2_70b", "gpt_oss_120b", "llama2_13b", "qwen3_32b", "llama2_7b", "qwen3_8b",
        "qwen3_1b7", "thalesian_cunei400m", "thalesian_akk300m", "umt5_base",
        "random", "llama2_7b_random", "llama2_13b_random", "llama2_70b_random", "tfidf"]
KEYS = {"mc": ("mc", "spearman_mean", "r2_mean", "ruler_MC_r8_stratified"),
        "mc_group": ("mc_group", "spearman_mean", "r2_mean", "ruler_MC_r8_group"),
        "loro": ("loro", "spearman", "r2", "leave_one_ruler_out")}


def rows_fragment(mode):
    blk, spk, r2k, proto = KEYS[mode]
    out = []
    for m in ARMS:
        for variant, lang, clean in (("akk_maximal", "AKK", "maximal"),
                                     ("eng_tier0", "EN", "tier0")):
            for pool in ("last", "mean", "text"):
                f = os.path.join(_WM, "akkadian", "results", "probes", m,
                                 f"{variant}.r8.year.{pool}.ridge.json")
                if not os.path.exists(f):
                    continue
                d = json.load(open(f)).get(blk)
                if not isinstance(d, dict):
                    continue
                for metric, key in (("spearman", spk), ("r2", r2k)):
                    v = d.get(key)
                    if v is None:
                        continue
                    out.append(dict(level="fragment", language=lang, salience="obscure",
                                    cleaning=clean, pooling=pool, probe="ridge",
                                    protocol=proto, metric=metric, target="year",
                                    arm=m, value=round(float(v), 4)))
                # mc_group also carries the PLS sweep — this is the deck's headline
                # read-out ("activation PLS Spearman", slide 2: k about 3-5), and it
                # separates the encoders from the n-gram floor far more sharply than
                # ridge does, so it must be available to the figures.
                pv = d.get("pls_spearman_mean")
                if pv is not None and pv == pv:
                    out.append(dict(level="fragment", language=lang, salience="obscure",
                                    cleaning=clean, pooling=pool, probe="pls",
                                    protocol=proto, metric="spearman", target="year",
                                    arm=m, value=round(float(pv), 4)))
    return out


def rows_entity_b():
    out = []
    for m in ARMS:
        for pool in ("ent_last", "ent_mean", "last", "mean", "text"):
            f = os.path.join(_WM, "akkadian", "results", "probes_entity", m,
                             f"assyrian_ruler.{pool}.json")
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            bl = str(d.get("best_layer"))
            blk = d["layers"].get(bl) or list(d["layers"].values())[0]
            for tag in ("bare", "all"):
                e = blk.get(tag)
                if not e:
                    continue
                for probe, pk in (("ridge", "ridge_mc"), ("pls5", "pls5_mc")):
                    s = e.get(pk)
                    if not s:
                        continue
                    for metric, key in (("spearman", "mc_rho"), ("r2", "mc_r2")):
                        v = s.get(key)
                        if v is None:
                            continue
                        out.append(dict(level="entity", language="EN", salience="obscure",
                                        cleaning=f"rows_{tag}", pooling=pool, probe=probe,
                                        protocol="entity_MC_heldout_entities",
                                        metric=metric, target="year", arm=m,
                                        value=round(float(v), 4)))
    return out


def rows_entity_a():
    out = []
    for m in ARMS:
        for et in ("historical_figure", "art", "headline"):
            for pool in ("last", "mean", "text"):
                f = os.path.join(_WM, "results", "probes", m, f"{et}.{pool}.ridge.json")
                if not os.path.exists(f):
                    continue
                d = json.load(open(f))
                for metric, key in (("spearman", "best_test_spearman"),
                                    ("r2", "best_test_r2")):
                    v = d.get(key)
                    if v is None:
                        continue
                    out.append(dict(level="entity", language="EN", salience="salient",
                                    cleaning=et, pooling=pool, probe="ridge",
                                    protocol="holdout_entities", metric=metric,
                                    target="year", arm=m, value=round(float(v), 4)))
    return out


# --------------------------------------------------------------------- read-out
# The deck does not use one probe everywhere, and neither should the figures:
#   cell A (salient entities)  ridge      — Gurnee & Tegmark's own probe
#   cell B (obscure entities)  pls5       — the k=5 PLS column of the entity MC
#   fragments                  pls        — p1_year_mc.csv's `pls_spearman_mean`,
#                                           which is where slide 4's .391 comes from
# `--readout deck` emits exactly that selection and relabels the winner `ridge` so the
# figure scripts need no per-cell probe logic. `--readout raw` keeps every probe row.
DECK_PROBE = {("fragment", "obscure"): "pls",
              ("entity", "obscure"): "pls5",
              ("entity", "salient"): "ridge"}


def apply_readout(rows):
    """Keep one probe per cell. PLS reports no R², so for the `r2` metric fall back to
    the ridge row of the same cell rather than dropping the cell from R² figures."""
    out, have = [], set()
    for r in rows:
        want = DECK_PROBE.get((r["level"], r["salience"]))
        if want is None or r["probe"] != want:
            continue
        out.append({**r, "probe": "ridge", "readout": want})
        have.add((r["level"], r["salience"], r["cleaning"], r["pooling"], r["arm"],
                  r["metric"]))
    for r in rows:
        key = (r["level"], r["salience"], r["cleaning"], r["pooling"], r["arm"],
               r["metric"])
        if r["metric"] == "r2" and r["probe"] == "ridge" and key not in have:
            out.append({**r, "readout": "ridge (PLS has no R²)"})
            have.add(key)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mc_group", choices=list(KEYS),
                    help="fragment-cell protocol block; mc_group matches the thesis deck")
    ap.add_argument("--readout", default="raw", choices=("raw", "deck"),
                    help="'deck' selects the per-cell probe the thesis actually reports")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = rows_fragment(args.mode) + rows_entity_b() + rows_entity_a()
    if args.readout == "deck":
        rows = apply_readout(rows)
    if not rows:
        print("no rows — has the mc_group job run yet?")
        return
    suffix = args.mode + ("__deck" if args.readout == "deck" else "")
    out = args.out or os.path.join(_HERE, f"TIDY_all_year_results__{suffix}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    nfrag = sum(1 for r in rows if r["level"] == "fragment")
    print(f"wrote {out}\n  {len(rows)} rows ({nfrag} fragment rows under '{args.mode}')")


if __name__ == "__main__":
    main()
