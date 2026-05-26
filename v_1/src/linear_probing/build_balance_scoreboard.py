#!/usr/bin/env python3
"""Consolidate balanced-vs-imbalanced results for EVERY model into one scoreboard.

Produces, per readout, a table with each model's best-config score on the
full imbalanced set vs the class-balanced Monte-Carlo (200 draws x 168 frags).

Readouts:
  - year_pls    : PLS year regression (Spearman, year-raw)
  - year_ridge  : Ridge year regression (cls_numeric probe, Spearman, year-raw)
  - ruler_cls   : ruler classification (Macro-F1)

Sources:
  full set   : orcc__probe_pls/pls_results_<m>.json            (PLS, metrics_per_k)
               orcc__probe_cls_numeric/cls_numeric_results_<m>.json (Ridge)
               orcc_round2_phase0/aggregated/phase0_summary.json (R1 ruler Macro-F1)
  balanced MC: orcc_round2_phase0/probes/<m>_<probe>__mc_balanced__summary.json

Writes balanced_mc_scoreboard.{json,csv} and prints markdown tables.
"""
import json
import csv
import math
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
PLS_DIR = RES / "orcc__probe_pls"
RIDGE_DIR = RES / "orcc__probe_cls_numeric"
MC_DIR = RES / "orcc_round2_phase0/probes"

MODELS = ["thalesian_cunei400m", "thalesian_akk300m", "mlm", "tfidf",
          "qwen", "qwen3_1b7", "qwen3_8b", "qwen3_32b", "random"]


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


# ---- full-set PLS: best year-raw Spearman over (cleaning,pool,layer,k) -------
def fullset_pls(model):
    p = PLS_DIR / f"pls_results_{model}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    best = None
    for key, rec in d.items():
        if not key.endswith("__year-raw"):
            continue
        for k, m in rec.get("metrics_per_k", {}).items():
            sp = m.get("spearman_mean")
            if _finite(sp) and (best is None or sp > best[0]):
                best = (sp, m.get("mae_mean"), rec.get("layer"), rec.get("cleaning"),
                        rec.get("pooling"), f"k={k}")
    return best  # (sp, mae, layer, cleaning, pool, k)


# ---- full-set Ridge: best year-raw Spearman over (cleaning,pool,layer) -------
def fullset_ridge(model):
    p = RIDGE_DIR / f"cls_numeric_results_{model}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    best = None
    for key, rec in d.items():
        if not key.endswith("__year-raw"):
            continue
        sp = rec.get("spearman_mean")
        if _finite(sp) and (best is None or sp > best[0]):
            best = (sp, rec.get("mae_mean"), rec.get("layer"),
                    rec.get("cleaning"), rec.get("pooling"), "ridge")
    return best


# ---- balanced MC: best over configs from a probe summary --------------------
def mc_best(model, probe, metric):
    """metric in {'year','ruler'}. Returns (mean,std,layer,cleaning,pool)."""
    p = MC_DIR / f"{model}_{probe}__mc_balanced__summary.json"
    if not p.exists():
        return None
    d = json.load(open(p)).get("per_config", {})
    best = None
    for key, rec in d.items():
        if metric == "year" and key.endswith("__year-raw"):
            v, s = rec.get("spearman_mean"), rec.get("spearman_std")
        elif metric == "ruler" and key.endswith("__ruler"):
            v, s = rec.get("macro_f1_mean"), rec.get("macro_f1_std")
        else:
            continue
        if _finite(v) and (best is None or v > best[0]):
            parts = key.split("__")  # model, cleaning, pool, Lnn, target
            best = (v, s, parts[3], parts[1], parts[2])
    return best


def r1_ruler(model):
    """Imbalanced R1 ruler Macro-F1 from phase0_summary if present."""
    p = RES / "orcc_round2_phase0/aggregated/phase0_summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    # search any nested record mentioning this model with an r1/macro_f1 field
    hits = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("method") == model or o.get("model") == model:
                hits.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    for h in hits:
        for fld in ("r1_macro_f1", "macro_f1_r1", "r1_macrof1"):
            if _finite(h.get(fld)):
                return h[fld]
    return None


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if _finite(x) else "—"


def main():
    out = {}
    rows_year, rows_ridge, rows_ruler = [], [], []
    for m in MODELS:
        fp, fr = fullset_pls(m), fullset_ridge(m)
        bp = mc_best(m, "pls", "year")
        br = mc_best(m, "cls_numeric", "year")
        # balanced ruler: prefer cls probe, fall back to pls's ruler config
        bru = mc_best(m, "cls", "ruler") or mc_best(m, "pls", "ruler")
        out[m] = dict(fullset_pls=fp, fullset_ridge=fr, mc_pls=bp,
                      mc_ridge=br, mc_ruler=bru)
        rows_year.append((m, fp, bp))
        rows_ridge.append((m, fr, br))
        rows_ruler.append((m, bru))

    # save machine-readable
    (RES / "balanced_mc_scoreboard.json").write_text(json.dumps(out, indent=2))
    with open(RES / "balanced_mc_scoreboard.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "readout", "fullset_sp_or_f1", "fullset_detail",
                    "balanced_mean", "balanced_std", "balanced_detail"])
        for m, fp, bp in rows_year:
            w.writerow([m, "year_pls", fp[0] if fp else "", fp[2:] if fp else "",
                        bp[0] if bp else "", bp[1] if bp else "", bp[2:] if bp else ""])
        for m, fr, br in rows_ridge:
            w.writerow([m, "year_ridge", fr[0] if fr else "", fr[2:] if fr else "",
                        br[0] if br else "", br[1] if br else "", br[2:] if br else ""])

    def ptable(title, rows, kind):
        print(f"\n### {title}")
        if kind == "year":
            print("| Model | Full-set Sp | best layer/cfg | Balanced Sp ± std | Δ (bal−full) |")
            print("|---|---|---|---|---|")
            srt = sorted(rows, key=lambda r: -(r[2][0] if r[2] else -9))
            for m, fs, bal in srt:
                delta = (bal[0]-fs[0]) if (fs and bal) else None
                cfg = f"L{fs[2]} {fs[3]}/{fs[4]} {fs[5]}" if fs else "—"
                print(f"| {m} | {fmt(fs[0]) if fs else '—'} | {cfg} | "
                      f"{fmt(bal[0]) if bal else '—'} ± {fmt(bal[1]) if bal else '—'} | "
                      f"{('%+.3f'%delta) if delta is not None else '—'} |")
        else:
            print("| Model | Balanced Macro-F1 ± std | best layer/cfg |")
            print("|---|---|---|")
            srt = sorted(rows, key=lambda r: -(r[1][0] if r[1] else -9))
            for m, bal in srt:
                cfg = f"{bal[2]} {bal[3]}/{bal[4]}" if bal else "—"
                print(f"| {m} | {fmt(bal[0]) if bal else '—'} ± "
                      f"{fmt(bal[1]) if bal else '—'} | {cfg} |")

    ptable("YEAR — PLS regression (Spearman, year-raw)", rows_year, "year")
    ptable("YEAR — Ridge regression (Spearman, year-raw)", rows_ridge, "year")
    ptable("RULER — classification (Macro-F1)", [(m, b) for m, b in rows_ruler], "ruler")
    print(f"\nSaved → {RES/'balanced_mc_scoreboard.json'} and .csv")


if __name__ == "__main__":
    main()
