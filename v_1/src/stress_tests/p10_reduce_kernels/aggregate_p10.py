"""P10 aggregation — does reducing first help the kernel/dial read chronology?

For each method×cleaning, tabulate the gkpls Spearman for every reducer×norm and flag
whether the best reduce+norm beats the `raw` anchor (= the P9/P8 result). Also the best
dial `pred`. Writes results/RESULTS_p10.md + summary_p10.csv.

    python aggregate_p10.py
"""
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def main():
    rows = []
    for fp in sorted(glob.glob(os.path.join(RES, "p10__*.json"))):
        d = json.load(open(fp))
        method = d["method"]
        for cl, blk in d.get("cleanings", {}).items():
            cfgs = blk.get("configs", {})
            for cfg, r in cfgs.items():
                if r.get("skipped"):
                    continue
                red, norm = cfg.split("/")
                dial = r.get("dial", {})
                best_pred = max((v["pred_mean"] for v in dial.values()
                                 if v["pred_mean"] == v["pred_mean"]), default=float("nan"))
                rows.append({
                    "method": method, "cleaning": cl, "reducer": red, "norm": norm,
                    "gkpls": r.get("gkpls", {}).get("spearman_mean", float("nan")),
                    "rbfkpls": r.get("rbfkpls", {}).get("spearman_mean", float("nan")),
                    "krr_geo": r.get("krr_geo", {}).get("spearman_mean", float("nan")),
                    "dial_pred": best_pred,
                })
    if not rows:
        print("no P10 results yet")
        return
    df = pd.DataFrame(rows)
    df.round(3).to_csv(os.path.join(RES, "summary_p10.csv"), index=False)

    lines = ["# P10 — reduce-then-kernel: does pre-reducing help?", "",
             "gkpls Spearman(year) under balanced-MC, per reducer×norm. **raw** = the "
             "P9/P8 anchor (no reduction). Δ = best reduce+norm − raw.", ""]
    for (method, cl), g in df.groupby(["method", "cleaning"]):
        raw = g[(g.reducer == "raw") & (g.norm == "none")]["gkpls"]
        raw = float(raw.iloc[0]) if len(raw) else float("nan")
        best = g.loc[g.gkpls.idxmax()] if g.gkpls.notna().any() else None
        lines.append(f"## {method} · {cl}  (raw={raw:.3f})")
        if best is not None:
            delta = best.gkpls - raw
            lines.append(f"best: **{best.reducer}/{best.norm}** gkpls={best.gkpls:.3f} "
                         f"(Δ={delta:+.3f}), dial_pred={best.dial_pred:.3f}")
        piv = g.pivot_table(index="reducer", columns="norm", values="gkpls")
        piv = piv.reindex(index=[r for r in ["raw", "pca", "pls", "umap"] if r in piv.index],
                          columns=[c for c in ["none", "zscore", "l2"] if c in piv.columns])
        lines += ["", piv.round(3).to_markdown(), ""]
    with open(os.path.join(RES, "RESULTS_p10.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print(f"\nwrote {RES}/RESULTS_p10.md")


if __name__ == "__main__":
    main()
