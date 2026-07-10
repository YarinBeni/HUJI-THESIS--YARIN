"""export_result_csvs.py — flatten every stress-test result JSON into one CSV per
experiment under results/csv/. Pure stdlib (json/csv); safe to run on a login node.

Outputs:
  results/csv/t9_knowledge.csv     results/csv/p1_year_gkf.csv
  results/csv/p2_geography.csv     results/csv/p1_year_mc.csv
  results/csv/p3_timeline.csv      results/csv/p1_maxking.csv
  results/csv/p7_ksparse.csv       results/csv/t10_gkf.csv  results/csv/t10_mc.csv
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ST = Path(__file__).resolve().parents[1]
OUT = ST / "results" / "csv"
ORDER = ["qwen3_1b7", "qwen3_8b", "qwen3_32b", "gpt_oss_120b",
         "thalesian_akk300m", "thalesian_cunei400m", "umt5_base", "mlm", "random"]


def jload(p): return json.loads(Path(p).read_text())
def okf(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else x
def mrank(m): return ORDER.index(m) if m in ORDER else 99
def write(name, header, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"wrote {OUT/name}  ({len(rows)} rows)")


def t9():
    rows = []
    for d in sorted([p.name.replace("direct_kp_", "") for p in ST.glob("redo_t9_knowledge/direct_kp_*")], key=mrank):
        b = ST / f"redo_t9_knowledge/direct_kp_{d}/scores"
        kp0 = jload(b/"kp0_metrics.json") if (b/"kp0_metrics.json").exists() else {}
        kp1 = jload(b/"kp1_metrics.json") if (b/"kp1_metrics.json").exists() else {}
        kp2 = jload(b/"kp2_metrics.json") if (b/"kp2_metrics.json").exists() else {}
        rows.append([d, okf(kp0.get("correct")), okf(kp0.get("total")), okf(kp0.get("accuracy")),
                     okf(kp0.get("parse_errors")), okf(kp1.get("aggregate_recall")),
                     okf(kp1.get("total_hits")), okf(kp1.get("total_targets")),
                     okf(kp2.get("hallucination_rate")), okf(kp2.get("gate_pass"))])
    write("t9_knowledge.csv",
          ["model", "kp0_correct", "kp0_total", "kp0_accuracy", "kp0_parse_errors",
           "kp1_recall", "kp1_hits", "kp1_targets", "kp2_hallucination_rate", "kp2_gate_pass"], rows)


def p2():
    rows = []
    for fp in sorted(ST.glob("p2_godey_geography/results/p2_geography__*.json"), key=lambda p: mrank(p.stem.split("__")[1])):
        d = jload(fp); m = d["method"]
        for cl, blk in d.get("cleanings", {}).items():
            if blk.get("missing"):
                continue
            bl = str(blk["best_layer_by_skill"]); pl = blk["per_layer"][bl]; g = pl["geo"]
            rows.append([m, cl, bl, okf(g.get("gc_km_mean")), okf(g.get("skill_vs_centroid")), okf(g.get("k")),
                         okf(pl.get("lat_spearman")), okf(pl.get("lat_ridge_spearman")),
                         okf(pl.get("lon_spearman")), okf(pl.get("lon_ridge_spearman"))])
    write("p2_geography.csv",
          ["model", "cleaning", "best_layer", "gc_km", "skill_vs_centroid", "pls_k",
           "lat_pls_spearman", "lat_ridge_spearman", "lon_pls_spearman", "lon_ridge_spearman"], rows)


def p1_gkf():
    rows = []
    for fp in sorted(ST.glob("p1_gurnee_tegmark/results/p1_year__*.json"), key=lambda p: mrank(p.stem.split("__")[1])):
        d = jload(fp); m = d["method"]
        for site in ["mean_tier0", "mean_maximal", "king_last", "king_mean"]:
            s = d.get(site, {})
            if s.get("missing") or s.get("insufficient_coverage") or "best_spearman" not in s:
                rows.append([m, site, "", "", "", ""]); continue
            pl = s.get("per_layer", {}).get(str(s["best_layer"]), {}).get("pls", {})
            rows.append([m, site, s.get("best_layer"), okf(s.get("best_spearman")),
                         okf(pl.get("spearman_std")), okf(s.get("mlp_spearman_at_best"))])
    write("p1_year_gkf.csv",
          ["model", "site", "best_layer", "spearman", "spearman_std_folds", "mlp_spearman"], rows)


def p1_mc():
    rows = []
    for fp in sorted(ST.glob("p1_gurnee_tegmark/results/mc/p1_year_mc__*.json"), key=lambda p: mrank(p.stem.split("__")[1])):
        d = jload(fp); m = d["method"]
        for site, blk in d.get("sites", {}).items():
            if blk.get("missing") or blk.get("insufficient"):
                rows.append([m, site, "", "", "", "", "", ""]); continue
            b = blk["best"]
            rows.append([m, site, blk.get("best_layer"), okf(b.get("best_k")), okf(b.get("spearman_mean")),
                         okf(b.get("spearman_std")), okf(b.get("ridge", {}).get("spearman_mean")),
                         okf(b.get("shuffled_spearman_mean"))])
    write("p1_year_mc.csv",
          ["model", "site", "best_layer", "pls_best_k", "pls_spearman_mean", "pls_spearman_std",
           "ridge_spearman", "shuffled_null"], rows)


def p1_maxking():
    rows = []
    for fp in sorted(ST.glob("p1_gurnee_tegmark/results/maxking/p1_maxking__*.json"), key=lambda p: mrank(p.stem.split("__")[1])):
        d = jload(fp); m = d["method"]
        for site, blk in d.get("sites", {}).items():
            if blk.get("missing") or blk.get("insufficient"):
                rows.append([m, site, "", "", "", "", "", "", ""]); continue
            b = blk["best"]; rc = b["ruler_clf"]; ys = b["year_strat"]
            acc10 = ys["per_k"].get(str(ys["best_k"]), {}).get("acc10_mean")
            rows.append([m, site, blk.get("best_layer"), okf(rc.get("macro_f1_mean")), okf(rc.get("macro_f1_std")),
                         okf(rc.get("chance_macro_f1")), okf(rc.get("shuffled_macro_f1")),
                         okf(ys.get("spearman_mean")), okf(acc10), okf(b.get("year_group", {}).get("spearman_mean"))])
    write("p1_maxking.csv",
          ["model", "site", "best_layer", "ruler_macro_f1", "ruler_macro_f1_std", "chance_macro_f1",
           "shuffled_macro_f1", "year_strat_spearman", "year_pm10_acc", "year_group_spearman"], rows)


def p3():
    rows = []
    for fp in sorted(ST.glob("p3_matter_of_time/results/p3_timeline__*.json"), key=lambda p: mrank(p.stem.split("__")[1])):
        d = jload(fp); m = d["method"]; best = None
        for L, v in d["per_layer"].items():
            a = max(v.get("3a_pca1_spearman", float("nan")), v.get("3a_isomap1_spearman", float("nan")))
            if a == a and (best is None or a > best[0]):
                best = (a, L, v.get("3a_pca1_spearman"), v.get("3a_isomap1_spearman"), v.get("3b_project_spearman"))
        if best:
            rows.append([m, best[1], okf(best[2]), okf(best[3]), okf(best[4])])
    write("p3_timeline.csv",
          ["model", "best_layer", "3a_pca1_spearman", "3a_isomap1_spearman", "3b_project_spearman"], rows)


def p7():
    rows = []
    for fp in sorted(ST.glob("p7_ksparse/results/p7_ksparse__*.json"), key=lambda p: mrank(p.stem.split("__")[1])):
        d = jload(fp); m = d["method"]
        bf1, bloc = 0.0, None
        for L, ks in d["per_layer"].items():
            for k, mm in ks.items():
                if mm["macro_f1"] > bf1:
                    bf1, bloc = mm["macro_f1"], (L, k)
        rows.append([m, okf(d.get("chance_acc")), okf(bf1), bloc[0] if bloc else "", bloc[1] if bloc else "",
                     okf(d.get("localization", {}).get("full_k_macro_f1")),
                     okf(d.get("localization", {}).get("k_reaching_90pct"))])
    write("p7_ksparse.csv",
          ["model", "chance_acc", "best_macro_f1", "best_layer", "best_k_neurons",
           "full_k_macro_f1", "k_reaching_90pct"], rows)


def t10():
    def best_over_layers(V, pool, mode):
        out = {}
        for pv, layers in V.items():
            b = None
            for L, pools in layers.items():
                pd_ = pools.get(pool, {})
                sp = pd_.get("spearman_mean") if mode == "mc" else pd_.get("pls", {}).get("spearman_mean")
                if sp is not None and sp == sp and (b is None or sp > b):
                    b = sp
            out[pv] = b
        return out
    for mode, glb, name in [("gkf", "*__t10_king_summary.json", "t10_gkf.csv"),
                            ("mc", "*__t10_mc_summary.json", "t10_mc.csv")]:
        rows = []
        for fp in sorted(ST.glob(f"redo_t10_prompt/results/{glb}"), key=lambda p: mrank(p.stem.split("__")[0])):
            d = jload(fp); m = d["model"]; V = d["variants"]
            for pool in ["mean", "king_last", "king_mean"]:
                bo = best_over_layers(V, pool, mode)
                rows.append([m, pool, okf(bo.get("pv0")), okf(bo.get("pv1")), okf(bo.get("pv2")), okf(bo.get("pv3"))])
        write(name, ["model", "pool", "pv0_bare", "pv1_framed", "pv2_fewshot", "pv3_cot"], rows)




# ---- gap-fix wave families (2026-07-05) ----

def p2_geo_mc():
    rows = []
    for fp in sorted(ST.glob("p2_godey_geography/results/mc/p2_geo_mc__*.json"),
                     key=lambda p: mrank(p.stem.split("__")[1])):
        d = jload(fp); m = d["method"]
        for cl, blk in d.get("cleanings", {}).items():
            if blk.get("missing") or blk.get("insufficient"):
                continue
            b = blk["best"]; r = b.get("ridge", {})
            rows.append([m, cl, blk.get("best_layer"), okf(b.get("best_k")),
                         okf(b.get("lat_spearman")), okf(r.get("lat_spearman")),
                         okf(b.get("lon_spearman")), okf(r.get("lon_spearman")),
                         okf(b.get("gc_km_mean")), okf(b.get("skill_mean")), okf(b.get("skill_std"))])
    write("p2_geo_mc.csv", ["model", "cleaning", "best_layer", "pls_best_k",
                            "lat_pls_spearman", "lat_ridge_spearman", "lon_pls_spearman",
                            "lon_ridge_spearman", "gc_km", "skill_mean", "skill_std"], rows)


def p3_pls():
    rows = []
    for fp in sorted(ST.glob("p3_matter_of_time/results/pls/p3pls__*.json"),
                     key=lambda p: mrank(p.stem.split("__")[1])):
        d = jload(fp); m = d["method"]
        for cl, blk in d.get("cleanings", {}).items():
            if blk.get("missing") or "best" not in blk:
                continue
            b = blk["best"]
            rows.append([m, cl, blk.get("best_layer"), okf(b.get("pls1_spearman")),
                         okf(b.get("geo1_spearman")), okf(b.get("proj_interp_spearman")),
                         okf(b.get("proj_nn1_spearman")), okf(b.get("ruler_macro_f1"))])
    write("p3_pls_timeline.csv", ["model", "cleaning", "best_layer", "anchors_pls1_spearman",
                                  "anchors_geo1_spearman", "texts_interp_spearman",
                                  "texts_nn1_spearman", "texts_ruler_macro_f1"], rows)


def p7_v2():
    rows = []
    for fp in sorted(ST.glob("p7_ksparse/results/v2/p7_v2__*.json")):
        d = jload(fp)
        if "per_layer" not in d:
            continue
        kmax = str(max(d["ks"]))
        bc, br = d["best_layer_cls"], d["best_layer_reg"]
        rows.append([d["method"], d["cleaning"],
                     okf(d["per_layer"][bc][kmax]["macro_f1"]), bc,
                     okf(d["per_layer"][br][kmax]["reg_spearman"]), br])
    rows.sort(key=lambda r: (mrank(r[0]), r[1]))
    write("p7_v2.csv", ["model", "cleaning", "cls_macro_f1_fullk", "cls_best_layer",
                        "reg_spearman_fullk", "reg_best_layer"], rows)


def t10_cleanings():
    rows = []
    for fp in sorted(ST.glob("redo_t10_prompt/results/*t10_mc_summary*.json")):
        d = jload(fp); m = d["model"]; cl = d.get("cleaning", "tier0")
        for pool in ["mean", "king_last"]:
            vals = []
            for pv in ["pv0", "pv1", "pv2", "pv3"]:
                layers = d["variants"].get(pv, {})
                b = max(((p[pool].get("spearman_mean") if not p[pool].get("skipped") else None) or -9)
                        for p in layers.values()) if layers else None
                vals.append(okf(b if (b is not None and b > -9) else None))
            rows.append([m, cl, pool] + vals)
    rows.sort(key=lambda r: (mrank(r[0]), r[1], r[2]))
    write("t10_mc_cleanings.csv", ["model", "cleaning", "pool", "pv0", "pv1", "pv2", "pv3"], rows)


def translation():
    rows = []
    for fp in sorted(ST.glob("translation/results/trans_mc__*.json"),
                     key=lambda p: mrank(p.stem.split("__")[1])):
        d = jload(fp); m = d["method"]
        for cl, blk in d.get("cleanings", {}).items():
            if blk.get("missing"):
                continue
            y = blk.get("year_best", {}); yr = y.get("ridge", {})
            g = blk.get("geo_best", {}); gr = g.get("ridge", {})
            rows.append([m, cl, okf(y.get("spearman_mean")), okf(y.get("spearman_std")),
                         okf(yr.get("spearman_mean")), okf(g.get("lat_spearman")),
                         okf(gr.get("lat_spearman")), okf(g.get("lon_spearman")),
                         okf(gr.get("lon_spearman"))])
    write("translation_mc.csv", ["model", "cleaning", "year_pls_spearman", "year_pls_std",
                                 "year_ridge_spearman", "lat_pls_spearman", "lat_ridge_spearman",
                                 "lon_pls_spearman", "lon_ridge_spearman"], rows)


def t11():
    CLEANINGS = ["tier0", "maximal", "maxking", "engtier0"]
    rows = []
    for fp in sorted(ST.glob("t11_gen_dating/results/t11_gen__*.json"),
                     key=lambda p: (mrank(p.stem.split("__")[1]),
                                    CLEANINGS.index(p.stem.split("__")[2])
                                    if p.stem.split("__")[2] in CLEANINGS else 99)):
        d = jload(fp); f = d["full_corpus"]; mc = d["mc_balanced"]
        sc = f.get("parse_status_counts", {})
        n_gen = f.get("n_generated") or 0
        declined = sum(sc.get(k, 0) for k in ("declined", "ce_only", "unparsed"))
        rows.append([d["model"], d["cleaning"],
                     okf(mc.get("spearman_mean")), okf(mc.get("spearman_std")),
                     okf(mc.get("n_draws_used")), okf(mc.get("mean_frags_per_draw")),
                     okf(f.get("spearman_all")), okf(f.get("n_scoreable")), n_gen,
                     okf(f["n_scoreable"] / n_gen if n_gen else None),
                     okf(declined / n_gen if n_gen else None),
                     okf(f.get("mae")), okf(f.get("acc@25yr")), okf(f.get("acc@50yr")),
                     okf(f.get("named_true_ruler_rate")),
                     okf(f.get("spearman_when_named")), okf(f.get("n_named")),
                     okf(f.get("spearman_when_unnamed")), okf(f.get("n_unnamed"))])
    write("t11_gen_dating.csv",
          ["model", "cleaning", "mc_spearman_mean", "mc_spearman_std", "mc_n_draws",
           "mc_frags_per_draw", "spearman_all", "n_scoreable", "n_generated",
           "scoreable_rate", "decline_rate", "mae", "acc@25yr", "acc@50yr",
           "named_true_ruler_rate", "spearman_when_named", "n_named",
           "spearman_when_unnamed", "n_unnamed"], rows)


def e5():
    rows = []
    for fp in sorted(ST.glob("e5_shuffle/results/e5_mc__*.json"),
                     key=lambda p: mrank(p.stem.split("__")[1])):
        d = jload(fp); m = d["method"]
        for cl, blk in d.get("cleanings", {}).items():
            u, s = blk.get("unshuf", {}), blk.get("shuf", {})
            ub, sb = u.get("best"), s.get("best")
            if not (ub and sb):
                rows.append([m, cl] + [""] * 10); continue
            rows.append([m, cl,
                         okf(ub.get("spearman_mean")), okf(ub.get("spearman_std")),
                         okf(ub.get("ridge", {}).get("spearman_mean")), okf(u.get("best_layer")),
                         okf(sb.get("spearman_mean")), okf(sb.get("spearman_std")),
                         okf(sb.get("ridge", {}).get("spearman_mean")), okf(s.get("best_layer")),
                         okf(blk.get("delta_spearman")), okf(blk.get("delta_ridge"))])
    rows.sort(key=lambda r: (mrank(r[0]), r[1]))
    write("e5_shuffle.csv",
          ["model", "cleaning", "unshuf_pls_spearman", "unshuf_pls_std", "unshuf_ridge_spearman",
           "unshuf_best_layer", "shuf_pls_spearman", "shuf_pls_std", "shuf_ridge_spearman",
           "shuf_best_layer", "delta_pls_spearman", "delta_ridge_spearman"], rows)


def p8():
    ORDER8 = ["tfidf"] + ORDER
    rows = []
    for fp in sorted(ST.glob("p8_lambda_probe/results/p8_lambda__*.json"),
                     key=lambda p: (ORDER8.index(p.stem.split("__")[1])
                                    if p.stem.split("__")[1] in ORDER8 else 99)):
        d = jload(fp); m = d["method"]
        for cl, blk in d.get("cleanings", {}).items():
            if not blk or blk.get("missing") or blk.get("skipped"):
                continue
            if "curves" in blk:                    # run_acts schema: best layers
                variants = blk["curves"]
            else:                                  # tfidf schema: k blocks
                variants = {kk: blk[kk]["per_lambda"] for kk in blk
                            if kk.startswith("k") and blk[kk].get("per_lambda")}
            for var, pl in variants.items():
                for lam in sorted(pl, key=float):
                    v = pl[lam]
                    rows.append([m, cl, var, lam,
                                 okf(v.get("align1_mean")), okf(v.get("align1_std")),
                                 okf(v.get("pred_mean")), okf(v.get("pred_std"))])
    write("p8_lambda.csv",
          ["model", "cleaning", "variant", "lambda", "align1_mean", "align1_std",
           "pred_mean", "pred_std"], rows)


def p9():
    rows = []
    for fp in sorted(ST.glob("p9_gkpls/results/p9_gkpls__*.json"),
                     key=lambda p: mrank(p.stem.split("__")[1])):
        d = jload(fp); m = d["method"]
        for cl, blk in d.get("cleanings", {}).items():
            if not blk or blk.get("missing") or blk.get("skipped"):
                continue
            b = blk["best"]
            rows.append([m, cl, blk.get("best_layer"),
                         okf(b["gkpls"].get("spearman_mean")), okf(b["gkpls"].get("spearman_std")),
                         okf(b["gkpls"].get("best_hyper")),
                         okf(b["rbfkpls"].get("spearman_mean")), okf(b["rbfkpls"].get("best_hyper")),
                         okf(b["krr_geo"].get("spearman_mean")), okf(b["krr_geo"].get("best_hyper"))])
    write("p9_gkpls.csv",
          ["model", "cleaning", "best_layer", "gkpls_spearman", "gkpls_std", "gkpls_a",
           "rbfkpls_spearman", "rbfkpls_a", "krr_geo_spearman", "krr_geo_lam"], rows)


if __name__ == "__main__":
    t9(); p2(); p1_gkf(); p1_mc(); p1_maxking(); p3(); p7(); t10()
    p2_geo_mc(); p3_pls(); p7_v2(); t10_cleanings(); translation()
    t11(); e5(); p8(); p9()
    print("done ->", OUT)
