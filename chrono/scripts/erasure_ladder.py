"""erasure_ladder.py — P1 single-variable erasure ladder on frozen features.

QUESTION. Three encoders and two text tiers agree that masking the ruler's
NAME does not hurt dating. So what does the representation date by? Erase ONE
metadata variable at a time with LEACE (fitted on the train fold only), read
the ridge probe out through the SLA protocol, and compare with the unerased
run: the drop attributes the chronological signal to that variable. `year10`
(year deciles) is the positive control -- erasing it must crush the signal;
`ruler` is the joint upper rung (ICC=1: ruler ~ year here).

Concepts: none | ruler | period | subgenre | provenance | length | year10.
Also reports the post-erasure readability of the erased variable (balanced
accuracy of a logistic probe, train->test), so a rung that fails to erase is
visible rather than silently mis-read.

Output: head-schema score parquets `ladder_<probe>_L<layer>mean_<lang>_<concept>-s0-f<k>`
(condition = 'orig') for aggregate_emin.py / battery, plus a ladder table.
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np, pandas as pd, yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chrono import common                                       # noqa: E402
from chrono.models.store import EmbStore                        # noqa: E402
from chrono.eval.erasure import LeaceEraser, z_readability      # noqa: E402
from chrono.eval.protocol import mc_balanced_rho, pooled_rho    # noqa: E402
from run_baseline_gate import fit_predict                       # noqa: E402

CONCEPTS = ["none", "ruler", "period", "subgenre", "provenance", "length", "year10"]


def concept_matrix(c: pd.DataFrame, concept: str):
    """(Z one-hot float64 [n,k], z categorical labels) for the corpus rows."""
    if concept == "none":
        return None, None
    if concept == "ruler":
        z = c["ruler"].astype(str)
    elif concept == "period":
        z = c["period"].fillna("unk").astype(str)
    elif concept == "subgenre":
        sg = c["sub_genre"].fillna("unk").astype(str); top = sg.value_counts().head(20).index
        z = sg.where(sg.isin(top), "other")
    elif concept == "provenance":
        pv = c["provenance"].fillna("unk").astype(str); top = pv.value_counts().head(20).index
        z = pv.where(pv.isin(top), "other")
    elif concept == "length":
        z = pd.qcut(np.log1p(c["n_words"].astype(float)), 5, duplicates="drop").astype(str)
    elif concept == "year10":
        z = pd.qcut(c["t"].astype(float), 10, duplicates="drop").astype(str)
    else:
        raise ValueError(concept)
    Z = pd.get_dummies(z, dtype=float).to_numpy()
    if concept == "length":   # basis expansion as in phase 2: log length + bins
        Z = np.hstack([np.log1p(c["n_words"].astype(float)).to_numpy()[:, None], Z])
    return Z.astype(np.float64), z.to_numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="an emin2_* config: features + langs")
    ap.add_argument("--probe", default="ridge", choices=["ridge", "pls"])
    ap.add_argument("--concepts", nargs="+", default=CONCEPTS)
    ap.add_argument("--corpus", default=os.path.join(common.ART, "corpus_chrono.parquet"))
    ap.add_argument("--splits-dir", default=os.path.join(common.ART, "splits"))
    ap.add_argument("--store-root", default=os.path.join(common.ART, "emb_store"))
    ap.add_argument("--out-dir", default="chrono/reports/tier0/scores")
    ap.add_argument("--table-out", default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config)); feats = cfg["features"]
    langs = cfg["views"].get("langs") or ["akk", "eng"]
    corpus = pd.read_parquet(args.corpus).reset_index(drop=True)
    with open(os.path.join(args.splits_dir, "gkf_ruler.json")) as f: gkf = json.load(f)
    with open(os.path.join(args.splits_dir, "mc_balanced.json")) as f: mc = json.load(f)
    store = EmbStore(args.store_root)
    ids = corpus["doc_id"].astype(str).tolist()
    t = corpus["t"].astype(float).to_numpy()
    # features = mean over the languages of this arm of the corpus ORIGINAL vector
    Xs = [store.get(feats["model"], feats["layer"], feats["site"],
                    [f"{d}::{lg}::orig" for d in ids]) for lg in langs]
    X = np.mean(Xs, axis=0).astype(np.float32)
    tag = f"L{feats['layer']}{feats['site']}_{'+'.join(langs)}"
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for concept in args.concepts:
        Z, z = concept_matrix(corpus, concept)
        oof = pd.Series(np.nan, index=pd.Index(ids, name="doc_id"), dtype=float)
        readab = []
        run = f"ladder_{args.probe}_{tag}_{concept}"
        for k, fold in enumerate(gkf["folds"]):
            tr = corpus["doc_id"].isin(set(fold["train"])).to_numpy()
            te = corpus["doc_id"].isin(set(fold["test"])).to_numpy()
            Xtr, Xte = X[tr], X[te]
            if Z is not None:
                er = LeaceEraser().fit(Xtr, Z[tr])
                Xtr, Xte = er(Xtr), er(Xte)
                if len(set(z[tr])) > 1 and len(set(z[te])) > 0:
                    readab.append(z_readability(Xtr, z[tr], Xte, z[te]))
            s_te, s_tr_mean = fit_predict(args.probe, Xtr, t[tr], Xte, 2)
            oof.loc[corpus["doc_id"][te]] = s_te - s_tr_mean
            sc = pd.DataFrame({"run_id": f"{run}-s0-f{k}", "doc_id": ids, "condition": "orig",
                               "s": np.where(te, oof.to_numpy(), np.nan), "fit": "oof", "fold": k,
                               "is_test": te, "s_rank": np.nan})
            # train-side rows carry the fold model's train scores for centring
            s_tr_all, _ = fit_predict(args.probe, Xtr, t[tr], Xtr, 2)
            sc.loc[tr, "s"] = s_tr_all
            sc.to_parquet(os.path.join(args.out_dir, f"{run}-s0-f{k}.parquet"), index=False)
        mcr = float(np.nanmean(mc_balanced_rho(oof, corpus, mc)))
        pr = float(pooled_rho(oof, corpus, gkf))
        rows.append(dict(concept=concept, mc_rho=mcr, gkf_pooled=pr,
                         z_readability=(float(np.mean(readab)) if readab else np.nan),
                         k=(0 if Z is None else Z.shape[1])))
        print(f"[ladder] {tag} {args.probe} erase={concept:<10s} mc {mcr:+.3f} pooled {pr:+.3f} "
              f"readability after {rows[-1]['z_readability']:.3f} (k={rows[-1]['k']})", flush=True)
    tab = pd.DataFrame(rows)
    out = args.table_out or os.path.join(os.path.dirname(args.out_dir), f"ladder_{args.probe}_{tag}.md")
    with open(out, "w") as f:
        f.write(f"# Erasure ladder — {feats['model']} L{feats['layer']} {feats['site']} · langs {langs} · {args.probe}\n\n")
        f.write("| erased | k | mc ρ | gkf pooled ρ | Δ mc vs none | z readable after (bal. acc) |\n|---|---|---|---|---|---|\n")
        base = tab.loc[tab.concept == "none", "mc_rho"].iloc[0] if (tab.concept == "none").any() else np.nan
        for r in tab.itertuples():
            f.write(f"| {r.concept} | {r.k} | {r.mc_rho:+.3f} | {r.gkf_pooled:+.3f} | {r.mc_rho - base:+.3f} | "
                    f"{'—' if np.isnan(r.z_readability) else f'{r.z_readability:.2f}'} |\n")
    print(open(out).read())


if __name__ == "__main__":
    main()
