"""J15 — P3 timeline v2: PLS-supervised anchor timeline + geodesic projection (CPU).

Yarin's proposed complement to the unsupervised 3a/3b (timeline_p3.py):

  1. Fit PLS(n_components=3) on the 153 ANCHOR embeddings vs their years — unlike
     PCA this extracts the directions that maximize COVARIANCE WITH YEAR, not raw
     variance. (Fit on anchors only: the corpus texts never touch the fit, so there
     is no text-label leakage.)
  2. The anchors' 3-D PLS scores form the "timeline manifold". Two 1-D readings:
       - pls1  : |Spearman(1st PLS component, year)| — the straight supervised axis.
       - geo1  : Isomap-1D over the 3-D anchor scores (geodesic distances along the
                 kNN graph) -> |Spearman(coordinate, year)| — the curved timeline.
  3. Project each ORCC text (mean-pooled, per cleaning) into the same 3-D PLS space
     and predict its year by INVERSE-DISTANCE-WEIGHTED INTERPOLATION over its m=5
     nearest anchors (plus the plain nearest-anchor m=1 for reference)
     -> Spearman(pred year, true year).
  4. Ruler classification: nearest RULER-anchor in the 3-D space -> predicted ruler
     -> macro-F1 over labeled texts (chance ~ 1/40).

Runs per model x cleaning (tier0 / maximal / maxking — anchors are English prompts
so only the TEXT side changes with cleaning) x layer.
Emits results/pls/p3pls__<method>.json.

Usage:  python timeline_p3_pls.py --method qwen3_8b
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from geo_loader import find_acts_dir, load_layer, available_layers, isomap_1d  # noqa: E402

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
CLEANINGS = ["tier0", "maximal", "maxking"]
N_COMP = 3
M_INTERP = 5


def _sp(a, b):
    from scipy.stats import spearmanr
    if len(set(np.asarray(a).tolist())) < 2:
        return float("nan")
    r = spearmanr(a, b).correlation
    return float(abs(r)) if r == r else float("nan")


def _sp_signed(a, b):
    from scipy.stats import spearmanr
    if len(set(np.asarray(a).tolist())) < 2:
        return float("nan")
    r = spearmanr(a, b).correlation
    return float(r) if r == r else float("nan")


def layer_block(A, ay, aruler, X, year, ruler, ymask):
    """One model/cleaning/layer. A (153,D) anchors; X (N,D) texts (corpus order)."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import normalize
    from sklearn.metrics import f1_score

    An = normalize(A, norm="l2")
    pls = PLSRegression(n_components=N_COMP).fit(An, ay)
    S = pls.transform(An)                     # (153, 3) anchor scores

    rec = {"pls1_spearman": _sp(S[:, 0], ay)}
    try:
        g1 = isomap_1d(S, k=min(10, len(S) - 1), metric="euclidean")
        rec["geo1_spearman"] = _sp(g1, ay)
    except Exception:
        rec["geo1_spearman"] = float("nan")

    # --- project texts into the anchor PLS space ---
    finite = np.isfinite(X).all(axis=1)
    use = ymask & finite
    if use.sum() < 20:
        rec["proj"] = {"insufficient": True}
        return rec
    T = pls.transform(normalize(X[use], norm="l2"))       # (n,3)
    yt = year[use]; rt = ruler[use]
    d2 = ((T[:, None, :] - S[None, :, :]) ** 2).sum(-1)   # (n,153) squared dists

    # nearest-anchor (m=1)
    nn = d2.argmin(axis=1)
    rec["proj_nn1_spearman"] = _sp_signed(ay[nn], yt)
    # inverse-distance interpolation over m nearest anchors
    m = min(M_INTERP, d2.shape[1])
    idx = np.argsort(d2, axis=1)[:, :m]
    dd = np.take_along_axis(d2, idx, axis=1) ** 0.5
    w = 1.0 / np.maximum(dd, 1e-9)
    w /= w.sum(axis=1, keepdims=True)
    pred = (w * ay[idx]).sum(axis=1)
    rec["proj_interp_spearman"] = _sp_signed(pred, yt)
    rec["proj_interp_mae"] = float(np.mean(np.abs(pred - yt)))

    # ruler classification via nearest RULER-anchor
    ridx = np.where(aruler != "")[0]
    if len(ridx):
        nnr = d2[:, ridx].argmin(axis=1)
        pred_r = aruler[ridx][nnr]
        lab = rt != ""
        rec["ruler_macro_f1"] = float(f1_score(rt[lab], pred_r[lab], average="macro",
                                               zero_division=0))
        rec["ruler_acc"] = float((pred_r[lab] == rt[lab]).mean())
        rec["n_ruler_anchors"] = int(len(ridx))
    rec["n_texts"] = int(use.sum())
    return rec


def run(args):
    anchor_dir = Path(args.anchors) / args.method
    if not anchor_dir.exists():
        print(f"no anchors for {args.method}"); return
    df = pd.read_parquet(args.corpus)
    year = df["year"].to_numpy(dtype=float)
    ymask = np.isfinite(year)
    ruler = df["ruler"].astype(str).to_numpy()

    # ruler label per anchor from the committed anchors.json (same row order as the
    # npz: extract_anchor_acts.py builds ruler anchors first, then year anchors)
    items = json.loads((anchor_dir / "anchors.json").read_text())
    aruler = np.array([str(it.get("ruler") or "") for it in items])

    out = {"method": args.method, "protocol": "pls3_geodesic_timeline",
           "n_components": N_COMP, "m_interp": M_INTERP, "cleanings": {}}
    for cleaning in CLEANINGS:
        d = find_acts_dir(args.method, cleaning, "mean")
        if d is None:
            out["cleanings"][cleaning] = {"missing": True}; continue
        text_layers = set(available_layers(d))
        per = {}
        for npz in sorted(anchor_dir.glob("L*.npz")):
            L = int(npz.stem[1:])
            if L not in text_layers:
                continue
            a = np.load(npz, allow_pickle=True)
            A = a["acts"]; ay = a["years"].astype(float)
            assert len(A) == len(aruler), "anchors.json / npz row mismatch"
            per[str(L)] = layer_block(A, ay, aruler, load_layer(d, L), year, ruler, ymask)
        valid = {L: r for L, r in per.items() if "proj_interp_spearman" in r}
        blk = {"per_layer": per}
        if valid:
            bestL = max(valid, key=lambda L: valid[L]["geo1_spearman"]
                        if valid[L]["geo1_spearman"] == valid[L]["geo1_spearman"] else -9)
            blk["best_layer"] = bestL
            blk["best"] = valid[bestL]
        out["cleanings"][cleaning] = blk

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"p3pls__{args.method}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for cl, blk in out["cleanings"].items():
        if blk.get("missing") or "best" not in blk:
            print(f"  {cl:8s}: missing/insufficient"); continue
        b = blk["best"]
        print(f"  {cl:8s}: L{blk['best_layer']} pls1={b['pls1_spearman']:.3f} "
              f"geo1={b['geo1_spearman']:.3f} | proj interp={b['proj_interp_spearman']:.3f} "
              f"nn1={b['proj_nn1_spearman']:.3f} | rulerF1={b.get('ruler_macro_f1', float('nan')):.3f}")
    print(f"wrote {fp}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--anchors", default=str(Path(__file__).resolve().parent / "anchors"))
    p.add_argument("--corpus", default=str(CORPUS))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "pls"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
