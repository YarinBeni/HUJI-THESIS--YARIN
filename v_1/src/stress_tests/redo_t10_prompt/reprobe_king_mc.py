"""J3 reprobe (MC) — T10 prompted-activation year-probe under Monte-Carlo balanced
draws. Pools mean + king_last + king_mean; king sites use the name-found mask.
Maps the (N_draws, N_corpus) draws onto the prompted npz rows via fragment_id.

Usage:
    python reprobe_king_mc.py --acts-dir <out>/prompted_king --model qwen3_8b
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from mc_probe import mc_year_probe  # noqa: E402

SUBSET = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"
POOLS = ["mean", "king_last", "king_mean"]


def _draw_rows_for_npz(draws, order, npz_fids, found=None):
    """For each draw, the corpus positions -> fragment_ids -> row indices in this
    npz (which may be a subset / reordered). Intersect with `found` for king pools."""
    fid_to_row = {f: j for j, f in enumerate(npz_fids)}
    out = []
    for d in range(draws.shape[0]):
        rows = []
        for i in np.where(draws[d])[0]:
            r = fid_to_row.get(order[i])
            if r is None:
                continue
            if found is not None and not found[r]:
                continue
            rows.append(r)
        out.append(np.array(rows, dtype=int))
    return out


def run(args):
    draws = np.load(args.draws)
    order = json.loads(Path(args.fragment_order).read_text())
    acts = Path(args.acts_dir)
    summary = {"model": args.model, "protocol": "mc_balanced",
               "cleaning": args.tag or "tier0", "variants": {}}
    for vdir in sorted(acts.glob("pv*")):
        v = vdir.name
        vres = {}
        for npz in sorted(vdir.glob("L*.npz")):
            L = int(npz.stem[1:])
            d = np.load(npz, allow_pickle=True)
            fids = [str(x) for x in d["fragment_ids"]]
            years = np.asarray(d["years"], dtype=float)
            rulers = d["rulers"].astype(str)
            found = np.asarray(d["found"], dtype=bool)
            row = {}
            for pool in POOLS:
                mask = None if pool == "mean" else found
                dr = _draw_rows_for_npz(draws, order, fids, mask)
                row[pool] = mc_year_probe(d[pool], years, rulers, dr)
            vres[str(L)] = row
        summary["variants"][v] = vres
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"__{args.tag}" if args.tag else ""
    fp = outdir / f"{args.model}__t10_mc_summary{suffix}.json"
    fp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for v, vres in summary["variants"].items():
        for pool in POOLS:
            best = max((vres[L][pool].get("spearman_mean", float("nan")) for L in vres
                        if not vres[L][pool].get("skipped")), default=float("nan"))
            print(f"  {args.model} {v:4s} {pool:9s} best-year-spearman(MC)={best:.3f}")
    print(f"wrote {fp}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acts-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", default="", help="cleaning tag for the output filename "
                   "(e.g. 'maximal' -> <model>__t10_mc_summary__maximal.json)")
    p.add_argument("--draws", default=str(SUBSET / "draws_matrix.npy"))
    p.add_argument("--fragment-order", default=str(SUBSET / "corpus_fragment_order.json"))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
