"""J4 — king-token activation extraction (tier0 only).

For one model, run every ORCC fragment's tier0 text through the model, locate the
COMMISSIONING ruler's name span (keyed on the known `ruler` label), and pool
`king_last` + `king_mean` at every layer. Also dumps `mean` (handy cross-check;
the canonical mean acts already exist on disk).

Outputs, in the canonical activations root so existing loaders find them:
    {acts_root}/{method}_tier0_kinglast/layer_{LL}.npz   (key: activations)
    {acts_root}/{method}_tier0_kingmean/layer_{LL}.npz
    {acts_root}/{method}_tier0_kinglast/king_coverage.npz (fragment_ids, years,
        rulers, found mask, corpus order)  -- written once, mirrored to kingmean

Rows where the name was NOT found are NaN (probe step J6 drops them); `found`
records coverage. Arrays are in orcc_corpus.parquet row order.

CLI:
    python extract_king_acts.py --hfid Qwen/Qwen3-8B --method qwen3_8b --arch causal
    python extract_king_acts.py --hfid google/umt5-base --method umt5_base --arch encoder
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]
sys.path.insert(0, str(_THIS.parents[1] / "shared"))

import king_token as kt          # noqa: E402
import extract_lib as xl         # noqa: E402

DEFAULT_CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
DEFAULT_ACTS = _REPO / "v_1/src/linear_probing/results/orcc__embed/activations"


def run(args):
    import torch
    df = pd.read_parquet(args.corpus)
    spellings = kt.load_spellings()
    tok, core, _model = xl.load_model(args.hfid, args.arch, random=args.random)

    out_last = Path(args.acts_root) / f"{args.method}_tier0_kinglast"
    out_mean = Path(args.acts_root) / f"{args.method}_tier0_kingmean"
    out_last.mkdir(parents=True, exist_ok=True)
    out_mean.mkdir(parents=True, exist_ok=True)

    n = len(df)
    layers = None
    buf_last: dict[int, list] = {}
    buf_kmean: dict[int, list] = {}
    found = np.zeros(n, dtype=bool)
    t0 = time.time()

    for i, row in enumerate(df.itertuples(index=False)):
        text = str(row.text_tier0)
        ruler = getattr(row, "ruler", None)
        sp = spellings.get(ruler, [])
        enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_tokens)
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask", torch.ones_like(input_ids))
        span = kt.locate_king_tokens(text, tok, sp, input_ids=input_ids[0].tolist()) if sp else None
        # guard against truncation putting the name out of range
        if span is not None and span[1] >= input_ids.shape[1]:
            span = None
        found[i] = span is not None

        input_ids = input_ids.to(core.device if hasattr(core, "device") else "cuda")
        attn = attn.to(input_ids.device)
        hs = xl.hidden_states_for(core, input_ids, attn)
        if layers is None:
            layers = list(range(len(hs)))
            for L in layers:
                buf_last[L] = []
                buf_kmean[L] = []
        for L in layers:
            pooled = xl.pool_all_sites(hs[L], attn[0], span)
            buf_last[L].append(pooled["king_last"])
            buf_kmean[L].append(pooled["king_mean"])
        if i % 200 == 0:
            print(f"[{args.method}] {i}/{n} found={found[:i+1].sum()} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    for L in layers:
        np.savez_compressed(out_last / f"layer_{L:02d}.npz",
                            activations=np.vstack(buf_last[L]).astype(np.float32))
        np.savez_compressed(out_mean / f"layer_{L:02d}.npz",
                            activations=np.vstack(buf_kmean[L]).astype(np.float32))

    # coverage as JSON (committed; the *.npz layer dumps are gitignored/cluster-only)
    cov = dict(
        fragment_ids=df["fragment_id"].astype(str).tolist(),
        years=[None if pd.isna(v) else int(v) for v in df["year"]],  # NA -> null (JSON-safe)
        rulers=df["ruler"].astype(str).tolist(),
        found=found.astype(int).tolist(),
    )
    for d in (out_last, out_mean):
        (d / "king_coverage.json").write_text(json.dumps(cov), encoding="utf-8")
        (d / "metadata.json").write_text(json.dumps({
            "method": args.method, "hfid": args.hfid, "arch": args.arch,
            "cleaning": "tier0", "pool": d.name.split("_")[-1],
            "n": int(n), "n_found": int(found.sum()),
            "coverage": round(float(found.mean()), 3), "n_layers": len(layers),
        }, indent=2), encoding="utf-8")

    print(f"[{args.method}] DONE  coverage={found.mean():.3f} "
          f"({found.sum()}/{n})  layers={len(layers)}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hfid", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--arch", required=True, choices=[xl.ARCH_CAUSAL, xl.ARCH_ENCODER])
    p.add_argument("--random", action="store_true",
                   help="random-init weights control (causal only)")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--acts-root", default=str(DEFAULT_ACTS))
    p.add_argument("--max-tokens", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
