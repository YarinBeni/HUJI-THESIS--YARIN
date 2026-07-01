"""J12 — activation extraction for the "maximal-with-kings" config.

For one model, run every ORCC fragment through the model on the maximal_keepking
cleaning (full maximal context, king-name span frozen intact) and pool all THREE
sites at every layer:  mean / king_last / king_mean. Unlike the tier0 king
extractor (extract_king_acts.py), here the mean pool is on the SAME maximal text as
the king pools, so the three sites are apples-to-apples.

Outputs (canonical activations root, cleaning tag = "maxking"):
    {acts_root}/{method}_maxking_mean/layer_{LL}.npz       (key: activations)
    {acts_root}/{method}_maxking_kinglast/layer_{LL}.npz
    {acts_root}/{method}_maxking_kingmean/layer_{LL}.npz
    {acts_root}/{method}_maxking_kinglast/king_coverage.json + metadata.json

Rows where the name was not found are NaN for the king sites; `found` records
coverage. Arrays are in orcc_corpus.parquet row order.

CLI:
    python extract_maxking_acts.py --hfid Qwen/Qwen3-8B --method qwen3_8b --arch causal
    python extract_maxking_acts.py --hfid Qwen/Qwen3-8B --method random --arch causal --random
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
from cleaning import clean_maximal_keepking  # noqa: E402

DEFAULT_CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
DEFAULT_ACTS = _REPO / "v_1/src/linear_probing/results/orcc__embed/activations"


def run(args):
    import torch
    df = pd.read_parquet(args.corpus)
    spellings = kt.load_spellings()
    tok, core, _model = xl.load_model(args.hfid, args.arch, random=args.random)

    out = {site: Path(args.acts_root) / f"{args.method}_maxking_{site}"
           for site in ("mean", "kinglast", "kingmean")}
    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)

    n = len(df)
    layers = None
    buf = {"mean": {}, "kinglast": {}, "kingmean": {}}
    found = np.zeros(n, dtype=bool)
    t0 = time.time()

    for i, row in enumerate(df.itertuples(index=False)):
        ruler = getattr(row, "ruler", None)
        sp = spellings.get(ruler, [])
        text, name = clean_maximal_keepking(str(row.text_tier0), sp)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_tokens)
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask", torch.ones_like(input_ids))
        span = (kt.locate_king_tokens(text, tok, sp, input_ids=input_ids[0].tolist())
                if name is not None else None)
        if span is not None and span[1] >= input_ids.shape[1]:
            span = None
        found[i] = span is not None

        input_ids = input_ids.to(core.device if hasattr(core, "device") else "cuda")
        attn = attn.to(input_ids.device)
        hs = xl.hidden_states_for(core, input_ids, attn)
        if layers is None:
            layers = list(range(len(hs)))
            for site in buf:
                for L in layers:
                    buf[site][L] = []
        for L in layers:
            pooled = xl.pool_all_sites(hs[L], attn[0], span)
            buf["mean"][L].append(pooled["mean"])
            buf["kinglast"][L].append(pooled["king_last"])
            buf["kingmean"][L].append(pooled["king_mean"])
        if i % 200 == 0:
            print(f"[{args.method}] {i}/{n} found={found[:i+1].sum()} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    site_key = {"mean": "mean", "kinglast": "kinglast", "kingmean": "kingmean"}
    for site, d in out.items():
        for L in layers:
            np.savez_compressed(d / f"layer_{L:02d}.npz",
                                activations=np.vstack(buf[site_key[site]][L]).astype(np.float32))

    cov = dict(
        fragment_ids=df["fragment_id"].astype(str).tolist(),
        years=[None if pd.isna(v) else int(v) for v in df["year"]],
        rulers=df["ruler"].astype(str).tolist(),
        found=found.astype(int).tolist(),
    )
    meta = {"method": args.method, "hfid": args.hfid, "arch": args.arch,
            "cleaning": "maximal_keepking", "config": "maximal-with-kings",
            "n": int(n), "n_found": int(found.sum()),
            "coverage": round(float(found.mean()), 3), "n_layers": len(layers)}
    for d in out.values():
        (d / "king_coverage.json").write_text(json.dumps(cov), encoding="utf-8")
        (d / "metadata.json").write_text(json.dumps({**meta, "pool": d.name.split("_")[-1]},
                                                    indent=2), encoding="utf-8")

    print(f"[{args.method}] DONE  coverage={found.mean():.3f} "
          f"({found.sum()}/{n})  layers={len(layers)}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hfid", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--arch", required=True, choices=[xl.ARCH_CAUSAL, xl.ARCH_ENCODER])
    p.add_argument("--random", action="store_true")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--acts-root", default=str(DEFAULT_ACTS))
    p.add_argument("--max-tokens", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
