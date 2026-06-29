"""J5 — P3 anchor extraction (GPU, small).

Embed the ruler + year ANCHOR prompts (from shared/anchors.py) through one model
and save mean-pooled embeddings per layer. ~150 short prompts -> fast. These are
the model's *explicit/declarative* time markers; J8 fits a 1-D timeline through
them (3a) and projects the ORCC texts onto it (3b).

Output: p3_matter_of_time/anchors/{method}/L{LL}.npz (acts, years, kinds) +
anchors.json. *.npz gitignored; anchors.json + metadata committed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

import anchors as anc          # noqa: E402
import extract_lib as xl       # noqa: E402

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"


def run(args):
    import torch
    df = pd.read_parquet(CORPUS)
    items = anc.build_ruler_anchors(df) + anc.build_year_anchors(df, step=int(args.year_step))
    tok, core, _ = xl.load_model(args.hfid, args.arch)

    outdir = Path(args.out) / args.method
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "anchors.json").write_text(json.dumps(items, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    buf: dict[int, list] = {}
    for j, it in enumerate(items):
        enc = tok(it["prompt"], return_tensors="pt", truncation=True, max_length=64)
        ids = enc["input_ids"]; attn = enc.get("attention_mask", torch.ones_like(ids))
        ids = ids.to(core.device if hasattr(core, "device") else "cuda")
        attn = attn.to(ids.device)
        hs = xl.hidden_states_for(core, ids, attn)
        for L in range(len(hs)):
            v = xl.pool_all_sites(hs[L], attn[0], None)["mean"]
            buf.setdefault(L, []).append(v)
        if j % 50 == 0:
            print(f"[{args.method}] anchor {j}/{len(items)}", flush=True)

    years = np.array([it["year"] for it in items], dtype=np.int32)
    kinds = np.array([it["kind"] for it in items])
    for L in buf:
        np.savez_compressed(outdir / f"L{L:02d}.npz",
                            acts=np.vstack(buf[L]).astype(np.float32),
                            years=years, kinds=kinds)
    (outdir / "metadata.json").write_text(json.dumps(
        {"method": args.method, "hfid": args.hfid, "arch": args.arch,
         "n_anchors": len(items), "n_layers": len(buf)}, indent=2), encoding="utf-8")
    print(f"[{args.method}] DONE {len(items)} anchors x {len(buf)} layers", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hfid", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--arch", required=True, choices=[xl.ARCH_CAUSAL, xl.ARCH_ENCODER])
    p.add_argument("--year-step", default=10)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "anchors"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
