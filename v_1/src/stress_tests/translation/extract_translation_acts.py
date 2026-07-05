"""J17 step 2 — embed the English translations (mean pool, every layer).

Reads translation/translations.parquet (from translate_thalesian.py) and runs the
English text through a model, mean-pooling per layer, exactly like the Akkadian
extractions. Cleaning tags: "engtier0" (translation of tier0) and "engmaximal"
(translation of maximal), so the standard loaders find the dirs:
    {acts_root}/{method}_engtier0_mean/layer_LL.npz
    {acts_root}/{method}_engmaximal_mean/layer_LL.npz

Usage:
    python extract_translation_acts.py --hfid Qwen/Qwen3-8B --method qwen3_8b --arch causal
    python extract_translation_acts.py --hfid Qwen/Qwen3-8B --method random --arch causal --random
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
import extract_lib as xl  # noqa: E402

TRANS = _THIS.parent / "translations.parquet"
DEFAULT_ACTS = _REPO / "v_1/src/linear_probing/results/orcc__embed/activations"


def run(args):
    import torch
    tdf = pd.read_parquet(TRANS)
    tok, core, _model = xl.load_model(args.hfid, args.arch, random=args.random)

    for col, tag in [("eng_tier0", "engtier0"), ("eng_maximal", "engmaximal")]:
        outdir = Path(args.acts_root) / f"{args.method}_{tag}_mean"
        outdir.mkdir(parents=True, exist_ok=True)
        texts = tdf[col].astype(str).tolist()
        buf, layers, t0 = {}, None, time.time()
        for i, text in enumerate(texts):
            if not text.strip():
                text = "..."
            enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_tokens)
            ids = enc["input_ids"]
            if ids.shape[1] == 0:
                fb = tok.eos_token_id if tok.eos_token_id is not None else (tok.pad_token_id or 0)
                ids = torch.tensor([[fb]])
            attn = torch.ones_like(ids)
            ids = ids.to(core.device if hasattr(core, "device") else "cuda")
            attn = attn.to(ids.device)
            hs = xl.hidden_states_for(core, ids, attn)
            if layers is None:
                layers = list(range(len(hs)))
                for L in layers:
                    buf[L] = []
            for L in layers:
                buf[L].append(xl.pool_all_sites(hs[L], attn[0], None)["mean"])
            if i % 200 == 0:
                print(f"[{args.method}/{tag}] {i}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
        for L in layers:
            np.savez_compressed(outdir / f"layer_{L:02d}.npz",
                                activations=np.vstack(buf[L]).astype(np.float32))
        (outdir / "metadata.json").write_text(json.dumps({
            "method": args.method, "hfid": args.hfid, "cleaning": tag,
            "source": "thalesian_english_translation", "n": len(texts),
            "n_layers": len(layers)}, indent=2), encoding="utf-8")
        print(f"[{args.method}/{tag}] DONE layers={len(layers)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hfid", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--arch", required=True, choices=[xl.ARCH_CAUSAL, xl.ARCH_ENCODER])
    p.add_argument("--random", action="store_true")
    p.add_argument("--acts-root", default=str(DEFAULT_ACTS))
    p.add_argument("--max-tokens", type=int, default=512)
    run(p.parse_args())
