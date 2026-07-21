"""W extraction: per-layer activations for one method over the G&T entity datasets.

Writes activations/{method}/{entity_type}/{site}.layer{L}.npz  (fp16, CSV row order;
npz are gitignored/cluster-local) and a committed metadata.json per entity_type.

    python extract_acts.py --method qwen3_8b                       # all 6 datasets
    python extract_acts.py --method llama2_70b --entity-type world_place
    python extract_acts.py --method qwen3_1b7 --limit 200          # smoke test
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wm_lib import entity_data  # noqa: E402
from wm_lib.registry import MODELS, RANDOM_SEED  # noqa: E402
from wm_lib import extract as ex  # noqa: E402
from wm_lib.tokenize_lib import encode_all  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ACTS_DIR = os.path.join(HERE, "activations")


def run_one(method, spec, tok, core, entity_type, args):
    t0 = time.time()
    df = entity_data.load_entity_df(entity_type)
    strings = entity_data.entity_strings(entity_type, df)
    if args.limit:
        df = df.iloc[:args.limit]
        strings = strings[:args.limit]
    prompt = entity_data.PROMPTS[entity_type][args.prompt]

    all_ids, prefix_len, n_trunc = encode_all(tok, prompt, strings, args.max_tokens)
    sites = args.sites.split(",") if args.sites else spec["sites"]

    pooled, kept_layers = ex.extract_dataset(
        core, tok, all_ids, prefix_len, sites=sites,
        layer_stride=spec["layer_stride"], batch_size=args.batch_size,
        n_rows=len(strings))

    out_dir = os.path.join(ACTS_DIR, method, entity_type)
    os.makedirs(out_dir, exist_ok=True)
    for site in sites:
        for li, arr in pooled[site].items():
            np.savez_compressed(
                os.path.join(out_dir, f"{site}.layer{li}.npz"), acts=arr)
    meta = {
        "method": method,
        "hfid": spec["hfid"],
        "entity_type": entity_type,
        "prompt": args.prompt,
        "n_rows": len(strings),
        "d": int(next(iter(pooled[sites[0]].values())).shape[1]),
        "layers": kept_layers,
        "layer_stride": spec["layer_stride"],
        "sites": sites,
        "max_tokens": args.max_tokens,
        "n_truncated": n_trunc,
        "random_init": bool(spec.get("random")),
        "seed": RANDOM_SEED if spec.get("random") else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {method}/{entity_type}: {len(kept_layers)} layers x "
          f"{len(sites)} sites, {n_trunc} truncated, {meta['elapsed_s']}s",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(MODELS))
    ap.add_argument("--entity-type", default="all",
                    choices=["all"] + entity_data.ENTITY_TYPES)
    ap.add_argument("--prompt", default="empty")
    ap.add_argument("--sites", default=None,
                    help="comma list overriding the registry default (last[,mean])")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test row cap")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    spec = MODELS[args.method]
    tok, core = ex.load_model(spec, dtype=args.dtype, seed=RANDOM_SEED)

    ets = entity_data.ENTITY_TYPES if args.entity_type == "all" else [args.entity_type]
    for et in ets:
        run_one(args.method, spec, tok, core, et, args)


if __name__ == "__main__":
    main()
