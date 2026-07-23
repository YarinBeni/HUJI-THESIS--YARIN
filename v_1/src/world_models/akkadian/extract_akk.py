"""WA extraction: last+mean-token activations for one decoder model over the Akkadian
fragments, for one text variant (akk_maximal or eng_maximal).

Reuses the validated wm_lib loaders/pooling. Writes
activations/{method}/{variant}/{site}.layer{L}.npz (fp16, fragment order) +
metadata.json (committed). npz are gitignored/cluster-local.

    python extract_akk.py --method qwen3_8b --variant akk_maximal
    python extract_akk.py --method llama2_70b --variant eng_maximal --limit 50   # smoke
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))   # world_models/ -> wm_lib
import akk_data as A                                    # noqa: E402
from wm_lib.registry import MODELS, RANDOM_SEED         # noqa: E402
from wm_lib import extract as ex                        # noqa: E402
from wm_lib.tokenize_lib import encode_all              # noqa: E402

ACTS_DIR = os.path.join(_HERE, "activations")
# decoder arms + the thesis encoders (added for the layer/PLS comparison; encoders
# have no causal last token, so only their `mean` pooling is meaningful downstream —
# the probe scripts skip the `last` site when it is absent/degenerate).
DECODER_METHODS = [
    "qwen3_1b7", "qwen3_8b", "qwen3_32b", "gpt_oss_120b", "random",
    "llama2_7b", "llama2_13b", "llama2_70b",
    "llama2_7b_random", "llama2_13b_random", "llama2_70b_random",
]
ENCODER_METHODS = ["thalesian_akk300m", "thalesian_cunei400m", "umt5_base"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=DECODER_METHODS + ENCODER_METHODS)
    ap.add_argument("--variant", required=True, choices=list(A.TEXT_VARIANTS))
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="fragments are longer than G&T names; 256 covers most")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    spec = MODELS[args.method]
    df = A.load_fragments()
    texts = A.entity_texts(df, args.variant)
    if args.limit:
        df, texts = df.iloc[:args.limit], texts[:args.limit]

    t0 = time.time()
    tok, core = ex.load_model(spec, dtype=args.dtype, seed=RANDOM_SEED)
    all_ids, prefix_len, n_trunc = encode_all(tok, "", texts, args.max_tokens)
    pooled, kept = ex.extract_dataset(
        core, tok, all_ids, prefix_len, sites=["last", "mean"],
        layer_stride=spec["layer_stride"], batch_size=args.batch_size,
        n_rows=len(texts))

    out_dir = os.path.join(ACTS_DIR, args.method, args.variant)
    os.makedirs(out_dir, exist_ok=True)
    for site in ("last", "mean"):
        for li, arr in pooled[site].items():
            np.savez_compressed(os.path.join(out_dir, f"{site}.layer{li}.npz"), acts=arr)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "method": args.method, "hfid": spec["hfid"], "variant": args.variant,
            "n_frags": len(texts), "d": int(next(iter(pooled["last"].values())).shape[1]),
            "layers": kept, "layer_stride": spec["layer_stride"],
            "sites": ["last", "mean"], "max_tokens": args.max_tokens,
            "n_truncated": n_trunc, "random_init": bool(spec.get("random")),
            "elapsed_s": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"[done] {args.method}/{args.variant}: {len(kept)} layers, "
          f"{n_trunc}/{len(texts)} truncated, {round(time.time()-t0,1)}s", flush=True)


if __name__ == "__main__":
    main()
