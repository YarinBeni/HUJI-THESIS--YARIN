"""W0: materialize the random-init Llama-2-70B checkpoint (seed 42).

from_config on 70B needs a ~137 GB CPU-RAM spike, so we do it once on a CPU node
and save safetensors shards to WM_MODELS_DIR/llama2_70b_random; extraction then
loads it with device_map=auto like any trained checkpoint. Tokenizer files are
copied alongside so the dir is fully self-contained.

    python build_random_llama.py                  # 70b (the one that needs this)
    python build_random_llama.py --size 13b       # optional: materialize smaller ones
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wm_lib.registry import MODELS, RANDOM_SEED, WM_MODELS_DIR  # noqa: E402
from wm_lib.extract import _snapshot_with_fallback  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="70b", choices=["7b", "13b", "70b"])
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    src = MODELS[f"llama2_{args.size}"]  # trained spec: hfid + gated fallback
    out_dir = os.path.join(WM_MODELS_DIR, f"llama2_{args.size}_random")
    if os.path.exists(os.path.join(out_dir, "config.json")):
        print(f"[skip] {out_dir} already exists")
        return

    path = _snapshot_with_fallback(src)
    cfg = AutoConfig.from_pretrained(path)
    print(f"[build] random-init Llama-2-{args.size} from {path}, seed {RANDOM_SEED}",
          flush=True)
    torch.manual_seed(RANDOM_SEED)
    model = AutoModelForCausalLM.from_config(
        cfg, torch_dtype=getattr(torch, args.dtype))
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(path).save_pretrained(out_dir)
    print(f"[done] saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
