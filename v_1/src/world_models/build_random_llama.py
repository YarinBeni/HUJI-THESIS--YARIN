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
    path = _snapshot_with_fallback(src)  # cached snapshot; config + tokenizer source

    # weights: skip the expensive 130GB rebuild if a prior run already wrote them
    # (a run that died at the tokenizer step below left config.json + shards intact)
    if os.path.exists(os.path.join(out_dir, "config.json")):
        print(f"[skip] weights already present at {out_dir}")
    else:
        cfg = AutoConfig.from_pretrained(path)
        print(f"[build] random-init Llama-2-{args.size} from {path}, seed {RANDOM_SEED}",
              flush=True)
        torch.manual_seed(RANDOM_SEED)
        model = AutoModelForCausalLM.from_config(
            cfg, torch_dtype=getattr(torch, args.dtype))
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
        print(f"[weights] saved to {out_dir}", flush=True)

    # tokenizer: BEST EFFORT only. transformers>=5 can't convert Llama-2's
    # tokenizer.model, so this usually fails — that's fine: the llama2_70b_random
    # registry entry sets tokenizer_hfid, so extraction loads the tokenizer from the
    # prebuilt-json repo, never from this dir. Only the config + weights matter here.
    if not os.path.exists(os.path.join(out_dir, "tokenizer_config.json")):
        for kw in ({}, {"use_fast": False}):
            try:
                AutoTokenizer.from_pretrained(path, **kw).save_pretrained(out_dir)
                print(f"[tokenizer] saved to {out_dir}", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"[tok] save {kw or 'fast'} failed ({type(e).__name__}); "
                      f"extraction will use tokenizer_hfid override", flush=True)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
