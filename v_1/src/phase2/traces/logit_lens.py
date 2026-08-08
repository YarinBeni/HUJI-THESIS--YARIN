"""E4.4 — logit-lens the year directions themselves. No erasure, no probes.

THE IDEA (from the parametric-knowledge-traces line): a probe direction is a
vector in residual space, and residual space projects onto the vocabulary. So ask
the direction directly: WHICH TOKENS does it point at? If the cell-A "year"
direction's vocabulary projection is dominated by royal names and toponyms rather
than temporal vocabulary (years, centuries, era words), that is intrinsic
evidence the signal is identity/name-mediated — obtained without training
anything, and immune to the ICC=1 degeneracy that blocks the erasure test.

Two directions per model, when present:
  * the frozen cell-A ridge direction (probe_wm best layer, raw coordinates);
  * E1's pairwise direction (trained on relative order only), converted back
    from standardized to raw coordinates via the fragment-activation sd.

Projection: v -> W_U @ (gamma * v/||v||), with gamma the final RMSNorm weight —
the standard direction-level logit lens. Both ends of the axis are reported
(negative end = "early", positive = "late", up to the probe's sign).

    python logit_lens.py --method olmo2_7b

Writes results/{method}.json. Needs model weights (CPU, ~16GB bf16) and the
cluster-local direction/activation stores.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
sys.path.insert(0, _WM)
from wm_lib import registry                              # noqa: E402

DIRS_A = os.path.join(_WM, "results", "directions")
DIRS_PAIR = os.path.abspath(os.path.join(_HERE, "..", "pairs", "results",
                                         "directions"))
AKK_ACTS = os.path.join(_WM, "akkadian", "activations")
RESULTS = os.path.join(_HERE, "results")
ENTITY = "historical_figure"


def load_unembed(method):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hfid = registry.MODELS[method]["hfid"]
    tok = AutoTokenizer.from_pretrained(hfid)
    model = AutoModelForCausalLM.from_pretrained(
        hfid, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    W_U = model.get_output_embeddings().weight.detach().float().numpy()
    # final norm weight: llama/olmo/qwen all expose model.model.norm
    norm = model.model.norm.weight.detach().float().numpy()
    del model
    return tok, W_U, norm


def top_tokens(v, W_U, norm, tok, k):
    v = v / (np.linalg.norm(v) + 1e-8)
    logits = W_U @ (norm * v)
    order = np.argsort(logits)
    def fmt(ids):
        return [{"token": tok.convert_ids_to_tokens(int(i)),
                 "logit": float(logits[i])} for i in ids]
    return {"positive_end": fmt(order[::-1][:k]), "negative_end": fmt(order[:k])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--variant", default="akk_maximal",
                    help="which E1 pairwise direction to lens")
    ap.add_argument("--topk", type=int, default=40)
    args = ap.parse_args()

    out = {"method": args.method, "directions": {}}
    tok, W_U, norm = load_unembed(args.method)
    print(f"[model] unembed {W_U.shape}, norm {norm.shape}", flush=True)

    # 1) frozen cell-A ridge direction (raw coordinates already)
    g = sorted(glob.glob(os.path.join(DIRS_A, args.method,
                                      f"{ENTITY}.*.layer*.npz")))
    if g:
        coef = np.load(g[0])["coef"].astype(np.float32).ravel()
        out["directions"][f"cellA:{os.path.basename(g[0])}"] = top_tokens(
            coef, W_U, norm, tok, args.topk)
        print(f"[lens] cell-A direction {os.path.basename(g[0])}", flush=True)
    else:
        print(f"[warn] no cell-A direction under {DIRS_A}/{args.method}",
              flush=True)

    # 2) E1 pairwise direction: standardized -> raw via fragment-activation sd
    for p in sorted(glob.glob(os.path.join(
            DIRS_PAIR, f"{args.method}.{args.variant}.mean.layer*.npz"))):
        L = int(re.search(r"layer(\d+)\.npz$", p).group(1))
        lay = os.path.join(AKK_ACTS, args.method, args.variant,
                           f"mean.layer{L}.npz")
        if not os.path.exists(lay):
            print(f"[warn] no fragment acts for layer {L}; skipping {p}",
                  flush=True)
            continue
        X = np.load(lay)["acts"].astype(np.float32)
        sd = X.std(axis=0) + 1e-8
        w_raw = np.load(p)["w"].astype(np.float32) / sd
        out["directions"][f"pairwise:{os.path.basename(p)}"] = top_tokens(
            w_raw, W_U, norm, tok, args.topk)
        print(f"[lens] pairwise direction {os.path.basename(p)}", flush=True)

    # calibration: what do RANDOM directions lens to? Without this baseline,
    # "the pairwise direction lenses to junk" is not interpretable — junk is
    # exactly what a random direction produces, and the claim needs to be
    # "cell-A looks temporal WHERE random looks like this".
    rng = np.random.default_rng(0)
    for i in range(3):
        v = rng.standard_normal(W_U.shape[1]).astype(np.float32)
        out["directions"][f"random_control_{i}"] = top_tokens(
            v, W_U, norm, tok, args.topk)
    print("[lens] 3 random-direction controls added", flush=True)

    if not out["directions"]:
        sys.exit("no directions found to lens")
    os.makedirs(RESULTS, exist_ok=True)
    pth = os.path.join(RESULTS, f"{args.method}.json")
    with open(pth, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] -> {pth}", flush=True)


if __name__ == "__main__":
    main()
