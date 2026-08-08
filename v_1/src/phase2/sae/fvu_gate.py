"""E5.1 — the FVU gate: can Qwen-Scope SAEs represent OUR activations at all?

Two mismatches could silently poison every downstream SAE claim: (a) the Scope
SAEs were trained on Qwen3-8B-BASE while our activations come from post-trained
Qwen/Qwen3-8B; (b) transliterated Akkadian is far outside the SAE's training
distribution. Both show up in one number — the fraction of variance unexplained
(FVU) of the SAE's reconstruction. This gate measures it per layer per dataset
BEFORE any feature is interpreted; E5.2 refuses to run on layers that fail it.

Datasets: cell-A entity activations (English, in-distribution-ish), the English
gloss fragments, and raw Akkadian fragments — all at site `last`, which is a
genuine token residual (a pooled mean is not a state the SAE ever saw).

LAYER-INDEX ALIGNMENT, resolved empirically: our npz files label hidden_states[1:]
as layer 1..N (output of block i-1), while Scope names SAEs by block index. The
right offset is decided by DATA, not convention: both offsets are tried on the
first requested layer and the one with lower cell-A FVU wins, and the choice is
recorded in the output for E5.2 to reuse.

    python fvu_gate.py                     # defaults: 8B SAE, layers 8..28
    python fvu_gate.py --layers 16 24

Writes results/fvu_gate.json. Downloads SAE weights from HuggingFace on first
run (cluster has network + HF cache).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
AKK_ACTS = os.path.join(_WM, "akkadian", "activations")
ENT_ACTS = os.path.join(_WM, "activations")
RESULTS = os.path.join(_HERE, "results")

REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100"
METHOD = "qwen3_8b"
ENTITY = "historical_figure"
K = 100                                    # the release's TopK L0


def load_sae(layer):
    """Fetch one layer's SAE and normalize its parameter names/orientations."""
    import torch
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO, f"layer{layer}.sae.pt")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    keys = {k.lower().replace(".", "_"): k for k in sd}
    def get(*cands):
        for c in cands:
            if c in keys:
                return sd[keys[c]].float()
        sys.exit(f"cannot find any of {cands} in SAE keys {list(sd)[:8]}")
    W_enc = get("w_enc", "encoder_weight", "enc_weight")
    b_enc = get("b_enc", "encoder_bias", "enc_bias")
    W_dec = get("w_dec", "decoder_weight", "dec_weight")
    b_dec = get("b_dec", "decoder_bias", "dec_bias")
    d_in = b_dec.shape[0]
    if W_enc.shape[0] == d_in:            # want (d_sae, d_in)
        W_enc = W_enc.T
    if W_dec.shape[1] != d_in:            # want (d_sae, d_in) rows = features
        W_dec = W_dec.T
    return W_enc, b_enc, W_dec, b_dec


def fvu(X, sae, k=K, batch=2048):
    import torch
    W_enc, b_enc, W_dec, b_dec = sae
    Xt = torch.from_numpy(X).float()
    num = 0.0
    for i in range(0, len(Xt), batch):
        xb = Xt[i:i + batch]
        pre = xb @ W_enc.T + b_enc
        val, idx = torch.topk(pre, k, dim=-1)
        val = torch.relu(val)
        rec = torch.zeros_like(pre).scatter_(1, idx, val) @ W_dec + b_dec
        num += float(((xb - rec) ** 2).sum())
    den = float(((Xt - Xt.mean(0)) ** 2).sum())
    return num / max(den, 1e-9)


def dataset_paths():
    return {
        "cellA_entities": os.path.join(ENT_ACTS, METHOD, ENTITY),
        "eng_tier0_frags": os.path.join(AKK_ACTS, METHOD, "eng_tier0"),
        "akk_maximal_frags": os.path.join(AKK_ACTS, METHOD, "akk_maximal"),
    }


def load_layer_acts(d, our_layer):
    p = os.path.join(d, f"last.layer{our_layer}.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["acts"].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 16, 20, 24, 28],
                    help="SAE block indices to gate")
    args = ap.parse_args()

    paths = dataset_paths()
    # settle the index offset on the first layer: SAE block L vs our file L or L+1
    L0 = args.layers[0]
    ref = paths["cellA_entities"]
    cands = {}
    sae0 = load_sae(L0)
    for off in (0, 1):
        X = load_layer_acts(ref, L0 + off)
        if X is not None:
            cands[off] = fvu(X, sae0)
    if not cands:
        sys.exit(f"no cell-A activations near layer {L0} under {ref}")
    offset = min(cands, key=cands.get)
    print(f"[align] SAE layer L <- our file layer L+{offset} "
          f"(fvu {cands})", flush=True)

    out = {"repo": REPO, "method": METHOD, "k": K, "offset": offset,
           "offset_probe": {str(k): v for k, v in cands.items()}, "fvu": {}}
    for L in args.layers:
        sae = sae0 if L == L0 else load_sae(L)
        row = {}
        for name, d in paths.items():
            X = load_layer_acts(d, L + offset)
            if X is None:
                row[name] = None
                continue
            row[name] = round(fvu(X, sae), 4)
        out["fvu"][str(L)] = row
        print(f"[fvu] SAE layer {L}: " +
              "  ".join(f"{n}={v}" for n, v in row.items()), flush=True)

    # the gate verdict E5.2 consumes: layers whose cell-A fvu is tolerable
    out["passing_layers"] = [int(L) for L, r in out["fvu"].items()
                             if (r.get("cellA_entities") or 1) < 0.5]
    os.makedirs(RESULTS, exist_ok=True)
    pth = os.path.join(RESULTS, "fvu_gate.json")
    with open(pth, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] passing layers: {out['passing_layers']} -> {pth}", flush=True)


if __name__ == "__main__":
    main()
