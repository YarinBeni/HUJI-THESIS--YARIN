"""SAE2 loader — the Adam Karvonen Qwen3-8B batch-TopK dictionary (Neuronpedia-
labeled), plus the step-0 instrument verification the handoff plan demands.

Everything here is deliberately paranoid: the release's repo name, layer list,
hook convention and inference rule were NOT independently confirmed, so this
module discovers them at runtime and RECORDS what it found instead of assuming.
If nothing matching is found, it fails with the list of attempts — that is the
blocking step-0 outcome, not an error to paper over.

Batch-TopK inference note: unlike per-sample TopK (Qwen-Scope), batch-TopK
trains with a batch-global budget and ships a THRESHOLD (theta) for inference:
z = relu(pre) * (pre > theta). If no threshold tensor is present we fall back
to per-sample topk(k=l0) and record the fallback — the FVU gate will judge it.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

CANDIDATE_REPOS = [
    "adamkarvonen/qwen3-8b-saes",
    "adamkarvonen/qwen3-saes",
    "adamkarvonen/qwen3_8b_saes",
    "adamkarvonen/saes-qwen3-8b",
]
PREFERRED_LAYERS = list(range(28, 12, -1))     # closest-to-24 first-ish; sorted later


def discover():
    """Find the release; return (repo, {layer: [filenames]}, notes).

    LESSON FROM THE FIRST F22 RUN: the repo carries SEVERAL configs per layer
    (38 files / 3 layers — different widths and L0s), and grabbing the first
    match landed on a 16k dictionary that failed the FVU gate at .82. All
    candidate files per layer are returned; the pipeline FVU-tests every one
    and keeps the best — the gate picks the config, not a filename heuristic."""
    from huggingface_hub import HfApi
    api = HfApi()
    tried = []
    for repo in CANDIDATE_REPOS:
        try:
            files = api.list_repo_files(repo)
        except Exception as e:                                    # noqa: BLE001
            tried.append(f"{repo}: {type(e).__name__}")
            continue
        layers = {}
        for f in files:
            m = re.search(r"(?:layer[_-]?|blocks[._])(\d+)", f)
            if m and f.endswith((".pt", ".safetensors", ".npz")):
                layers.setdefault(int(m.group(1)), []).append(f)
        if layers:
            return repo, layers, {"tried": tried, "n_files": len(files)}
        tried.append(f"{repo}: no layer-pattern files")
    sys.exit("step-0 BLOCKED: no Karvonen qwen3-8b SAE release found.\n"
             "Tried: " + "; ".join(tried) + "\n"
             "Find the real repo via neuronpedia.org/qwen3-8b (source "
             "resid-batchtopk-65k__l0-80) and add it to CANDIDATE_REPOS.")


def trainer_config(repo, filename):
    """The trainer's config.json (k/l0, width, hook point) if shipped next to
    the weights — lets a human match the trainer to a Neuronpedia source id."""
    from huggingface_hub import hf_hub_download
    cfg = os.path.join(os.path.dirname(filename), "config.json")
    try:
        with open(hf_hub_download(repo, cfg)) as f:
            return json.load(f)
    except Exception:                                             # noqa: BLE001
        return None


def load(repo, filename):
    """Load one layer's SAE; normalize names; detect the inference rule."""
    import torch
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, filename)
    if p.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(p)
    else:
        sd = torch.load(p, map_location="cpu", weights_only=True)
    keys = {k.lower().replace(".", "_"): k for k in sd}

    def get(*cands, required=True):
        for c in cands:
            if c in keys:
                return sd[keys[c]].float()
        if required:
            sys.exit(f"SAE keys not understood: {sorted(sd)[:10]}")
        return None
    W_enc = get("w_enc", "encoder_weight", "enc_weight", "encoder_w")
    b_enc = get("b_enc", "encoder_bias", "enc_bias")
    W_dec = get("w_dec", "decoder_weight", "dec_weight", "decoder_w")
    b_dec = get("b_dec", "decoder_bias", "dec_bias")
    theta = get("threshold", "theta", "topk_threshold", required=False)
    d_in = b_dec.shape[0]
    if W_enc.shape[0] == d_in:
        W_enc = W_enc.T                       # want (d_sae, d_in)
    if W_dec.shape[1] != d_in:
        W_dec = W_dec.T                       # want (d_sae, d_in)
    mode = "threshold" if theta is not None else "topk_fallback"
    return {"W_enc": W_enc, "b_enc": b_enc, "W_dec": W_dec, "b_dec": b_dec,
            "theta": theta, "mode": mode, "d_in": int(d_in),
            "d_sae": int(W_enc.shape[0])}


def encode(X, sae, k_fallback=80, batch=2048):
    """Dense (n, d_sae) feature activations under the detected inference rule."""
    import torch
    Xt = torch.from_numpy(np.asarray(X, np.float32))
    outs = []
    for i in range(0, len(Xt), batch):
        pre = Xt[i:i + batch] @ sae["W_enc"].T + sae["b_enc"]
        if sae["mode"] == "threshold":
            th = sae["theta"]
            th = th if th.ndim else th.expand(pre.shape[-1])
            z = torch.relu(pre) * (pre > th)
        else:
            val, idx = torch.topk(pre, k_fallback, dim=-1)
            z = torch.zeros_like(pre).scatter_(-1, idx, torch.relu(val))
        outs.append(z)
    return torch.cat(outs)


def decode(Z, sae):
    return Z @ sae["W_dec"] + sae["b_dec"]


def fvu(X, sae):
    import torch
    Xt = torch.from_numpy(np.asarray(X, np.float32))
    rec = decode(encode(X, sae), sae)
    return float(((Xt - rec) ** 2).sum() / ((Xt - Xt.mean(0)) ** 2).sum())
