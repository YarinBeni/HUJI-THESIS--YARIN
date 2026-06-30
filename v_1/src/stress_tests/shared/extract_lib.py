"""Shared model-loading + per-fragment pooling for the stress-test extractors
(J3 prompted reprobe, J4 king-token extraction).

Handles both architectures already on disk in this project:
  * causal     — Qwen3 / gpt-oss  (AutoModelForCausalLM; read the base transformer
                 `model.model` to skip the LM head and avoid OOM on big-vocab models,
                 exactly as round2_phase1b/extract_prompted_acts.py does).
  * encoder    — Thalesian / uMT5 (AutoModelForSeq2SeqLM; read `model.get_encoder()`).

Pooling sites: mean (masked), king_last, king_mean (king_* via shared.king_token).
king_* are tier0-only and partial-coverage: when the name isn't found the king
vectors are NaN and `found=False` for that fragment.
"""
from __future__ import annotations

import numpy as np

ARCH_CAUSAL = "causal"
ARCH_ENCODER = "encoder"


def _ensure_local(hf_id: str, arch: str) -> str:
    """Download a clean snapshot (with retries) and return its local path.

    HF serves a TRUNCATED config.json (813 bytes, no `model_type`) for
    google/umt5-base to unauthenticated clients, which makes AutoConfig fail with
    'Unrecognized model ... should have a model_type key' — and force_download does
    NOT fix it (the hub keeps returning the truncated body). Since every encoder
    model we use is umt5-family, we patch `model_type: "umt5"` into the cached
    config when it's missing. Loading from the local path then succeeds.
    """
    import json
    import os
    import time
    from huggingface_hub import snapshot_download

    last = None
    for i in range(5):
        try:
            path = snapshot_download(hf_id, force_download=(i == 0))
            cfg = os.path.join(path, "config.json")
            if os.path.exists(cfg):
                with open(cfg) as f:
                    d = json.load(f)
                if not d.get("model_type"):
                    if arch != ARCH_ENCODER:
                        raise ValueError(f"{hf_id} config has no model_type and is not an encoder")
                    d["model_type"] = "umt5"
                    with open(cfg, "w") as f:
                        json.dump(d, f)
                    print(f"[fix] injected model_type=umt5 into {hf_id} config.json", flush=True)
            return path
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[download] {hf_id} attempt {i}: {type(e).__name__}: {e}", flush=True)
            time.sleep(15)
    raise RuntimeError(f"could not prepare {hf_id}: {last}")


def load_model(hf_id: str, arch: str, dtype="bfloat16"):
    """Returns (tokenizer, core_module, full_model). core_module(input_ids,
    attention_mask, output_hidden_states=True) yields .hidden_states of length
    n_layers+1 (index 0 = embeddings)."""
    import os
    import torch
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    path = _ensure_local(hf_id, arch)   # local snapshot, config patched if needed
    td = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if arch == ARCH_CAUSAL:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=td, device_map="auto", output_hidden_states=True)
        core = getattr(model, "model", model)  # base transformer; skips LM head
    elif arch == ARCH_ENCODER:
        # Proven load chain from round2_phase3/extract_enc_activations.py:
        # Auto first; on failure use the explicit UMT5/MT5/T5 class, which bypasses
        # AutoConfig's model_type registry (works even with the truncated config).
        import transformers
        kw = dict(torch_dtype=td, device_map="auto", output_hidden_states=True)
        try:
            from transformers import AutoModelForSeq2SeqLM
            model = AutoModelForSeq2SeqLM.from_pretrained(path, **kw)
        except Exception as e:  # noqa: BLE001
            print(f"[load] Auto failed ({type(e).__name__}); trying explicit seq2seq classes", flush=True)
            model = None
            for cls_name in ("UMT5ForConditionalGeneration",
                             "MT5ForConditionalGeneration",
                             "T5ForConditionalGeneration"):
                cls = getattr(transformers, cls_name, None)
                if cls is None:
                    continue
                try:
                    model = cls.from_pretrained(path, **kw)
                    print(f"[load] loaded via {cls_name}", flush=True)
                    break
                except Exception as e2:  # noqa: BLE001
                    print(f"[load] {cls_name} failed: {type(e2).__name__}: {e2}", flush=True)
            if model is None:
                raise
        core = model.get_encoder()
    else:
        raise ValueError(f"unknown arch {arch!r}")
    model.eval()
    return tok, core, model


def hidden_states_for(core, input_ids, attention_mask):
    """Single forward pass -> tuple of (1, seq, hidden) hidden states."""
    import torch
    with torch.no_grad():
        out = core(input_ids=input_ids, attention_mask=attention_mask,
                   output_hidden_states=True, use_cache=False)
    return out.hidden_states


def pool_all_sites(hs_layer, attn_row, king_span):
    """Pool one layer's hidden state (1, seq, hidden) at the three sites.
    Returns dict site -> vector (or NaN vector for king_* when king_span is None)."""
    h = hs_layer[0]  # (seq, hidden)
    hnp = h.detach().float().cpu().numpy()
    a = attn_row.detach().float().cpu().numpy().reshape(-1, 1)
    mean = (hnp * a).sum(0) / max(float(a.sum()), 1.0)
    D = hnp.shape[-1]
    if king_span is None:
        nan = np.full(D, np.nan, dtype=np.float32)
        return {"mean": mean.astype(np.float32), "king_last": nan, "king_mean": nan}
    s, e = king_span
    return {
        "mean": mean.astype(np.float32),
        "king_last": hnp[e].astype(np.float32),
        "king_mean": hnp[s:e + 1].mean(0).astype(np.float32),
    }
