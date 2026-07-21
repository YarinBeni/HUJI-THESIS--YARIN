"""Model loading + activation extraction for the W section.

Loader lineage: stress_tests/shared/extract_lib.py (causal via `model.model` to skip
the LM head, encoder via get_encoder(), umt5 truncated-config patch, download
retries). Extended here rather than imported because W needs three extra behaviors
the shared lib deliberately doesn't have: local-directory checkpoints (materialized
random-70B), gated-repo fallback (meta-llama -> NousResearch), and multi-GPU random
init. The shared lib is load-bearing for the J jobs — not touched.
"""
import json
import os
import time

import numpy as np

ARCH_CAUSAL = "causal"
ARCH_ENCODER = "encoder"


def _snapshot(hf_id: str, arch: str) -> str:
    """Local snapshot path with retries; patches umt5's truncated config
    (see shared/extract_lib._ensure_local for the war story)."""
    if os.path.isdir(hf_id):
        return hf_id
    from huggingface_hub import snapshot_download
    last = None
    for i in range(5):
        try:
            path = snapshot_download(hf_id)
            cfg = os.path.join(path, "config.json")
            if os.path.exists(cfg):
                with open(cfg) as f:
                    d = json.load(f)
                if not d.get("model_type"):
                    if arch != ARCH_ENCODER:
                        raise ValueError(
                            f"{hf_id} config has no model_type and is not an encoder")
                    d["model_type"] = "umt5"
                    with open(cfg, "w") as f:
                        json.dump(d, f)
                    print(f"[fix] injected model_type=umt5 into {hf_id}", flush=True)
            return path
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[download] {hf_id} attempt {i}: {type(e).__name__}: {e}", flush=True)
            time.sleep(15)
    raise RuntimeError(f"could not prepare {hf_id}: {last}")


def _snapshot_with_fallback(spec: dict) -> str:
    try:
        return _snapshot(spec["hfid"], spec["arch"])
    except Exception as e:  # noqa: BLE001
        fb = spec.get("fallback_hfid")
        if not fb:
            raise
        print(f"[load] {spec['hfid']} failed ({type(e).__name__}); "
              f"falling back to {fb}", flush=True)
        return _snapshot(fb, spec["arch"])


def _load_tokenizer(path, tokenizer_hfid=None):
    """Load a tokenizer. transformers>=5 cannot convert Llama-2's SentencePiece
    tokenizer.model on ANY path (fast/slow/LlamaTokenizerFast all route through the
    tiktoken loader and crash on "Error parsing line b'\\x0e'"), so the Llama arms
    set tokenizer_hfid to a repo that ships a prebuilt tokenizer.json — tried FIRST,
    loads with no conversion. Falls back to the model's own dir for everything else.
    Every failure is printed so the log names the real cause."""
    from transformers import AutoTokenizer
    errs = []
    sources = ([("tokenizer_hfid", tokenizer_hfid)] if tokenizer_hfid else []) + \
              [("model_dir", path)]
    for label, src in sources:
        for fast in (True, False):
            try:
                return AutoTokenizer.from_pretrained(src, use_fast=fast)
            except Exception as e:  # noqa: BLE001
                errs.append(f"{label}/{'fast' if fast else 'slow'}="
                            f"{type(e).__name__}: {e}")
                print(f"[tok] {label} use_fast={fast} failed -> {errs[-1]}",
                      flush=True)
    raise RuntimeError("all tokenizer paths failed: " + " || ".join(errs))


def load_model(spec: dict, dtype: str = "bfloat16", seed: int = 42):
    """Returns (tokenizer, core_module). core(input_ids, attention_mask,
    output_hidden_states=True).hidden_states has n_layers+1 entries (0=embeddings)."""
    import torch
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    td = getattr(torch, dtype)
    path = _snapshot_with_fallback(spec)
    tok = _load_tokenizer(path, spec.get("tokenizer_hfid"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if spec.get("random"):
        # random-weights control: init from config only, fixed seed (thesis
        # convention, matches J12c / 01b_extract_random_baseline). Fits on one GPU
        # for <=13B; the 70B random arm is a pre-materialized checkpoint instead.
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(path)
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=td)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        return tok, getattr(model, "model", model)

    if spec["arch"] == ARCH_CAUSAL:
        from transformers import AutoModelForCausalLM
        try:
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=td, device_map="auto",
                output_hidden_states=True, attn_implementation="sdpa")
        except Exception as e:  # noqa: BLE001
            print(f"[load] sdpa failed ({type(e).__name__}); default attention",
                  flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=td, device_map="auto", output_hidden_states=True)
        core = getattr(model, "model", model)
    elif spec["arch"] == ARCH_ENCODER:
        import transformers
        kw = dict(torch_dtype=td, device_map="auto", output_hidden_states=True)
        try:
            from transformers import AutoModelForSeq2SeqLM
            model = AutoModelForSeq2SeqLM.from_pretrained(path, **kw)
        except Exception as e:  # noqa: BLE001
            print(f"[load] Auto failed ({type(e).__name__}); explicit seq2seq classes",
                  flush=True)
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
                    print(f"[load] {cls_name} failed: {type(e2).__name__}: {e2}",
                          flush=True)
            if model is None:
                raise
        core = model.get_encoder()
    else:
        raise ValueError(f"unknown arch {spec['arch']!r}")
    model.eval()
    return tok, core


def extract_dataset(core, tok, all_ids, prefix_len, *, sites, layer_stride,
                    batch_size, n_rows, device=None):
    """Forward every row, pool per layer/site over entity tokens.

    Returns {site: {layer_ix: (n_rows, d) float16 array}} with layer_ix counted
    1..L over transformer blocks (embedding layer skipped), keeping every
    layer_stride-th layer *from the top-down convention of range(1, L+1)*.
    """
    import torch
    from .tokenize_lib import batches

    out = {s: {} for s in sites}
    kept_layers = None
    with torch.no_grad():
        for batch in batches(all_ids, prefix_len, tok.pad_token_id, batch_size):
            dev = device or next(core.parameters()).device
            ids = batch.input_ids.to(dev)
            attn = batch.attention_mask.to(dev)
            res = core(input_ids=ids, attention_mask=attn,
                       output_hidden_states=True, use_cache=False)
            hs = res.hidden_states  # (L+1) x (B, T, d), index 0 = embeddings
            if kept_layers is None:
                L = len(hs) - 1
                kept_layers = list(range(1, L + 1, layer_stride))
                if kept_layers[-1] != L:      # always keep the final layer
                    kept_layers.append(L)
                d = hs[0].shape[-1]
                for s in sites:
                    for li in kept_layers:
                        out[s][li] = np.zeros((n_rows, d), dtype=np.float16)
            for li in kept_layers:
                # with device_map=auto later layers live on other GPUs; keep the
                # indexing tensors on each hidden state's own device
                h = hs[li]
                hdev = h.device
                if "last" in sites:
                    bidx = torch.arange(ids.shape[0], device=hdev)
                    v = h[bidx, batch.last_entity_ix.to(hdev)]
                    out["last"][li][batch.orig_rows] = (
                        v.float().cpu().numpy().astype(np.float16))
                if "mean" in sites:
                    ent_f = batch.entity_mask.to(hdev).unsqueeze(-1).to(h.dtype)
                    v = (h * ent_f).sum(dim=1) / ent_f.sum(dim=1).clamp(min=1.0)
                    out["mean"][li][batch.orig_rows] = (
                        v.float().cpu().numpy().astype(np.float16))
    return out, kept_layers
