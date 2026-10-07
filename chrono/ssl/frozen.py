"""Frozen token-level encoder for the hybrid family: returns one layer's token
states for a batch of texts, on the GPU, bf16, no grad. Loads through the
world-models registry (same hfid/fallback logic as extract_embeddings)."""
from __future__ import annotations
import os, sys, torch
from chrono import common

WM = os.path.join(common.REPO, "v_1", "src", "world_models")


class FrozenTokenEncoder:
    def __init__(self, key: str, layer: int, max_len: int = 192, device: str | None = None):
        if WM not in sys.path:
            sys.path.insert(0, WM)
        from wm_lib.registry import MODELS
        from wm_lib import extract as ex
        self.spec = MODELS[key]; self.layer = layer; self.max_len = max_len
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok, self.core = ex.load_model(self.spec, dtype=("bfloat16" if self.dev == "cuda" else "float32"))
        self.causal = self.spec["arch"] == "causal"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        with torch.no_grad():
            s, _ = self("a")
        self.d = s.shape[-1]

    @torch.no_grad()
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if self.causal:
            enc = self.tok(list(texts), add_special_tokens=False, return_attention_mask=False)["input_ids"]
            bos = self.tok.bos_token_id
            rows = [(([bos] if bos is not None else []) + list(ids))[: self.max_len] for ids in enc]
            T = max(len(r) for r in rows); pad = self.tok.pad_token_id
            ids = torch.full((len(rows), T), pad, dtype=torch.long); attn = torch.zeros((len(rows), T), dtype=torch.long)
            for i, r in enumerate(rows):
                ids[i, :len(r)] = torch.tensor(r); attn[i, :len(r)] = 1
            out = self.core(input_ids=ids.to(self.dev), attention_mask=attn.to(self.dev), output_hidden_states=True, use_cache=False)
            mask = attn.bool().to(self.dev)
            if bos is not None:
                mask[:, 0] = False          # exclude BOS from pooling, as the M.Sc. did
        else:
            b = self.tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
            out = self.core(input_ids=b["input_ids"].to(self.dev), attention_mask=b["attention_mask"].to(self.dev),
                            output_hidden_states=True)
            mask = b["attention_mask"].bool().to(self.dev)
        return out.hidden_states[self.layer], mask
