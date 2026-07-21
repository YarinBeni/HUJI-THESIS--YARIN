"""Tokenization for the W extraction jobs, generic over tokenizers.

Mirrors world-models feature_datasets/common.make_prompt_dataset semantics:
`add_special_tokens=False`, BOS prepended iff the tokenizer defines one (Llama yes,
Qwen/gpt-oss/T5-family no), entity tokens = everything except BOS and padding (the
`empty` prompt contributes no prefix tokens; a non-empty prompt's prefix length is
measured by tokenizing the prompt alone, matching their shared-prefix-column trick).

We batch by sorted length (big speedup on the short-name datasets) and return the
original ordering info so pooled activations land in CSV row order.
"""
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Batch:
    input_ids: "object"        # LongTensor (B, T)
    attention_mask: "object"   # LongTensor (B, T)
    entity_mask: "object"      # BoolTensor (B, T)
    last_entity_ix: "object"   # LongTensor (B,) index of last entity token
    orig_rows: np.ndarray      # (B,) row indices into the entity CSV


def encode_all(tokenizer, prompt: str, strings: List[str], max_tokens: int = 96):
    """Tokenize prompt+string for every row. Returns (list_of_id_lists,
    prefix_len, n_truncated). prefix_len counts BOS + prompt tokens."""
    bos = tokenizer.bos_token_id
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
    prefix = ([bos] if bos is not None else []) + prompt_ids
    n_trunc = 0
    all_ids = []
    enc = tokenizer(
        [prompt + s for s in strings], add_special_tokens=False,
        return_attention_mask=False)["input_ids"]
    for ids in enc:
        ids = ([bos] if bos is not None else []) + ids
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
            n_trunc += 1
        all_ids.append(ids)
    return all_ids, len(prefix), n_trunc


def batches(all_ids, prefix_len, pad_id, batch_size: int):
    """Yield length-sorted padded Batches covering every row exactly once."""
    import torch

    order = np.argsort([len(x) for x in all_ids], kind="stable")
    for start in range(0, len(order), batch_size):
        rows = order[start:start + batch_size]
        chunk = [all_ids[r] for r in rows]
        T = max(len(x) for x in chunk)
        B = len(chunk)
        ids = torch.full((B, T), pad_id, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        ent = torch.zeros((B, T), dtype=torch.bool)
        last_ix = torch.zeros(B, dtype=torch.long)
        for i, x in enumerate(chunk):
            L = len(x)
            ids[i, :L] = torch.tensor(x, dtype=torch.long)
            attn[i, :L] = 1
            # entity tokens: after the (BOS+prompt) prefix, before padding.
            # A fully-truncated-to-prefix row (can't happen with empty prompt +
            # max_tokens>=2) would fall back to the last real token.
            e0 = min(prefix_len, L - 1)
            ent[i, e0:L] = True
            last_ix[i] = L - 1
        yield Batch(ids, attn, ent, last_ix, rows)
