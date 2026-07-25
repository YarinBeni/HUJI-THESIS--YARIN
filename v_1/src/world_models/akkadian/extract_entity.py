"""WB extraction — activations for the CELL-B *entity-level* datasets.

Separate from `extract_acts.py` because these rows carry an **entity span inside a
carrier sentence**, so we pool four ways instead of two:

    ent_last  — last token of the entity span   (G&T's entity-last-token protocol)
    ent_mean  — mean over the entity span
    last      — last token of the whole string  (G&T's `headline` protocol)
    mean      — mean over the whole string

`template=bare` rows have span == whole string, so ent_last == last there by
construction — that row is the exact paper-faithful probe and the sentence rows are
the extension.

Span -> token mapping uses the tokenizer's offset mapping when available (all fast
tokenizers) and falls back to a prefix-retokenisation measurement otherwise, which is
what Llama-2's slow SentencePiece path needs.

Writes activations/{method}/{entity_type}/{site}.layer{L}.npz (gitignored) plus a
committed metadata.json.

    python extract_entity.py --method qwen3_8b
    python extract_entity.py --method llama2_70b --entity-type assyrian_ruler
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # world_models/
from wm_lib import extract as ex                     # noqa: E402
from wm_lib.registry import MODELS, RANDOM_SEED      # noqa: E402

ENTITY_TYPES = ["assyrian_ruler", "mesopotamian_place"]
SITES = ["ent_last", "ent_mean", "last", "mean"]
DATA_DIR = os.path.join(os.path.dirname(_HERE), "data", "entity_datasets")
ACTS_DIR = os.path.join(os.path.dirname(_HERE), "activations")


def load_df(entity_type):
    return pd.read_csv(os.path.join(DATA_DIR, f"{entity_type}.csv"))


def _span_tokens_offsets(tok, strings, spans, max_tokens):
    """Fast path: char->token via offset mapping."""
    enc = tok(list(strings), add_special_tokens=False,
              return_offsets_mapping=True, return_attention_mask=False)
    out = []
    for ids, offs, (c0, c1) in zip(enc["input_ids"], enc["offset_mapping"], spans):
        t0, t1 = None, None
        for ti, (a, b) in enumerate(offs):
            if b <= a:            # zero-width piece (some BPEs emit these)
                continue
            if t0 is None and b > c0:
                t0 = ti
            if a < c1:
                t1 = ti
        out.append((ids, t0 if t0 is not None else 0,
                    t1 if t1 is not None else len(ids) - 1))
    return out


def _span_tokens_prefix(tok, strings, spans, max_tokens):
    """Fallback: measure the span by retokenising the prefixes around it. Exact for
    the bare template; for sentence rows it is accurate to the piece that straddles
    the boundary, which is the same convention the offset path uses."""
    out = []
    for s, (c0, c1) in zip(strings, spans):
        ids = tok.encode(s, add_special_tokens=False)
        n_before = len(tok.encode(s[:c0], add_special_tokens=False)) if c0 else 0
        n_through = len(tok.encode(s[:c1], add_special_tokens=False))
        t0 = min(n_before, len(ids) - 1)
        t1 = min(max(n_through - 1, t0), len(ids) - 1)
        out.append((ids, t0, t1))
    return out


def encode_with_spans(tok, strings, spans, max_tokens):
    """Returns (all_ids, ent_t0, ent_t1, n_truncated) with BOS prepended when the
    tokenizer defines one (indices are shifted to match)."""
    try:
        rows = _span_tokens_offsets(tok, strings, spans, max_tokens)
    except Exception as e:  # noqa: BLE001
        print(f"[tok] offset mapping unavailable ({type(e).__name__}); "
              f"using prefix measurement", flush=True)
        rows = _span_tokens_prefix(tok, strings, spans, max_tokens)

    bos = tok.bos_token_id
    shift = 1 if bos is not None else 0
    all_ids, t0s, t1s, n_trunc = [], [], [], 0
    for ids, t0, t1 in rows:
        ids = ([bos] if bos is not None else []) + list(ids)
        t0, t1 = t0 + shift, t1 + shift
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
            n_trunc += 1
        last = len(ids) - 1
        t0, t1 = min(t0, last), min(t1, last)
        if t1 < t0:
            t0 = t1
        all_ids.append(ids)
        t0s.append(t0)
        t1s.append(t1)
    return all_ids, np.array(t0s), np.array(t1s), n_trunc


def extract(core, tok, all_ids, ent_t0, ent_t1, *, layer_stride, batch_size):
    """Forward all rows; pool the four sites per layer. Row order preserved."""
    import torch

    n_rows = len(all_ids)
    order = np.argsort([len(x) for x in all_ids], kind="stable")
    out = {s: {} for s in SITES}
    kept_layers = None
    pad_id = tok.pad_token_id

    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            chunk = [all_ids[r] for r in rows]
            T = max(len(x) for x in chunk)
            B = len(chunk)
            ids = torch.full((B, T), pad_id, dtype=torch.long)
            attn = torch.zeros((B, T), dtype=torch.long)
            ent = torch.zeros((B, T), dtype=torch.bool)
            full = torch.zeros((B, T), dtype=torch.bool)
            last_ix = torch.zeros(B, dtype=torch.long)
            ent_ix = torch.zeros(B, dtype=torch.long)
            for i, (r, x) in enumerate(zip(rows, chunk)):
                L = len(x)
                ids[i, :L] = torch.tensor(x, dtype=torch.long)
                attn[i, :L] = 1
                full[i, (1 if tok.bos_token_id is not None else 0):L] = True
                ent[i, ent_t0[r]:ent_t1[r] + 1] = True
                last_ix[i] = L - 1
                ent_ix[i] = ent_t1[r]

            dev = next(core.parameters()).device
            res = core(input_ids=ids.to(dev), attention_mask=attn.to(dev),
                       output_hidden_states=True, use_cache=False)
            hs = res.hidden_states
            if kept_layers is None:
                L = len(hs) - 1
                kept_layers = list(range(1, L + 1, layer_stride))
                if kept_layers[-1] != L:
                    kept_layers.append(L)
                d = hs[0].shape[-1]
                for s in SITES:
                    for li in kept_layers:
                        out[s][li] = np.zeros((n_rows, d), dtype=np.float16)

            for li in kept_layers:
                h = hs[li]
                hdev = h.device
                bidx = torch.arange(B, device=hdev)
                pick = {
                    "ent_last": h[bidx, ent_ix.to(hdev)],
                    "last": h[bidx, last_ix.to(hdev)],
                }
                for site, mask in (("ent_mean", ent), ("mean", full)):
                    m = mask.to(hdev).unsqueeze(-1).to(h.dtype)
                    pick[site] = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
                for site, v in pick.items():
                    out[site][li][rows] = v.float().cpu().numpy().astype(np.float16)
    return out, kept_layers


def run_one(method, spec, tok, core, entity_type, args):
    t0 = time.time()
    df = load_df(entity_type)
    if args.limit:
        df = df.iloc[:args.limit]
    strings = df.entity_string.astype(str).tolist()
    spans = list(zip(df.ent_start.astype(int), df.ent_end.astype(int)))

    all_ids, et0, et1, n_trunc = encode_with_spans(
        tok, strings, spans, args.max_tokens)
    pooled, kept_layers = extract(
        core, tok, all_ids, et0, et1,
        layer_stride=spec["layer_stride"], batch_size=args.batch_size)

    out_dir = os.path.join(ACTS_DIR, method, entity_type)
    os.makedirs(out_dir, exist_ok=True)
    for site in SITES:
        for li, arr in pooled[site].items():
            np.savez_compressed(
                os.path.join(out_dir, f"{site}.layer{li}.npz"), acts=arr)
    meta = {
        "method": method, "hfid": spec["hfid"], "entity_type": entity_type,
        "n_rows": len(df), "n_entities": int(df.entity_ix.nunique()),
        "templates": sorted(df.template.unique().tolist()),
        "d": int(next(iter(pooled[SITES[0]].values())).shape[1]),
        "layers": kept_layers, "layer_stride": spec["layer_stride"],
        "sites": SITES, "max_tokens": args.max_tokens, "n_truncated": n_trunc,
        "random_init": bool(spec.get("random")),
        "seed": RANDOM_SEED if spec.get("random") else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {method}/{entity_type}: {len(kept_layers)} layers x {len(SITES)} "
          f"sites, {meta['n_entities']} entities, {n_trunc} truncated, "
          f"{meta['elapsed_s']}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(MODELS))
    ap.add_argument("--entity-type", default="all", choices=["all"] + ENTITY_TYPES)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    spec = MODELS[args.method]
    tok, core = ex.load_model(spec, dtype=args.dtype, seed=RANDOM_SEED)
    ets = ENTITY_TYPES if args.entity_type == "all" else [args.entity_type]
    for et in ets:
        run_one(args.method, spec, tok, core, et, args)


if __name__ == "__main__":
    main()
