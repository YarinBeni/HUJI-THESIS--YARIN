"""extract_prompted_acts.py — Phase 1b prompted-activation extractor.

For each ORCC fragment, build the prompt for a given variant (pv0/pv1/pv2/pv3),
run a forward pass with output_hidden_states=True, and pool the hidden state at
the LAST token of the target `<<FRAG>>...<</FRAG>>` span.

Outputs ONE consolidated NPZ per (variant, layer):
    {out_dir}/prompted_activations/{variant}/L{LL}.npz

NPZ keys (rich Phase 1b schema):
    acts           — (N, D) float32  pooled activations
    fragment_ids   — (N,)   <U32     ORCC fragment_id strings
    rulers         — (N,)   <U64     ruler corpus labels
    years          — (N,)   int32    year BCE (positive int)
    span_end_token — (N,)   int32    token index where activation was pooled

ALSO writes a Round-1-compatible `layer_{LL}.npz` alongside (key=`activations`)
plus `metadata.json` so the existing 06_aggregate_*/07_plot_* scripts keep
working without modification.

CLI:
    python extract_prompted_acts.py --variant pv1 --layer 15 --out_dir <path>
    python extract_prompted_acts.py --variant pv0 --layer -1 --out_dir <path>

Layer convention (matches hidden_states tuple from a transformers CausalLM):
    layer 0  = embedding layer hidden state
    layer 1.. = transformer block outputs
    layer -1 = last layer (final hidden state)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

_THIS_FILE = pathlib.Path(__file__).resolve()
_THIS_DIR = _THIS_FILE.parent
_REPO_ROOT = _THIS_FILE.parents[4]
sys.path.insert(0, str(_THIS_DIR))

from pv_parse import parse_prompt_md  # noqa: E402
from run_pv import (  # noqa: E402
    DEFAULT_CORPUS,
    DEFAULT_DRAWS_MATRIX,
    DEFAULT_FRAGMENT_ORDER,
    DEFAULT_MODEL,
    DEFAULT_OUT_ROOT,
    DEFAULT_PROMPTS_DIR,
    FEWSHOT_VARIANTS,
    SEED,
    build_chat_prompt,
    compute_span_token_indices,
    render_user_prompt,
    select_fewshot_examples,
)


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir) / "prompted_activations" / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ---- Prompt ----
    prompt_path = pathlib.Path(args.prompts_dir) / f"{args.variant}.md"
    prompt = parse_prompt_md(str(prompt_path))

    # ---- Corpus ----
    df = pd.read_parquet(args.corpus)
    print(f"[corpus] loaded {len(df)} fragments", flush=True)

    # ---- Model ----
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[model] loading {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()
    print(f"[model] loaded.  device={next(model.parameters()).device}", flush=True)

    # ---- Few-shot pool (pv2) ----
    fewshot_examples = None
    if args.variant in FEWSHOT_VARIANTS:
        fewshot_examples = select_fewshot_examples(
            df=df,
            draws_matrix_path=pathlib.Path(args.draws_matrix),
            fragment_order_path=pathlib.Path(args.fragment_order),
            tokenizer=tokenizer,
            n_examples=5,
            truncate_tokens=150,
            seed=SEED,
        )

    # ---- Determine layer index ----
    n_layers_total: int | None = None  # set after first forward
    layer_arg = args.layer

    # ---- Forward-pass loop ----
    n = len(df)
    acts_list: list[np.ndarray] = []
    fragment_ids: list[str] = []
    rulers: list[str] = []
    years: list[int] = []
    span_end_tokens: list[int] = []
    hidden_dim: int | None = None
    resolved_layer: int | None = None

    print(f"[fwd] starting; n={n}  layer_arg={layer_arg}", flush=True)
    for i, row in enumerate(df.itertuples(index=False)):
        fragment_text = str(row.text_tier0)
        user_prompt = render_user_prompt(prompt["user_template"], fragment_text, fewshot_examples)
        prompt_str, input_ids = build_chat_prompt(
            tokenizer, prompt["system_prompt"], user_prompt, args.variant,
        )
        try:
            _span_start, span_end = compute_span_token_indices(tokenizer, prompt_str, input_ids)
        except Exception as e:
            print(f"  [span-err] {row.fragment_id}: {e}", flush=True)
            continue

        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple length (n_transformer_layers + 1)
        if n_layers_total is None:
            n_layers_total = len(hs)
            hidden_dim = hs[0].shape[-1]
            resolved_layer = layer_arg if layer_arg >= 0 else n_layers_total + layer_arg
            assert 0 <= resolved_layer < n_layers_total, (
                f"layer_arg={layer_arg} resolves to {resolved_layer}, "
                f"out of range [0,{n_layers_total - 1}]"
            )
            print(f"[shape] n_hidden_states={n_layers_total}  hidden_dim={hidden_dim}  "
                  f"resolved layer={resolved_layer}", flush=True)

        layer_hs = hs[resolved_layer]   # (1, T, D)
        vec = layer_hs[0, span_end, :].cpu().float().numpy()
        acts_list.append(vec)
        fragment_ids.append(str(row.fragment_id))
        rulers.append(str(row.ruler))
        years.append(int(row.year) if pd.notna(row.year) else -1)
        span_end_tokens.append(int(span_end))

        del out, hs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (i + 1) % 50 == 0 or i + 1 == n:
            elapsed = (time.time() - t0) / 60
            print(f"  [fwd] {i + 1}/{n}  ({elapsed:.1f} min)", flush=True)

    if not acts_list:
        raise RuntimeError("No activations collected — span detection failed for ALL fragments?")

    acts = np.stack(acts_list, axis=0).astype(np.float32)
    fragment_ids_arr = np.asarray(fragment_ids)
    rulers_arr = np.asarray(rulers)
    years_arr = np.asarray(years, dtype=np.int32)
    span_end_arr = np.asarray(span_end_tokens, dtype=np.int32)
    print(f"[stack] acts.shape={acts.shape}", flush=True)

    # ---- Save rich NPZ (Phase 1b schema) ----
    LL = f"{resolved_layer:02d}"
    rich_path = out_dir / f"L{LL}.npz"
    np.savez_compressed(
        rich_path,
        acts=acts,
        fragment_ids=fragment_ids_arr,
        rulers=rulers_arr,
        years=years_arr,
        span_end_token=span_end_arr,
    )
    print(f"[save] rich npz -> {rich_path}", flush=True)

    # ---- Save Round-1-compatible NPZ + metadata.json ----
    # Round 1 used key 'activations' (see 01_extract_activations.py:137); we
    # write a sibling file under the same dir so existing aggregation scripts
    # (06_aggregate_*) can load it as drop-in.
    r1_path = out_dir / f"layer_{LL}.npz"
    np.savez_compressed(r1_path, activations=acts)
    print(f"[save] r1-compat npz -> {r1_path}", flush=True)

    # metadata.json (only update — don't clobber other layers' entries)
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}
    meta.setdefault("model_id", args.model_path)
    meta.setdefault("model_short_name", "qwen2.5-7b-instruct")
    meta.setdefault("variant", args.variant)
    meta.setdefault("n_texts", int(acts.shape[0]))
    meta.setdefault("hidden_dim", int(acts.shape[1]))
    meta.setdefault("n_layers", int(n_layers_total))
    meta.setdefault("pooling", "span_end_token")
    meta.setdefault("fragment_ids", fragment_ids)
    meta.setdefault("rulers", rulers)
    meta.setdefault("years", years)
    meta["timestamp"] = datetime.now().isoformat()
    layers_done = set(meta.get("layers_done", []))
    layers_done.add(int(resolved_layer))
    meta["layers_done"] = sorted(layers_done)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[save] metadata -> {meta_path}", flush=True)
    elapsed = (time.time() - t0) / 60
    print(f"[done] variant={args.variant}  layer={resolved_layer}  {elapsed:.1f} min", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1b prompted-activation extractor")
    p.add_argument("--variant", required=True, choices=["pv0", "pv1", "pv2", "pv3"])
    p.add_argument("--layer", type=int, required=True,
                   help="Layer index into hidden_states tuple. 0=embedding, -1=last.")
    p.add_argument("--model_path", default=DEFAULT_MODEL,
                   help=f"HF model path (default env QWEN_MODEL_PATH or {DEFAULT_MODEL})")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--prompts_dir", default=str(DEFAULT_PROMPTS_DIR))
    p.add_argument("--draws_matrix", default=str(DEFAULT_DRAWS_MATRIX))
    p.add_argument("--fragment_order", default=str(DEFAULT_FRAGMENT_ORDER))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
