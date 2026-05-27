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

    # ---- Resolve layer indices (set after first forward) ----
    n_layers_total: int | None = None
    layers_requested: list[int] = [int(x) for x in str(args.layers).split(",") if x.strip() != ""]
    resolved_layers: list[int] | None = None

    # ---- Forward-pass loop ----
    # One forward pass yields all hidden_states; we pool BOTH last and mean
    # at every requested layer for free. Per-pooling buffers indexed by layer.
    n = len(df)
    fragment_ids: list[str] = []
    rulers: list[str] = []
    years: list[int] = []
    span_start_tokens: list[int] = []
    span_end_tokens: list[int] = []
    # acts_by_pool_by_layer[pool][layer_idx] -> list of vectors
    acts_by_pool_by_layer: dict[str, dict[int, list[np.ndarray]]] = {
        "last": {},
        "mean": {},
    }
    hidden_dim: int | None = None

    print(f"[fwd] starting; n={n}  layers_requested={layers_requested}", flush=True)
    for i, row in enumerate(df.itertuples(index=False)):
        fragment_text = str(row.text_tier0)
        user_prompt = render_user_prompt(prompt["user_template"], fragment_text, fewshot_examples)
        prompt_str, input_ids = build_chat_prompt(
            tokenizer, prompt["system_prompt"], user_prompt, args.variant,
        )
        try:
            span_start, span_end = compute_span_token_indices(tokenizer, prompt_str, input_ids)
        except Exception as e:
            print(f"  [span-err] {row.fragment_id}: {e}", flush=True)
            continue

        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            # We only read hidden_states. Calling the full *ForCausalLM forward
            # also computes the LM-head logits, a (seq, vocab) projection that
            # allocates several GiB for large-vocab models — enough to OOM an
            # 80GB H100 already ~74GB-full with Qwen3-32B's weights. Running the
            # base transformer (model.model) skips the head entirely.
            core = getattr(model, "model", model)
            out = core(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple length (n_transformer_layers + 1)
        if n_layers_total is None:
            n_layers_total = len(hs)
            hidden_dim = hs[0].shape[-1]
            resolved_layers = []
            for la in layers_requested:
                r = la if la >= 0 else n_layers_total + la
                assert 0 <= r < n_layers_total, (
                    f"layer={la} resolves to {r}, out of range [0,{n_layers_total - 1}]"
                )
                resolved_layers.append(r)
            for pool in ("last", "mean"):
                for r in resolved_layers:
                    acts_by_pool_by_layer[pool][r] = []
            print(f"[shape] n_hidden_states={n_layers_total}  hidden_dim={hidden_dim}  "
                  f"resolved_layers={resolved_layers}", flush=True)

        for r in resolved_layers:
            layer_hs = hs[r]  # (1, T, D)
            v_last = layer_hs[0, span_end, :].cpu().float().numpy()
            v_mean = layer_hs[0, span_start:span_end + 1, :].mean(dim=0).cpu().float().numpy()
            acts_by_pool_by_layer["last"][r].append(v_last)
            acts_by_pool_by_layer["mean"][r].append(v_mean)

        fragment_ids.append(str(row.fragment_id))
        rulers.append(str(row.ruler))
        years.append(int(row.year) if pd.notna(row.year) else -1)
        span_start_tokens.append(int(span_start))
        span_end_tokens.append(int(span_end))

        del out, hs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (i + 1) % 50 == 0 or i + 1 == n:
            elapsed = (time.time() - t0) / 60
            print(f"  [fwd] {i + 1}/{n}  ({elapsed:.1f} min)", flush=True)

    if not fragment_ids:
        raise RuntimeError("No activations collected — span detection failed for ALL fragments?")

    fragment_ids_arr = np.asarray(fragment_ids)
    rulers_arr = np.asarray(rulers)
    years_arr = np.asarray(years, dtype=np.int32)
    span_start_arr = np.asarray(span_start_tokens, dtype=np.int32)
    span_end_arr = np.asarray(span_end_tokens, dtype=np.int32)

    # ---- Save per-pooling, per-layer NPZs ----
    for pool in ("last", "mean"):
        pool_dir = out_dir / pool
        pool_dir.mkdir(parents=True, exist_ok=True)
        for r in resolved_layers:
            acts = np.stack(acts_by_pool_by_layer[pool][r], axis=0).astype(np.float32)
            LL = f"{r:02d}"
            rich_path = pool_dir / f"L{LL}.npz"
            np.savez_compressed(
                rich_path,
                acts=acts,
                fragment_ids=fragment_ids_arr,
                rulers=rulers_arr,
                years=years_arr,
                span_start_token=span_start_arr,
                span_end_token=span_end_arr,
            )
            # Round-1-compatible sibling for drop-in to 05_compute_*
            r1_path = pool_dir / f"layer_{LL}.npz"
            np.savez_compressed(r1_path, activations=acts)
        print(f"[save] pool={pool}  layers={resolved_layers}  -> {pool_dir}", flush=True)

    # metadata.json — keep one per variant, document both poolings + layers
    meta = {
        "model_id": args.model_path,
        "model_short_name": "qwen2.5-7b-instruct",
        "variant": args.variant,
        "n_texts": int(len(fragment_ids)),
        "hidden_dim": int(hidden_dim),
        "n_layers_total": int(n_layers_total),
        "layers_extracted": sorted(resolved_layers),
        "poolings": ["last", "mean"],
        "fragment_ids": fragment_ids,
        "rulers": rulers,
        "years": years,
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[save] metadata -> {meta_path}", flush=True)
    elapsed = (time.time() - t0) / 60
    print(f"[done] variant={args.variant}  layers={resolved_layers}  poolings=[last,mean]  "
          f"{elapsed:.1f} min", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1b prompted-activation extractor")
    p.add_argument("--variant", required=True, choices=["pv0", "pv1", "pv2", "pv3"])
    p.add_argument("--layers", required=True,
                   help="Comma-separated layer indices into hidden_states tuple "
                        "(e.g. '0,4,10,15,22,28'). 0=embedding, -1=last; "
                        "all 6 captured in a single forward pass per fragment.")
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
