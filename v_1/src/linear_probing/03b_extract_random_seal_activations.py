"""
Step 3b — Extract Random-Weights Qwen Activations for SEAL Fragments.
Same architecture and tokenizer as Qwen2.5-7B-Instruct, but with randomly
initialized weights. Mirrors the 01 / 01b pattern for the letters pipeline.

Output dir: results/seal_round4/activations/random_{tier0|maximal}/
"""

import argparse
import json
import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime

from utils import mean_pool, last_token_pool, RESULTS_DIR, SEED

SEAL_ACTS_DIR = RESULTS_DIR / 'seal_round4' / 'activations'
DEFAULT_MODEL = 'Qwen/Qwen2.5-7B-Instruct'


def run(args):
    t0 = time.time()

    # ── Load data ───────────────────────────────────────────────────────────
    parquet_path = Path(args.input_parquet)
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} fragments from {parquet_path}")
    print(f"Loaded {len(df)} fragments (no length assertion)")
    assert args.text_col in df.columns, (
        f"Column '{args.text_col}' not in parquet. Available: {df.columns.tolist()}"
    )

    texts = df[args.text_col].tolist()
    fragment_ids = df['fragment_id'].tolist() if 'fragment_id' in df.columns else list(range(len(df)))
    print(f"Text column: {args.text_col}  |  N={len(texts)}")

    # ── Load tokenizer (pretrained) + model (random weights) ───────────────
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

    print(f"Loading RANDOM-WEIGHTS model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize from config only (no pretrained weights)
    config = AutoConfig.from_pretrained(args.model)
    config.output_hidden_states = True

    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(dtype=torch.bfloat16, device='cuda')
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Random model initialized: {n_params / 1e9:.1f}B params on "
          f"{next(model.parameters()).device}")

    # ── Extract activations at all layers ───────────────────────────────────
    batch_size = args.batch_size
    n_batches = (len(texts) + batch_size - 1) // batch_size

    per_layer_acts = None
    n_layers = None
    hidden_dim = None

    print(f"Processing {len(texts)} texts in {n_batches} batches "
          f"(batch_size={batch_size}, max_length={args.max_length})...")

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(model.device)

            attention_mask = inputs['attention_mask']
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            if per_layer_acts is None:
                n_layers = len(hidden_states)
                hidden_dim = hidden_states[0].shape[-1]
                per_layer_acts = [[] for _ in range(n_layers)]
                print(f"  Detected {n_layers} hidden states "
                      f"({n_layers - 1} transformer layers + embedding), "
                      f"hidden_dim={hidden_dim}")

            for layer_idx in range(n_layers):
                if args.pooling == "mean":
                    pooled = mean_pool(hidden_states[layer_idx], attention_mask)
                else:
                    pooled = last_token_pool(hidden_states[layer_idx], attention_mask)
                per_layer_acts[layer_idx].append(pooled.cpu().float().numpy())

            del outputs, hidden_states
            torch.cuda.empty_cache()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                elapsed = time.time() - t0
                print(f"  Batch {batch_idx + 1}/{n_batches} done "
                      f"({elapsed / 60:.1f} min elapsed)")

    # ── Save layer files ────────────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        suffix = args.text_col.replace('text_', '')
        out_dir = SEAL_ACTS_DIR / f'random_{suffix}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {n_layers} layer files to {out_dir}/")
    for layer_idx in range(n_layers):
        X = np.concatenate(per_layer_acts[layer_idx], axis=0)  # (384, hidden_dim)
        assert X.shape == (len(df), hidden_dim), (
            f"Layer {layer_idx}: expected ({len(df)}, {hidden_dim}), got {X.shape}"
        )
        assert not np.any(np.isnan(X)), f"Layer {layer_idx}: NaN detected!"
        npz_path = out_dir / f'layer_{layer_idx:02d}.npz'
        np.savez_compressed(npz_path, activations=X)

    # Save metadata
    metadata = {
        'model_id': args.model,
        'model_type': 'random_weights',
        'random_seed': SEED,
        'text_col': args.text_col,
        'n_texts': len(df),
        'n_layers': n_layers,
        'hidden_dim': int(hidden_dim),
        'max_length': args.max_length,
        'batch_size': batch_size,
        'fragment_ids': [str(fid) for fid in fragment_ids],
        'timestamp': datetime.now().isoformat(),
    }
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\nDone! {n_layers} layer files saved to {out_dir}")
    print(f"Total wall time: {elapsed / 60:.1f} min")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Step 3b: Extract random-weights Qwen activations for SEAL fragments')
    parser.add_argument(
        '--input-parquet', type=str,
        default='v_1/data/evaluation/corpora/seal_corpus.parquet',
        help='Path to seal_corpus.parquet (default: v_1/data/evaluation/corpora/seal_corpus.parquet)',
    )
    parser.add_argument(
        '--text-col', type=str, required=True,
        choices=['text_tier0', 'text_maximal'],
        help='Pre-cleaned text column to embed',
    )
    parser.add_argument(
        '--model', type=str, default=DEFAULT_MODEL,
        help=f'HuggingFace model ID for config + tokenizer (default: {DEFAULT_MODEL})',
    )
    parser.add_argument(
        '--batch-size', type=int, default=8,
        help='Batch size for inference (default: 8)',
    )
    parser.add_argument(
        '--max-length', type=int, default=512,
        help='Max token length for truncation (default: 512)',
    )
    parser.add_argument(
        '--pooling', choices=['mean', 'last'], default='mean',
        help='Pooling strategy: mean (default) or last token',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Override default output directory path',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
