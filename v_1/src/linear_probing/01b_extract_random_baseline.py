"""
Step 1b — Extract Activations from a Random-Weights Model.
Same architecture and tokenizer as the pretrained model, but with
randomly initialized weights. This isolates how much of the linear
probe's accuracy comes from pretraining vs. architecture + tokenizer.
"""

import argparse
import json
import time
import numpy as np
import torch
from datetime import datetime

from utils import (
    load_letters, clean_tier0, clean_maximal, model_short_name,
    mean_pool, last_token_pool,
    activations_dir, RESULTS_DIR, SEED,
)


def run(args):
    t0 = time.time()

    # ── Load data ───────────────────────────────────────────────────────────
    df = load_letters()
    print(f"Loaded {len(df)} letters.")
    print(f"Period distribution:\n{df['period'].value_counts().to_string()}\n")

    # Apply cleaning
    if args.cleaning == 'tier0':
        df['text_clean'] = df['text'].apply(clean_tier0)
    elif args.cleaning == 'maximal':
        df['text_clean'] = df['text'].apply(clean_maximal)
    else:
        raise ValueError(f"Unknown cleaning mode: {args.cleaning}")
    print(f"Cleaning mode: {args.cleaning}")

    # ── Load tokenizer (pretrained) + model (random weights) ───────────────
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

    short_name = model_short_name(args.model) + '-random'
    print(f"Loading RANDOM-WEIGHTS model: {args.model} (short: {short_name})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize model from config only (random weights, NOT pretrained)
    config = AutoConfig.from_pretrained(args.model)
    config.output_hidden_states = True

    # Set seed for reproducible random initialization
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(dtype=torch.bfloat16, device='cuda')
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Random model initialized: {n_params / 1e9:.1f}B params on {next(model.parameters()).device}")

    # ── Extract activations at all layers ───────────────────────────────────
    texts = df['text_clean'].tolist()
    batch_size = args.batch_size
    n_batches = (len(texts) + batch_size - 1) // batch_size

    per_layer_acts = None
    all_token_counts = []
    n_layers = None
    hidden_dim = None

    pool_fn = mean_pool if args.pooling == 'mean' else last_token_pool
    print(f"Pooling method: {args.pooling}")
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
            for j in range(len(batch_texts)):
                n_tok = int(attention_mask[j].sum().item())
                all_token_counts.append(n_tok)

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
                pooled = pool_fn(hidden_states[layer_idx], attention_mask)
                per_layer_acts[layer_idx].append(pooled.cpu().float().numpy())

            del outputs, hidden_states
            torch.cuda.empty_cache()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
                elapsed = time.time() - t0
                print(f"  Batch {batch_idx + 1}/{n_batches} done "
                      f"({elapsed / 60:.1f} min elapsed)")

    # ── Stack and save ──────────────────────────────────────────────────────
    cleaning_tag = f"{args.cleaning}_{args.pooling}" if args.pooling != 'mean' else args.cleaning
    out_dir = activations_dir(short_name, cleaning_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {n_layers} layer files to {out_dir}/")
    for layer_idx in range(n_layers):
        X = np.concatenate(per_layer_acts[layer_idx], axis=0)
        assert X.shape == (len(df), hidden_dim), \
            f"Layer {layer_idx}: expected ({len(df)}, {hidden_dim}), got {X.shape}"
        assert not np.any(np.isnan(X)), f"Layer {layer_idx}: NaN detected!"

        npz_path = out_dir / f'layer_{layer_idx:02d}.npz'
        np.savez_compressed(npz_path, activations=X)

    # Save metadata
    if 'fragment_id' in df.columns:
        text_ids = df['fragment_id'].tolist()
    else:
        text_ids = list(range(len(df)))

    metadata = {
        'model_id': args.model,
        'model_short_name': short_name,
        'model_type': 'random_weights',
        'random_seed': SEED,
        'cleaning': args.cleaning,
        'pooling': args.pooling,
        'n_texts': len(df),
        'n_layers': n_layers,
        'hidden_dim': int(hidden_dim),
        'max_length': args.max_length,
        'batch_size': batch_size,
        'text_ids': [str(tid) for tid in text_ids],
        'period_labels': df['period'].tolist(),
        'token_counts': all_token_counts,
        'timestamp': datetime.now().isoformat(),
    }

    meta_path = out_dir / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\nDone! {n_layers} layer files saved.")
    print(f"Total wall time: {elapsed / 60:.1f} min")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Step 1b: Extract activations from random-weights model')
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model ID (uses config + tokenizer only)')
    parser.add_argument('--cleaning', type=str, required=True,
                        choices=['tier0', 'maximal'],
                        help='Cleaning mode: tier0 or maximal')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'last_token'],
                        help='Pooling method: mean or last_token (default: mean)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for inference (default: 8)')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Max token length for truncation (default: 512)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
