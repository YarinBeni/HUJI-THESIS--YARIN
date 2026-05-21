"""
Round 2 Phase 3 — Extract encoder-side activations from a HuggingFace
seq2seq (encoder-decoder) model — specifically the Akkadian-finetuned
UMT5 variants from Thalesian (AKK_300m, cuneiformBase-400m).

Mirrors the I/O contract of 03_extract_seal_activations.py so that the
downstream aggregators (06_aggregate_cls.py / 06_aggregate_pls.py) can
pick up these activations transparently.

Key differences from the Qwen extractor:
- Uses AutoModelForSeq2SeqLM (encoder-decoder), not AutoModelForCausalLM.
- Calls model.encoder(...) — we never run the decoder.
- T5/UMT5 tokenizers already have pad_token set; no eos_token fallback.
- metadata.json has model_type="seq2seq_encoder".
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Make the linear_probing utils importable (this script lives one level deeper).
_THIS_DIR = Path(__file__).resolve().parent
_LP_DIR = _THIS_DIR.parent
if str(_LP_DIR) not in sys.path:
    sys.path.insert(0, str(_LP_DIR))

from utils import mean_pool, last_token_pool, RESULTS_DIR, model_short_name  # noqa: E402

ORCC_ACTS_DIR = RESULTS_DIR / 'orcc__embed' / 'activations'
DEFAULT_MODEL = 'Thalesian/AKK_300m'


def run(args):
    t0 = time.time()

    # ── Load data ───────────────────────────────────────────────────────────
    parquet_path = Path(args.input_parquet)
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} fragments from {parquet_path}")
    assert args.text_col in df.columns, (
        f"Column '{args.text_col}' not in parquet. Available: {df.columns.tolist()}"
    )

    texts = df[args.text_col].tolist()
    fragment_ids = (
        df['fragment_id'].tolist() if 'fragment_id' in df.columns else list(range(len(df)))
    )
    print(f"Text column: {args.text_col}  |  N={len(texts)}")

    # ── Load model + tokenizer ──────────────────────────────────────────────
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print(f"Loading seq2seq model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # T5/UMT5 should have pad_token already. Defensive check, fallback if missing.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print("  WARNING: tokenizer.pad_token was None; falling back to eos_token")
        else:
            raise RuntimeError("Tokenizer has neither pad_token nor eos_token.")

    model_kwargs = dict(torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model_kwargs['device_map'] = 'auto'
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded. Device: {device}")

    # ── Extract activations at all encoder layers ───────────────────────────
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
            ).to(device)

            attention_mask = inputs['attention_mask']
            encoder_outputs = model.encoder(
                input_ids=inputs['input_ids'],
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = encoder_outputs.hidden_states  # (n_enc_layers+1,) tuple

            if per_layer_acts is None:
                n_layers = len(hidden_states)
                hidden_dim = hidden_states[0].shape[-1]
                per_layer_acts = [[] for _ in range(n_layers)]
                print(f"  Detected {n_layers} encoder hidden states "
                      f"({n_layers - 1} encoder layers + embedding), "
                      f"hidden_dim={hidden_dim}")

            for layer_idx in range(n_layers):
                if args.pooling == "mean":
                    pooled = mean_pool(hidden_states[layer_idx], attention_mask)
                else:
                    pooled = last_token_pool(hidden_states[layer_idx], attention_mask)
                per_layer_acts[layer_idx].append(pooled.cpu().float().numpy())

            del encoder_outputs, hidden_states
            if torch.cuda.is_available():
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
        out_dir = ORCC_ACTS_DIR / f'{model_short_name(args.model)}_{suffix}_{args.pooling}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {n_layers} layer files to {out_dir}/")
    for layer_idx in range(n_layers):
        X = np.concatenate(per_layer_acts[layer_idx], axis=0)
        assert X.shape == (len(df), hidden_dim), (
            f"Layer {layer_idx}: expected ({len(df)}, {hidden_dim}), got {X.shape}"
        )
        assert not np.any(np.isnan(X)), f"Layer {layer_idx}: NaN detected!"
        npz_path = out_dir / f'layer_{layer_idx:02d}.npz'
        np.savez_compressed(npz_path, activations=X)

    # ── Save metadata ───────────────────────────────────────────────────────
    metadata = {
        'model_id': args.model,
        'model_type': 'seq2seq_encoder',
        'text_col': args.text_col,
        'pooling': args.pooling,
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
        description='Round 2 Phase 3: Extract encoder activations from a '
                    'seq2seq (UMT5-family) Akkadian-finetuned model.'
    )
    parser.add_argument(
        '--input-parquet', type=str,
        default='v_1/data/evaluation/corpora/orcc_corpus.parquet',
        help='Path to the corpus parquet (default: orcc_corpus.parquet)',
    )
    parser.add_argument(
        '--text-col', type=str, required=True,
        choices=['text_tier0', 'text_maximal'],
        help='Pre-cleaned text column to embed',
    )
    parser.add_argument(
        '--model', type=str, default=DEFAULT_MODEL,
        help=f'HuggingFace model ID (default: {DEFAULT_MODEL})',
    )
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='Batch size for inference (default: 16; UMT5 is small)',
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
