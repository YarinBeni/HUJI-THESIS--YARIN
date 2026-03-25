"""
Step 0 — Tokenization Sanity Check.
Load a HuggingFace tokenizer, tokenize all 4,957 Akkadian letters,
print 10 samples, and save full statistics.
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

from utils import (
    load_letters, clean_tier0, model_short_name,
    RESULTS_DIR, PERIODS, SEED,
)


def run(args):
    # ── Load data ───────────────────────────────────────────────────────────
    df = load_letters()
    df['text_clean'] = df['text'].apply(clean_tier0)
    print(f"Loaded {len(df)} letters.")
    print(f"Period distribution:\n{df['period'].value_counts().to_string()}\n")

    # ── Load tokenizer ──────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    short_name = model_short_name(args.model)
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}\n")

    # ── Tokenize ALL texts for statistics ───────────────────────────────────
    all_token_counts = []
    all_byte_fallback_counts = []
    all_unk_counts = []

    for i, text in enumerate(df['text_clean']):
        enc = tokenizer(text, add_special_tokens=False)
        ids = enc['input_ids']
        token_strs = tokenizer.convert_ids_to_tokens(ids)
        n_tokens = len(ids)

        # Count byte-fallback tokens (e.g., <0x...> or single bytes)
        n_byte = sum(1 for t in token_strs if t.startswith('<0x') or t.startswith('Ġ<0x'))

        # Count unknown tokens
        unk_id = tokenizer.unk_token_id
        n_unk = sum(1 for tid in ids if tid == unk_id) if unk_id is not None else 0

        all_token_counts.append(n_tokens)
        all_byte_fallback_counts.append(n_byte)
        all_unk_counts.append(n_unk)

    df['token_count'] = all_token_counts
    df['byte_fallback_count'] = all_byte_fallback_counts
    df['unk_count'] = all_unk_counts

    # ── Print overall statistics ────────────────────────────────────────────
    print("=" * 70)
    print("OVERALL TOKENIZATION STATISTICS")
    print("=" * 70)
    tc = np.array(all_token_counts)
    print(f"  Total texts:           {len(tc)}")
    print(f"  Mean tokens/text:      {tc.mean():.1f}")
    print(f"  Median tokens/text:    {np.median(tc):.1f}")
    print(f"  Std tokens/text:       {tc.std():.1f}")
    print(f"  Min tokens:            {tc.min()}")
    print(f"  Max tokens:            {tc.max()}")
    print(f"  Total byte-fallback:   {sum(all_byte_fallback_counts)}")
    print(f"  Total unk tokens:      {sum(all_unk_counts)}")
    print()

    # Per-period statistics
    print("PER-PERIOD TOKEN STATISTICS:")
    per_period_stats = {}
    for period in PERIODS:
        mask = df['period'] == period
        counts = df.loc[mask, 'token_count'].values
        stats = {
            'count': int(mask.sum()),
            'mean': float(np.mean(counts)),
            'median': float(np.median(counts)),
            'std': float(np.std(counts)),
            'min': int(np.min(counts)),
            'max': int(np.max(counts)),
        }
        per_period_stats[period] = stats
        print(f"  {period}: n={stats['count']}, mean={stats['mean']:.1f}, "
              f"median={stats['median']:.1f}, std={stats['std']:.1f}")

    # ── Print 10 sample tokenizations ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAMPLE TOKENIZATIONS (3 OB, 4 NA, 3 LB)")
    print("=" * 70)

    # Select samples: 3 OB, 4 NA, 3 LB — mix of short and long
    samples = []
    rng = np.random.RandomState(SEED)
    for period, n_sample in [('OB', 3), ('NA', 4), ('LB', 3)]:
        period_df = df[df['period'] == period]
        # Sort by token count to get a spread
        sorted_idx = period_df['token_count'].argsort()
        period_sorted = period_df.iloc[sorted_idx]
        # Pick from beginning, middle, end for diversity
        indices = np.linspace(0, len(period_sorted) - 1, n_sample, dtype=int)
        samples.extend(period_sorted.index[indices].tolist())

    sample_tokenizations = []
    for idx in samples:
        text = df.loc[idx, 'text_clean']
        period = df.loc[idx, 'period']
        enc = tokenizer(text, add_special_tokens=False)
        ids = enc['input_ids']
        token_strs = tokenizer.convert_ids_to_tokens(ids)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)

        print(f"\n--- [{period}] (idx={idx}, {len(ids)} tokens) ---")
        print(f"  Raw text:    {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"  Tokens:      {token_strs[:30]}{'...' if len(token_strs) > 30 else ''}")
        print(f"  Decoded:     {decoded[:200]}{'...' if len(decoded) > 200 else ''}")
        print(f"  Token count: {len(ids)}")

        sample_tokenizations.append({
            'index': int(idx),
            'period': period,
            'text': text[:500],
            'tokens': token_strs[:50],
            'decoded': decoded[:500],
            'token_count': len(ids),
            'n_byte_fallback': sum(1 for t in token_strs if t.startswith('<0x')),
        })

    # ── Save results ────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        'model_id': args.model,
        'model_short_name': short_name,
        'vocab_size': tokenizer.vocab_size,
        'n_texts': len(df),
        'overall_stats': {
            'mean_tokens': float(tc.mean()),
            'median_tokens': float(np.median(tc)),
            'std_tokens': float(tc.std()),
            'min_tokens': int(tc.min()),
            'max_tokens': int(tc.max()),
            'total_byte_fallback': int(sum(all_byte_fallback_counts)),
            'total_unk': int(sum(all_unk_counts)),
        },
        'per_period_stats': per_period_stats,
        'per_text_token_counts': [int(c) for c in all_token_counts],
        'sample_tokenizations': sample_tokenizations,
    }

    out_path = RESULTS_DIR / 'tokenization_check.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Step 0: Tokenization sanity check')
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
