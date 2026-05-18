"""
Step 0.5 — Quick EDA on Final-Layer Embeddings.
Extract final-layer embeddings (mean-pooled AND last-token) for all 4,957 texts,
run PCA + t-SNE (skip UMAP if not installed), save plots and embeddings.
"""

import argparse
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import (
    load_letters, clean_tier0, model_short_name, mean_pool, last_token_pool,
    RESULTS_DIR, PERIODS, PERIOD_COLORS, SEED,
)

POOLING_METHODS = ['mean', 'last_token']


def extract_embeddings(model, tokenizer, texts, batch_size, pooling):
    """Extract final-layer embeddings using the specified pooling method."""
    pool_fn = mean_pool if pooling == 'mean' else last_token_pool
    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"Extracting final-layer embeddings ({pooling} pooling) in {n_batches} batches...")

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)

            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            attention_mask = inputs['attention_mask']

            pooled = pool_fn(last_hidden, attention_mask)
            all_embeddings.append(pooled.cpu().float().numpy())

            if (i // batch_size + 1) % 50 == 0:
                print(f"  Batch {i // batch_size + 1}/{n_batches}")

            del outputs, last_hidden, pooled
            torch.cuda.empty_cache()

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Embeddings shape: {embeddings.shape}")
    assert embeddings.shape[0] == len(texts), f"Expected {len(texts)} rows, got {embeddings.shape[0]}"
    assert not np.any(np.isnan(embeddings)), "NaN in embeddings!"
    return embeddings


def make_plot(embeddings, periods, short_name, pooling_label, plots_dir):
    """Run PCA + t-SNE, save scatter plot."""
    print("Running PCA (2D)...")
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(embeddings)

    print("Running t-SNE (2D, perplexity=40)...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=SEED, max_iter=1000)
    X_tsne = tsne.fit_transform(embeddings)

    # Try UMAP
    X_umap = None
    try:
        import umap
        print("Running UMAP (2D)...")
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED)
        X_umap = reducer.fit_transform(embeddings)
    except ImportError:
        print("umap-learn not installed, skipping UMAP.")

    # Plot
    n_panels = 3 if X_umap is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 2:
        axes = list(axes)

    panels = [('PCA', X_pca), ('t-SNE', X_tsne)]
    if X_umap is not None:
        panels.append(('UMAP', X_umap))

    for ax, (title, X_2d) in zip(axes, panels):
        for period in PERIODS:
            mask = periods == period
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=PERIOD_COLORS[period],
                label=f'{period} (n={mask.sum()})',
                alpha=0.35, s=7, linewidths=0, rasterized=True,
            )
        ax.set_title(f'{title} — Final Layer', fontsize=12)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.legend(markerscale=4, fontsize=9, loc='best', framealpha=0.8)

    fig.suptitle(f'{short_name} — Final-Layer Embeddings ({pooling_label} pooling, tier0 clean)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    plot_path = plots_dir / f'quick_eda_final_layer_{pooling_label}.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved plot to {plot_path}")


def run(args):
    t0 = time.time()

    # ── Load data ───────────────────────────────────────────────────────────
    df = load_letters()
    df['text_clean'] = df['text'].apply(clean_tier0)
    print(f"Loaded {len(df)} letters.")

    # ── Load model + tokenizer ──────────────────────────────────────────────
    from transformers import AutoTokenizer, AutoModelForCausalLM

    short_name = model_short_name(args.model)
    print(f"Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()
    print(f"Model loaded. Device: {next(model.parameters()).device}")

    texts = df['text_clean'].tolist()
    periods = df['period'].values

    act_dir = RESULTS_DIR / 'activations' / short_name
    act_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = RESULTS_DIR / 'letters__probe_cls__period' / 'figures'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract + plot for each pooling method ────────────────────────────
    for pooling in POOLING_METHODS:
        embeddings = extract_embeddings(model, tokenizer, texts, args.batch_size, pooling)

        # Save embeddings
        emb_path = act_dir / f'final_layer_only_{pooling}.npz'
        np.savez_compressed(emb_path, activations=embeddings)
        print(f"Saved embeddings to {emb_path}")

        # Plot
        make_plot(embeddings, periods, short_name, pooling, plots_dir)

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed / 60:.1f} min")


def parse_args():
    parser = argparse.ArgumentParser(description='Step 0.5: Quick EDA on final-layer embeddings')
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model ID')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for inference (default: 8)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
