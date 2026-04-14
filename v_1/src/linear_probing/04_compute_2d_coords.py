"""
Step 4 — Compute 2D Coordinates for All SEAL Activation Layers.

For each of 4 activation dirs × 29 layers:
  - Load layer_XX.npz (384 × 3584 float32)
  - Run t-SNE (perplexity=30, max_iter=1000, random_state=42)
  - Run PCA (2 components, random_state=42)

Produces 232 keys (2 methods × 2 cleanings × 29 layers × 2 reductions).
Key format: {method}__{cleaning}__L{NN:02d}__{reduction}
  method:    qwen | random
  cleaning:  tier0 | maximal
  reduction: tsne | pca

Output: results/seal_round4/seal_qwen_coords.json
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

from utils import RESULTS_DIR

SEAL_ACTS_DIR = RESULTS_DIR / 'seal_round4' / 'activations'
OUTPUT_PATH = RESULTS_DIR / 'seal_round4' / 'seal_qwen_coords.json'

# 4 activation directories and their (method, cleaning) tags
ACTIVATION_CONFIGS = [
    ('qwen_tier0',    'qwen',   'tier0'),
    ('qwen_maximal',  'qwen',   'maximal'),
    ('random_tier0',  'random', 'tier0'),
    ('random_maximal','random', 'maximal'),
]

N_TEXTS = 384
N_LAYERS = 29   # L00..L28 (embedding + 28 transformer layers for Qwen2.5-7B)


def run_tsne(X):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    return tsne.fit_transform(X)


def run_pca(X):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)


def validate_coords(key, coords):
    assert len(coords) == N_TEXTS, (
        f"Key '{key}': expected {N_TEXTS} points, got {len(coords)}"
    )
    arr = np.array(coords)
    assert arr.shape == (N_TEXTS, 2), (
        f"Key '{key}': expected shape ({N_TEXTS}, 2), got {arr.shape}"
    )
    assert not np.any(np.isnan(arr)), f"Key '{key}': NaN detected!"
    assert not np.any(np.isinf(arr)), f"Key '{key}': Inf detected!"


def main():
    t_start = time.time()

    # Verify all activation dirs exist before starting
    missing = []
    for dir_name, _, _ in ACTIVATION_CONFIGS:
        act_dir = SEAL_ACTS_DIR / dir_name
        if not act_dir.exists():
            missing.append(str(act_dir))
    if missing:
        print("ERROR: Missing activation directories:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    embeddings = {}
    total_combos = len(ACTIVATION_CONFIGS) * N_LAYERS * 2  # ×2 for tsne+pca
    done = 0

    for dir_name, method, cleaning in ACTIVATION_CONFIGS:
        act_dir = SEAL_ACTS_DIR / dir_name
        print(f"\n{'='*60}")
        print(f"Processing: {dir_name}  ({method} / {cleaning})")
        print(f"{'='*60}")

        for layer_idx in range(N_LAYERS):
            npz_path = act_dir / f'layer_{layer_idx:02d}.npz'
            if not npz_path.exists():
                print(f"  ERROR: {npz_path} not found — aborting.")
                sys.exit(1)

            X = np.load(npz_path)['activations'].astype(np.float32)
            assert X.shape == (N_TEXTS, X.shape[1]), (
                f"Unexpected shape {X.shape} for {npz_path}"
            )
            layer_tag = f'L{layer_idx:02d}'

            # t-SNE
            t0 = time.time()
            coords_tsne = run_tsne(X).tolist()
            key_tsne = f'{method}__{cleaning}__{layer_tag}__tsne'
            validate_coords(key_tsne, coords_tsne)
            embeddings[key_tsne] = coords_tsne
            done += 1

            # PCA
            coords_pca = run_pca(X).tolist()
            key_pca = f'{method}__{cleaning}__{layer_tag}__pca'
            validate_coords(key_pca, coords_pca)
            embeddings[key_pca] = coords_pca
            done += 1

            elapsed = time.time() - t0
            print(f"  Layer {layer_idx:02d}: t-SNE + PCA done in {elapsed:.1f}s  "
                  f"[{done}/{total_combos} total]")

    # Final validation
    assert len(embeddings) == total_combos, (
        f"Expected {total_combos} keys, got {len(embeddings)}"
    )

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump({'embeddings': embeddings}, f)

    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done! {len(embeddings)} keys saved to {OUTPUT_PATH}")
    print(f"Total wall time: {total_elapsed / 60:.1f} min")

    # Summary
    methods = sorted({k.split('__')[0] for k in embeddings})
    cleanings = sorted({k.split('__')[1] for k in embeddings})
    reductions = sorted({k.split('__')[3] for k in embeddings})
    print(f"Methods: {methods}")
    print(f"Cleanings: {cleanings}")
    print(f"Reductions: {reductions}")
    print(f"Layers per combination: {N_LAYERS} (L00–L{N_LAYERS-1:02d})")


if __name__ == '__main__':
    main()
