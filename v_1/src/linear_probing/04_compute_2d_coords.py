"""
Step 4 — Compute 2D Coordinates for All SEAL Activation Layers.

For each of 4 activation dirs × N layers:
  - Load layer_XX.npz (N_TEXTS × hidden_dim float32)
  - Run t-SNE (perplexity=30, max_iter=1000, random_state=42)
  - Run PCA (2 components, random_state=42)
  - Optionally run UMAP (--include-umap)

Default produces 232 keys (2 methods × 2 cleanings × 29 layers × 2 reductions).
Key format: {method}__{cleaning}__L{NN:02d}[__last]__{reduction}
  method:    qwen | random
  cleaning:  tier0 | maximal
  reduction: tsne | pca | umap

Output: results/seal_round4/seal_qwen_coords.json  (or --output-path)
"""

import argparse
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

N_TEXTS = 384   # default; overridden dynamically in main()
N_LAYERS = 29   # L00..L28 (embedding + 28 transformer layers for Qwen2.5-7B)


def run_tsne(X):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    return tsne.fit_transform(X)


def run_pca(X):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)


def run_umap(X):
    import umap as umap_lib
    reducer = umap_lib.UMAP(n_components=2, n_neighbors=15,
                            min_dist=0.1, random_state=42)
    return reducer.fit_transform(X.astype("float64"))


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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Step 4: Compute 2D coordinates for activation layers')
    parser.add_argument('--include-umap', action='store_true', default=False)
    parser.add_argument('--pooling', choices=['mean', 'last'], default='mean',
        help="When 'last', inserts '__last' before the reduction token in each key")
    parser.add_argument('--input-base', type=str, default=None,
        help='Base directory that contains the activation subdirs. '
             'Defaults to SEAL_ACTS_DIR.')
    parser.add_argument('--input-dirs', nargs='+', default=None,
        help='List of activation subdir names. Overrides ACTIVATION_CONFIGS.')
    parser.add_argument('--method-tags', nargs='+', default=None,
        help="Parallel to --input-dirs: 'method__cleaning' tag for each dir.")
    parser.add_argument('--output-path', type=str, default=None,
        help='Output JSON path. Defaults to the existing OUTPUT_PATH.')
    return parser.parse_args()


def main():
    global N_TEXTS
    args = parse_args()

    # Build activation configs
    if args.input_dirs and args.method_tags:
        ACTIVATION_CONFIGS_active = [
            (dir_name, tag.split('__')[0], tag.split('__')[1])
            for dir_name, tag in zip(args.input_dirs, args.method_tags)
        ]
    else:
        ACTIVATION_CONFIGS_active = ACTIVATION_CONFIGS

    base_dir = Path(args.input_base) if args.input_base else SEAL_ACTS_DIR
    out_path = Path(args.output_path) if args.output_path else OUTPUT_PATH
    pooling_infix = '__last' if args.pooling == 'last' else ''

    t_start = time.time()

    # Verify all activation dirs exist before starting
    missing = []
    for dir_name, _, _ in ACTIVATION_CONFIGS_active:
        act_dir = base_dir / dir_name
        if not act_dir.exists():
            missing.append(str(act_dir))
    if missing:
        print("ERROR: Missing activation directories:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    # Dynamically detect N_TEXTS from first .npz file
    first_dir = base_dir / ACTIVATION_CONFIGS_active[0][0]
    first_npz = next(first_dir.glob('layer_*.npz'), None)
    if first_npz is None:
        print(f"ERROR: No .npz files found in {first_dir}")
        sys.exit(1)
    N_TEXTS = np.load(first_npz)['activations'].shape[0]
    print(f"Detected N_TEXTS={N_TEXTS} from {first_npz.name}")

    embeddings = {}
    n_reductions = 2 + (1 if args.include_umap else 0)
    total_combos = len(ACTIVATION_CONFIGS_active) * N_LAYERS * n_reductions
    done = 0

    for dir_name, method, cleaning in ACTIVATION_CONFIGS_active:
        act_dir = base_dir / dir_name
        print(f"\n{'='*60}")
        print(f"Processing: {dir_name}  ({method} / {cleaning})")
        print(f"{'='*60}")

        for layer_idx in range(N_LAYERS):
            npz_path = act_dir / f'layer_{layer_idx:02d}.npz'
            if not npz_path.exists():
                print(f"  ERROR: {npz_path} not found — aborting.")
                sys.exit(1)

            X = np.load(npz_path)['activations'].astype(np.float32)
            assert X.shape[0] == N_TEXTS, (
                f"Unexpected shape {X.shape} for {npz_path}: expected {N_TEXTS} rows"
            )
            layer_tag = f'L{layer_idx:02d}'

            # t-SNE
            t0 = time.time()
            coords_tsne = run_tsne(X).tolist()
            key_tsne = f'{method}__{cleaning}__{layer_tag}{pooling_infix}__tsne'
            validate_coords(key_tsne, coords_tsne)
            embeddings[key_tsne] = coords_tsne
            done += 1

            # PCA
            coords_pca = run_pca(X).tolist()
            key_pca = f'{method}__{cleaning}__{layer_tag}{pooling_infix}__pca'
            validate_coords(key_pca, coords_pca)
            embeddings[key_pca] = coords_pca
            done += 1

            # UMAP (optional)
            if args.include_umap:
                coords_umap = run_umap(X)
                key_umap = f'{method}__{cleaning}__{layer_tag}{pooling_infix}__umap'
                validate_coords(key_umap, coords_umap.tolist())
                embeddings[key_umap] = coords_umap.tolist()
                done += 1

            elapsed = time.time() - t0
            umap_label = ' + UMAP' if args.include_umap else ''
            print(f"  Layer {layer_idx:02d}: t-SNE + PCA{umap_label} done in {elapsed:.1f}s  "
                  f"[{done}/{total_combos} total]")

    # Final validation
    assert len(embeddings) == total_combos, (
        f"Expected {total_combos} keys, got {len(embeddings)}"
    )

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'embeddings': embeddings}, f)

    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done! {len(embeddings)} keys saved to {out_path}")
    print(f"Total wall time: {total_elapsed / 60:.1f} min")

    # Summary
    methods = sorted({k.split('__')[0] for k in embeddings})
    cleanings = sorted({k.split('__')[1] for k in embeddings})
    reductions = sorted({k.split('__')[-1] for k in embeddings})
    print(f"Methods: {methods}")
    print(f"Cleanings: {cleanings}")
    print(f"Reductions: {reductions}")
    print(f"Layers per combination: {N_LAYERS} (L00–L{N_LAYERS-1:02d})")


if __name__ == '__main__':
    main()
