#!/usr/bin/env python3
"""
03_extract_seal_embeddings.py — Plan D-extraction: Akkadian MLM embeddings for 384 SEAL fragments.

Preprocessing: df['text'].str.replace('-', ' ') converts word-level transliteration
(e.g. 'GAB-RI še20-e-mi3') to sign-level tokens ('GAB RI še20 e mi3') that match
the training corpus format.  Uses the `text` column (clean_value joined), NOT
text_tier0/text_maximal — those apply character-level cleaning that may strip signs.

Outputs 10 keys (5 layers × 2 reductions), saved to seal_mlm_coords.json:
  mlm__tier0__L{00,04,08,12,16}__{tsne,pca}

Naming convention: labelled 'tier0' since `text` (clean_value) is the minimal
cleaned form; the GUI's graceful degradation will show 'not yet available' for
the mlm__maximal__* slots.

Usage (from repo root):
    python v_1/src/archive/baseline_mlm/03_extract_seal_embeddings.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- Paths ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent           # .../baseline_mlm/
REPO_ROOT  = Path(__file__).resolve().parents[4]       # .../HUJI-THESIS--YARIN/

# Make data_utils and model importable from the same directory
sys.path.insert(0, str(SCRIPT_DIR))
from data_utils import load_vocabulary, tokenize_text  # noqa: E402
from model import AeneasConfig, AeneasForMLM            # noqa: E402

CHECKPOINT = REPO_ROOT / "v_1/models/baseline_retrained/baseline_best.pt"
VOCAB      = REPO_ROOT / "v_1/data/training_ready/vocab.json"
PARQUET    = REPO_ROOT / "v_1/data/evaluation/corpora/seal_corpus.parquet"
OUT_DIR    = REPO_ROOT / "v_1/src/linear_probing/results/seal_round4"
OUT_JSON   = OUT_DIR / "seal_mlm_coords.json"

ANALYSIS_LAYERS = [0, 4, 8, 12, 16]
MAX_LENGTH      = 512
BATCH_SIZE      = 64
SEED            = 42

# ----------------------------------------------------------------------------


def load_model(device: str) -> AeneasForMLM:
    ckpt   = torch.load(CHECKPOINT, map_location="cpu")
    config = AeneasConfig.from_dict(ckpt["config"])
    model  = AeneasForMLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    print(f"  epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}, "
          f"d_model={config.d_model}, num_layers={config.num_layers}")
    return model


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over non-PAD positions. (B,L,D) × (B,L) → (B,D)."""
    mask_f = mask.float().unsqueeze(-1)           # (B, L, 1)
    return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)


def extract_hidden_states(
    texts: list,
    sign_to_id: dict,
    model: AeneasForMLM,
    device: str,
) -> dict:
    """Return {layer_idx: np.ndarray (N, d_model)} mean-pooled hidden states."""
    # tokenize_text pads every sequence to MAX_LENGTH, so we can stack directly
    all_ids, all_mask = [], []
    for text in texts:
        ids, mask = tokenize_text(text, sign_to_id, max_length=MAX_LENGTH)
        all_ids.append(ids)
        all_mask.append(mask)

    ids_t  = torch.tensor(all_ids,  dtype=torch.long)   # (N, MAX_LENGTH)
    mask_t = torch.tensor(all_mask, dtype=torch.long)   # (N, MAX_LENGTH)

    accum = {layer: [] for layer in ANALYSIS_LAYERS}

    for start in range(0, len(texts), BATCH_SIZE):
        b_ids  = ids_t[start : start + BATCH_SIZE].to(device)
        b_mask = mask_t[start : start + BATCH_SIZE].to(device)

        with torch.no_grad():
            out = model(
                b_ids, b_mask,
                output_hidden_states=True,
                hidden_states_layers=ANALYSIS_LAYERS,
            )

        for layer, hs in out["hidden_states"].items():
            accum[layer].append(mean_pool(hs, b_mask).cpu().numpy())

        done = min(start + BATCH_SIZE, len(texts))
        print(f"  {done}/{len(texts)}", end="\r", flush=True)

    print()
    return {layer: np.concatenate(arrs) for layer, arrs in accum.items()}


def reduce_2d(emb: np.ndarray, layer: int) -> dict:
    """t-SNE + PCA on (N, d_model) → two key/coord pairs."""
    tag  = f"L{layer:02d}"
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=SEED).fit_transform(emb)
    pca  = PCA(n_components=2, random_state=SEED).fit_transform(emb)
    return {
        f"mlm__tier0__{tag}__tsne": tsne.tolist(),
        f"mlm__tier0__{tag}__pca":  pca.tolist(),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Vocab
    print("\n[1/5] Loading vocabulary...")
    sign_to_id, _ = load_vocabulary(str(VOCAB))
    print(f"  vocab_size={len(sign_to_id):,}")

    # 2. Model
    print("\n[2/5] Loading model checkpoint...")
    model = load_model(device)

    # 3. SEAL corpus
    print("\n[3/5] Loading SEAL corpus and preprocessing...")
    df    = pd.read_parquet(PARQUET)
    texts = df["text"].str.replace("-", " ", regex=False).tolist()
    avg_signs = sum(len(t.split()) for t in texts) / len(texts)
    print(f"  {len(texts)} fragments, avg {avg_signs:.0f} signs/fragment after hyphen split")

    # 4. Extraction
    print(f"\n[4/5] Extracting hidden states at layers {ANALYSIS_LAYERS}...")
    hidden = extract_hidden_states(texts, sign_to_id, model, device)
    for layer, arr in hidden.items():
        print(f"  L{layer:02d}: shape={arr.shape}, mean={arr.mean():.4f}, std={arr.std():.4f}")

    # 5. 2-D reduction
    print("\n[5/5] t-SNE + PCA per layer...")
    coords: dict = {}
    for layer, arr in hidden.items():
        print(f"  L{layer:02d}...", end=" ", flush=True)
        coords.update(reduce_2d(arr, layer))
        print("done")

    # Validate
    N = len(df)
    for key, vals in coords.items():
        flat = np.array(vals, dtype=float)
        assert len(vals) == N,           f"{key}: expected {N} rows, got {len(vals)}"
        assert flat.shape[1] == 2,       f"{key}: expected 2 columns"
        assert not np.isnan(flat).any(), f"{key}: NaN detected"
        assert not np.isinf(flat).any(), f"{key}: Inf detected"
    print(f"\n  {len(coords)} keys validated ✓")
    print(f"  keys: {sorted(coords.keys())}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(coords, f)
    size_mb = OUT_JSON.stat().st_size / 1e6
    print(f"\nSaved {OUT_JSON.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")
    print("✅ Done")


if __name__ == "__main__":
    main()
