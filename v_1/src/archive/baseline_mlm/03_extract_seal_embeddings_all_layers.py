#!/usr/bin/env python3
"""
03_extract_seal_embeddings_all_layers.py — Raw MLM activations for all 17 layers.

Unlike 03_extract_seal_embeddings.py (which produces 2D visualisation coords),
this script saves mean-pooled hidden states as layer_NN.npz files in the same
convention used by the Qwen extraction scripts, so they can be consumed by the
PLS pipeline (05_compute_pls_mlm.py).

Layer numbering mirrors the Torso forward pass:
  L00 = after embedding (before any transformer block)
  L01–L16 = after each of the 16 transformer blocks (17 total)

Output directory layout (example for SEAL):
  results/seal_round4/activations/mlm_tier0/
    layer_00.npz  ...  layer_16.npz   # {"activations": (N, 384)} float32
    metadata.json

Text preprocessing: df[text_col].str.replace('-', ' ') converts word-level
transliteration to sign-level tokens matching the training corpus format.
Use --text-col text  for SEAL (default), --text-col text_tier0  for ORCC.

Usage (from repo root):
  # SEAL
  python v_1/src/archive/baseline_mlm/03_extract_seal_embeddings_all_layers.py

  # ORCC
  python v_1/src/archive/baseline_mlm/03_extract_seal_embeddings_all_layers.py \\
      --input-parquet v_1/data/evaluation/corpora/orcc_corpus.parquet \\
      --text-col text_tier0 \\
      --output-dir v_1/src/linear_probing/results/orcc_round1/activations/mlm_tier0

  # Force re-extraction of all layers even if .npz already exists
  python ... --force
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent       # .../baseline_mlm/
REPO_ROOT  = Path(__file__).resolve().parents[4]   # .../HUJI-THESIS--YARIN/

sys.path.insert(0, str(SCRIPT_DIR))
from data_utils import load_vocabulary, tokenize_text  # noqa: E402
from model import AeneasConfig, AeneasForMLM            # noqa: E402

CHECKPOINT   = REPO_ROOT / "v_1/models/baseline_retrained/baseline_best.pt"
VOCAB        = REPO_ROOT / "v_1/data/training_ready/vocab.json"
SEAL_PARQUET = REPO_ROOT / "v_1/data/evaluation/corpora/seal_corpus.parquet"
SEAL_OUT_DIR = (REPO_ROOT
                / "v_1/src/linear_probing/results/seal_round4/activations/mlm_tier0")

MAX_LENGTH = 512
BATCH_SIZE = 64
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract Akkadian MLM activations for all layers (raw .npz)")
    p.add_argument("--input-parquet", default=str(SEAL_PARQUET),
                   help="Path to input parquet (default: seal_corpus.parquet)")
    p.add_argument("--text-col",      default="text",
                   help="Text column to use; use 'text_tier0' for ORCC (default: text)")
    p.add_argument("--output-dir",    default=str(SEAL_OUT_DIR),
                   help="Directory to write layer_NN.npz + metadata.json")
    p.add_argument("--force",         action="store_true",
                   help="Re-extract layers even if .npz already exists")
    return p.parse_args()


def load_model(device: str):
    ckpt   = torch.load(CHECKPOINT, map_location="cpu")
    config = AeneasConfig.from_dict(ckpt["config"])
    model  = AeneasForMLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    print(f"  d_model={config.d_model}, num_layers={config.num_layers}, "
          f"epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    return model, config


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over non-PAD positions. (B,L,D) × (B,L) → (B,D)."""
    mask_f = mask.float().unsqueeze(-1)
    return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)


def extract_layers(
    texts: list,
    sign_to_id: dict,
    model: AeneasForMLM,
    layers_to_extract: list,
    device: str,
) -> dict:
    """Return {layer_idx: np.ndarray (N, d_model)} mean-pooled hidden states."""
    all_ids, all_mask = [], []
    for text in texts:
        ids, mask = tokenize_text(text, sign_to_id, max_length=MAX_LENGTH)
        all_ids.append(ids)
        all_mask.append(mask)

    ids_t  = torch.tensor(all_ids,  dtype=torch.long)
    mask_t = torch.tensor(all_mask, dtype=torch.long)

    accum = {layer: [] for layer in layers_to_extract}

    for start in range(0, len(texts), BATCH_SIZE):
        b_ids  = ids_t[start : start + BATCH_SIZE].to(device)
        b_mask = mask_t[start : start + BATCH_SIZE].to(device)

        with torch.no_grad():
            out = model(
                b_ids, b_mask,
                output_hidden_states=True,
                hidden_states_layers=layers_to_extract,
            )

        for layer, hs in out["hidden_states"].items():
            accum[layer].append(mean_pool(hs, b_mask).cpu().float().numpy())

        done = min(start + BATCH_SIZE, len(texts))
        print(f"  {done}/{len(texts)}", end="\r", flush=True)

    print()
    return {layer: np.concatenate(arrs) for layer, arrs in accum.items()}


def main():
    args       = parse_args()
    out_dir    = Path(args.output_dir)
    in_parquet = Path(args.input_parquet)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:     {device}")
    print(f"Input:      {in_parquet}")
    print(f"Text col:   {args.text_col}")
    print(f"Output dir: {out_dir}")
    print(f"Force:      {args.force}")

    # 1. Vocabulary
    print("\n[1/4] Loading vocabulary...")
    sign_to_id, _ = load_vocabulary(str(VOCAB))
    print(f"  vocab_size={len(sign_to_id):,}")

    # 2. Model
    print("\n[2/4] Loading model checkpoint...")
    model, config = load_model(device)
    all_layers = list(range(config.num_layers + 1))  # 0..16 inclusive (17 total)
    print(f"  Layers available: {all_layers[0]}..{all_layers[-1]} ({len(all_layers)} total)")

    # Determine which layers still need extraction
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        todo = all_layers
    else:
        todo = [l for l in all_layers
                if not (out_dir / f"layer_{l:02d}.npz").exists()]
        skipped = [l for l in all_layers if l not in todo]
        if skipped:
            print(f"  Skipping {len(skipped)} already-extracted layers "
                  f"(--force to redo): {skipped}")

    if not todo:
        print("  All layers already extracted. Nothing to do.")
        return
    print(f"  Will extract {len(todo)} layers: {todo}")

    # 3. Corpus
    print("\n[3/4] Loading corpus...")
    df = pd.read_parquet(in_parquet)
    assert args.text_col in df.columns, (
        f"Column '{args.text_col}' not found. Available: {df.columns.tolist()}"
    )
    texts = df[args.text_col].str.replace("-", " ", regex=False).tolist()
    fragment_ids = (df["fragment_id"].astype(str).tolist()
                    if "fragment_id" in df.columns else [str(i) for i in range(len(df))])
    avg_signs = sum(len(t.split()) for t in texts) / max(len(texts), 1)
    print(f"  {len(texts)} fragments, avg {avg_signs:.0f} signs/fragment")

    # 4. Extraction
    print(f"\n[4/4] Extracting hidden states at layers {todo}...")
    hidden = extract_layers(texts, sign_to_id, model, todo, device)

    print("\nSaving .npz files...")
    for layer in sorted(hidden.keys()):
        arr = hidden[layer]
        assert arr.shape == (len(df), config.d_model), (
            f"Layer {layer}: expected ({len(df)}, {config.d_model}), got {arr.shape}"
        )
        assert not np.isnan(arr).any(), f"Layer {layer}: NaN detected"
        out_path = out_dir / f"layer_{layer:02d}.npz"
        np.savez_compressed(out_path, activations=arr)
        print(f"  L{layer:02d}: {arr.shape}  →  {out_path.name}")

    # Write / merge metadata
    metadata_path = out_dir / "metadata.json"
    if metadata_path.exists() and not args.force:
        with open(metadata_path) as f:
            existing_meta = json.load(f)
        prev_extracted = set(existing_meta.get("layers_extracted", []))
    else:
        existing_meta  = {}
        prev_extracted = set()

    layers_now = prev_extracted | set(hidden.keys())
    metadata = {
        **existing_meta,
        "model_id":        "AeneasForMLM",
        "checkpoint":      str(CHECKPOINT),
        "text_col":        args.text_col,
        "n_texts":         len(df),
        "n_layers_total":  len(all_layers),
        "hidden_dim":      int(config.d_model),
        "layers_extracted": sorted(layers_now),
        "fragment_ids":    fragment_ids,
        "timestamp":       datetime.now().isoformat(),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nMetadata saved → {metadata_path}")
    print(f"  layers_extracted = {metadata['layers_extracted']}")
    print("✅ Done")


if __name__ == "__main__":
    main()
