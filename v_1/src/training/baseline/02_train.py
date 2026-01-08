#!/usr/bin/env python3
"""
Training script for Akkadian MLM model.

This script:
1. Loads prepared data (fragment texts + vocabulary)
2. Creates model and saves initial weights
3. Trains with validation
4. Saves best and last checkpoints
5. Extracts embeddings and hidden states pre/post training

Usage:
    cd v_1/src/training/baseline
    python 02_train.py --data_dir ../../../data/prepared --output_dir ../../../models/baseline

    # Or from repo root:
    python v_1/src/training/baseline/02_train.py
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_utils import (
    AkkadianMLMDataset,
    load_vocabulary,
    collate_fn,
)
from model import AeneasConfig, AeneasForMLM, create_model


# Analysis layers for SAE (every 4th layer from 0 to 16)
ANALYSIS_LAYERS = [0, 4, 8, 12, 16]


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_checkpoint(
    model: AeneasForMLM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: Path,
    config: AeneasConfig,
):
    """Save model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'config': config.to_dict(),
    }, path)


def extract_embeddings_and_hidden_states(
    model: AeneasForMLM,
    eval_dataset: AkkadianMLMDataset,
    device: str,
    batch_size: int = 16,
) -> Dict:
    """
    Extract embeddings and hidden states on the eval subset.

    Returns:
        Dict with:
            - 'embedding_matrix': [vocab_size, d_model]
            - 'hidden_states': Dict[layer_idx -> [num_tokens, d_model]]
            - 'token_ids': [num_tokens]
            - 'attention_mask': [num_tokens]
    """
    model.eval()

    # Get embedding matrix
    embedding_matrix = model.get_embedding_weights().cpu()

    # Collect hidden states
    all_hidden_states = {layer: [] for layer in ANALYSIS_LAYERS}
    all_token_ids = []
    all_attention_masks = []

    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids,
                attention_mask,
                output_hidden_states=True,
                hidden_states_layers=ANALYSIS_LAYERS
            )

            # Store hidden states (flatten batch and seq dimensions)
            for layer_idx, hs in outputs['hidden_states'].items():
                # hs: [batch, seq_len, d_model]
                # Flatten to [batch * seq_len, d_model]
                all_hidden_states[layer_idx].append(hs.cpu())

            all_token_ids.append(input_ids.cpu())
            all_attention_masks.append(attention_mask.cpu())

    # Concatenate all batches
    result = {
        'embedding_matrix': embedding_matrix,
        'hidden_states': {
            layer: torch.cat(tensors, dim=0)
            for layer, tensors in all_hidden_states.items()
        },
        'token_ids': torch.cat(all_token_ids, dim=0),
        'attention_mask': torch.cat(all_attention_masks, dim=0),
    }

    return result


def save_analysis_artifacts(
    artifacts: Dict,
    output_dir: Path,
    prefix: str,
):
    """Save embeddings and hidden states to disk."""
    # Save embedding matrix
    torch.save(
        artifacts['embedding_matrix'],
        output_dir / f"{prefix}_embeddings.pt"
    )

    # Save hidden states for each layer
    for layer_idx, hs in artifacts['hidden_states'].items():
        torch.save(
            hs,
            output_dir / f"{prefix}_hidden_states_layer_{layer_idx}.pt"
        )

    # Save token IDs and masks for reference
    torch.save({
        'token_ids': artifacts['token_ids'],
        'attention_mask': artifacts['attention_mask'],
    }, output_dir / f"{prefix}_tokens.pt")


def train_epoch(
    model: AeneasForMLM,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    log_interval: int = 100,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                  f"Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

    return total_loss / num_batches


def validate(
    model: AeneasForMLM,
    val_loader: DataLoader,
    device: str,
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels)
            total_loss += outputs['loss'].item()
            num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train Akkadian MLM model")
    parser.add_argument("--data_dir", type=Path, default=Path("v_1/data/prepared"))
    parser.add_argument("--output_dir", type=Path, default=Path("v_1/models/baseline"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Device
    device = args.device or get_device()
    print(f"Using device: {device}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AKKADIAN MLM TRAINING")
    print("=" * 60)
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max length: {args.max_length}")

    # Load vocabulary
    print("\n📚 Loading vocabulary...")
    sign_to_id, id_to_sign = load_vocabulary(args.data_dir / "vocab.json")
    vocab_size = len(sign_to_id)
    print(f"   Vocabulary size: {vocab_size:,}")

    # Load data
    print("\n📂 Loading data...")
    train_df = pd.read_parquet(args.data_dir / "train_fragments.parquet")
    val_df = pd.read_parquet(args.data_dir / "val_fragments.parquet")
    eval_df = pd.read_parquet(args.data_dir / "eval_subset.parquet")

    print(f"   Train fragments: {len(train_df):,}")
    print(f"   Val fragments: {len(val_df):,}")
    print(f"   Eval subset: {len(eval_df):,}")

    # Create datasets
    print("\n🔧 Creating datasets...")
    train_dataset = AkkadianMLMDataset(
        train_df, sign_to_id, max_length=args.max_length, seed=args.seed
    )
    val_dataset = AkkadianMLMDataset(
        val_df, sign_to_id, max_length=args.max_length, seed=args.seed
    )
    eval_dataset = AkkadianMLMDataset(
        eval_df, sign_to_id, max_length=args.max_length, seed=args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    print(f"   Train batches: {len(train_loader):,}")
    print(f"   Val batches: {len(val_loader):,}")

    # Create model
    print("\n🏗️ Creating model...")
    model = create_model(vocab_size=vocab_size)
    model = model.to(device)
    print(f"   Parameters: {model.get_num_params():,}")
    print(f"   Config: d_model={model.config.d_model}, layers={model.config.num_layers}")

    # Save initial weights
    print("\n💾 Saving initial weights...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config.to_dict(),
    }, args.output_dir / "baseline_init.pt")

    # Extract pre-training embeddings and hidden states
    print("\n🔬 Extracting pre-training analysis artifacts...")
    pre_artifacts = extract_embeddings_and_hidden_states(
        model, eval_dataset, device, batch_size=args.batch_size
    )
    save_analysis_artifacts(pre_artifacts, args.output_dir, "baseline_pre")
    print(f"   Saved embeddings: {pre_artifacts['embedding_matrix'].shape}")
    print(f"   Saved hidden states for layers: {list(pre_artifacts['hidden_states'].keys())}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\n🚀 Starting training...")
    best_val_loss = float('inf')
    training_stats = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, args.log_interval
        )

        # Validate
        val_loss = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        print(f"\n📊 Epoch {epoch}/{args.epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"time={epoch_time:.1f}s")

        training_stats.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': scheduler.get_last_lr()[0],
            'time': epoch_time,
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                args.output_dir / "baseline_best.pt",
                model.config
            )
            print("   💾 Saved new best model")

    # Save last model
    save_checkpoint(
        model, optimizer, args.epochs, val_loss,
        args.output_dir / "baseline_last.pt",
        model.config
    )

    # Extract post-training embeddings and hidden states
    print("\n🔬 Extracting post-training analysis artifacts...")
    post_artifacts = extract_embeddings_and_hidden_states(
        model, eval_dataset, device, batch_size=args.batch_size
    )
    save_analysis_artifacts(post_artifacts, args.output_dir, "baseline_post")

    # Save training stats
    with open(args.output_dir / "training_stats.json", "w") as f:
        json.dump({
            'stats': training_stats,
            'config': model.config.to_dict(),
            'best_val_loss': best_val_loss,
            'args': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'max_length': args.max_length,
                'seed': args.seed,
            },
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"\nOutput files:")
    for f in sorted(args.output_dir.glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   {f.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
