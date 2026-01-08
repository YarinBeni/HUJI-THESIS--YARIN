#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenient training launcher for Akkadian MLM model.

This script sets up paths and launches training with optimized settings
for Apple Silicon MPS or CUDA GPUs.

Usage:
    python3 v_1/run_training.py                    # Default settings
    python3 v_1/run_training.py --epochs 20        # Custom epochs
    python3 v_1/run_training.py --fast             # Quick test (1 epoch)
    python3 v_1/run_training.py --benchmark        # Find optimal batch size
"""

import argparse
import os
import sys
from pathlib import Path

# Add the training module to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / "src" / "training" / "baseline"))

import torch
import math
from tqdm import tqdm


def compute_mrr(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for MLM predictions.

    For each masked position, finds the rank of the correct token in the
    model's predictions and computes 1/rank. MRR is the mean of these values.

    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        labels: Ground truth labels [batch_size, seq_len], -100 for non-masked positions

    Returns:
        MRR score (float between 0 and 1, higher is better)
    """
    # Only compute for masked positions (labels != -100)
    mask = labels != -100

    if mask.sum() == 0:
        return 0.0

    # Get predictions for masked positions
    masked_logits = logits[mask]  # [num_masked, vocab_size]
    masked_labels = labels[mask]  # [num_masked]

    # Sort logits in descending order to get rankings
    # argsort of argsort gives ranks (0-indexed)
    sorted_indices = masked_logits.argsort(dim=-1, descending=True)

    # Find rank of correct token for each masked position
    # ranks are 1-indexed for MRR calculation
    ranks = torch.zeros(masked_labels.size(0), device=logits.device)
    for i in range(masked_labels.size(0)):
        correct_token = masked_labels[i]
        # Find position of correct token in sorted predictions
        rank = (sorted_indices[i] == correct_token).nonzero(as_tuple=True)[0].item() + 1
        ranks[i] = rank

    # MRR = mean of 1/rank
    reciprocal_ranks = 1.0 / ranks
    mrr = reciprocal_ranks.mean().item()

    return mrr


def compute_mrr_batch(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Faster batched MRR computation using vectorized operations.
    """
    mask = labels != -100

    if mask.sum() == 0:
        return 0.0

    # Get predictions for masked positions
    masked_logits = logits[mask]  # [num_masked, vocab_size]
    masked_labels = labels[mask]  # [num_masked]

    # Get the logit value of the correct token for each masked position
    correct_logits = masked_logits.gather(dim=-1, index=masked_labels.unsqueeze(-1)).squeeze(-1)

    # Count how many tokens have higher logits than the correct one (this gives rank-1)
    # rank = 1 + number of tokens with higher logit
    ranks = (masked_logits > correct_logits.unsqueeze(-1)).sum(dim=-1) + 1

    # MRR = mean of 1/rank
    mrr = (1.0 / ranks.float()).mean().item()

    return mrr


def get_optimal_settings():
    """Get optimal batch size and settings for available hardware."""

    settings = {
        "batch_size": 8,
        "num_workers": 0,
        "max_length": 512,
        "gradient_accumulation": 1,
    }

    if torch.cuda.is_available():
        # CUDA GPU - check memory
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)} ({gpu_mem_gb:.1f} GB)")

        if gpu_mem_gb >= 24:
            settings["batch_size"] = 64
            settings["max_length"] = 768
        elif gpu_mem_gb >= 16:
            settings["batch_size"] = 32
            settings["max_length"] = 512
        elif gpu_mem_gb >= 8:
            settings["batch_size"] = 16
            settings["max_length"] = 512
        else:
            settings["batch_size"] = 8
            settings["max_length"] = 256

        settings["num_workers"] = 4

    elif torch.backends.mps.is_available():
        # Apple Silicon MPS
        print("Apple Silicon MPS detected")
        # MPS works best with moderate batch sizes
        # Too large causes memory issues, too small is inefficient
        settings["batch_size"] = 16  # Good balance for M1/M2/M3
        settings["max_length"] = 512
        settings["num_workers"] = 0  # MPS doesn't benefit from workers

    else:
        # CPU only
        print("No GPU detected, using CPU (will be slow)")
        settings["batch_size"] = 4
        settings["max_length"] = 256
        settings["num_workers"] = 2

    return settings


def benchmark_batch_size(model, device, sign_to_id, max_length=512):
    """Find the optimal batch size by testing increasingly larger batches."""
    from data_utils import collate_fn

    print("\n" + "=" * 60)
    print("BENCHMARKING BATCH SIZE")
    print("=" * 60)
    print("Testing batch sizes to find the largest that fits in memory...")
    print("(This will take a minute)\n")

    # Create dummy data
    vocab_size = len(sign_to_id)
    batch_sizes_to_test = [8, 16, 24, 32, 48, 64, 96, 128]

    best_batch_size = 8
    best_throughput = 0
    results = []

    for batch_size in batch_sizes_to_test:
        try:
            # Create dummy batch
            dummy_ids = torch.randint(0, vocab_size, (batch_size, max_length)).to(device)
            dummy_mask = torch.ones(batch_size, max_length).to(device)
            dummy_labels = torch.randint(0, vocab_size, (batch_size, max_length)).to(device)

            # Warm up
            model.train()
            for _ in range(2):
                outputs = model(dummy_ids, dummy_mask, dummy_labels)
                outputs['loss'].backward()
                model.zero_grad()

            # Time it
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()

            import time
            start = time.time()
            n_iters = 5
            for _ in range(n_iters):
                outputs = model(dummy_ids, dummy_mask, dummy_labels)
                outputs['loss'].backward()
                model.zero_grad()

            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()

            elapsed = time.time() - start
            samples_per_sec = (batch_size * n_iters) / elapsed

            results.append({
                'batch_size': batch_size,
                'samples_per_sec': samples_per_sec,
                'time_per_batch': elapsed / n_iters,
            })

            print(f"  Batch size {batch_size:3d}: {samples_per_sec:.1f} samples/sec, {elapsed/n_iters*1000:.0f}ms/batch ✓")

            if samples_per_sec > best_throughput:
                best_throughput = samples_per_sec
                best_batch_size = batch_size

            # Clear memory
            del dummy_ids, dummy_mask, dummy_labels, outputs
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "MPS" in str(e):
                print(f"  Batch size {batch_size:3d}: OUT OF MEMORY ✗")
                if device == "mps":
                    torch.mps.empty_cache()
                elif device == "cuda":
                    torch.cuda.empty_cache()
                break
            else:
                raise e

    print(f"\n  → Optimal batch size: {best_batch_size} ({best_throughput:.1f} samples/sec)")
    print("=" * 60)

    return best_batch_size, results


def main():
    parser = argparse.ArgumentParser(description="Launch Akkadian MLM training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--fast", action="store_true", help="Quick test run (1 epoch)")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark to find optimal batch size")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max_length", type=int, default=None, help="Override max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Get optimal settings for hardware
    settings = get_optimal_settings()

    # Apply overrides
    if args.batch_size:
        settings["batch_size"] = args.batch_size
    if args.max_length:
        settings["max_length"] = args.max_length
    if args.fast:
        args.epochs = 1
        # Don't reduce batch size for --fast anymore, use optimal settings

    print(f"\nTraining settings:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {settings['batch_size']}")
    print(f"  Max length: {settings['max_length']}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Num workers: {settings['num_workers']}")
    print()

    # Set up paths
    data_dir = script_dir / "data" / "prepared"
    output_dir = script_dir / "models" / "baseline"

    # Check if data is prepared
    if not (data_dir / "vocab.json").exists():
        print("Data not prepared! Running data preparation first...")
        os.chdir(script_dir / "src" / "training" / "baseline")
        os.system(f"python 01_prepare_data.py --data_dir {script_dir / 'data' / 'processed' / 'unified'} --output_dir {data_dir}")
        os.chdir(script_dir.parent)

    # Import and run training
    from data_utils import AkkadianMLMDataset, load_vocabulary, collate_fn
    from model import create_model, AeneasConfig

    import pandas as pd
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import time
    import json
    from datetime import datetime

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    print("\nLoading vocabulary...")
    sign_to_id, id_to_sign = load_vocabulary(data_dir / "vocab.json")
    vocab_size = len(sign_to_id)
    print(f"  Vocabulary size: {vocab_size:,}")

    # Load data
    print("\nLoading data...")
    train_df = pd.read_parquet(data_dir / "train_fragments.parquet")
    val_df = pd.read_parquet(data_dir / "val_fragments.parquet")
    eval_df = pd.read_parquet(data_dir / "eval_subset.parquet")

    print(f"  Train: {len(train_df):,} fragments")
    print(f"  Val: {len(val_df):,} fragments")
    print(f"  Eval subset: {len(eval_df):,} fragments")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = AkkadianMLMDataset(
        train_df, sign_to_id, max_length=settings["max_length"], seed=args.seed
    )
    val_dataset = AkkadianMLMDataset(
        val_df, sign_to_id, max_length=settings["max_length"], seed=args.seed
    )
    eval_dataset = AkkadianMLMDataset(
        eval_df, sign_to_id, max_length=settings["max_length"], seed=args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=settings["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=settings["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=settings["num_workers"],
    )

    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches: {len(val_loader):,}")

    # Create model
    print("\nCreating model...")
    model = create_model(vocab_size=vocab_size)
    model = model.to(device)
    print(f"  Parameters: {model.get_num_params():,}")

    # Run benchmark if requested
    if args.benchmark:
        optimal_batch, benchmark_results = benchmark_batch_size(
            model, device, sign_to_id, max_length=settings["max_length"]
        )
        settings["batch_size"] = optimal_batch
        # Recreate data loaders with optimal batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=settings["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=settings["num_workers"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=settings["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=settings["num_workers"],
        )
        print(f"\nUsing benchmarked batch size: {settings['batch_size']}")
        print(f"  Train batches: {len(train_loader):,}")
        print(f"  Val batches: {len(val_loader):,}")

    # Save initial weights
    print("\nSaving initial weights...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config.to_dict(),
    }, output_dir / "baseline_init.pt")

    # Extract pre-training hidden states
    print("\nExtracting pre-training hidden states...")
    ANALYSIS_LAYERS = [0, 4, 8, 12, 16]

    model.eval()
    eval_loader = DataLoader(eval_dataset, batch_size=settings["batch_size"], shuffle=False, collate_fn=collate_fn)

    pre_hidden_states = {layer: [] for layer in ANALYSIS_LAYERS}
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Extracting pre-train states"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask, output_hidden_states=True, hidden_states_layers=ANALYSIS_LAYERS)
            for layer_idx, hs in outputs['hidden_states'].items():
                pre_hidden_states[layer_idx].append(hs.cpu())

    # Save pre-training artifacts
    torch.save(model.get_embedding_weights().cpu(), output_dir / "baseline_pre_embeddings.pt")
    for layer_idx, tensors in pre_hidden_states.items():
        torch.save(torch.cat(tensors, dim=0), output_dir / f"baseline_pre_hidden_states_layer_{layer_idx}.pt")
    print(f"  Saved pre-training embeddings and hidden states for layers {ANALYSIS_LAYERS}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    best_val_loss = float('inf')
    training_stats = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        total_loss = 0.0
        num_batches = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs} [Train]",
            leave=True,
            dynamic_ncols=True,
        )

        for batch in train_pbar:
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

            # Update progress bar with current loss
            train_pbar.set_postfix({
                'loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        train_loss = total_loss / num_batches

        # Validate with Perplexity and MRR
        model.eval()
        val_loss = 0.0
        val_mrr_sum = 0.0
        val_batches = 0
        total_masked_tokens = 0

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{args.epochs} [Val]",
            leave=True,
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask, labels)

                batch_loss = outputs['loss'].item()
                val_loss += batch_loss
                val_batches += 1

                # Compute MRR for this batch
                logits = outputs['logits']
                batch_mrr = compute_mrr_batch(logits, labels)
                num_masked = (labels != -100).sum().item()
                val_mrr_sum += batch_mrr * num_masked
                total_masked_tokens += num_masked

                # Update progress bar
                current_ppl = math.exp(val_loss / val_batches)
                current_mrr = val_mrr_sum / total_masked_tokens if total_masked_tokens > 0 else 0
                val_pbar.set_postfix({
                    'loss': f"{val_loss / val_batches:.4f}",
                    'ppl': f"{current_ppl:.2f}",
                    'mrr': f"{current_mrr:.4f}"
                })

        val_loss /= val_batches
        val_perplexity = math.exp(val_loss)
        val_mrr = val_mrr_sum / total_masked_tokens if total_masked_tokens > 0 else 0

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Compute train perplexity
        train_perplexity = math.exp(train_loss)

        print(f"\n→ Epoch {epoch}/{args.epochs}:")
        print(f"    Train: loss={train_loss:.4f}, perplexity={train_perplexity:.2f}")
        print(f"    Val:   loss={val_loss:.4f}, perplexity={val_perplexity:.2f}, MRR={val_mrr:.4f}")
        print(f"    Time: {epoch_time:.1f}s")

        training_stats.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_perplexity': train_perplexity,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
            'val_mrr': val_mrr,
            'lr': scheduler.get_last_lr()[0],
            'time': epoch_time,
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'config': model.config.to_dict(),
            }, output_dir / "baseline_best.pt")
            print("  Saved new best model!")

    # Save last model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.epochs,
        'val_loss': val_loss,
        'config': model.config.to_dict(),
    }, output_dir / "baseline_last.pt")

    # Extract post-training hidden states
    print("\nExtracting post-training hidden states...")
    model.eval()
    post_hidden_states = {layer: [] for layer in ANALYSIS_LAYERS}
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Extracting post-train states"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask, output_hidden_states=True, hidden_states_layers=ANALYSIS_LAYERS)
            for layer_idx, hs in outputs['hidden_states'].items():
                post_hidden_states[layer_idx].append(hs.cpu())

    # Save post-training artifacts
    torch.save(model.get_embedding_weights().cpu(), output_dir / "baseline_post_embeddings.pt")
    for layer_idx, tensors in post_hidden_states.items():
        torch.save(torch.cat(tensors, dim=0), output_dir / f"baseline_post_hidden_states_layer_{layer_idx}.pt")

    # Get best metrics from training stats
    best_epoch_stats = min(training_stats, key=lambda x: x['val_loss'])
    best_val_perplexity = best_epoch_stats['val_perplexity']
    best_val_mrr = best_epoch_stats['val_mrr']

    # Save training stats
    with open(output_dir / "training_stats.json", "w") as f:
        json.dump({
            'stats': training_stats,
            'config': model.config.to_dict(),
            'best_val_loss': best_val_loss,
            'best_val_perplexity': best_val_perplexity,
            'best_val_mrr': best_val_mrr,
            'settings': settings,
            'args': {
                'epochs': args.epochs,
                'lr': args.lr,
                'seed': args.seed,
            },
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest validation metrics (epoch {best_epoch_stats['epoch']}):")
    print(f"  Loss:       {best_val_loss:.4f}")
    print(f"  Perplexity: {best_val_perplexity:.2f}")
    print(f"  MRR:        {best_val_mrr:.4f}")
    print(f"\nOutput files in {output_dir}:")
    for f in sorted(output_dir.glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
