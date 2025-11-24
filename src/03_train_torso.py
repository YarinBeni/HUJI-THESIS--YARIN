#!/usr/bin/env python3
"""Train the custom rotary transformer for restoration."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk

from modeling_restoration import RestorationConfig, RestorationModel

import time
import psutil


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def compute_vocab_size(dataset) -> int:
    max_id = 0
    for split in dataset.keys():
        inputs = np.array(dataset[split]["input_ids"], dtype=np.int64)
        max_id = max(max_id, int(inputs.max()))
        label_values = np.array(dataset[split]["labels"], dtype=np.int64)
        label_values = label_values[label_values != -100]
        if label_values.size:
            max_id = max(max_id, int(label_values.max()))
    return max_id + 1


def maybe_slice(ds, max_examples):
    if max_examples and max_examples < len(ds):
        return ds.select(range(max_examples))
    return ds


def main():
    print("DEBUG: main() function started")
    start_time = time.time()

    def log_memory(device):
        """Log current memory usage."""
        if device == "mps":
            import torch.mps
            mps_mem = torch.mps.current_allocated_memory() / (1024**3)  # GB
            print(f"   MPS Memory: {mps_mem:.2f} GB allocated")
        elif device.startswith("cuda"):
            mem_allocated = torch.cuda.memory_allocated(device) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(device) / (1024**3)
            print(f"   GPU Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
        # System memory for all
        vm = psutil.virtual_memory()
        print(f"   System Memory: {vm.percent}% used ({vm.available / (1024**3):.2f} GB available)")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    args = parser.parse_args()

    print("🚀 Starting model training...")
    print(f"   Dataset: {args.dataset}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Device: {args.device}")
    print(f"   Max train examples: {args.max_train_examples or 'None (full dataset)'}")
    print(f"   Max eval examples: {args.max_eval_examples or 'None (full dataset)'}")

    load_start = time.time()
    dataset = load_from_disk(str(args.dataset))
    vocab_size = compute_vocab_size(dataset)
    load_time = time.time() - load_start
    print(f"   Dataset loaded in {load_time:.2f}s")
    print(f"   Vocab size: {vocab_size}")

    config = RestorationConfig(vocab_size=vocab_size)
    model = RestorationModel(config).to(args.device)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"   Model initialized: {model_params:,} parameters")
    log_memory(args.device)

    original_train_size = len(dataset["train"])
    original_val_size = len(dataset["validation"])
    dataset["train"] = maybe_slice(dataset["train"], args.max_train_examples)
    dataset["validation"] = maybe_slice(dataset["validation"], args.max_eval_examples)
    train_size = len(dataset["train"])
    val_size = len(dataset["validation"])
    print(f"   Dataset sizes: Train={train_size}/{original_train_size}, Val={val_size}/{original_val_size}")
    if args.max_train_examples or args.max_eval_examples:
        print("   ⚠️  WARNING: Using subset of data—training may be fast but undertrained!")

    dataset.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset["validation"], batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"   DataLoaders: Train batches={len(train_loader)}, Val batches={len(val_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"   Optimizer: AdamW (lr={args.lr})")

    best_val = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("   Memory before training loop:")
    log_memory(args.device)

    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        step_count = 0

        for step, batch in enumerate(train_loader, 1):
            step_start = time.time()
            for key in batch:
                batch[key] = batch[key].to(args.device)
            loss, _ = model.compute_loss(**batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            step_count += 1

            # Log every 10 steps or if small dataset
            if step % 10 == 0 or train_size < 500:
                step_time = time.time() - step_start
                print(f"     Step {step}/{len(train_loader)}: loss={loss.item():.4f} ({step_time:.2f}s/step)")
                if step % 20 == 0:  # Log memory every 20 steps
                    print("     Memory at step:")
                    log_memory(args.device)

        avg_loss = total_loss / max(1, step_count)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                for key in batch:
                    batch[key] = batch[key].to(args.device)
                loss, _ = model.compute_loss(**batch)
                val_loss += loss.item()
                val_steps += 1
        val_loss /= max(1, val_steps)

        print(f"✅ Epoch {epoch}/{args.epochs}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f} ({epoch_time:.2f}s)")
        print(f"   Memory: {psutil.virtual_memory().percent}% used | GPU: {torch.cuda.memory_allocated(args.device) if torch.cuda.is_available() else 'N/A'}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "vocab_size": vocab_size,
            }, args.output_dir / "best_model.pt")
            print("   💾 Saved new best model")

    total_training_time = time.time() - start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"\n📊 Training Summary:")
    print(f"   Total time: {total_training_time:.2f}s ({total_training_time/3600:.2f}h)")
    print(f"   Avg epoch time: {avg_epoch_time:.2f}s")
    print(f"   Final best val loss: {best_val:.4f}")
    print(f"   Data used: {'Subset' if args.max_train_examples else 'Full'} training set")
    print("   Final Memory Usage:")
    log_memory(args.device)
    if args.max_train_examples:
        print("   🔍 DIAGNOSIS: Training is fast because you're using only a tiny subset! Remove --max_train_examples for full training.")

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
    }, args.output_dir / "last_model.pt")
    print(f"   💾 Final model saved to {args.output_dir}/last_model.pt")


if __name__ == "__main__":
    main()
