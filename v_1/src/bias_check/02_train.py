#!/usr/bin/env python3
"""
Step 2: Train all 8 model variants for the bias check.

Loads TF-IDF features → trains each model → saves best checkpoint + training history.

Usage:
    python 02_train.py
    python 02_train.py --debug              # fast run (10%% of data, 5 epochs)
    python 02_train.py --models mlp_1layer mlp_3layer
    python 02_train.py --device cpu
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import load_npz
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FEATURES_DIR,
    MODELS_DIR,
    METRICS_DIR,
    TRAINING_HISTORY_JSON,
    MODEL_VARIANTS,
    NUM_CLASSES,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_EPOCHS,
    EARLY_STOP_PATIENCE,
    LR_SCHEDULER_PATIENCE,
)
from models import build_model, count_parameters


def load_split(split: str, debug: bool = False):
    """Load sparse TF-IDF matrix + labels, return dense tensors."""
    X = load_npz(FEATURES_DIR / f"{split}.npz")
    y = np.load(FEATURES_DIR / f"y_{split}.npy")
    if debug:
        n = max(64, len(y) // 5)
        X = X[:n]
        y = y[:n]
    X_dense = torch.tensor(X.toarray(), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_dense, y_tensor


def compute_class_weights(y_train: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency class weights to handle imbalance."""
    counts = torch.bincount(y_train, minlength=NUM_CLASSES).float()
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * NUM_CLASSES
    return weights


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def train_model(name, n_attn, n_mlp, train_loader, val_loader,
                criterion, device, max_epochs):
    """Train a single model variant, return history dict."""
    print(f"\n  Building {name} (attn_blocks={n_attn}, mlp_layers={n_mlp})...")
    model = build_model(name).to(device)
    n_params = count_parameters(model)
    print(f"    Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_SCHEDULER_PATIENCE, factor=0.5
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    best_val_loss = float("inf")
    best_val_acc  = 0.0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_val_acc  = va_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0 or patience_counter == 0:
            elapsed = time.time() - t0
            print(f"    Epoch {epoch:3d}/{max_epochs}  "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
                  f"val_loss={va_loss:.4f}  val_acc={va_acc:.3f}  "
                  f"({elapsed:.1f}s)")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"    Early stop at epoch {epoch} (best epoch={best_epoch})")
            break

    print(f"    Best → val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.3f}  "
          f"(epoch {best_epoch})")

    return {
        "name": name,
        "n_params": n_params,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train bias check classifiers")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of model names to train (default: all)")
    parser.add_argument("--device", default=None,
                        help="Device: cuda / cpu (default: auto-detect)")
    parser.add_argument("--debug", action="store_true",
                        help="Fast run: 5 epochs, small data subset")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 2: Train Bias Check Models")
    print("=" * 60)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Check features exist
    for split in ["train", "val"]:
        npz = FEATURES_DIR / f"{split}.npz"
        if not npz.exists():
            print(f"\nError: {npz} not found. Run 01_featurize.py first.")
            sys.exit(1)

    # Load data
    print("\nLoading features...")
    X_train, y_train = load_split("train", debug=args.debug)
    X_val,   y_val   = load_split("val",   debug=args.debug)
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}")

    # Select variants
    variants = MODEL_VARIANTS
    if args.models:
        variants = [v for v in MODEL_VARIANTS if v[0] in args.models]
        if not variants:
            print(f"Error: no matching models in {args.models}")
            sys.exit(1)

    max_epochs = 5 if args.debug else MAX_EPOCHS

    # Train
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Build data loaders and loss once (shared across all models)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset   = TensorDataset(X_val,   y_val)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    class_weights = compute_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    all_history = {}
    print(f"\nTraining {len(variants)} model(s)...")

    for name, n_attn, n_mlp in variants:
        result = train_model(
            name, n_attn, n_mlp,
            train_loader, val_loader, criterion,
            device=device,
            max_epochs=max_epochs,
        )
        all_history[name] = result

    # Save training history
    print(f"\nSaving training history to {TRAINING_HISTORY_JSON}...")
    with open(TRAINING_HISTORY_JSON, "w") as f:
        json.dump(all_history, f, indent=2)

    # Summary table
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"{'Model':<18} {'Params':>10} {'Best Val Loss':>14} {'Best Val Acc':>13} {'Epochs':>7}")
    print("-" * 65)
    for name, result in all_history.items():
        print(f"{name:<18} {result['n_params']:>10,} "
              f"{result['best_val_loss']:>14.4f} "
              f"{result['best_val_acc']:>12.3f} "
              f"{result['total_epochs']:>7}")

    print(f"\nCheckpoints saved to {MODELS_DIR}/")
    print("=" * 60)
    print("Done! Run 03_evaluate.py next.")
    print("=" * 60)


if __name__ == "__main__":
    main()
