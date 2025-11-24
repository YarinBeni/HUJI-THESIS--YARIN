#!/usr/bin/env python3
"""Memory-efficient SAE training using ORIGINAL RestorationModel (not TransformerLens).

This version uses the lightweight original PyTorch model to avoid memory issues.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))
from modeling_restoration import RestorationConfig, RestorationModel


def collect_activations_from_layer(
    model: RestorationModel,
    dataset,
    layer_idx: int,
    max_samples: int = 100,
    device: str = "cpu"
) -> torch.Tensor:
    """Extract activations from specific layer using hooks on original model.
    
    Args:
        model: Original RestorationModel
        dataset: Dataset with input_ids
        layer_idx: Layer to extract from (0-15)
        max_samples: Max samples to process
        device: Device
        
    Returns:
        Activations tensor [num_tokens, d_model]
    """
    print(f"\n📊 Collecting activations from layer {layer_idx}...")
    
    # Limit dataset
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    print(f"   Using {max_samples} samples")
    
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Hook to capture layer output
    captured = []
    
    def hook_fn(module, input, output):
        # output is the residual after this layer
        captured.append(output.detach().cpu())
    
    # Register hook on target layer
    hook = model.layers[layer_idx].register_forward_hook(hook_fn)
    
    model.eval()
    with torch.no_grad():
        for idx, item in enumerate(tqdm(dataset, desc="Extracting")):
            # Process ONE sample at a time (most memory efficient)
            input_ids = item["input_ids"].unsqueeze(0).to(device)  # [1, seq_len]
            
            # Forward pass triggers hook
            _ = model(input_ids)
            
            # Immediately move to CPU and clear
            if device in ["mps", "cuda"]:
                if device == "mps":
                    torch.mps.empty_cache()
                else:
                    torch.cuda.empty_cache()
    
    hook.remove()
    
    # Concatenate: [num_samples, seq_len, d_model] -> [num_samples * seq_len, d_model]
    all_acts = torch.cat(captured, dim=0)  # [num_samples, seq_len, d_model]
    all_acts_flat = all_acts.reshape(-1, all_acts.shape[-1])  # [N, d_model]
    
    print(f"   ✓ Shape: {all_acts_flat.shape}")
    print(f"   ✓ Memory: {all_acts_flat.element_size() * all_acts_flat.nelement() / 1024**2:.1f} MB")
    
    return all_acts_flat


def train_sae(
    activations: torch.Tensor,
    d_in: int,
    expansion_factor: int = 4,
    l1_coef: float = 1e-3,
    batch_size: int = 256,
    num_epochs: int = 50,
    lr: float = 1e-4,
    device: str = "cpu"
) -> dict:
    """Train sparse autoencoder.
    
    Returns:
        Dict with trained weights and losses
    """
    print(f"\n🚀 Training SAE...")
    print(f"   Samples: {activations.shape[0]:,}")
    print(f"   d_in: {d_in}, d_sae: {d_in * expansion_factor}")
    print(f"   Batch size: {batch_size}, Epochs: {num_epochs}")
    
    d_sae = d_in * expansion_factor
    
    # Initialize weights
    W_enc = nn.Parameter(torch.randn(d_in, d_sae, device=device) * 0.01)
    b_enc = nn.Parameter(torch.zeros(d_sae, device=device))
    W_dec = nn.Parameter(torch.randn(d_sae, d_in, device=device) * 0.01)
    b_dec = nn.Parameter(torch.zeros(d_in, device=device))
    
    # Normalize decoder
    with torch.no_grad():
        W_dec.data = W_dec.data / W_dec.data.norm(dim=1, keepdim=True)
    
    optimizer = torch.optim.Adam([W_enc, b_enc, W_dec, b_dec], lr=lr)
    
    # Training
    num_batches = len(activations) // batch_size
    all_losses = []
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Shuffle
        perm = torch.randperm(activations.shape[0])
        acts_shuffled = activations[perm]
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i in pbar:
            start = i * batch_size
            end = min(start + batch_size, len(acts_shuffled))
            batch = acts_shuffled[start:end].to(device)
            
            # Forward
            hidden = torch.relu(batch @ W_enc + b_enc)
            recon = hidden @ W_dec + b_dec
            
            # Loss
            recon_loss = (batch - recon).pow(2).mean()
            l1_loss = hidden.abs().mean()
            loss = recon_loss + l1_coef * l1_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W_enc, b_enc, W_dec, b_dec], 1.0)
            optimizer.step()
            
            # Renormalize decoder
            with torch.no_grad():
                W_dec.data = W_dec.data / W_dec.data.norm(dim=1, keepdim=True)
            
            epoch_losses.append(loss.item())
            
            if i % 10 == 0:
                sparsity = (hidden > 0).float().mean()
                pbar.set_postfix({
                    'loss': f'{loss.item():.3f}',
                    'recon': f'{recon_loss.item():.3f}',
                    'spar': f'{sparsity:.1%}'
                })
            
            del batch, hidden, recon
        
        avg_loss = np.mean(epoch_losses)
        all_losses.append(avg_loss)
        print(f"   Epoch {epoch+1}: loss={avg_loss:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print(f"   Early stop at epoch {epoch+1}")
                break
    
    return {
        'W_enc': W_enc.cpu(),
        'b_enc': b_enc.cpu(),
        'W_dec': W_dec.cpu(),
        'b_dec': b_dec.cpu(),
        'losses': all_losses,
        'config': {'d_in': d_in, 'd_sae': d_sae, 'l1_coef': l1_coef}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default="models/torso_restoration/best_model.pt")
    parser.add_argument("--dataset", type=Path, default="data/restoration_dataset")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--l1-coef", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default="models/sae_results")
    args = parser.parse_args()
    
    print("="*60)
    print("🧠 SAE Training (Memory-Optimized)")
    print("="*60)
    
    # Load ORIGINAL model (much lighter than TransformerLens!)
    print(f"\n📦 Loading model from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = RestorationConfig(**ckpt["config"])
    model = RestorationModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()
    print(f"   ✓ Loaded (37M params, d_model={config.d_model})")
    
    # Load dataset
    print(f"\n📚 Loading dataset...")
    dataset = load_from_disk(str(args.dataset))["validation"]
    print(f"   ✓ {len(dataset)} samples")
    
    # Collect activations
    activations = collect_activations_from_layer(
        model, dataset, args.layer, args.max_samples, args.device
    )
    
    # Train SAE
    sae_state = train_sae(
        activations,
        d_in=config.d_model,
        expansion_factor=args.expansion,
        l1_coef=args.l1_coef,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=1e-4,
        device=args.device
    )
    
    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    save_path = args.output / f"sae_layer{args.layer}_optimized.pt"
    torch.save(sae_state, save_path)
    print(f"\n💾 Saved to: {save_path}")
    
    # Quick analysis
    print(f"\n📊 Analysis:")
    print(f"   Final loss: {sae_state['losses'][-1]:.4f}")
    print(f"   Training epochs: {len(sae_state['losses'])}")
    
    # Compute sparsity on sample
    with torch.no_grad():
        sample = activations[:1000].to(args.device)
        hidden = torch.relu(sample @ sae_state['W_enc'].to(args.device) + sae_state['b_enc'].to(args.device))
        sparsity = (hidden > 0).float().mean()
        print(f"   Feature sparsity: {sparsity:.2%}")
        print(f"   Avg active features: {(hidden > 0).sum(1).float().mean():.0f}/{sae_state['config']['d_sae']}")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()

