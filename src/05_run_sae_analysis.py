#!/usr/bin/env python3
"""Train Sparse Autoencoders (SAEs) on RestorationModel activations for interpretability.

This script:
1. Loads the TransformerLens-converted model
2. Extracts activations from target layers
3. Trains SAE to discover interpretable features
4. Analyzes learned features for linguistic patterns
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformer_lens import HookedTransformer
import numpy as np
from tqdm import tqdm

# SAELens imports
try:
    from sae_lens import SAE, LanguageModelSAERunnerConfig, language_model_sae_runner
    SAELENS_AVAILABLE = True
except ImportError:
    print("⚠️  SAELens not available. Install with: pip install sae-lens")
    SAELENS_AVAILABLE = False


class ActivationCollector:
    """Collect activations from a specific layer of the model."""
    
    def __init__(self, model: HookedTransformer, layer_idx: int, device: str = "cpu"):
        """Initialize collector.
        
        Args:
            model: TransformerLens model
            layer_idx: Which layer to extract from (0-15 for our model)
            device: Device to run on
        """
        self.model = model
        self.layer_idx = layer_idx
        self.device = device
        self.hook_name = f"blocks.{layer_idx}.hook_resid_post"
        
    def collect_from_dataset(
        self, 
        dataset, 
        max_samples: int = 1000,
        batch_size: int = 8
    ) -> torch.Tensor:
        """Collect activations from dataset.
        
        Args:
            dataset: HuggingFace dataset with 'input_ids' field
            max_samples: Maximum number of samples to process
            batch_size: Batch size for forward passes
            
        Returns:
            Tensor of shape [num_samples * seq_len, d_model]
        """
        print(f"\n📊 Collecting activations from layer {self.layer_idx}...")
        print(f"   Hook name: {self.hook_name}")
        
        # Limit dataset size
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            print(f"   Using {max_samples} samples (limited for memory)")
        else:
            print(f"   Using all {len(dataset)} samples")
        
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        activations_list = []
        self.model.eval()
        
        # Use TransformerLens's built-in hook system (memory efficient)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting activations")):
                input_ids = batch["input_ids"].to(self.device)
                
                # Run model and extract ONLY the specific layer we need
                # Use names_filter to avoid caching all layers
                _, cache = self.model.run_with_cache(
                    input_ids,
                    names_filter=lambda name: name == self.hook_name
                )
                
                # Extract target layer activations
                layer_acts = cache[self.hook_name]  # [batch, seq, d_model]
                
                # Flatten to [batch * seq, d_model] and move to CPU immediately
                layer_acts_flat = layer_acts.reshape(-1, layer_acts.shape[-1]).cpu()
                activations_list.append(layer_acts_flat)
                
                # Free memory immediately
                del cache, layer_acts, layer_acts_flat
                
                # Periodically consolidate to avoid list overhead
                if (batch_idx + 1) % 5 == 0 and len(activations_list) > 5:
                    partial = torch.cat(activations_list[:5], dim=0)
                    activations_list = [partial] + activations_list[5:]
                
                # Clear GPU/MPS cache if needed
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
        
        # Concatenate all activations
        all_activations = torch.cat(activations_list, dim=0)
        
        print(f"   ✓ Collected shape: {all_activations.shape}")
        print(f"   ✓ Memory: {all_activations.element_size() * all_activations.nelement() / 1024**2:.1f} MB")
        
        return all_activations


class SAETrainer:
    """Train and analyze Sparse Autoencoders."""
    
    def __init__(
        self,
        d_in: int,
        expansion_factor: int = 4,
        l1_coefficient: float = 1e-3,
        device: str = "cpu"
    ):
        """Initialize SAE trainer.
        
        Args:
            d_in: Input dimension (d_model)
            expansion_factor: SAE hidden size = d_in * expansion_factor
            l1_coefficient: Sparsity penalty
            device: Device to train on
        """
        self.d_in = d_in
        self.d_sae = d_in * expansion_factor
        self.l1_coefficient = l1_coefficient
        self.device = device
        
        print(f"\n🔧 SAE Configuration:")
        print(f"   Input dim: {d_in}")
        print(f"   SAE dim: {self.d_sae} (expansion {expansion_factor}x)")
        print(f"   L1 coefficient: {l1_coefficient}")
        print(f"   Device: {device}")
    
    def train(
        self,
        activations: torch.Tensor,
        batch_size: int = 4,
        num_epochs: int = 5,
        lr: float = 1e-4,
        save_path: Path = None
    ) -> Dict:
        """Train SAE on activations.
        
        Args:
            activations: Tensor of shape [num_samples, d_in]
            batch_size: Training batch size
            num_epochs: Number of training epochs
            lr: Learning rate
            save_path: Where to save trained SAE
            
        Returns:
            Dictionary with training metrics and SAE
        """
        print(f"\n🚀 Training SAE with custom implementation...")
        print(f"   Training samples: {activations.shape[0]:,}")
        print(f"   Batch size: {batch_size}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning rate: {lr}")
        
        # Create simple SAE (not using full LanguageModelSAERunner due to complexity)
        # Instead, we'll implement basic SAE training
        
        # Initialize SAE weights as nn.Parameters for proper leaf tensor handling
        import torch.nn as nn
        W_enc = nn.Parameter(torch.randn(self.d_in, self.d_sae, device=self.device) * 0.01)
        b_enc = nn.Parameter(torch.zeros(self.d_sae, device=self.device))
        W_dec = nn.Parameter(torch.randn(self.d_sae, self.d_in, device=self.device) * 0.01)
        b_dec = nn.Parameter(torch.zeros(self.d_in, device=self.device))
        
        # Normalize decoder weights (important for SAEs)
        with torch.no_grad():
            W_dec.data = W_dec.data / W_dec.data.norm(dim=1, keepdim=True)
        
        # Optimizer
        optimizer = torch.optim.Adam([W_enc, b_enc, W_dec, b_dec], lr=lr)
        
        # Training loop with better memory management
        num_batches = len(activations) // batch_size
        training_losses = []
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 5  # Early stopping patience
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Shuffle activations
            perm = torch.randperm(activations.shape[0])
            activations_shuffled = activations[perm]
            
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
            for i in pbar:
                # Get batch
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(activations_shuffled))
                batch = activations_shuffled[start_idx:end_idx].to(self.device)
                
                # Forward pass
                # Encode: x -> ReLU(W_enc @ x + b_enc)
                hidden = torch.relu(batch @ W_enc + b_enc)
                
                # Decode: hidden -> W_dec @ hidden + b_dec
                reconstructed = hidden @ W_dec + b_dec
                
                # Loss: reconstruction + L1 sparsity
                reconstruction_loss = (batch - reconstructed).pow(2).mean()
                sparsity_loss = hidden.abs().mean()
                loss = reconstruction_loss + self.l1_coefficient * sparsity_loss
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_([W_enc, b_enc, W_dec, b_dec], max_norm=1.0)
                
                optimizer.step()
                
                # Renormalize decoder weights
                with torch.no_grad():
                    W_dec.data = W_dec.data / W_dec.data.norm(dim=1, keepdim=True)
                
                # Track
                epoch_losses.append(loss.item())
                
                # Update progress bar every 10 batches to reduce overhead
                if i % 10 == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'recon': f'{reconstruction_loss.item():.4f}',
                        'l1': f'{sparsity_loss.item():.4f}',
                        'sparsity': f'{(hidden > 0).float().mean():.2%}'
                    })
                
                # Free memory
                del batch, hidden, reconstructed
            
            avg_loss = np.mean(epoch_losses)
            training_losses.append(avg_loss)
            print(f"   Epoch {epoch+1} avg loss: {avg_loss:.4f}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best checkpoint
                if save_path:
                    torch.save({
                        'W_enc': W_enc.cpu(),
                        'b_enc': b_enc.cpu(),
                        'W_dec': W_dec.cpu(),
                        'b_dec': b_dec.cpu(),
                        'config': {
                            'd_in': self.d_in,
                            'd_sae': self.d_sae,
                            'l1_coefficient': self.l1_coefficient,
                        },
                        'epoch': epoch,
                        'loss': avg_loss
                    }, str(save_path).replace('.pt', '_best.pt'))
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"   Early stopping at epoch {epoch+1} (patience={max_patience})")
                    break
        
        # Save SAE
        sae_state = {
            'W_enc': W_enc.cpu(),
            'b_enc': b_enc.cpu(),
            'W_dec': W_dec.cpu(),
            'b_dec': b_dec.cpu(),
            'config': {
                'd_in': self.d_in,
                'd_sae': self.d_sae,
                'l1_coefficient': self.l1_coefficient,
            },
            'training_losses': training_losses
        }
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sae_state, save_path)
            print(f"\n💾 Saved SAE to: {save_path}")
        
        return sae_state
    
    def analyze_features(
        self,
        sae_state: Dict,
        activations: torch.Tensor,
        top_k: int = 20
    ) -> Dict:
        """Analyze learned SAE features.
        
        Args:
            sae_state: Trained SAE state dict
            activations: Sample activations to analyze
            top_k: Number of top features to report
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\n🔍 Analyzing learned features...")
        
        W_enc = sae_state['W_enc'].to(self.device)
        b_enc = sae_state['b_enc'].to(self.device)
        W_dec = sae_state['W_dec'].to(self.device)
        
        # Encode sample activations
        sample = activations[:1000].to(self.device)  # Use subset for analysis
        with torch.no_grad():
            hidden = torch.relu(sample @ W_enc + b_enc)
        
        # Compute feature statistics
        feature_activations = hidden.cpu()
        
        # Sparsity: % of times each feature is active (> 0)
        feature_sparsity = (feature_activations > 0).float().mean(dim=0)
        
        # Average activation when active
        feature_mean_when_active = []
        for i in range(self.d_sae):
            active_mask = feature_activations[:, i] > 0
            if active_mask.sum() > 0:
                feature_mean_when_active.append(feature_activations[:, i][active_mask].mean().item())
            else:
                feature_mean_when_active.append(0.0)
        feature_mean_when_active = torch.tensor(feature_mean_when_active)
        
        # Find most active features
        top_indices = feature_mean_when_active.topk(top_k).indices
        
        print(f"\n   Top {top_k} features by average activation:")
        print(f"   {'Feature':<10} {'Sparsity':<12} {'Avg When Active':<18} {'Decoder Norm':<15}")
        print(f"   {'-'*10} {'-'*12} {'-'*18} {'-'*15}")
        
        for idx in top_indices:
            idx = idx.item()
            sparsity = feature_sparsity[idx].item()
            mean_act = feature_mean_when_active[idx].item()
            dec_norm = W_dec[idx].norm().item()
            print(f"   {idx:<10} {sparsity:<12.2%} {mean_act:<18.4f} {dec_norm:<15.4f}")
        
        # Overall statistics
        print(f"\n   Overall feature statistics:")
        print(f"   - Active features per sample: {(feature_activations > 0).sum(dim=1).float().mean():.1f}")
        print(f"   - Feature sparsity (avg): {feature_sparsity.mean():.2%}")
        print(f"   - Dead features (never active): {(feature_sparsity == 0).sum()}/{self.d_sae}")
        
        return {
            'feature_sparsity': feature_sparsity,
            'feature_mean_when_active': feature_mean_when_active,
            'top_features': top_indices.tolist(),
            'dead_features': (feature_sparsity == 0).sum().item()
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train SAE for interpretability analysis")
    parser.add_argument(
        "--model",
        type=Path,
        default="models/torso_restoration/tl_model.pt",
        help="Path to TransformerLens model"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/restoration_dataset",
        help="Path to preprocessed dataset"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=8,
        help="Which layer to analyze (0-15)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum samples to use for activation collection"
    )
    parser.add_argument(
        "--expansion-factor",
        type=int,
        default=4,
        help="SAE expansion factor (hidden_size = d_model * factor)"
    )
    parser.add_argument(
        "--l1-coef",
        type=float,
        default=1e-3,
        help="L1 sparsity coefficient"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device to use"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="models/sae_results",
        help="Output directory for SAE and results"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.model.exists():
        print(f"❌ Model not found at {args.model}")
        print(f"   Run: python src/04_convert_to_transformerlens.py first")
        return
    
    if not args.dataset.exists():
        print(f"❌ Dataset not found at {args.dataset}")
        print(f"   Run: python src/02_preprocess_dataset.py first")
        return
    
    # Setup
    print("="*60)
    print("🧠 Sparse Autoencoder Training for Interpretability")
    print("="*60)
    
    # Load model
    print(f"\n📦 Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)
    model = checkpoint['model']
    model.to(args.device)
    model.eval()
    print(f"   ✓ Model loaded (d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers})")
    
    # Load dataset
    print(f"\n📚 Loading dataset from {args.dataset}...")
    dataset = load_from_disk(str(args.dataset))
    val_dataset = dataset["validation"]
    print(f"   ✓ Loaded {len(val_dataset)} validation samples")
    
    # Step 1: Collect activations
    collector = ActivationCollector(model, args.layer, args.device)
    activations = collector.collect_from_dataset(
        val_dataset,
        max_samples=args.max_samples,
        batch_size=1  # Use batch size of 1 for activation extraction to minimize memory
    )
    
    # Step 2: Train SAE
    trainer = SAETrainer(
        d_in=model.cfg.d_model,
        expansion_factor=args.expansion_factor,
        l1_coefficient=args.l1_coef,
        device=args.device
    )
    
    sae_save_path = args.output_dir / f"sae_layer{args.layer}.pt"
    sae_state = trainer.train(
        activations,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        save_path=sae_save_path
    )
    
    # Step 3: Analyze features
    if sae_state:
        analysis = trainer.analyze_features(sae_state, activations, top_k=20)
        
        # Save analysis results
        results_path = args.output_dir / f"analysis_layer{args.layer}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                'layer': args.layer,
                'config': sae_state['config'],
                'training_losses': sae_state['training_losses'],
                'top_features': analysis['top_features'],
                'dead_features': analysis['dead_features'],
                'avg_active_features': (activations.shape[0] * analysis['feature_sparsity'].mean()).item()
            }, f, indent=2)
        
        print(f"\n💾 Saved analysis to: {results_path}")
    
    print("\n" + "="*60)
    print("✨ Analysis complete!")
    print("="*60)
    print("\nNext steps:")
    if sae_state:
        print(f"1. Inspect learned features in: {sae_save_path}")
        print(f"2. Review analysis results in: {results_path}")
        print("3. Try different layers (--layer 4, --layer 12) to see how features evolve")
        print("4. Investigate top features to understand what linguistic patterns they capture")
    else:
        print("1. SAE training requires implementing the full training loop")
        print("2. Activations successfully collected - can be used with external SAE tools")
        print(f"3. Activations shape: {activations.shape}")


if __name__ == "__main__":
    main()

