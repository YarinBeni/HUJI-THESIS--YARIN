#!/usr/bin/env python3
"""Inspect and analyze trained SAE features to understand what the model learned.

This script helps you investigate:
1. Which features are most active
2. What input patterns activate each feature
3. How features relate to linguistic patterns (Akkadian, Sumerian, morphology)
4. Feature clustering and relationships
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))
from modeling_restoration import RestorationConfig, RestorationModel
from common_atf import strip_atf


class SAEInspector:
    """Analyze trained SAE to understand learned features."""
    
    def __init__(
        self,
        sae_path: Path,
        model_path: Path,
        dataset_path: Path,
        device: str = "cpu"
    ):
        """Initialize inspector.
        
        Args:
            sae_path: Path to trained SAE
            model_path: Path to trained model
            dataset_path: Path to dataset
            device: Device to use
        """
        self.device = device
        
        # Load SAE
        print(f"📦 Loading SAE from {sae_path}...")
        self.sae = torch.load(sae_path, map_location=device, weights_only=False)
        self.W_enc = self.sae['W_enc'].to(device)
        self.b_enc = self.sae['b_enc'].to(device)
        self.W_dec = self.sae['W_dec'].to(device)
        self.b_dec = self.sae['b_dec'].to(device)
        self.d_in = self.sae['config']['d_in']
        self.d_sae = self.sae['config']['d_sae']
        print(f"   ✓ SAE: {self.d_in} → {self.d_sae} features")
        
        # Load model
        print(f"\n📦 Loading model from {model_path}...")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        self.config = RestorationConfig(**ckpt['config'])
        self.model = RestorationModel(self.config)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print(f"   ✓ Model: {self.config.n_layers} layers, vocab={self.config.vocab_size}")
        
        # Load dataset
        print(f"\n📚 Loading dataset from {dataset_path}...")
        dataset = load_from_disk(str(dataset_path))
        self.val_dataset = dataset['validation']
        print(f"   ✓ {len(self.val_dataset)} validation samples")
        
        # Load raw fragments for text inspection
        fragments_path = Path("data/eBL_fragments.json")
        if fragments_path.exists():
            import json
            with open(fragments_path) as f:
                self.fragments = json.load(f)
            print(f"   ✓ {len(self.fragments)} raw fragments loaded")
        else:
            self.fragments = None
            print(f"   ⚠️  Raw fragments not found (won't show original text)")
    
    def compute_feature_activations(
        self,
        layer_idx: int,
        num_samples: int = 100
    ) -> tuple[torch.Tensor, list]:
        """Compute which features activate for each sample.
        
        Args:
            layer_idx: Which layer the SAE was trained on
            num_samples: Number of samples to analyze
            
        Returns:
            feature_acts: [num_samples * seq_len, d_sae]
            sample_texts: List of original texts
        """
        print(f"\n🔍 Computing feature activations for layer {layer_idx}...")
        
        # Setup hook to capture activations
        captured = []
        
        def hook_fn(module, input, output):
            captured.append(output.detach().cpu())
        
        hook = self.model.layers[layer_idx].register_forward_hook(hook_fn)
        
        # Process samples
        self.val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        sample_texts = []
        
        with torch.no_grad():
            for idx in tqdm(range(min(num_samples, len(self.val_dataset))), desc="Processing"):
                item = self.val_dataset[idx]
                input_ids = item['input_ids'].unsqueeze(0).to(self.device)
                
                # Get original text
                text = ''.join([chr(x) for x in item['input_ids'].tolist() if x != 32])  # Skip padding
                sample_texts.append(text)
                
                # Forward pass
                _ = self.model(input_ids)
        
        hook.remove()
        
        # Concatenate activations: [num_samples, seq_len, d_model]
        layer_acts = torch.cat(captured, dim=0)
        layer_acts_flat = layer_acts.reshape(-1, self.d_in)  # [N, d_in]
        
        # Encode through SAE: get feature activations
        feature_acts = torch.relu(layer_acts_flat @ self.W_enc + self.b_enc)  # [N, d_sae]
        
        print(f"   ✓ Feature activations: {feature_acts.shape}")
        print(f"   ✓ Average active features: {(feature_acts > 0).sum(1).float().mean():.1f}/{self.d_sae}")
        
        return feature_acts, sample_texts
    
    def find_top_features(self, feature_acts: torch.Tensor, top_k: int = 20) -> dict:
        """Find most important features.
        
        Args:
            feature_acts: Feature activation tensor
            top_k: Number of top features to return
            
        Returns:
            Dictionary with feature statistics
        """
        print(f"\n📊 Analyzing top {top_k} features...")
        
        # Compute statistics
        feature_active_count = (feature_acts > 0).sum(0)  # How often each feature fires
        feature_mean_activation = feature_acts.mean(0)  # Average activation
        feature_max_activation = feature_acts.max(0)[0]  # Max activation
        
        # Find top features by mean activation
        top_indices = feature_mean_activation.topk(top_k).indices
        
        print(f"\n{'Feature':<10} {'Fire Rate':<12} {'Mean Act':<12} {'Max Act':<12}")
        print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*12}")
        
        feature_info = {}
        for idx in top_indices:
            idx_val = idx.item()
            fire_rate = feature_active_count[idx].item() / len(feature_acts)
            mean_act = feature_mean_activation[idx].item()
            max_act = feature_max_activation[idx].item()
            
            print(f"{idx_val:<10} {fire_rate:<12.2%} {mean_act:<12.4f} {max_act:<12.4f}")
            
            feature_info[idx_val] = {
                'fire_rate': fire_rate,
                'mean_activation': mean_act,
                'max_activation': max_act
            }
        
        return feature_info
    
    def inspect_feature(
        self,
        feature_idx: int,
        feature_acts: torch.Tensor,
        sample_texts: list,
        top_n: int = 10
    ) -> dict:
        """Inspect a specific feature to see what activates it.
        
        Args:
            feature_idx: Which feature to inspect
            feature_acts: All feature activations [num_tokens, d_sae]
            sample_texts: Original texts
            top_n: Number of top activations to show
            
        Returns:
            Dictionary with analysis
        """
        print(f"\n🔬 Inspecting Feature {feature_idx}")
        print("="*60)
        
        # Get activations for this feature
        feat_acts = feature_acts[:, feature_idx]  # [num_tokens]
        
        # Find top activating positions
        top_values, top_indices = feat_acts.topk(top_n)
        
        print(f"\nTop {top_n} activations:")
        print(f"{'Rank':<6} {'Activation':<15} {'Context':<50}")
        print(f"{'-'*6} {'-'*15} {'-'*50}")
        
        seq_len = 768  # From preprocessing
        contexts = []
        
        for rank, (val, idx) in enumerate(zip(top_values, top_indices), 1):
            idx_val = idx.item()
            val_item = val.item()
            
            # Find which sample and position
            sample_idx = idx_val // seq_len
            pos_in_sample = idx_val % seq_len
            
            if sample_idx < len(sample_texts):
                text = sample_texts[sample_idx]
                
                # Get context around this position (±20 chars)
                start = max(0, pos_in_sample - 20)
                end = min(len(text), pos_in_sample + 20)
                context = text[start:end]
                
                # Highlight the specific character
                char_at_pos = text[pos_in_sample] if pos_in_sample < len(text) else ' '
                context_display = context.replace(char_at_pos, f"[{char_at_pos}]", 1)
                
                print(f"{rank:<6} {val_item:<15.4f} {context_display:<50}")
                contexts.append(context)
        
        # Analyze common patterns
        print(f"\n🔍 Pattern Analysis:")
        
        # Character frequency in top contexts
        all_chars = ''.join(contexts)
        char_freq = Counter(all_chars)
        print(f"   Most common characters: {char_freq.most_common(10)}")
        
        # Look for linguistic markers
        markers = {
            'Akkadian genitive': '-i',
            'Akkadian accusative': '-a',
            'Sumerian ergative': '-e',
            'Sumerian locative': '-a',
            'Determinative': 'd ',
            'Number sign': 'm ',
        }
        
        print(f"\n   Linguistic markers found:")
        for marker_name, marker_pattern in markers.items():
            count = sum(1 for ctx in contexts if marker_pattern in ctx.lower())
            if count > 0:
                print(f"     {marker_name}: {count}/{len(contexts)} contexts")
        
        return {
            'feature_idx': feature_idx,
            'top_activations': top_values.tolist(),
            'contexts': contexts[:5],  # Save first 5
            'char_distribution': dict(char_freq.most_common(20))
        }
    
    def find_features_for_pattern(
        self,
        pattern: str,
        feature_acts: torch.Tensor,
        sample_texts: list,
        top_k: int = 10
    ) -> list:
        """Find which features activate most for a specific text pattern.
        
        Args:
            pattern: Text pattern to search for (e.g., '-i', 'lugal')
            feature_acts: Feature activations
            sample_texts: Original texts
            top_k: Number of top features to return
            
        Returns:
            List of (feature_idx, avg_activation) tuples
        """
        print(f"\n🔎 Finding features for pattern: '{pattern}'")
        
        # Find positions where pattern occurs
        seq_len = 768
        pattern_positions = []
        
        for sample_idx, text in enumerate(sample_texts):
            pos = 0
            while pos < len(text):
                idx = text.find(pattern, pos)
                if idx == -1:
                    break
                token_idx = sample_idx * seq_len + idx
                if token_idx < len(feature_acts):
                    pattern_positions.append(token_idx)
                pos = idx + 1
        
        if not pattern_positions:
            print(f"   ⚠️  Pattern '{pattern}' not found in samples")
            return []
        
        print(f"   Found {len(pattern_positions)} occurrences")
        
        # Get feature activations at these positions
        pattern_acts = feature_acts[pattern_positions]  # [num_occurrences, d_sae]
        
        # Find features with highest average activation for this pattern
        avg_acts = pattern_acts.mean(0)  # [d_sae]
        top_features = avg_acts.topk(top_k)
        
        print(f"\n   Top {top_k} features for pattern '{pattern}':")
        print(f"   {'Feature':<10} {'Avg Activation':<20} {'Fire Rate':<15}")
        print(f"   {'-'*10} {'-'*20} {'-'*15}")
        
        results = []
        for idx, val in zip(top_features.indices, top_features.values):
            idx_val = idx.item()
            val_item = val.item()
            fire_rate = (pattern_acts[:, idx] > 0).float().mean().item()
            print(f"   {idx_val:<10} {val_item:<20.4f} {fire_rate:<15.2%}")
            results.append((idx_val, val_item, fire_rate))
        
        return results
    
    def compare_akkadian_vs_sumerian(
        self,
        feature_acts: torch.Tensor,
        sample_texts: list
    ):
        """Find features that distinguish Akkadian from Sumerian.
        
        Args:
            feature_acts: Feature activations
            sample_texts: Original texts
        """
        print(f"\n🏛️  Comparing Akkadian vs. Sumerian Features")
        print("="*60)
        
        # Akkadian markers: -i (genitive), -am (accusative), -u (nominative)
        # Sumerian markers: -e (ergative), -ak (genitive), -ø (absolutive)
        
        akkadian_patterns = ['-i ', '-am', '-u ', 'ša ', 'ana ']
        sumerian_patterns = ['-e ', '-ak', 'lugal', 'dingir', '-bi']
        
        print("\n📖 Akkadian-specific features:")
        for pattern in akkadian_patterns[:3]:
            self.find_features_for_pattern(pattern, feature_acts, sample_texts, top_k=5)
        
        print("\n📖 Sumerian-specific features:")
        for pattern in sumerian_patterns[:3]:
            self.find_features_for_pattern(pattern, feature_acts, sample_texts, top_k=5)
    
    def analyze_decoder_directions(self, top_k_features: list = None):
        """Analyze what each feature's decoder direction represents.
        
        Args:
            top_k_features: List of feature indices to analyze (or None for top 10)
        """
        print(f"\n🧭 Analyzing Decoder Directions")
        print("="*60)
        
        if top_k_features is None:
            # Use features with highest decoder norm
            decoder_norms = self.W_dec.norm(dim=1)
            top_k_features = decoder_norms.topk(10).indices.tolist()
        
        print(f"\nAnalyzing {len(top_k_features)} features...")
        
        for feat_idx in top_k_features[:5]:  # Show first 5
            print(f"\n--- Feature {feat_idx} ---")
            
            # Get decoder direction
            decoder_vec = self.W_dec[feat_idx]  # [d_in]
            
            # Find which dimensions this feature writes to most strongly
            top_dims = decoder_vec.abs().topk(10)
            
            print(f"   Top dimensions this feature affects:")
            for dim_idx, val in zip(top_dims.indices[:5], top_dims.values[:5]):
                print(f"     Dim {dim_idx.item()}: {val.item():.4f}")
            
            # Compute decoder norm
            norm = decoder_vec.norm().item()
            print(f"   Decoder norm: {norm:.4f}")
    
    def visualize_feature_coactivation(
        self,
        feature_acts: torch.Tensor,
        top_n_features: int = 50
    ):
        """Find which features tend to activate together.
        
        Args:
            feature_acts: Feature activations
            top_n_features: Number of top features to analyze
        """
        print(f"\n🔗 Feature Co-activation Analysis")
        print("="*60)
        
        # Get top N most active features
        feature_freq = (feature_acts > 0).sum(0)
        top_features = feature_freq.topk(top_n_features).indices
        
        # Binary activation matrix for top features
        binary_acts = (feature_acts[:, top_features] > 0).float()  # [num_tokens, top_n]
        
        # Compute correlation between features
        correlation = torch.corrcoef(binary_acts.t())  # [top_n, top_n]
        
        # Find highly correlated pairs (excluding self)
        print(f"\n   Top feature pairs that co-activate:")
        print(f"   {'Feature 1':<12} {'Feature 2':<12} {'Correlation':<15}")
        print(f"   {'-'*12} {'-'*12} {'-'*15}")
        
        pairs_found = 0
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                corr = correlation[i, j].item()
                if corr > 0.5:  # High correlation threshold
                    feat1 = top_features[i].item()
                    feat2 = top_features[j].item()
                    print(f"   {feat1:<12} {feat2:<12} {corr:<15.3f}")
                    pairs_found += 1
                    if pairs_found >= 10:
                        break
            if pairs_found >= 10:
                break
        
        if pairs_found == 0:
            print(f"   No highly correlated pairs found (correlation > 0.5)")


def main():
    parser = argparse.ArgumentParser(description="Inspect trained SAE features")
    parser.add_argument(
        "--sae",
        type=Path,
        default="models/sae_results/sae_layer8_optimized.pt",
        help="Path to trained SAE"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default="models/torso_restoration/best_model.pt",
        help="Path to trained model"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/restoration_dataset",
        help="Path to dataset"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=8,
        help="Which layer the SAE was trained on"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to analyze"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--feature",
        type=int,
        help="Specific feature to inspect in detail"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Text pattern to find features for (e.g., '-i', 'lugal')"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔍 SAE Feature Inspector")
    print("="*60)
    
    # Initialize inspector
    inspector = SAEInspector(args.sae, args.model, args.dataset, args.device)
    
    # Compute feature activations on validation data
    feature_acts, sample_texts = inspector.compute_feature_activations(
        args.layer,
        args.num_samples
    )
    
    # Analysis 1: Find top features overall
    top_features = inspector.find_top_features(feature_acts, top_k=20)
    
    # Analysis 2: Inspect specific feature if requested
    if args.feature is not None:
        inspector.inspect_feature(args.feature, feature_acts, sample_texts, top_n=10)
    
    # Analysis 3: Search for pattern if requested
    if args.pattern:
        inspector.find_features_for_pattern(args.pattern, feature_acts, sample_texts, top_k=10)
    
    # Analysis 4: Compare Akkadian vs Sumerian
    inspector.compare_akkadian_vs_sumerian(feature_acts, sample_texts)
    
    # Analysis 5: Decoder directions
    inspector.analyze_decoder_directions(list(top_features.keys())[:10])
    
    # Analysis 6: Co-activation patterns
    inspector.visualize_feature_coactivation(feature_acts, top_n_features=50)
    
    print("\n" + "="*60)
    print("✨ Inspection Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Inspect specific features:")
    print(f"   python src/07_inspect_sae.py --feature 96")
    print("\n2. Search for linguistic patterns:")
    print(f"   python src/07_inspect_sae.py --pattern '-i'")
    print(f"   python src/07_inspect_sae.py --pattern 'lugal'")
    print("\n3. Compare different layers:")
    print(f"   # First train SAE on layer 2 and layer 14, then inspect each")


if __name__ == "__main__":
    main()

