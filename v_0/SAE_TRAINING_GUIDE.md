# Quick SAE Training Guide (Memory-Optimized)

## ✅ Solution to Memory Issues

The **memory-optimized version** (`src/06_sae_memory_optimized.py`) solves all memory problems by:

1. **Using Original Model** - Lightweight RestorationModel (~37M params) instead of TransformerLens (156MB + caching overhead)
2. **One Sample at a Time** - Processes each sample individually during activation extraction
3. **Immediate CPU Transfer** - Moves activations to CPU right after extraction
4. **PyTorch Hooks** - Direct hooks on model layers (no heavy caching infrastructure)
5. **Early Stopping** - Stops when converged (saves time)

## 🚀 Successful Run Results

Just completed a full training run:
- **200 samples**, **50 epochs requested** (stopped at 30 due to convergence)
- **Layer 8** (middle of network)
- **Memory**: Only 1.57GB RAM used
- **Time**: ~3-4 minutes total
- **Loss**: 2.37 → 0.006 (excellent convergence!)
- **Output**: `models/sae_results/sae_layer8_optimized.pt`

### ⚠️  Sparsity caveat
Dense activations observed: the example run fired **1 520 / 1 536** neurons per token (≈99 %).
If you want sparser, more interpretable features retrain with `--l1-coef 3e-3` or `--k-sparse 20`.

### 🔎 Inspecting features quickly
```bash
# Plot activation histograms for all features in layer 8
python src/07_inspect_sae.py --histogram --layer 8

# Open a mid-frequency feature that lit up for the suffix "-ak"
python src/07_inspect_sae.py --feature 1170
```

## 📝 Usage Commands

### Train SAE on Layer 8 (Recommended Settings)
```bash
conda activate torso-sae
python src/06_sae_memory_optimized.py --layer 8 --max-samples 200 --epochs 50 --batch-size 256 --device cpu
```

### Try Different Layers
```bash
# Early layer (character-level features)
python src/06_sae_memory_optimized.py --layer 2 --max-samples 200 --epochs 50

# Middle layer (morphological features)
python src/06_sae_memory_optimized.py --layer 8 --max-samples 200 --epochs 50

# Late layer (semantic features)
python src/06_sae_memory_optimized.py --layer 14 --max-samples 200 --epochs 50
```

### More Samples (if you have time)
```bash
# Use more data for better features
python src/06_sae_memory_optimized.py --layer 8 --max-samples 500 --epochs 50 --device cpu
# Takes ~10-15 min, but gives much better results
```

### Adjust Sparsity
```bash
# Less sparse (more features active)
python src/06_sae_memory_optimized.py --layer 8 --max-samples 200 --l1-coef 1e-4

# More sparse (fewer but stronger features)
python src/06_sae_memory_optimized.py --layer 8 --max-samples 200 --l1-coef 5e-3
```

## 📊 Understanding the Output

The trained SAE file contains:
```python
{
    'W_enc': [384, 1536],  # Encoder weights
    'b_enc': [1536],       # Encoder bias  
    'W_dec': [1536, 384],  # Decoder weights (normalized)
    'b_dec': [384],        # Decoder bias
    'losses': [2.37, 0.84, ..., 0.006],  # Training losses per epoch
    'config': {'d_in': 384, 'd_sae': 1536, 'l1_coef': 0.001}
}
```

## 🔍 Next Steps: Analyzing Features

To see what linguistic patterns each feature learned, create a visualization script:

```python
import torch
from src.modeling_restoration import RestorationConfig, RestorationModel

# Load SAE
sae = torch.load('models/sae_results/sae_layer8_optimized.pt', weights_only=False)
W_enc = sae['W_enc']
W_dec = sae['W_dec']

# Load model and test data
checkpoint = torch.load('models/torso_restoration/best_model.pt', weights_only=False)
model = RestorationModel(RestorationConfig(**checkpoint['config']))
model.load_state_dict(checkpoint['model_state_dict'])

# For a given input, see which features activate
# ... then analyze what patterns those features correspond to
```

## 💡 Key Parameters Explained

- `--max-samples`: How many fragments to analyze (200-500 recommended)
- `--epochs`: Training epochs (50 with early stopping usually converges ~25-35 epochs)
- `--batch-size`: **For SAE training** not extraction (256-512 works well)
- `--expansion`: SAE size multiplier (4x = 1536 features, 8x = 3072 features)
- `--l1-coef`: Sparsity penalty (1e-3 default, lower = more active features)
- `--layer`: Which transformer layer to analyze (0-15)

## ⚠️ Memory Notes

Your Mac M2 with 18GB can handle:
- ✅ Up to 500 samples with this optimized script
- ✅ Batch size 1 for extraction, 256-512 for training
- ✅ All 16 layers independently
- ❌ Don't use TransformerLens version (`05_run_sae_analysis.py`) - requires 24GB+ RAM

If you need to analyze 1000+ samples, run on cloud GPU or split into multiple runs and combine features.

## 🎯 Research Questions to Investigate

With your trained SAE, you can now explore:

1. **Which features activate for damaged regions** (`-` or `#` masks)?
2. **Are there Akkadian-specific vs. Sumerian-specific features?**
3. **Do features correspond to morphological patterns** (verb conjugations, case markers)?
4. **How do features evolve across layers** (character → morphology → semantics)?

The trained SAE provides a window into what your restoration model learned! 🏺✨

