# Model Format & SAELens Usage Guide

## Overview
This guide explains the format of the trained RestorationModel and how to use it with SAELens for mechanistic interpretability analysis.

---

## Model Format

### Trained Model Location
- **Best model**: `models/torso_restoration/best_model.pt`
- **Last model**: `models/torso_restoration/last_model.pt`
- **TransformerLens converted**: `models/torso_restoration/tl_model.pt`

### File Format (.pt files)

The `.pt` files are PyTorch serialized checkpoints containing:

```python
{
    "model_state_dict": dict,  # Model weights as OrderedDict[str, Tensor]
    "config": dict,            # Model hyperparameters
    "vocab_size": int          # Character vocabulary size (11812)
}
```

#### Config Dictionary
```python
{
    "vocab_size": 11812,      # Byte-level character vocab
    "max_position": 768,      # Maximum sequence length
    "d_model": 384,          # Hidden dimension
    "n_heads": 8,            # Attention heads
    "d_ff": 1536,            # FFN dimension (4x d_model)
    "n_layers": 16,          # Transformer layers
    "dropout": 0.1,          # Dropout rate
    "rotary_base": 10000     # RoPE base frequency
}
```

#### State Dict Structure
```
embed.weight: [11812, 384]           # Token embeddings
layers.0.ln1.weight: [384]          # Layer norm 1
layers.0.ln1.bias: [384]
layers.0.attn.q_proj.weight: [384, 384]  # Query projection
layers.0.attn.k_proj.weight: [384, 384]  # Key projection
layers.0.attn.v_proj.weight: [384, 384]  # Value projection
layers.0.attn.out_proj.weight: [384, 384]  # Output projection
layers.0.ln2.weight: [384]          # Layer norm 2
layers.0.ln2.bias: [384]
layers.0.ff.0.weight: [1536, 384]   # FFN layer 1
layers.0.ff.0.bias: [1536]
layers.0.ff.2.weight: [384, 1536]   # FFN layer 2
layers.0.ff.2.bias: [384]
... (repeat for layers 1-15)
final_ln.weight: [384]              # Final layer norm
final_ln.bias: [384]
head.weight: [11812, 384]           # Output projection (unembedding)
head.bias: [11812]
```

**Note on Positional Embeddings**: This model uses **RoPE (Rotary Position Embeddings)**, computed on-the-fly during forward pass. There are no learned positional embedding weights to load.

### Model Architecture

```
RestorationModel (37M parameters)
├── embed: Embedding(11812, 384)
├── layers: ModuleList (16 layers)
│   └── TransformerBlock
│       ├── ln1: LayerNorm(384)
│       ├── attn: MultiHeadSelfAttention (with RoPE)
│       │   ├── q_proj: Linear(384, 384)
│       │   ├── k_proj: Linear(384, 384)
│       │   ├── v_proj: Linear(384, 384)
│       │   └── out_proj: Linear(384, 384)
│       ├── ln2: LayerNorm(384)
│       └── ff: Sequential
│           ├── Linear(384, 1536)
│           ├── GELU()
│           └── Linear(1536, 384)
├── final_ln: LayerNorm(384)
└── head: Linear(384, 11812)
```

---

## Loading the Model

### Option 1: Original PyTorch Format

```python
import torch
from src.modeling_restoration import RestorationConfig, RestorationModel

# Load checkpoint
checkpoint = torch.load("models/torso_restoration/best_model.pt", map_location="cpu")

# Create model
config = RestorationConfig(**checkpoint["config"])
model = RestorationModel(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Use model
input_ids = torch.randint(0, config.vocab_size, (1, 100))  # Batch of 1, seq_len 100
logits = model(input_ids)  # Shape: [1, 100, 11812]
```

### Option 2: TransformerLens Format (Recommended for SAELens)

```python
import torch
from transformer_lens import HookedTransformer

# Load converted model
checkpoint = torch.load("models/torso_restoration/tl_model.pt", map_location="cpu")
model = checkpoint['model']
model.eval()

# Run with cache to capture activations
input_ids = torch.randint(0, 11812, (1, 100))
logits, cache = model.run_with_cache(input_ids)

# Access layer activations
layer_8_output = cache["blocks.8.hook_resid_post"]  # Shape: [1, 100, 384]
```

---

## Using SAELens for Interpretability

SAELens trains Sparse Autoencoders (SAEs) on model activations to discover interpretable features. Here's how to use it with our model.

### Installation (Already Done)

```bash
conda activate torso-sae
# SAELens and TransformerLens already installed in environment
```

### Step 1: Load Model and Dataset

```python
import torch
from transformer_lens import HookedTransformer
from datasets import load_from_disk

# Load converted model
checkpoint = torch.load("models/torso_restoration/tl_model.pt")
model = checkpoint['model']
model.eval()

# Load validation dataset (for collecting activations)
dataset = load_from_disk("data/restoration_dataset")["validation"]
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
```

### Step 2: Extract Activations from Target Layer

```python
from torch.utils.data import DataLoader

# Choose which layer to analyze (e.g., middle layer 8)
target_layer = 8
activations = []

# Collect activations
dataloader = DataLoader(dataset, batch_size=8)
with torch.no_grad():
    for batch in dataloader:
        input_ids = batch["input_ids"]
        _, cache = model.run_with_cache(input_ids)
        
        # Extract activations from target layer
        layer_acts = cache[f"blocks.{target_layer}.hook_resid_post"]
        activations.append(layer_acts.cpu())

# Concatenate all activations: [num_samples, seq_len, d_model]
all_activations = torch.cat(activations, dim=0)
print(f"Collected activations: {all_activations.shape}")  # e.g., [N, 768, 384]
```

### Step 3: Train SAE on Activations

```python
from sae_lens import LanguageModelSAERunnerConfig, language_model_sae_runner

# Configure SAE training
cfg = LanguageModelSAERunnerConfig(
    # Model details
    model_name="custom_restoration_model",
    hook_name=f"blocks.{target_layer}.hook_resid_post",  # Where to extract activations
    d_in=384,  # Input dimension (d_model)
    
    # SAE architecture
    expansion_factor=4,  # SAE hidden size = 384 * 4 = 1536 features
    
    # Training hyperparameters
    l1_coefficient=1e-3,  # Sparsity penalty
    lr=1e-4,
    batch_size=4,  # Smaller due to Mac M2 memory constraints
    total_training_tokens=10_000_000,  # Adjust based on dataset size
    
    # Logging
    log_to_wandb=False,  # Set to True if using Weights & Biases
    device="mps",  # or "cpu" or "cuda"
)

# Train SAE
# Note: This will take significant time and memory
sparse_autoencoder = language_model_sae_runner(cfg)
```

### Step 4: Analyze Learned Features

```python
# Load trained SAE
sae = sparse_autoencoder  # From training step

# Get feature activations for a sample
sample_input = torch.randint(0, 11812, (1, 100))
_, cache = model.run_with_cache(sample_input)
layer_acts = cache[f"blocks.{target_layer}.hook_resid_post"]

# Encode to SAE features
feature_acts = sae.encode(layer_acts)  # Shape: [1, 100, 1536]
print(f"Feature activation sparsity: {(feature_acts > 0).float().mean():.2%}")

# Find top-k active features
top_k = 10
top_features = torch.topk(feature_acts[0].mean(0), k=top_k)
print(f"Top {top_k} features: {top_features.indices.tolist()}")
print(f"Activations: {top_features.values.tolist()}")

# Decode to see what each feature represents
for feat_idx in top_features.indices[:3]:
    # Get feature direction
    feat_vector = sae.W_dec[feat_idx]  # Shape: [384]
    
    # Project onto vocabulary to see which tokens it corresponds to
    # (if you have embedding matrix access)
    print(f"\nFeature {feat_idx}:")
    # Analyze what linguistic patterns this feature captures
```

---

## Interpretability Research Questions

Once you have trained SAEs, you can investigate:

1. **Character Restoration Patterns**
   - Which features activate for damaged text regions?
   - Do specific features specialize in restoring Akkadian vs. Sumerian?

2. **Linguistic Feature Discovery**
   - Are there features for verb forms, noun cases, or specific morphemes?
   - Do features correspond to loanwords or foreign text?

3. **Attention Head Analysis**
   - Which attention heads focus on grammatical agreement?
   - Are there induction heads that copy patterns from earlier in the text?

4. **Layer-wise Progression**
   - Do early layers focus on character-level features?
   - Do later layers encode higher-level semantic features?

---

## Memory Considerations (Mac M2 18GB)

Your Mac M2 has 18GB unified memory. Based on training experience:

- **Model loading**: ~1GB
- **Single forward pass** (batch_size=32, seq_len=768): ~4-6GB
- **SAE training**: Can require 8-12GB depending on batch size and expansion factor

**Recommendations**:
- Use `batch_size=4` for SAE training
- Use `expansion_factor=4` (not 8) to keep SAE hidden size manageable
- Consider training on a subset of validation data (e.g., first 1000 samples)
- Use `device="cpu"` if MPS runs out of memory (slower but stable)
- Train SAE on one layer at a time, not all layers simultaneously

---

## Troubleshooting

### Issue: "RuntimeError: MPS out of memory"
**Solution**: Reduce batch size or use CPU
```python
cfg = LanguageModelSAERunnerConfig(
    batch_size=2,  # Reduce from 4
    device="cpu",   # Switch to CPU
    ...
)
```

### Issue: "Model forward pass takes too long"
**Solution**: Use a subset of data for activation collection
```python
# Limit to first 500 samples
dataset_subset = dataset.select(range(500))
```

### Issue: "SAE features all zero"
**Solution**: Reduce L1 coefficient (too much sparsity)
```python
cfg = LanguageModelSAERunnerConfig(
    l1_coefficient=1e-4,  # Reduce from 1e-3
    ...
)
```

---

## Next Steps

1. **Fix RoPE Bug** (Optional): If you want full numerical verification, update `modeling_restoration.py` line 24-25:
   ```python
   # Before
   cos = cos[:seq_len]
   # After
   cos = cos[:, :, :seq_len, :]
   ```

2. **Create SAE Training Script**: Write `src/05_run_sae_analysis.py` following the examples above

3. **Analyze Features**: After training SAE, analyze which features correspond to linguistic patterns

4. **Visualize**: Use TransformerLens's visualization tools to plot attention patterns and feature activations

---

## References

- **TransformerLens Docs**: https://neelnanda-io.github.io/TransformerLens/
- **SAELens Docs**: https://jbloomaus.github.io/SAELens/
- **Our Project Plan**: `RESTORATION_PROJECT_PLAN.md`
- **Progress Log**: `PROGRESS.md`

> **Mac-M2 note**   The full SAELens runner shown below needs >24 GB RAM due to TransformerLens caching.
> For laptops use our lightweight alternative:
> ```bash
> python src/06_sae_memory_optimized.py --layer 8 --max-samples 200 --epochs 50 --device cpu
> ```
> It hooks `x_resid_post` directly in the original model, so attention-head tensors are already merged.

