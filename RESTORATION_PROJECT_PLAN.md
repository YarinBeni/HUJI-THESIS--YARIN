# Akkadian Restoration Project – Easiest Path

A concise, beginner-friendly checklist for building a **text-only restoration model** (ByT5 fine-tune) and running **SAELens** for mechanistic interpretability.

---

## 0. Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10+ | Use the `evaCun` conda env we already created. |
| PyTorch | ≥2.1 | CUDA / MPS enabled. |
| HuggingFace Transformers | ≥4.40 |
| Datasets | ≥2.19 |
| SAELens | latest main branch |
| wandb (optional) | latest | For experiment tracking. |

Install once:
```bash
conda activate evaCun
pip install torch transformers datasets saelens wandb
```

---

## 1. Repository Layout
```
lititure-review/
├── data/
│   └── eBL_fragments.json          # downloaded via download_evaCun_dataset.py
├── src/
│   ├── 01_build_vocab.py           # build char vocab & tokenizer
│   ├── 02_preprocess_dataset.py    # create masked training examples (HF Dataset)
│   ├── 03_train_byt5.py            # fine-tune google/byt5-small
│   └── 04_run_saelens.py           # train & analyse sparse autoencoder
├── models/
│   └── byt5-restoration/           # final HF model directory (config + pytorch_model.bin)
├── notebooks/
│   └── demo_inference.ipynb        # nice demo for historians
├── RESTORATION_PROJECT_PLAN.md     # <-- this file
└── requirements.txt                # pinned versions
```

---

## 2. Step-by-Step Checklist

### Step 1 – Build Character Vocabulary
1. Read **eBL_fragments.json**.
2. Extract `atf` strings, strip ATF markup (regex).
3. Collect all unique Unicode characters; keep top-N if needed.
4. Save `char_vocab.json` with mapping `{char: id}`.
5. Update special tokens: `<pad>=0`, `<s>=1`, `</s>=2`, `-` for single mask, `#` for span mask.

👉 Run:
```bash
python src/01_build_vocab.py --input data/eBL_fragments.json --output data/char_vocab.json
```

### Step 2 – Pre-process Dataset
1. Create an HF `Dataset` with fields: `input_ids`, `labels`.
2. Mask 15-25 % of chars: 70 % single (`-`), 30 % spans (`#`, length 2-10).
3. Split 80/10/10 by fragment id → train/val/test `.arrow` files.

👉 Run:
```bash
python src/02_preprocess_dataset.py \
  --fragments data/eBL_fragments.json \
  --vocab    data/char_vocab.json \
  --out_dir  data/restoration_dataset
```

### Step 3 – Train Custom Torso Decoder
1. Load our custom `RestorationModel` (T5-style with rotary embeddings).
2. Compute vocab size from dataset (ord-based encoding).
3. Train with PyTorch:
   * batch_size=8 (adjust for memory)
   * epochs=5-10 (early stop on val loss)
   * lr=3e-4, AdamW optimizer
   * loss: cross-entropy on masked positions (ignore_index=-100)
   * gradient clipping (1.0)
4. Save best checkpoint to `models/torso_restoration/best_model.pt`.

👉 Run (small test first):
```bash
python src/03_train_torso.py \
  --dataset data/restoration_dataset \
  --output_dir models/torso_restoration \
  --epochs 1 --batch_size 2 --max_train_examples 100 --max_eval_examples 50
```

### Step 4 – Quick Inference Demo (Notebook)
1. Load model + tokenizer.
2. Provide damaged text → show top-k restorations.

👉 Open `notebooks/demo_inference.ipynb`.

### Step 5 – Train Sparse Autoencoder (Memory-Optimised)
1. Use the lightweight trainer:
   ```bash
   python src/06_sae_memory_optimized.py --layer 8 --max-samples 200 --epochs 50 --batch-size 256 --device cpu
   ```
2. **Optional:** increase samples to 500 for better features.

#### 5b – Inspect Features
```bash
# Plot histogram of feature activations (layer 8)
python src/07_inspect_sae.py --histogram --layer 8

# Inspect a suffix-sensitive feature (e.g., 1170)
python src/07_inspect_sae.py --feature 1170
```

> Note: initial run showed very dense activations – retrain with `--l1-coef 3e-3` for better sparsity.

### Step 6 – Documentation & Deliverables
* `README.md` – how to reproduce.
* Trained model + tokenizer on HF Hub.
* SAE artifacts + dashboard screenshots.

---

## 3. Dependencies File (`requirements.txt`)
```
torch>=2.1
transformers>=4.40
datasets>=2.19
saelens @ git+https://github.com/jbloomAus/SAELens.git
wandb
regex
```
