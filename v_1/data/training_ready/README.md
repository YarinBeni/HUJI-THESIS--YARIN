# training_ready/ — Tokenized Model-Ready Data

The final stage before training. Contains tokenized fragments, vocabulary, and a small evaluation subset. Generated from `unified/` by the data preparation script.

## Files

| File | Description |
|------|-------------|
| `train_fragments.parquet` | Tokenized training fragments (32,343 texts) |
| `val_fragments.parquet` | Tokenized validation fragments (4,042 texts) |
| `test_fragments.parquet` | Tokenized test fragments (4,044 texts) |
| `vocab.json` | Sign-level vocabulary — 14,797 tokens (including 5 special tokens) |
| `metadata.json` | Config used to generate this prepared data (vocab size, split sizes, seed, source data_dir) |
| `eval_subset.parquet` | 500-sample random subset drawn from test set — used for fast evaluation during training |
| `eval_subset_ids.json` | Fragment IDs in the eval subset (for reproducibility) |

## Generation

**Script:** `src/archive/baseline_mlm/01_prepare_data.py`

**Input:** `unified/` (train/val/test parquets)

**Key params:** vocab built from training set only; seed=42; eval_subset_size=500

## Consumed By

- `src/archive/baseline_mlm/02_train.py` — main training script reads all fragments + vocab
