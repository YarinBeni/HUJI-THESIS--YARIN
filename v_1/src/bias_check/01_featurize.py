#!/usr/bin/env python3
"""
Step 1: Featurize texts for bias check classifiers.

Loads texts_for_evaluation.parquet → TF-IDF char n-gram features → stratified splits.
Saves sparse .npz matrices and fitted vectorizer.

Usage:
    python 01_featurize.py
    python 01_featurize.py --debug   # 10% of data, for local testing
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TEXTS_PARQUET,
    FEATURES_DIR,
    LABEL2IDX,
    LABELS,
    TFIDF_KWARGS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SPLIT_SEED,
)


def main():
    parser = argparse.ArgumentParser(description="Featurize texts for bias check")
    parser.add_argument("--debug", action="store_true",
                        help="Use 10%% of data for fast local testing")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: Featurize")
    print("=" * 60)

    # Load source data
    if not TEXTS_PARQUET.exists():
        print(f"\nError: {TEXTS_PARQUET} not found")
        print("Run v_1/src/evaluation/02_prepare_texts.py first.")
        sys.exit(1)

    print(f"\nLoading {TEXTS_PARQUET}...")
    df = pd.read_parquet(TEXTS_PARQUET, columns=["full_text", "period"])
    print(f"  Loaded {len(df):,} texts")

    # Filter to our three target periods
    before = len(df)
    df = df[df["period"].isin(LABELS)].copy()
    removed = before - len(df)
    if removed > 0:
        print(f"  Dropped {removed} texts with period outside {LABELS}")
    print(f"  Texts for classification: {len(df):,}")

    if args.debug:
        df = df.sample(frac=0.10, random_state=SPLIT_SEED).reset_index(drop=True)
        print(f"  [DEBUG] Subsampled to {len(df):,} texts (10%)")

    # Encode labels
    df["label"] = df["period"].map(LABEL2IDX)
    texts = df["full_text"]
    labels = df["label"].values

    # Print overall distribution
    print("\nClass distribution:")
    for i, cls in enumerate(LABELS):
        n = (labels == i).sum()
        print(f"  {cls}: {n:,} ({n / len(labels):.1%})")

    # -------------------------------------------------------------------------
    # Stratified splits: 70 / 15 / 15
    # Two-step: (train+val) vs test, then train vs val
    # -------------------------------------------------------------------------
    train_val_ratio = TRAIN_RATIO + VAL_RATIO   # 0.85
    val_frac_of_trainval = VAL_RATIO / train_val_ratio  # 0.15 / 0.85

    idx = np.arange(len(texts))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=TEST_RATIO, stratify=labels, random_state=SPLIT_SEED
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_frac_of_trainval,
        stratify=labels[idx_trainval],
        random_state=SPLIT_SEED,
    )

    texts_train = texts.iloc[idx_train]
    texts_val   = texts.iloc[idx_val]
    texts_test  = texts.iloc[idx_test]
    y_train = labels[idx_train]
    y_val   = labels[idx_val]
    y_test  = labels[idx_test]

    print(f"\nSplit sizes:  train={len(texts_train):,}  val={len(texts_val):,}  test={len(texts_test):,}")

    # Verify stratification
    for split_name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        dist = " | ".join(
            f"{LABELS[i]}: {(y_split == i).sum()} ({(y_split == i).mean():.1%})"
            for i in range(len(LABELS))
        )
        print(f"  {split_name}: {dist}")

    # -------------------------------------------------------------------------
    # TF-IDF: fit on train only, transform all splits
    # -------------------------------------------------------------------------
    print(f"\nFitting TfidfVectorizer on train ({len(texts_train):,} texts)...")
    vectorizer = TfidfVectorizer(**TFIDF_KWARGS)
    X_train = vectorizer.fit_transform(texts_train)
    X_val   = vectorizer.transform(texts_val)
    X_test  = vectorizer.transform(texts_test)

    print(f"  Feature matrix shape: {X_train.shape} (train)")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"  Non-zero entries (train): {X_train.nnz:,} "
          f"({X_train.nnz / X_train.shape[0]:.0f} avg per text)")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving features to {FEATURES_DIR}/...")
    save_npz(FEATURES_DIR / "train.npz", X_train)
    save_npz(FEATURES_DIR / "val.npz",   X_val)
    save_npz(FEATURES_DIR / "test.npz",  X_test)
    np.save(FEATURES_DIR / "y_train.npy", y_train)
    np.save(FEATURES_DIR / "y_val.npy",   y_val)
    np.save(FEATURES_DIR / "y_test.npy",  y_test)

    with open(FEATURES_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("  Saved: train.npz, val.npz, test.npz")
    print("  Saved: y_train.npy, y_val.npy, y_test.npy")
    print("  Saved: vectorizer.pkl")

    print("\n" + "=" * 60)
    print("Done! Features ready for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
