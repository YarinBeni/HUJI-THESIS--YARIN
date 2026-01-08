"""
Data utilities for Akkadian MLM training.

This module provides functions to:
1. Load parquet data and reconstruct fragment-level text
2. Build sign vocabulary
3. Create PyTorch datasets for MLM training
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset


# Special tokens for MLM
SPECIAL_TOKENS = {
    "pad": "[PAD]",
    "unk": "[UNK]",
    "cls": "[CLS]",
    "sep": "[SEP]",
    "mask": "[MASK]",
}

SPECIAL_TOKEN_IDS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
}


def load_parquet_split(parquet_path: str) -> pd.DataFrame:
    """Load a parquet file."""
    return pd.read_parquet(parquet_path)


def reconstruct_fragment_text(fragment_df: pd.DataFrame) -> str:
    """
    Reconstruct text for a single fragment from its rows.

    Each row has a `value_signs` column containing space-separated signs.
    We sort by (line_num, word_idx), then join all value_signs with spaces.

    Example:
        Row 1: value_signs = "a na"
        Row 2: value_signs = "be li"
        Row 3: value_signs = "u"

        Output: "a na be li u"

    The resulting string can be split on spaces to get individual sign tokens.

    Args:
        fragment_df: DataFrame containing rows for a single fragment

    Returns:
        Space-separated string of all signs in the fragment
    """
    # Sort by line number and word position
    fragment_df = fragment_df.sort_values(['line_num', 'word_idx'])

    # Get all value_signs and join with space
    signs_list = fragment_df['value_signs'].dropna().tolist()

    # Join all value_signs (each already contains space-separated signs)
    text = ' '.join(signs_list)

    return text


def build_fragment_texts(
    df: pd.DataFrame,
    include_metadata: bool = False
) -> pd.DataFrame:
    """
    Build a DataFrame with one row per fragment, containing reconstructed text.

    Args:
        df: Raw parquet DataFrame with one row per word
        include_metadata: If True, include source, domain columns

    Returns:
        DataFrame with columns: fragment_id, text, num_signs, [source, domain]
    """
    results = []

    for fragment_id, group in df.groupby('fragment_id'):
        text = reconstruct_fragment_text(group)
        signs = text.split()

        row = {
            'fragment_id': fragment_id,
            'text': text,
            'num_signs': len(signs),
        }

        if include_metadata:
            row['source'] = group['source'].iloc[0]
            # Domain might be a list string, take first value
            domain = group['domain'].iloc[0] if 'domain' in group.columns else ''
            row['domain'] = domain if pd.notna(domain) else ''

        results.append(row)

    return pd.DataFrame(results)


def build_sign_vocabulary(
    df: pd.DataFrame,
    min_freq: int = 1,
    save_path: Optional[str] = None
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build a vocabulary mapping from all unique signs in the dataset.

    Args:
        df: DataFrame with 'value_signs' column (one row per word)
        min_freq: Minimum frequency for a sign to be included
        save_path: Optional path to save vocab as JSON

    Returns:
        Tuple of (sign_to_id, id_to_sign) dictionaries
    """
    # Count all signs
    sign_counter = Counter()

    for value_signs in df['value_signs'].dropna():
        signs = value_signs.split()
        sign_counter.update(signs)

    # Filter by minimum frequency
    valid_signs = [sign for sign, count in sign_counter.items() if count >= min_freq]
    valid_signs = sorted(valid_signs)  # Sort for reproducibility

    # Build vocabulary starting after special tokens
    sign_to_id = dict(SPECIAL_TOKEN_IDS)  # Start with special tokens
    next_id = len(SPECIAL_TOKEN_IDS)

    for sign in valid_signs:
        if sign not in sign_to_id:
            sign_to_id[sign] = next_id
            next_id += 1

    # Build reverse mapping
    id_to_sign = {v: k for k, v in sign_to_id.items()}

    # Save if path provided
    if save_path:
        vocab_data = {
            'sign_to_id': sign_to_id,
            'id_to_sign': {str(k): v for k, v in id_to_sign.items()},  # JSON keys must be strings
            'vocab_size': len(sign_to_id),
            'num_special_tokens': len(SPECIAL_TOKEN_IDS),
            'num_signs': len(valid_signs),
        }
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    return sign_to_id, id_to_sign


def load_vocabulary(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load vocabulary from JSON file."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    sign_to_id = vocab_data['sign_to_id']
    id_to_sign = {int(k): v for k, v in vocab_data['id_to_sign'].items()}

    return sign_to_id, id_to_sign


def tokenize_text(
    text: str,
    sign_to_id: Dict[str, int],
    max_length: int = 768,
    add_special_tokens: bool = True
) -> Tuple[List[int], List[int]]:
    """
    Tokenize a text string into token IDs.

    Args:
        text: Space-separated string of signs
        sign_to_id: Vocabulary mapping
        max_length: Maximum sequence length (including special tokens)
        add_special_tokens: Whether to add [CLS] and [SEP]

    Returns:
        Tuple of (input_ids, attention_mask)
    """
    signs = text.split()

    # Convert to IDs
    unk_id = sign_to_id['[UNK]']
    token_ids = [sign_to_id.get(sign, unk_id) for sign in signs]

    # Add special tokens
    if add_special_tokens:
        cls_id = sign_to_id['[CLS]']
        sep_id = sign_to_id['[SEP]']
        # Truncate to make room for special tokens
        max_content_length = max_length - 2
        token_ids = token_ids[:max_content_length]
        token_ids = [cls_id] + token_ids + [sep_id]
    else:
        token_ids = token_ids[:max_length]

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = [1] * len(token_ids)

    # Pad to max_length
    pad_id = sign_to_id['[PAD]']
    padding_length = max_length - len(token_ids)
    if padding_length > 0:
        token_ids = token_ids + [pad_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length

    return token_ids, attention_mask


class AkkadianMLMDataset(Dataset):
    """
    PyTorch Dataset for Akkadian MLM training.

    Each item is a fragment, tokenized and ready for MLM.
    Masking is applied dynamically during __getitem__.
    """

    def __init__(
        self,
        fragment_texts_df: pd.DataFrame,
        sign_to_id: Dict[str, int],
        max_length: int = 768,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            fragment_texts_df: DataFrame with 'fragment_id' and 'text' columns
            sign_to_id: Vocabulary mapping
            max_length: Maximum sequence length
            mask_prob: Probability of masking a token (default 15%)
            mask_token_prob: Of masked tokens, probability of replacing with [MASK] (default 80%)
            random_token_prob: Of masked tokens, probability of replacing with random token (default 10%)
            seed: Random seed for reproducibility
        """
        self.texts = fragment_texts_df['text'].tolist()
        self.fragment_ids = fragment_texts_df['fragment_id'].tolist()
        self.sign_to_id = sign_to_id
        self.id_to_sign = {v: k for k, v in sign_to_id.items()}
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob
        self.vocab_size = len(sign_to_id)

        # IDs for special tokens
        self.pad_id = sign_to_id['[PAD]']
        self.mask_id = sign_to_id['[MASK]']
        self.cls_id = sign_to_id['[CLS]']
        self.sep_id = sign_to_id['[SEP]']

        # Random generator for reproducibility
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        input_ids, attention_mask = tokenize_text(
            text, self.sign_to_id, self.max_length, add_special_tokens=True
        )

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # Apply MLM masking
        input_ids, labels = self._apply_mlm_masking(input_ids, attention_mask)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def _apply_mlm_masking(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply BERT-style MLM masking.

        - 15% of tokens are selected for prediction
        - Of those: 80% replaced with [MASK], 10% random token, 10% unchanged
        - Labels are -100 for non-masked positions (ignored in loss)
        """
        labels = input_ids.clone()

        # Create mask for positions that can be masked (not special tokens or padding)
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        special_tokens_mask[input_ids == self.pad_id] = True
        special_tokens_mask[input_ids == self.cls_id] = True
        special_tokens_mask[input_ids == self.sep_id] = True

        # Probability matrix for masking
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        probability_matrix[special_tokens_mask] = 0.0

        # Sample which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix, generator=self.rng).bool()

        # Set labels to -100 for non-masked tokens
        labels[~masked_indices] = -100

        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, self.mask_token_prob), generator=self.rng
        ).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_id

        # 10% of the time, replace with random token
        random_prob = self.random_token_prob / (1 - self.mask_token_prob)
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, random_prob), generator=self.rng
        ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(SPECIAL_TOKEN_IDS), self.vocab_size, input_ids.shape,
            dtype=torch.long, generator=self.rng
        )
        input_ids[indices_random] = random_words[indices_random]

        # 10% of the time, keep original (already done - no change needed)

        return input_ids, labels


def create_eval_subset(
    fragment_texts_df: pd.DataFrame,
    n_fragments: int = 500,
    seed: int = 42,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a fixed evaluation subset for pre/post training analysis.

    Args:
        fragment_texts_df: DataFrame with fragment texts
        n_fragments: Number of fragments to include
        seed: Random seed for reproducibility
        save_path: Optional path to save fragment IDs

    Returns:
        DataFrame with the selected fragments
    """
    # Sample deterministically
    eval_subset = fragment_texts_df.sample(
        n=min(n_fragments, len(fragment_texts_df)),
        random_state=seed
    )

    if save_path:
        # Save just the fragment IDs for reproducibility
        eval_ids = eval_subset['fragment_id'].tolist()
        with open(save_path, 'w') as f:
            json.dump({'fragment_ids': eval_ids, 'seed': seed}, f, indent=2)

    return eval_subset


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }


if __name__ == "__main__":
    # Quick test
    import sys

    print("Testing data_utils.py...")

    # Load a small sample
    train_path = "v_1/data/processed/unified/train.parquet"
    df = pd.read_parquet(train_path)

    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {df.columns.tolist()}")

    # Build fragment texts
    print("\nBuilding fragment texts...")
    fragment_df = build_fragment_texts(df.head(1000), include_metadata=True)
    print(f"Built {len(fragment_df)} fragment texts")
    print(f"Sample:\n{fragment_df.head()}")

    # Build vocabulary
    print("\nBuilding vocabulary...")
    sign_to_id, id_to_sign = build_sign_vocabulary(df)
    print(f"Vocabulary size: {len(sign_to_id)}")
    print(f"First 10 signs: {list(sign_to_id.items())[:10]}")

    print("\n✅ All tests passed!")
