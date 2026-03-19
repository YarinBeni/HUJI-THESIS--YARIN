"""
PyTorch model classes for bias check classifiers.

Two architectures:
  BiasCheckMLP          — pure MLP with BatchNorm + Dropout
  BiasCheckAttentionMLP — transformer encoder blocks + MLP head
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    HIDDEN_DIM,
    ATTN_DIM,
    ATTN_HEADS,
    DROPOUT,
    INPUT_DIM,
    NUM_CLASSES,
    MODEL_VARIANTS,
)


class BiasCheckMLP(nn.Module):
    """
    Multi-layer perceptron with BatchNorm + ReLU + Dropout per layer.

    Architecture:
        Linear(input_dim → hidden_dim) → BN → ReLU → Dropout
        [Linear(hidden_dim → hidden_dim) → BN → ReLU → Dropout] × (num_layers - 1)
        Linear(hidden_dim → num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        assert num_layers >= 1

        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            layers += [
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_features = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.hidden(x))


class _TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block: MHA + FFN, both with residual + LayerNorm."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x


class BiasCheckAttentionMLP(nn.Module):
    """
    Transformer encoder blocks followed by an MLP head.

    Architecture:
        Reshape (batch, input_dim) → (batch, seq_len, attn_dim)
            where seq_len = input_dim // attn_dim (no trainable projection)
        [TransformerEncoderBlock] × num_attn_blocks
        Mean-pool over seq_len → (batch, attn_dim)
        BiasCheckMLP(attn_dim → num_classes, num_layers=num_mlp_layers)
    """

    def __init__(
        self,
        input_dim: int,
        attn_dim: int,
        num_heads: int,
        num_attn_blocks: int,
        hidden_dim: int,
        num_mlp_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        assert input_dim % attn_dim == 0, (
            f"input_dim ({input_dim}) must be divisible by attn_dim ({attn_dim})"
        )
        assert attn_dim % num_heads == 0, (
            f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.seq_len = input_dim // attn_dim
        self.attn_dim = attn_dim

        # No large projection linear — reshape TF-IDF chunks directly into sequence
        self.encoder = nn.Sequential(
            *[_TransformerEncoderBlock(attn_dim, num_heads, dropout)
              for _ in range(num_attn_blocks)]
        )
        self.mlp_head = BiasCheckMLP(
            input_dim=attn_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        x = x.view(x.size(0), self.seq_len, self.attn_dim)  # (batch, seq_len, attn_dim)
        x = self.encoder(x)                                   # (batch, seq_len, attn_dim)
        x = x.mean(dim=1)                                     # (batch, attn_dim)
        return self.mlp_head(x)


def build_model(name: str, input_dim: int = INPUT_DIM) -> nn.Module:
    """
    Factory: build a model by name using MODEL_VARIANTS from config.

    Args:
        name: model name (e.g. "mlp_3layer", "attn2_mlp3")
        input_dim: TF-IDF feature dimension

    Returns:
        Initialized (untrained) model
    """
    variant = {v[0]: v for v in MODEL_VARIANTS}.get(name)
    if variant is None:
        raise ValueError(
            f"Unknown model: {name!r}. Valid names: {[v[0] for v in MODEL_VARIANTS]}"
        )
    _, n_attn, n_mlp = variant

    if n_attn == 0:
        return BiasCheckMLP(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=n_mlp,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
        )
    else:
        return BiasCheckAttentionMLP(
            input_dim=input_dim,
            attn_dim=ATTN_DIM,
            num_heads=ATTN_HEADS,
            num_attn_blocks=n_attn,
            hidden_dim=HIDDEN_DIM,
            num_mlp_layers=n_mlp,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
