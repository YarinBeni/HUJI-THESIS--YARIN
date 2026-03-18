"""
Simplified Aeneas Twin - Akkadian MLM Model

Architecture based on the Aeneas paper (Assael et al., 2025).
See: yarin/justification/justification_aeneas_twin_architecture.md

Key specifications:
- d_model = 384
- d_ff = 1,536
- d_kv = 32 (per-head dimension)
- num_heads = 8
- num_layers = 16
- Rotary Positional Embeddings (RoPE)
- Pre-Norm (RMSNorm / T5-style LayerNorm)
- 2-layer MLP restoration head
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AeneasConfig:
    """Configuration for Simplified Aeneas Twin model."""
    vocab_size: int = 15000  # Will be set from vocabulary
    d_model: int = 384
    d_ff: int = 1536
    d_kv: int = 32  # Per-head Q/K/V dimension (NOT d_model // num_heads)
    num_heads: int = 8
    num_layers: int = 16
    max_seq_len: int = 768
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    rotary_base: int = 10000

    @property
    def inner_dim(self) -> int:
        """Total dimension for attention (num_heads * d_kv)."""
        return self.num_heads * self.d_kv  # 8 * 32 = 256

    def to_dict(self) -> Dict:
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'd_kv': self.d_kv,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout,
            'layer_norm_eps': self.layer_norm_eps,
            'rotary_base': self.rotary_base,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'AeneasConfig':
        return cls(**d)


# =============================================================================
# Rotary Positional Embeddings (RoPE)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding as used in Aeneas/RoFormer."""

    def __init__(self, dim: int, max_seq_len: int = 1024, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for cos/sin (will be computed on first forward)
        self.register_buffer("cos_cache", None, persistent=False)
        self.register_buffer("sin_cache", None, persistent=False)

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cos/sin cache if needed."""
        if self.cos_cache is not None and self.cos_cache.shape[1] >= seq_len:
            return

        # Compute frequencies for all positions
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(dtype))

        # Double the frequencies (for rotation of pairs)
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]

        # Reshape for broadcasting: [1, 1, seq_len, dim]
        self.cos_cache = emb.cos()[None, None, :, :]
        self.sin_cache = emb.sin()[None, None, :, :]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos/sin embeddings for the sequence.

        Args:
            x: Input tensor of shape [batch, heads, seq_len, head_dim]

        Returns:
            Tuple of (cos, sin) each of shape [1, 1, seq_len, head_dim]
        """
        seq_len = x.shape[2]
        self._update_cache(seq_len, x.device, x.dtype)
        return self.cos_cache[:, :, :seq_len, :], self.sin_cache[:, :, :seq_len, :]


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.

    Args:
        q, k: Tensors of shape [batch, heads, seq_len, head_dim]
        cos, sin: Tensors of shape [1, 1, seq_len, head_dim]

    Returns:
        Rotated q and k tensors
    """
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# Building Blocks
# =============================================================================

class RMSNorm(nn.Module):
    """T5-style RMS Layer Normalization (no mean centering, no bias)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class AeneasAttention(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(self, config: AeneasConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_kv = config.d_kv
        self.inner_dim = config.inner_dim  # num_heads * d_kv = 256
        self.scale = self.d_kv ** -0.5

        # Projections: d_model -> inner_dim (not d_model -> d_model!)
        self.q_proj = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.d_kv, config.max_seq_len, config.rotary_base)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len] with 1 for valid, 0 for padding

        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        B, L, D = x.shape

        # Project and reshape to [batch, heads, seq_len, d_kv]
        q = self.q_proj(x).view(B, L, self.num_heads, self.d_kv).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.d_kv).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.d_kv).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention scores: [batch, heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask (convert from [B, L] to [B, 1, 1, L])
        if attention_mask is not None:
            # Mask: 1 for valid, 0 for padding -> convert to large negative for padding
            mask = (1.0 - attention_mask[:, None, None, :].float()) * -1e9
            attn_scores = attn_scores + mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # [B, H, L, d_kv]
        context = context.transpose(1, 2).contiguous().view(B, L, self.inner_dim)

        # Output projection
        return self.out_proj(context)


class AeneasMLP(nn.Module):
    """Feed-forward MLP block."""

    def __init__(self, config: AeneasConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class AeneasBlock(nn.Module):
    """Single transformer block with Pre-Norm architecture."""

    def __init__(self, config: AeneasConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(config.d_model, config.layer_norm_eps)
        self.attn = AeneasAttention(config)
        self.norm2 = RMSNorm(config.d_model, config.layer_norm_eps)
        self.mlp = AeneasMLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm: x = x + Dropout(Attn(Norm(x)))
        x = x + self.dropout(self.attn(self.norm1(x), attention_mask))
        # Pre-norm: x = x + Dropout(MLP(Norm(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


# =============================================================================
# Main Model
# =============================================================================

class AeneasTorso(nn.Module):
    """The main transformer backbone (torso)."""

    def __init__(self, config: AeneasConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            AeneasBlock(config, layer_idx=i) for i in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        hidden_states_layers: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[int, torch.Tensor]]]:
        """
        Forward pass through the torso.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            output_hidden_states: Whether to return hidden states
            hidden_states_layers: Which layer indices to return (e.g., [0, 4, 8, 12, 16])
                                  Layer 0 = after embedding, Layer i = after block i

        Returns:
            Tuple of:
                - Final hidden states: [batch, seq_len, d_model]
                - Dict mapping layer_idx -> hidden_states (if output_hidden_states=True)
        """
        x = self.embed(input_ids)
        x = self.dropout(x)

        hidden_states_dict = {}

        # Store layer 0 (after embedding) if requested
        if output_hidden_states and hidden_states_layers and 0 in hidden_states_layers:
            hidden_states_dict[0] = x.clone()

        # Pass through transformer blocks
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask)

            # Store hidden states if requested (layer i+1 = after block i)
            layer_num = i + 1
            if output_hidden_states and hidden_states_layers and layer_num in hidden_states_layers:
                hidden_states_dict[layer_num] = x.clone()

        x = self.final_norm(x)

        return x, hidden_states_dict if output_hidden_states else None


class AeneasRestorationHead(nn.Module):
    """Two-layer MLP head for restoration (MLM) task."""

    def __init__(self, config: AeneasConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            Logits: [batch, seq_len, vocab_size]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class AeneasForMLM(nn.Module):
    """Complete model for Masked Language Modeling."""

    def __init__(self, config: AeneasConfig):
        super().__init__()
        self.config = config
        self.torso = AeneasTorso(config)
        self.head = AeneasRestorationHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        hidden_states_layers: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len] with -100 for non-masked positions
            output_hidden_states: Whether to return hidden states
            hidden_states_layers: Which layers to return (e.g., [0, 4, 8, 12, 16])

        Returns:
            Dict with 'logits', optionally 'loss' and 'hidden_states'
        """
        # Get hidden states from torso
        hidden_states, hidden_states_dict = self.torso(
            input_ids,
            attention_mask,
            output_hidden_states,
            hidden_states_layers
        )

        # Get logits from head
        logits = self.head(hidden_states)

        outputs = {'logits': logits}

        # Compute loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            outputs['loss'] = loss

        # Add hidden states if requested
        if output_hidden_states and hidden_states_dict:
            outputs['hidden_states'] = hidden_states_dict

        return outputs

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_embedding_weights(self) -> torch.Tensor:
        """Return the token embedding matrix."""
        return self.torso.embed.weight.data.clone()


def create_model(vocab_size: int, **kwargs) -> AeneasForMLM:
    """Factory function to create model with given vocab size."""
    config = AeneasConfig(vocab_size=vocab_size, **kwargs)
    model = AeneasForMLM(config)
    return model


if __name__ == "__main__":
    # Quick test
    print("Testing Simplified Aeneas Twin model...")

    config = AeneasConfig(vocab_size=15000)
    model = AeneasForMLM(config)

    print(f"\nConfig:")
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")

    print(f"\nModel parameters: {model.get_num_params():,}")

    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = input_ids.clone()
    labels[labels != 4] = -100  # Only predict some positions

    outputs = model(input_ids, attention_mask, labels)
    print(f"\nOutput keys: {outputs.keys()}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")

    # Test hidden states extraction
    outputs_with_hidden = model(
        input_ids, attention_mask, labels,
        output_hidden_states=True,
        hidden_states_layers=[0, 4, 8, 12, 16]
    )
    print(f"\nHidden states layers: {list(outputs_with_hidden['hidden_states'].keys())}")
    for layer_idx, hs in outputs_with_hidden['hidden_states'].items():
        print(f"  Layer {layer_idx}: {hs.shape}")

    print("\n✅ All tests passed!")
