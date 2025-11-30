"""Custom restoration transformer with rotary embeddings."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_rotary_cache(dim: int, max_position: int, base: int, device):
    """Return cos/sin caches shaped [1, 1, seq_len, dim] for easy broadcasting with (B,H,S,D)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(max_position, device=device).float()
    freqs = torch.einsum("i,j->ij", positions, inv_freq)  # [seq, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq, dim]
    cos = emb.cos()[None, None, :, :]  # [1,1,seq,dim]
    sin = emb.sin()[None, None, :, :]  # [1,1,seq,dim]
    return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, seq_len, head_dim = x.shape
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


@dataclass
class RestorationConfig:
    vocab_size: int
    max_position: int = 768
    d_model: int = 384
    n_heads: int = 8
    d_ff: int = 1536
    n_layers: int = 16
    dropout: float = 0.1
    rotary_base: int = 10000


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: RestorationConfig):
        super().__init__()
        self.num_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, config: RestorationConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, attn_mask=None):
        attn_out = self.attn(self.ln1(x), cos, sin, attn_mask)
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.ln2(x))
        return x + self.dropout(ff_out)


class RestorationModel(nn.Module):
    def __init__(self, config: RestorationConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        self.max_position = config.max_position
        self.rotary_dim = config.d_model // config.n_heads
        self.register_buffer("cos_cache", None, persistent=False)
        self.register_buffer("sin_cache", None, persistent=False)

    def _get_rotary(self, device):
        if self.cos_cache is None or self.cos_cache.device != device:
            cos, sin = _build_rotary_cache(self.rotary_dim, self.max_position, self.config.rotary_base, device)
            self.cos_cache = cos
            self.sin_cache = sin
        return self.cos_cache, self.sin_cache

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        bsz, seq_len = input_ids.shape
        x = self.embed(input_ids)
        x = self.dropout(x)
        cos, sin = self._get_rotary(x.device)
        attn_mask = None
        if attention_mask is not None:
            mask = (1.0 - attention_mask[:, None, None, :]) * -1e9
            attn_mask = mask
        for layer_idx, layer in enumerate(self.layers, 1):
            x = layer(x, cos, sin, attn_mask)
            # Debug: Log layer activations if batch size is small (e.g., debugging mode)
            if bsz <= 4 and layer_idx == 1:  # Only log first layer to avoid spam
                print(f"      Layer {layer_idx}: x shape {x.shape}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        x = self.final_ln(x)
        logits = self.head(x)
        return logits

    def compute_loss(self, input_ids, labels, attention_mask=None):
        logits = self.forward(input_ids, attention_mask)
        loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)
        return loss, logits
