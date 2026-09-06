"""From-scratch Transformer encoder family for the SSL scaling sweep.

SIZES (parameters excluding the sign embedding, which adds ~vocab x d):
  XS   d 64   L 2   ff 256    ~0.1 M   (tests only)
  S    d 256  L 4   ff 1024   ~3.2 M   -> ~8 M with a 17k-sign vocab
  M    d 384  L 8   ff 1536   ~14 M    -> ~21 M
  L    d 512  L 12  ff 2048   ~38 M    -> ~47 M
  XL   d 768  L 12  ff 3072   ~85 M    -> ~98 M
Pre-LN, learned positions, mean pooling over non-pad tokens -> h; projector
MLP -> p (for the SSL loss); a scalar axis s is kept for the later
fine-tune so the object has the same (h, s, p) interface as AdapterHead.
"""
from __future__ import annotations
import torch
from torch import nn

SIZES = {"XS": dict(d=64, L=2, ff=256, heads=2), "S": dict(d=256, L=4, ff=1024, heads=4),
         "M": dict(d=384, L=8, ff=1536, heads=6), "L": dict(d=512, L=12, ff=2048, heads=8),
         "XL": dict(d=768, L=12, ff=3072, heads=12)}


class SignEncoder(nn.Module):
    def __init__(self, vocab_size: int, size: str = "S", max_len: int = 192, d_proj: int = 256, dropout: float = 0.1):
        super().__init__()
        c = SIZES[size]; d = c["d"]
        self.tok = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, c["heads"], c["ff"], dropout=dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, c["L"])
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d_proj))
        self.axis = nn.Linear(d, 1)
        self.d = d

    def forward(self, ids: torch.Tensor):
        mask = ids.ne(0)
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        x = self.enc(x, src_key_padding_mask=~mask)
        x = self.norm(x)
        m = mask.unsqueeze(-1).to(x.dtype)
        h = (x * m).sum(1) / m.sum(1).clamp_min(1.0)
        return h, self.axis(h).squeeze(-1), self.proj(h)

    def n_params(self, with_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n if with_embedding else n - self.tok.weight.numel() - self.pos.weight.numel()


class HybridEncoder(nn.Module):
    """Third family (PI suggestion, 2026-09-03): the FROZEN encoder's token
    states are the input, a fresh Transformer of the same S/M/L/XL family is
    trained on top with SSL. Between the shallow adapter (one pooled vector ->
    MLP) and the from-scratch model (raw signs): it inherits everything the
    frozen model learned from its much larger pretraining data, but learns a
    new COMPOSITION of the tokens instead of a plain mean. Token masking is
    done in embedding space with a learned <mask> vector, so it works for any
    frozen tokenizer."""

    def __init__(self, d_frozen: int, size: str = "S", d_proj: int = 256, dropout: float = 0.1):
        super().__init__()
        c = SIZES[size]; d = c["d"]
        self.inp = nn.Linear(d_frozen, d)
        self.mask_vec = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, c["heads"], c["ff"], dropout=dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, c["L"])
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d_proj))
        self.axis = nn.Linear(d, 1)
        self.d = d

    def forward(self, states: torch.Tensor, mask: torch.Tensor, drop: torch.Tensor | None = None):
        """states [B,T,d_frozen], mask [B,T] bool (real tokens), drop [B,T] bool
        positions whose token vector is replaced by the learned mask vector."""
        x = self.inp(states.to(self.inp.weight.dtype))
        if drop is not None:
            x = torch.where(drop.unsqueeze(-1), self.mask_vec.to(x.dtype).expand_as(x), x)
        x = self.enc(x, src_key_padding_mask=~mask)
        x = self.norm(x)
        m = mask.unsqueeze(-1).to(x.dtype)
        h = (x * m).sum(1) / m.sum(1).clamp_min(1.0)
        return h, self.axis(h).squeeze(-1), self.proj(h)

    def n_params(self, with_embedding=True):
        return sum(p.numel() for p in self.parameters())
