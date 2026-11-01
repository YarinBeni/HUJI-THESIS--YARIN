"""Sign-level tokenizer for Akkadian transliteration.

Words are split on hyphens and dots into signs; determinatives ({d}, {ki},
(munus)…) and logograms stay as tokens; case is kept (upper-case logograms
carry meaning). Vocabulary from the TRAIN split only, tokens seen < min_count
times map to <unk>. Deterministic, dependency-free, fast enough to run inside
the data loader (views are generated on the fly).
"""
from __future__ import annotations
import json, re
from collections import Counter

_SPLIT = re.compile(r"[-.\s]+")
SPECIALS = ["<pad>", "<unk>", "<cls>", "<mask>"]


def signs(text: str) -> list[str]:
    return [t for t in _SPLIT.split(text) if t]


class SignTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab; self.pad, self.unk, self.cls, self.mask = 0, 1, 2, 3

    @classmethod
    def fit(cls, texts, min_count: int = 2, max_size: int = 40000) -> "SignTokenizer":
        c = Counter(); [c.update(signs(t)) for t in texts]
        toks = [t for t, n in c.most_common(max_size) if n >= min_count]
        return cls({t: i for i, t in enumerate(SPECIALS + toks)})

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.vocab.get(t, self.unk) for t in signs(text)][: max_len - 1]
        return [self.cls] + ids

    def save(self, path): json.dump(self.vocab, open(path, "w"))
    @classmethod
    def load(cls, path): return cls(json.load(open(path)))
    def __len__(self): return len(self.vocab)
