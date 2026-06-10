"""Tokenizer pre-EDA for the Akkadian NTP fine-tune (Task 5, 03.06 meeting).

Question this answers: how badly do the stock Qwen3 / gpt-oss tokenizers
fragment Akkadian transliteration, what do they already cover, and how much
would a domain BPE (-> vocab expansion) buy us?

Inputs:  v_1/data/unified/{train,val,test}.parquet (word-level rows)
Outputs: v_1/src/finetune/eda/results/
           tokenizer_eda.json            all computed stats
           candidate_tokens_<model>.txt  domain-BPE tokens absent from model vocab
           TOKENIZER_EDA.md              human-readable report

Run locally (no GPU needed):
    venv/bin/python v_1/src/finetune/eda/tokenizer_eda.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "v_1" / "data" / "unified"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

TOKENIZERS = {
    "qwen3": "Qwen/Qwen3-8B",          # identical tokenizer across Qwen3 1.7B/8B/32B
    "gpt_oss": "openai/gpt-oss-120b",  # o200k_harmony, identical for gpt-oss-20b
}

DOMAIN_BPE_SIZES = [4_000, 8_000, 16_000]
N_CANDIDATES = 12_000  # candidate new tokens to dump per model


# --- tier0 cleaning (same as corpus builds / linear_probing/utils.py) -------
def clean_tier0(t: str) -> str:
    t = re.sub(r"@[a-z0-9]+", "", t)
    t = t.replace("\xa0", " ")
    t = t.replace("ₓ", "")
    return t


def build_fragment_texts() -> pd.DataFrame:
    """Fragment-level texts: space-joined clean_value (fallback raw), tier0-cleaned.

    Mirrors v_1/src/corpus/03_build_orcc_corpus.py text assembly.
    """
    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(DATA / f"{split}.parquet")
        df["word"] = df["value_clean"].fillna(df["value_raw"]).astype(str)
        df = df.sort_values(["fragment_id", "line_num", "word_idx"])
        g = df.groupby("fragment_id", sort=False)
        frag = pd.DataFrame({
            "text": g["word"].apply(" ".join),
            "n_words": g["word"].size(),
            "n_signs": g["value_signs"].apply(
                lambda s: int(sum(len(str(x).split()) for x in s))),
            "source": g["source"].first(),
        })
        frag["split"] = split
        frag["text"] = frag["text"].apply(clean_tier0)
        frames.append(frag.reset_index())
    return pd.concat(frames, ignore_index=True)


def is_byte_fallback(piece_str: str) -> bool:
    """Token decodes to (partial) invalid UTF-8 -> it's a raw-byte fragment."""
    return "�" in piece_str


def analyze_tokenizer(name: str, model_id: str, frags: pd.DataFrame,
                      word_counter: Counter) -> dict:
    tok = AutoTokenizer.from_pretrained(model_id)
    texts = frags["text"].tolist()

    enc = tok(texts, add_special_tokens=False)["input_ids"]
    n_tokens_per_frag = [len(ids) for ids in enc]
    frags[f"n_tok_{name}"] = n_tokens_per_frag

    total_tokens = int(sum(n_tokens_per_frag))
    total_words = int(frags["n_words"].sum())
    total_signs = int(frags["n_signs"].sum())
    total_chars = int(frags["text"].str.len().sum())

    # global token-id frequency + byte-fallback share
    tid_counter: Counter = Counter()
    for ids in enc:
        tid_counter.update(ids)
    decoded = {tid: tok.decode([tid]) for tid in tid_counter}
    fallback_tokens = sum(c for tid, c in tid_counter.items()
                          if is_byte_fallback(decoded[tid]))
    unique_used = len(tid_counter)

    # token length distribution (chars per decoded token, weighted by freq)
    len_counter: Counter = Counter()
    for tid, c in tid_counter.items():
        len_counter[len(decoded[tid].strip())] += c

    # how the special characters fare
    special_chars = {}
    for ch in ["š", "ṣ", "ṭ", "ḫ", "ī", "ā", "ū", "ē", "₂", "₃", "₄", "ʾ"]:
        ids = tok.encode(ch, add_special_tokens=False)
        ids_mid = tok.encode("a" + ch, add_special_tokens=False)
        special_chars[ch] = {
            "alone_n_tokens": len(ids),
            "after_a_n_tokens": len(ids_mid) ,
            "alone_pieces": [tok.decode([i]) for i in ids],
        }

    # coverage of the most frequent word forms
    top_words = word_counter.most_common(2000)
    single, double, more = 0, 0, 0
    examples = []
    for w, c in top_words:
        n = len(tok.encode(" " + w, add_special_tokens=False))
        if n == 1:
            single += 1
        elif n == 2:
            double += 1
        else:
            more += 1
        if len(examples) < 40:
            pieces = [tok.decode([i]) for i in
                      tok.encode(" " + w, add_special_tokens=False)]
            examples.append({"word": w, "freq": c, "n_tokens": n,
                             "pieces": pieces})

    per_frag = pd.Series(n_tokens_per_frag)
    return {
        "model_id": model_id,
        "vocab_size": len(tok),
        "total_tokens": total_tokens,
        "tokens_per_word": round(total_tokens / total_words, 3),
        "tokens_per_sign": round(total_tokens / total_signs, 3),
        "chars_per_token": round(total_chars / total_tokens, 3),
        "unique_tokens_used": unique_used,
        "pct_byte_fallback_tokens": round(100 * fallback_tokens / total_tokens, 2),
        "token_char_len_dist": {str(k): int(v) for k, v in
                                sorted(len_counter.items())},
        "frag_tokens_quantiles": {q: int(per_frag.quantile(q)) for q in
                                  (0.5, 0.9, 0.95, 0.99)},
        "frags_over_2048_tokens": int((per_frag > 2048).sum()),
        "top2000_words_single_token": single,
        "top2000_words_two_tokens": double,
        "top2000_words_3plus_tokens": more,
        "special_chars": special_chars,
        "top_word_examples": examples,
    }


def train_domain_bpe(texts: list[str], vocab_size: int):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    bpe = Tokenizer(models.BPE(unk_token=None))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=[], show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    bpe.train_from_iterator(texts, trainer=trainer)
    return bpe


def main() -> None:
    print("[eda] building fragment texts ...")
    frags = build_fragment_texts()
    print(f"[eda] {len(frags)} fragments, {frags['n_words'].sum():,} words, "
          f"{frags['text'].str.len().sum():,} chars")

    word_counter: Counter = Counter()
    for t in frags["text"]:
        word_counter.update(t.split())

    results: dict = {
        "corpus": {
            "n_fragments": int(len(frags)),
            "n_words": int(frags["n_words"].sum()),
            "n_signs": int(frags["n_signs"].sum()),
            "n_chars": int(frags["text"].str.len().sum()),
            "n_unique_words": len(word_counter),
            "by_split": frags.groupby("split")["n_words"].sum().to_dict(),
        },
        "tokenizers": {},
        "domain_bpe": {},
    }

    for name, model_id in TOKENIZERS.items():
        print(f"[eda] analyzing {name} ({model_id}) ...")
        results["tokenizers"][name] = analyze_tokenizer(
            name, model_id, frags, word_counter)

    # --- domain BPE: achievable-fertility lower bound + candidate tokens ----
    texts = frags["text"].tolist()
    total_words = int(frags["n_words"].sum())
    biggest = None
    for vs in DOMAIN_BPE_SIZES:
        print(f"[eda] training domain BPE vocab={vs} ...")
        bpe = train_domain_bpe(texts, vs)
        n_tok = sum(len(bpe.encode(t).ids) for t in texts)
        results["domain_bpe"][str(vs)] = {
            "total_tokens": int(n_tok),
            "tokens_per_word": round(n_tok / total_words, 3),
        }
        biggest = bpe

    # candidate new tokens = domain-BPE vocab entries that the stock vocabs
    # do not have as a single token
    print("[eda] computing candidate token lists ...")
    from tokenizers import decoders
    bld = decoders.ByteLevel()
    vocab_items = biggest.get_vocab()  # byte-level piece -> id

    def piece_to_str(piece: str) -> str:
        try:
            return bld.decode([piece])
        except Exception:
            return ""

    decoded_vocab = []
    for piece, _id in vocab_items.items():
        s = piece_to_str(piece)
        if len(s.strip()) >= 3:
            decoded_vocab.append(s)

    for name, model_id in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(model_id)
        existing = set()
        cands = []
        for s in decoded_vocab:
            if len(tok.encode(s, add_special_tokens=False)) == 1:
                existing.add(s)
            else:
                cands.append(s)
        cands.sort(key=lambda s: -len(s))
        path = OUT / f"candidate_tokens_{name}.txt"
        path.write_text("\n".join(cands[:N_CANDIDATES]), encoding="utf-8")
        results["tokenizers"][name]["domain_bpe_pieces_already_single_token"] = \
            len(existing)
        results["tokenizers"][name]["domain_bpe_pieces_missing"] = len(cands)
        print(f"[eda] {name}: {len(existing)} domain pieces already single-token, "
              f"{len(cands)} missing -> {path.name}")

    with open(OUT / "tokenizer_eda.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[eda] wrote {OUT / 'tokenizer_eda.json'}")


if __name__ == "__main__":
    main()
