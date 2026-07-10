"""E5 — word-shuffle control: does word ORDER matter to the embedding signal?

If the balanced-MC year Spearman survives randomly permuting each fragment's
words, the "signal" the probes read is lexical / bag-of-tokens — the model is
not composing the text — which matches the TF-IDF-equivalence finding from the
other direction (TF-IDF is order-free by construction).

Design (order is the ONLY difference):
  * texts per cleaning exactly as in T11 (tier0 / maximal / maxking / engtier0);
  * cap at --max-frag-words FIRST (same first-N words in both variants), THEN
    shuffle (seeded, deterministic) -> content identical, order differs;
  * BOTH variants are extracted here with identical settings (tokenizer cap,
    pooling), so the shuf-vs-unshuf delta is immune to any historical
    extraction-setting mismatch. unshuf also sanity-checks against the old rows.

Output acts dirs (found by geo_loader.find_acts_dir):
  {acts_root}/{method}_shuf{cleaning}_mean/layer_LL.npz
  {acts_root}/{method}_unshuf{cleaning}_mean/layer_LL.npz

Usage:
  python extract_shuffled_acts.py --hfid Qwen/Qwen3-8B --method qwen3_8b \
      --arch causal --max-frag-words 300 --max-tokens 2048
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]
sys.path.insert(0, str(_THIS.parents[1] / "shared"))
sys.path.insert(0, str(_THIS.parents[1] / "t11_gen_dating"))
import extract_lib as xl                       # noqa: E402
from generate_dates import fragment_texts      # noqa: E402  (same texts as T11)

DEFAULT_CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
DEFAULT_TRANSLATIONS = _THIS.parents[1] / "translation/translations.parquet"
DEFAULT_ACTS = _REPO / "v_1/src/linear_probing/results/orcc__embed/activations"

CLEANINGS = ["tier0", "maximal", "maxking", "engtier0"]
SEED = 42


def build_variants(df, cleaning, translations, max_frag_words):
    """-> (unshuf_texts, shuf_texts): word-capped first, then shuffled."""
    texts, _ = fragment_texts(df, cleaning, translations)
    rng = np.random.default_rng(SEED)   # corpus order fixed -> deterministic
    unshuf, shuf = [], []
    for t in texts:
        words = (t.strip() or "...").split()
        if max_frag_words and len(words) > max_frag_words:
            words = words[:max_frag_words]
        unshuf.append(" ".join(words))
        perm = list(words)
        rng.shuffle(perm)
        shuf.append(" ".join(perm))
    return unshuf, shuf


def embed_all(texts, tok, core, max_tokens):
    """-> {layer: (N, D) float32} mean-pooled per layer."""
    import torch
    buf, layers, t0 = {}, None, time.time()
    for i, text in enumerate(texts):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens)
        ids = enc["input_ids"]
        if ids.shape[1] == 0:
            fb = tok.eos_token_id if tok.eos_token_id is not None else (tok.pad_token_id or 0)
            ids = torch.tensor([[fb]])
        attn = torch.ones_like(ids)
        ids = ids.to(core.device if hasattr(core, "device") else "cuda")
        attn = attn.to(ids.device)
        hs = xl.hidden_states_for(core, ids, attn)
        if layers is None:
            layers = list(range(len(hs)))
            for L in layers:
                buf[L] = []
        for L in layers:
            buf[L].append(xl.pool_all_sites(hs[L], attn[0], None)["mean"])
        if i % 200 == 0:
            print(f"    {i}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    return {L: np.vstack(buf[L]).astype(np.float32) for L in layers}


def run(args):
    df = pd.read_parquet(args.corpus)
    tok, core, _model = xl.load_model(args.hfid, args.arch, random=args.random)

    for cleaning in args.cleanings.split(","):
        assert cleaning in CLEANINGS, cleaning
        unshuf, shuf = build_variants(df, cleaning, Path(args.translations),
                                      args.max_frag_words)
        for tag, texts in [(f"unshuf{cleaning}", unshuf), (f"shuf{cleaning}", shuf)]:
            outdir = Path(args.acts_root) / f"{args.method}_{tag}_mean"
            if not args.overwrite and outdir.is_dir() and any(outdir.glob("layer_*.npz")):
                print(f"[{args.method}/{tag}] exists — skip", flush=True)
                continue
            outdir.mkdir(parents=True, exist_ok=True)
            print(f"[{args.method}/{tag}] embedding {len(texts)} fragments", flush=True)
            acts = embed_all(texts, tok, core, args.max_tokens)
            for L, X in acts.items():
                np.savez_compressed(outdir / f"layer_{L:02d}.npz", activations=X)
            (outdir / "metadata.json").write_text(json.dumps({
                "method": args.method, "hfid": args.hfid, "cleaning": tag,
                "experiment": "e5_word_shuffle", "seed": SEED,
                "max_frag_words": args.max_frag_words, "max_tokens": args.max_tokens,
                "n": len(texts), "n_layers": len(acts)}, indent=2), encoding="utf-8")
            print(f"[{args.method}/{tag}] DONE layers={len(acts)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hfid", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--arch", required=True, choices=[xl.ARCH_CAUSAL, xl.ARCH_ENCODER])
    p.add_argument("--random", action="store_true")
    p.add_argument("--cleanings", default=",".join(CLEANINGS),
                   help="comma list; default all four")
    p.add_argument("--max-frag-words", type=int, default=300,
                   help="word cap applied BEFORE shuffling (both variants see the "
                        "same words; only order differs)")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="tokenizer safety cap; set so the word cap fits (encoder "
                        "models: 512 -> use a smaller --max-frag-words)")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--translations", default=str(DEFAULT_TRANSLATIONS))
    p.add_argument("--acts-root", default=str(DEFAULT_ACTS))
    p.add_argument("--overwrite", action="store_true")
    run(p.parse_args())
