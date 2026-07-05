"""J17 step 1 — translate the ORCC corpus to English with Thalesian/cuneiformBase-400m.

Model-card facts this script honors (huggingface.co/Thalesian/cuneiformBase-400m):
  * multilingual T5 for ancient-script translation; context window 512 tokens and
    "longer texts should be split into individual lines" -> we CHUNK each fragment
    (~20 words per chunk ~ one inscription line; the corpus text column lost line
    boundaries, so word-count chunking is the best proxy), translate chunk by
    chunk, and re-join with spaces.
  * script variants are named in the prompts ("simple transliteration",
    "complex transliteration", cuneiform/syllabary). Our columns map to:
      tier0   = scholarly transliteration with diacritics/indices/logograms
                -> "complex transliteration" prefix
      maximal = stripped/normalized -> "simple transliteration" prefix
  * caveat noted on the card: out-of-domain Akkadian degrades; ORCC royal
    inscriptions are ORACC scholarly transliterations (in-domain family).

The exact prefix strings still come from the card's usage section; a PREFIX PROBE
translates 3 sample chunks under every candidate at start — check the log and
override --prefix-tier0 / --prefix-maximal in the sbatch if needed.

Writes committed translation/translations.parquet
  (fragment_id, eng_tier0, eng_maximal) + translations_preview.json.

Usage:
    python translate_thalesian.py [--prefix-tier0 "..."] [--prefix-maximal "..."]
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
import extract_lib as xl  # noqa: E402  (robust HF loader + umt5 config patch)

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
HFID = "Thalesian/cuneiformBase-400m"
OUT = _THIS.parent / "translations.parquet"
CHUNK_WORDS = 20

CANDIDATE_PREFIXES = [
    "Translate Akkadian complex transliteration to English: ",
    "Translate Akkadian simple transliteration to English: ",
    "Translate Akkadian transliteration to English: ",
    "Translate Akkadian cuneiform to English: ",
    "Translate Akkadian to English: ",
]


def chunks(text: str, n: int = CHUNK_WORDS) -> list[str]:
    w = text.split()
    if not w:
        return ["..."]
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)]


def generate(model, tok, inputs, max_new=128, beams=2):
    import torch
    enc = tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, num_beams=beams)
    return tok.batch_decode(out, skip_special_tokens=True)


def translate_column(model, tok, texts, prefix, batch_size):
    """Chunk every text, translate all chunks batched, re-join per text."""
    all_chunks, owner = [], []
    for i, t in enumerate(texts):
        for c in chunks(str(t)):
            all_chunks.append(prefix + c)
            owner.append(i)
    outs, t0 = [], time.time()
    for i in range(0, len(all_chunks), batch_size):
        outs.extend(generate(model, tok, all_chunks[i:i + batch_size]))
        if (i // batch_size) % 20 == 0:
            print(f"  chunk {i}/{len(all_chunks)} ({time.time()-t0:.0f}s)", flush=True)
    joined = [[] for _ in texts]
    for o, i in zip(outs, owner):
        joined[i].append(o.strip())
    print(f"  DONE {len(all_chunks)} chunks for {len(texts)} texts ({time.time()-t0:.0f}s)", flush=True)
    return [" ".join(j) for j in joined]


def main(args):
    df = pd.read_parquet(CORPUS)
    tok, _core, model = xl.load_model(HFID, xl.ARCH_ENCODER)
    model.eval()

    # ---- prefix probe: 3 sample chunks x every candidate ----
    probe = [chunks(str(df.iloc[i]["text_tier0"]))[0] for i in (0, 100, 500)]
    print("=" * 70 + "\nPREFIX PROBE — pick the candidate that yields fluent English:")
    for p in CANDIDATE_PREFIXES:
        outs = generate(model, tok, [p + c for c in probe], max_new=64, beams=1)
        print(f"\n--- prefix = {p!r}")
        for o in outs:
            print("   ->", o[:160])
    print("=" * 70)
    print(f"USING tier0 prefix   = {args.prefix_tier0!r}")
    print(f"USING maximal prefix = {args.prefix_maximal!r}", flush=True)

    res = {"fragment_id": df["fragment_id"].astype(str).tolist()}
    for col, name, prefix in [("text_tier0", "eng_tier0", args.prefix_tier0),
                              ("text_maximal", "eng_maximal", args.prefix_maximal)]:
        print(f"[{name}] translating with prefix {prefix!r}", flush=True)
        res[name] = translate_column(model, tok, df[col].astype(str).tolist(),
                                     prefix, args.batch_size)

    out_df = pd.DataFrame(res)
    out_df.to_parquet(OUT, index=False)
    prev = out_df.head(3).to_dict(orient="records")
    (_THIS.parent / "translations_preview.json").write_text(
        json.dumps({"prefix_tier0": args.prefix_tier0,
                    "prefix_maximal": args.prefix_maximal,
                    "chunk_words": CHUNK_WORDS, "rows": prev}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB) + preview")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-tier0", default=CANDIDATE_PREFIXES[0],
                    help="tier0 = scholarly/complex transliteration")
    ap.add_argument("--prefix-maximal", default=CANDIDATE_PREFIXES[1],
                    help="maximal = stripped/simple transliteration")
    ap.add_argument("--batch-size", type=int, default=32)
    main(ap.parse_args())
