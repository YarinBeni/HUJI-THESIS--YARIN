"""J17 step 1 — translate the ORCC corpus to English with Thalesian/cuneiformBase-400m.

Translates BOTH text columns (tier0 + maximal) for all 1,202 fragments and writes a
small committed parquet: translation/translations.parquet with columns
  fragment_id, eng_tier0, eng_maximal
plus translations_preview.json (first 5 rows, human-checkable).

The model is a umT5 seq2seq (same checkpoint we already use as an encoder for
embeddings); translation = tokenize("<prefix><akkadian>") -> generate().

TASK PREFIX: T5-family models are prefix-conditioned and the exact string comes
from the HF model card (unreachable from the dev sandbox; check
https://huggingface.co/Thalesian/cuneiformBase-400m). The script therefore
  1) prints a PREFIX PROBE at start: 3 sample fragments translated under every
     candidate prefix — eyeball the log; whichever yields fluent English is right;
  2) uses --prefix (default: first candidate). Override in the sbatch if the
     probe shows a different candidate works better.

Usage:
    python translate_thalesian.py [--prefix "..."] [--batch-size 16]
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
import extract_lib as xl  # noqa: E402  (reuses the robust HF loader + umt5 config patch)

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
HFID = "Thalesian/cuneiformBase-400m"
OUT = _THIS.parent / "translations.parquet"

CANDIDATE_PREFIXES = [
    "Translate Akkadian simple transliteration to English: ",
    "Translate Akkadian grouped transliteration to English: ",
    "Translate Akkadian to English: ",
    "translate Akkadian to English: ",
    "",
]


def translate_batch(model, tok, texts, prefix, max_new=256, beams=2):
    import torch
    inp = [prefix + (t if t.strip() else "...") for t in texts]
    enc = tok(inp, return_tensors="pt", padding=True, truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, num_beams=beams)
    return tok.batch_decode(out, skip_special_tokens=True)


def main(args):
    df = pd.read_parquet(CORPUS)
    tok, _core, model = xl.load_model(HFID, xl.ARCH_ENCODER)
    model.eval()

    # ---- prefix probe: 3 samples x every candidate, for the log ----
    probe = [str(df.iloc[i]["text_tier0"])[:300] for i in (0, 100, 500)]
    print("=" * 70 + "\nPREFIX PROBE — pick the candidate that yields fluent English:")
    for p in CANDIDATE_PREFIXES:
        outs = translate_batch(model, tok, probe, p, max_new=96, beams=1)
        print(f"\n--- prefix = {p!r}")
        for o in outs:
            print("   ->", o[:160])
    print("=" * 70 + f"\nUSING prefix = {args.prefix!r}", flush=True)

    res = {"fragment_id": df["fragment_id"].astype(str).tolist()}
    for col, name in [("text_tier0", "eng_tier0"), ("text_maximal", "eng_maximal")]:
        texts = df[col].astype(str).tolist()
        outs, t0 = [], time.time()
        for i in range(0, len(texts), args.batch_size):
            outs.extend(translate_batch(model, tok, texts[i:i + args.batch_size],
                                        args.prefix))
            if (i // args.batch_size) % 10 == 0:
                print(f"[{name}] {i}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
        res[name] = outs
        print(f"[{name}] DONE {len(outs)} ({time.time()-t0:.0f}s)", flush=True)

    out_df = pd.DataFrame(res)
    out_df.to_parquet(OUT, index=False)
    prev = out_df.head(5).to_dict(orient="records")
    (_THIS.parent / "translations_preview.json").write_text(
        json.dumps({"prefix_used": args.prefix, "rows": prev}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB) + preview")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=CANDIDATE_PREFIXES[0])
    ap.add_argument("--batch-size", type=int, default=16)
    main(ap.parse_args())
