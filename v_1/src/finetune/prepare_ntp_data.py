"""Build the Akkadian NTP fine-tune corpus (Task 5).

Fragment-level texts from the canonical unified splits, constructed exactly
like the ORCC corpus build (v_1/src/corpus/03_build_orcc_corpus.py): words
sorted by (line_num, word_idx), space-joined `value_clean` with fallback to
`value_raw`, tier0-cleaned. Train split -> CPT training data, val split ->
perplexity eval. Test split is left untouched.

Also reports the overlap with the ORCC probing corpus (leakage accounting —
see eda/results/TOKENIZER_EDA.md §5).

Run (CPU, ~2 min):
    python v_1/src/finetune/prepare_ntp_data.py
Outputs:
    v_1/data/finetune/ntp_train.parquet   (fragment_id, text, n_words, source)
    v_1/data/finetune/ntp_val.parquet
    v_1/data/finetune/metadata.json
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
UNIFIED = REPO / "v_1" / "data" / "unified"
ORCC_PARQUET = REPO / "v_1" / "data" / "evaluation" / "corpora" / "orcc_corpus.parquet"
OUT_DIR = REPO / "v_1" / "data" / "finetune"


def clean_tier0(t: str) -> str:
    """Same as v_1/src/linear_probing/utils.py — minimal markup strip."""
    t = re.sub(r"@[a-z0-9]+", "", t)
    t = t.replace("\xa0", " ")
    t = t.replace("ₓ", "")
    return t


def build_split(split: str) -> pd.DataFrame:
    df = pd.read_parquet(UNIFIED / f"{split}.parquet")
    df["word"] = df["value_clean"].fillna(df["value_raw"]).astype(str)
    df = df.sort_values(["fragment_id", "line_num", "word_idx"])
    g = df.groupby("fragment_id", sort=False)
    frag = pd.DataFrame({
        "text": g["word"].apply(" ".join),
        "n_words": g["word"].size(),
        "source": g["source"].first(),
    }).reset_index()
    frag["text"] = frag["text"].apply(clean_tier0)
    frag = frag[frag["text"].str.strip().str.len() > 0].reset_index(drop=True)
    return frag


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict = {"created": datetime.now().isoformat(),
                  "construction": "value_clean (fallback value_raw), sorted "
                                  "(line_num, word_idx), space-joined, tier0"}

    orcc_ids = set(pd.read_parquet(ORCC_PARQUET, columns=["fragment_id"])
                   ["fragment_id"].astype(str)) if ORCC_PARQUET.exists() else set()

    for split, out_name in (("train", "ntp_train"), ("val", "ntp_val")):
        frag = build_split(split)
        out = OUT_DIR / f"{out_name}.parquet"
        frag.to_parquet(out, index=False)
        overlap = len(orcc_ids & set(frag["fragment_id"].astype(str)))
        meta[split] = {
            "n_fragments": int(len(frag)),
            "n_words": int(frag["n_words"].sum()),
            "n_chars": int(frag["text"].str.len().sum()),
            "orcc_probe_overlap": overlap,
        }
        print(f"[{split}] {len(frag)} fragments, {frag['n_words'].sum():,} words "
              f"-> {out}  (ORCC-probe overlap: {overlap})")

    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {OUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
