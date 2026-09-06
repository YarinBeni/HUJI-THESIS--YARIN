"""Freeze the five evaluation splits (P0.2).

Reads chrono/artifacts/corpus_chrono.parquet (run make_corpus.py first)
and writes chrono/artifacts/splits/{gkf_ruler, mc_balanced, loro,
source_held_out, object_held_out}.json in the canonical byte-stable
encoding, so a rebuilt split can be diffed against the frozen one by
file hash alone.

    python chrono/scripts/make_splits.py
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from chrono import common                       # noqa: E402
from chrono.data import splits                  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",
                    default=os.path.join(common.ART,
                                         "corpus_chrono.parquet"))
    ap.add_argument("--out-dir",
                    default=os.path.join(common.ART, "splits"))
    ap.add_argument("--seed", type=int, default=splits.SEED)
    args = ap.parse_args()

    corpus = pd.read_parquet(args.corpus)
    for name, sp in build_items(corpus, args.seed):
        path = splits.write_split(sp, args.out_dir)
        sizes = sorted({len(f["test"]) for f in sp["folds"]})
        print(f"[split] {name}: {len(sp['folds'])} folds, "
              f"test sizes {sizes[:6]} -> {path}")


def build_items(corpus, seed):
    built = splits.build_all(corpus, seed=seed)
    return [(n, built[n]) for n in splits.SPLIT_NAMES]


if __name__ == "__main__":
    main()
