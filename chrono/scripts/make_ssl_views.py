"""make_ssl_views.py — views of every SSL-corpus text for self-supervised
pretraining (PLAN_SCALE_SSL, S2).

No ruler spans exist outside the royal inscriptions, so the menu is the
label-free part of the augmentation library plus one generic corruption:
  crop16 / crop32  contiguous window of n words
  drop_span        delete one ~10% span
  orthonorm        orthographic normalisation (sign indices, determinatives)
  tokmask          replace 15% of words by <MASK>  (BERT-style; also the
                   'context' input for the JEPA objective, whose target is the
                   embedding of the same text WITHOUT the mask)
The clean text is view '' (orig). Two seeds per stochastic op. Output rows:
uid, source, augs, seed, view_id ('ssl::<uid>::<augs>+s<seed>'), text.
"""
from __future__ import annotations
import argparse, os, sys, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chrono.augment.ops import OPS                             # noqa: E402
from chrono import common                                       # noqa: E402

_WORD = re.compile(r"\S+")
MENU = ["", "crop16", "crop32", "drop_span", "orthonorm", "tokmask"]


def tokmask(text, rng, p=0.15):
    words = text.split()
    if len(words) < 4:
        return text
    m = rng.random(len(words)) < p
    if not m.any():
        m[rng.integers(len(words))] = True
    return " ".join("<MASK>" if k else w for w, k in zip(words, m))


def make(text, aug, seed, uid):
    rng = np.random.default_rng(abs(hash((uid, aug, seed))) % (2**32))
    if aug == "":
        return text
    if aug == "tokmask":
        return tokmask(text, rng)
    out, _ = OPS[aug](text, {}, rng)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(common.REPO, "chrono", "artifacts_ssl", "corpus_all.parquet"))
    ap.add_argument("--out", default=os.path.join(common.REPO, "chrono", "artifacts_ssl", "views_ssl.parquet"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "dated"])
    args = ap.parse_args(argv)
    c = pd.read_parquet(args.corpus, columns=["uid", "source", "text", "split"])
    c = c[c["split"].isin(args.splits)]
    rows = []
    for uid, src, text in zip(c["uid"], c["source"], c["text"]):
        for aug in MENU:
            for sd in (args.seeds if aug else [0]):
                rows.append((uid, src, aug, sd, f"ssl::{uid}::{aug}+s{sd}", make(text, aug, sd, uid)))
    v = pd.DataFrame(rows, columns=["uid", "source", "augs", "seed", "view_id", "text"])
    v["n_words"] = v["text"].str.split().str.len()
    # identical views waste GPU and teach nothing: report the rate
    same = (v.merge(c[["uid", "text"]].rename(columns={"text": "orig"}), on="uid")["text"]
            == v.merge(c[["uid", "text"]].rename(columns={"text": "orig"}), on="uid")["orig"])
    v.to_parquet(args.out, index=False)
    print(f"[views] {len(v):,} views for {c.uid.nunique():,} texts; identical-to-orig rate "
          f"(non-orig views) {same[v.augs.ne('').to_numpy()].mean():.3f}; -> {args.out}")
    print(v.groupby("augs")["n_words"].median().to_dict())


if __name__ == "__main__":
    main()
