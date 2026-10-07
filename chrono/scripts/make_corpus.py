"""Build the canonical corpus artifacts (P0.1).

Reads the raw ORCC parquet plus the eng_tier0 translations, applies the
eligibility contract of chrono/data/contract.py, and writes
chrono/artifacts/{corpus_chrono.parquet, ruler_table.parquet}. The
census and span coverage are printed so any drift is visible in logs;
build_corpus() itself hard-fails if the census leaves 1,187/40/47.

    python chrono/scripts/make_corpus.py
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from chrono import common                       # noqa: E402
from chrono.data import contract                # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orcc", default=common.ORCC)
    ap.add_argument("--trans", default=contract.TRANS)
    ap.add_argument("--out-dir", default=common.ART)
    ap.add_argument("--akk-tier", default="maximal",
                    choices=sorted(contract.AKK_TIERS),
                    help="Akkadian text tier (see contract.AKK_TIERS)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = contract.build_corpus(args.orcc, args.trans, akk_tier=args.akk_tier)
    p_corpus = os.path.join(args.out_dir, "corpus_chrono.parquet")
    df.to_parquet(p_corpus, index=False)

    rt = contract.build_ruler_table(df)
    p_ruler = os.path.join(args.out_dir, "ruler_table.parquet")
    rt.to_parquet(p_ruler, index=False)

    cov_eng = (df["ruler_spans_eng"].str.len() > 0).mean()
    cov_akk = (df["ruler_spans_akk"].str.len() > 0).mean()
    print(f"[corpus] {len(df)} rows | {df.ruler.nunique()} rulers | "
          f"{df.t.nunique()} distinct t | t in "
          f"[{df.t.min():.0f}, {df.t.max():.0f}]")
    print(f"[spans]  eng coverage {cov_eng:.3f} | akk coverage "
          f"{cov_akk:.3f}")
    print(f"[write]  {p_corpus}")
    print(f"[write]  {p_ruler} ({len(rt)} rulers)")


if __name__ == "__main__":
    main()
