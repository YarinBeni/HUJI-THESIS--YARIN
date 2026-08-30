"""Materialize views.parquet (+ confounds.parquet) from a chrono corpus.

WHAT. Runs the A2 augmentation engine over every corpus row and writes
the exact views.parquet schema of SLA section 4, plus the confound audit
table next to it. Import-safe: nothing is read until main() runs.

    python chrono/scripts/make_views.py \
        --corpus chrono/artifacts/corpus_chrono.parquet \
        --out chrono/artifacts/views.parquet --menu default
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from chrono.augment import audit, engine                 # noqa: E402

MENUS = {"default": engine.DEFAULT_MENU, "mild": engine.MENU_MILD}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True,
                    help="corpus parquet (corpus_chrono schema)")
    ap.add_argument("--out", required=True, help="views parquet path")
    ap.add_argument("--menu", default="default", choices=sorted(MENUS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--audit-out", default="",
                    help="confound table path (default: confounds.parquet "
                         "next to --out)")
    args = ap.parse_args(argv)

    import pandas as pd
    corpus = pd.read_parquet(args.corpus)
    views = engine.build_views(corpus, MENUS[args.menu], args.seeds)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    views.to_parquet(args.out, index=False)
    audit_out = args.audit_out or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "confounds.parquet")
    table = audit.confound_table(views, corpus)
    table.to_parquet(audit_out, index=False)
    print(f"{len(views)} views from {corpus.doc_id.nunique()} docs "
          f"-> {args.out}\nconfound audit -> {audit_out}")


if __name__ == "__main__":
    main()
