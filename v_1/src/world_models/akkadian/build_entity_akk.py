"""WB-akk — the missing cell of the entity matrix: ruler NAMES in Akkadian.

The entity ladder so far held language fixed at the name level (CELL-B probes
"Ashurbanipal" in English) and varied language only at fragment level, so the
document "cliff" confounds two candidate causes: the language/script of the
surface form, and the span/aggregation. This builds the fourth cell — the same
rulers, the same templates, the same split, but the entity string is the
ruler's ATTESTED TRANSLITERATED NAME, harvested from his own inscriptions.

Harvesting: the chrono corpus ships a pre-masked Akkadian tier where personal
names are replaced by `[PN]` (chrono/augment/engine.py, review fix F1).
Aligning `text_akk` against `text_akk_masked` recovers the exact substrings
the mask covered — the attested spellings. A royal name is then a spelling
that (a) appears in >= MIN_DOCS of the ruler's documents and (b) appears
predominantly (>= DOMINANCE) in THIS ruler's documents — fathers and
predecessors are also named in royal inscriptions, so raw frequency alone
would harvest genealogy.

The split (`is_test`) is copied per-ruler from the English dataset, so the
English<->Akkadian comparison is fold-matched: a ruler is in test in both
languages or in neither.

Output: ../data/entity_datasets/assyrian_ruler_akk.csv (same schema as
assyrian_ruler.csv) and a human-readable harvest report on stdout.

    python build_entity_akk.py              # writes the CSV
    python build_entity_akk.py --report     # harvest report only
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
CORPUS = os.path.join(_ROOT, "chrono", "artifacts_tier0", "corpus_chrono.parquet")
ENG_CSV = os.path.join(os.path.dirname(_HERE), "data", "entity_datasets", "assyrian_ruler.csv")
OUT_CSV = os.path.join(os.path.dirname(_HERE), "data", "entity_datasets", "assyrian_ruler_akk.csv")

PN = "[PN]"
MIN_DOCS = 2          # a spelling must recur (1 for rulers with <3 docs)
DOMINANCE = 0.7       # share of the spelling's docs that belong to this ruler
MAX_SPELLINGS = 2     # keep at most this many spellings per ruler
# Same carrier sentences as the English set, so the only change is the name.
TEMPLATES = ["{e}", "The king {e} ruled over the land.",
             "An inscription commissioned by {e}."]
TEMPLATE_IDS = ["bare", "t1", "t2"]
_DAMAGE = re.compile(r"[\[\]⌈⌉⸢⸣«»!?#<>]")


def pn_strings(akk: str, masked: str) -> list:
    """The substrings of `akk` that `masked` covers with [PN], by alignment."""
    segs = masked.split(PN)
    if len(segs) < 2:
        return []
    out, pos = [], 0
    j = akk.find(segs[0], pos)
    if j != 0 and segs[0].strip():
        return []                                   # texts don't align; skip doc
    pos = j + len(segs[0])
    for seg in segs[1:]:
        if seg == "":
            continue                                # adjacent PNs: unrecoverable split
        k = akk.find(seg, pos)
        if k < pos:
            return []                               # mis-alignment; skip doc
        out.append(akk[pos:k])
        pos = k + len(seg)
    if masked.endswith(PN):
        out.append(akk[pos:])
    return out


def clean(s: str) -> str:
    s = _DAMAGE.sub("", s)
    return re.sub(r"\s+", " ", s).strip(" -–—.,;:")


def harvest(corpus: pd.DataFrame):
    """ruler -> [(spelling, n_docs)], after the dominance filter."""
    by_spell = defaultdict(lambda: defaultdict(set))   # spelling -> ruler -> docs
    n_docs, n_aligned = 0, 0
    for r in corpus.itertuples():
        m = str(r.text_akk_masked or "")
        if PN not in m:
            continue
        n_docs += 1
        names = pn_strings(str(r.text_akk or ""), m)
        if names:
            n_aligned += 1
        # TITULARY PRIOR (first harvest's lesson): counting every PN hands the
        # dominance filter the king's ENEMIES — Teumman outnumbers Ashurbanipal
        # inside Ashurbanipal's own annals, Merodach-baladan outnumbers
        # Sennacherib. A royal inscription OPENS with the royal name, so only
        # the first personal name of each document is a candidate spelling.
        for s in {clean(n) for n in names[:1]}:
            if 3 <= len(s) <= 60:
                by_spell[s][r.ruler].add(r.doc_id)
    print(f"[harvest] {n_aligned}/{n_docs} PN-bearing docs aligned", flush=True)
    per_ruler = defaultdict(list)
    docs_of = corpus.groupby("ruler")["doc_id"].nunique()
    for s, owners in by_spell.items():
        total = sum(len(d) for d in owners.values())
        ruler, docs = max(owners.items(), key=lambda kv: len(kv[1]))
        need = MIN_DOCS if docs_of.get(ruler, 0) >= 3 else 1
        if len(docs) >= need and len(docs) / total >= DOMINANCE:
            per_ruler[ruler].append((s, len(docs)))
    return {ru: sorted(v, key=lambda x: -x[1])[:MAX_SPELLINGS]
            for ru, v in per_ruler.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)

    corpus = pd.read_parquet(CORPUS)
    spells = harvest(corpus)
    eng = pd.read_csv(ENG_CSV)
    is_test = eng.drop_duplicates("name").set_index("name")["is_test"].to_dict()
    year = (-corpus.groupby("ruler")["t"].median()).to_dict()   # positive BC years
    ntex = corpus.groupby("ruler")["doc_id"].nunique().to_dict()

    print("\n[report] ruler -> attested spellings (n docs):")
    for ru in sorted(spells):
        print(f"  {ru:<28} " + " | ".join(f"{s!r} ({n})" for s, n in spells[ru]))
    missing = sorted(set(corpus.ruler.unique()) - set(spells))
    print(f"[report] rulers with no surviving spelling: {len(missing)}: {missing}")
    if args.report:
        return

    rows, eix = [], 0
    for ru in sorted(spells):
        if ru not in is_test:
            continue                       # keep the fold-matched set only
        for s, _n in spells[ru]:
            for tid, tpl in zip(TEMPLATE_IDS, TEMPLATES):
                text = tpl.format(e=s)
                c0 = tpl.index("{e}")
                rows.append(dict(name=ru, entity_ix=eix, template=f"{tid}",
                                 entity_string=text, ent_start=c0,
                                 ent_end=c0 + len(s),
                                 death_year=float(year[ru]),
                                 n_texts=int(ntex.get(ru, 0)),
                                 is_test=bool(is_test[ru])))
        eix += 1
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[build] wrote {OUT_CSV}: {len(df)} rows, "
          f"{df.name.nunique()} rulers, {int(df.is_test.sum())} test rows")


if __name__ == "__main__":
    main()
