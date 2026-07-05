"""translation_fidelity.py — does the Thalesian English translation name the right king?

For each fragment: does eng_tier0 / eng_maximal contain the commissioning ruler's
English name (variant list below)? Reported overall and CONDITIONAL on the Akkadian
tier0 actually containing the name (per ruler_spellings.csv). Also counts wrong-king
hallucinations (famous kings NOT the commissioner, e.g. Ashurnasirpal II).

Usage:  python v_1/src/stress_tests/eda/translation_fidelity.py
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import pandas as pd

ST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ST / "shared"))
from king_token import find_name_word, load_spellings  # noqa: E402

ENG = {
    "Esarhaddon": ["esarhaddon"], "Sennacherib": ["sennacherib"],
    "Ashurbanipal": ["ashurbanipal", "assurbanipal"], "Sargon II": ["sargon"],
    "Tiglath-pileser III": ["tiglath"], "Sîn-šarru-iškun": ["sin-sarru", "sin-shar", "sinshar"],
    "Nebuchadnezzar II": ["nebuchadnezzar", "nebuchadrezzar"], "Nabonidus": ["nabonidus"]}
WRONG = ["ashurnasirpal", "shalmaneser", "adad-nirari", "tukulti"]


def norm(s):
    n = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in n if not unicodedata.combining(c)).lower()


def main():
    tr = pd.read_parquet(ST / "translation/translations.parquet")
    df = pd.read_parquet(ST.parents[2] / "data/evaluation/corpora/orcc_corpus.parquet")
    m = df.merge(tr, on="fragment_id")
    spell = load_spellings()
    print(f"{'ruler':22s} {'n':>4} {'akk-in-t0':>9} {'et0-right':>9} {'et0-wrong':>9} "
          f"{'emax-right':>10} {'emax-wrong':>10} {'et0|akk':>8}")
    for r, variants in ENG.items():
        sub = m[m["ruler"] == r]
        if not len(sub):
            continue
        sp = spell.get(r, [])
        akk = (sub["text_tier0"].apply(lambda t: find_name_word(str(t), sp) is not None)
               if sp else pd.Series(False, index=sub.index))
        et0 = sub["eng_tier0"].apply(norm); emx = sub["eng_maximal"].apply(norm)
        rt0 = et0.apply(lambda t: any(v in t for v in variants))
        rmx = emx.apply(lambda t: any(v in t for v in variants))
        wt0 = et0.apply(lambda t: any(w in t for w in WRONG))
        wmx = emx.apply(lambda t: any(w in t for w in WRONG))
        cond = rt0[akk].mean() if akk.any() else float("nan")
        print(f"{r:22s} {len(sub):>4} {akk.mean():>9.2f} {rt0.mean():>9.2f} {wt0.mean():>9.2f} "
              f"{rmx.mean():>10.2f} {wmx.mean():>10.2f} {cond:>8.2f}")


if __name__ == "__main__":
    main()
