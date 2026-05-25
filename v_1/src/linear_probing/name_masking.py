#!/usr/bin/env python3
"""Canonical personal-name masking for the ORCC dating control.

The control question: does TF-IDF (or any model) date Akkadian texts by reading
*who* the text is about (royal/personal names) or by period orthography that
survives masking those names?

Akkadian transliteration here is space-separated WORDS, hyphen-separated SIGNS
within a word (e.g. `m-d-30-PAP-MEŠ-SU` is ONE whitespace token). Personal names
are flagged by determinatives:

  m-   male personal-name determinative  -> the whole token is a personal name
       (catches the Assyrian kings: Sennacherib m-d-30-PAP-MEŠ-SU,
        Ashurbanipal m-AN-ŠAR2-DU3-A, Sargon m-LUGAL-GI-NA, Esarhaddon ...).

  d-   divine determinative. Marks *every* god, so it is NOT a name by itself.
       BUT god-head + a predicate = a theophoric SENTENCE-NAME of a person
       (e.g. d-AG-NIG2-DU-URU3 = Nabû-kudurri-uṣur = Nebuchadnezzar). These
       leak past m- masking and were empirically the top dating features
       (Nebuchadnezzar, Nabonidus, Nabopolassar). We mask god-head + predicate;
       we KEEP the bare god (d-AG, d-da-gan, d-15 = Ištar, d-maš = Ninurta ...).

Discriminator (auditable, not hand-listed):
  - `m-<...>`                       -> mask  (any male personal name)
  - `f-<...>`                       -> mask  (any female personal name)
  - `d-<godhead>-<predicate...>`    -> mask  (theophoric personal name)
    where <godhead> is a name-forming theophoric element. Bare `d-<godhead>`
    (no further signs) is the GOD and is kept.

This is intentionally conservative toward REMOVING names: a theophoric name of a
non-royal official is masked too, because the control claim is "dating does not
depend on ANY personal name," not just royal ones.
"""
import re

# --- male / female personal-name determinatives: whole token is a name ---
_PERS_DET = re.compile(r'(?<!\S)[mf]-\S+')

# --- theophoric god-heads that form personal (sentence) names. A bare god of
# this head is kept; the head FOLLOWED by >=1 further sign is a person's name.
# Case-insensitive; covers the common gods that head Babylonian/Assyrian royal
# and personal names. The trailing `-\S+` is the mandatory predicate, so the
# bare god token (d-AG, d-30, d-AMAR-UTU, d-aš-šur, ...) is NOT matched.
_THEO_HEADS = [
    r'ag', r'na-bi-um', r'muati', r'pa',          # Nabû  (the empirical leak)
    r'amar-utu',                                   # Marduk
    r'aš-šur', r'a-šur', r'an-šar2',               # Aššur (personal, e.g. Aššur-X)
    r'30', r'en-zu', r'sin',                       # Sîn
    r'utu', r'šam-aš', r'šam-ši',                  # Šamaš
    r'en', r'en-lil2', r'nin-urta', r'maš',        # Bēl / Enlil / Ninurta
]
_THEO = re.compile(
    r'(?<!\S)d-(?:' + '|'.join(_THEO_HEADS) + r')-\S+',
    re.IGNORECASE,
)

REPL = "[PN]"


def mask_personal_names(text: str) -> str:
    """Replace male/female and theophoric personal-name tokens with [PN]."""
    t = _PERS_DET.sub(REPL, str(text))
    t = _THEO.sub(REPL, t)
    return t


# ---------------------------------------------------------------------------
# Audit mode: `python name_masking.py [corpus.parquet]` prints every distinct
# token the masker touches (with corpus frequency) so the blocklist is fully
# reviewable, then a SAFETY SCAN of the top dating features AFTER masking to
# confirm no king name still leaks.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    from collections import Counter
    import pandas as pd
    import numpy as np

    root = Path(__file__).resolve().parents[3]
    pq = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        root / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
    df = pd.read_parquet(pq)
    df = df[df["year"].notna()].copy()
    txt = df["text_tier0"].fillna("").astype(str)

    toks = Counter(" ".join(txt).split())
    masked_tokens = {t: n for t, n in toks.items()
                     if mask_personal_names(t) == REPL}
    print(f"=== MASKED token inventory ({len(masked_tokens)} distinct) ===")
    for t, n in sorted(masked_tokens.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {t}")

    total = sum(toks.values())
    mtot = sum(masked_tokens.values())
    print(f"\nmasked {mtot:,}/{total:,} tokens ({100*mtot/total:.2f}%)\n")

    # SAFETY SCAN: top dating features after masking must be vocabulary, not names
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    masked_txt = txt.apply(mask_personal_names)
    y = df["year"].astype(float).values
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=5)
    X = vec.fit_transform(masked_txt)
    feats = np.array(vec.get_feature_names_out())
    coef = Ridge(alpha=1.0).fit(X, y).coef_
    print("=== SAFETY SCAN: top dating features AFTER masking ===")
    print("(should be places/titles/spelling, NOT personal names)")
    print("LATE:", [feats[i] for i in np.argsort(coef)[::-1][:15]])
    print("EARLY:", [feats[i] for i in np.argsort(coef)[:15]])
