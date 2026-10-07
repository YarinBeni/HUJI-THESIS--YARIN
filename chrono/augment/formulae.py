"""Rule library for strip_formula: dateable royal-inscription boilerplate.

WHY THESE RULES. The F28 erasure ladder ranked ruler identity, then period,
then object type as the strongest shortcuts a probe can ride; titulary and
opening formulae are the textual carriers of the first two (Esarhaddon's
"LUGAL GAL-u2 LUGAL dan-nu LUGAL kiš-ša2-ti" dates a fragment by convention,
not by content). strip_formula removes exactly those spans so a lateness
score must survive without them.

WHAT THE RULES COVER. Three registers, because the corpus carries three:
tier-0 transliteration with logograms (`text_tier0` in the real ORCC
parquet — the only register available locally, so tier-0 rules are derived
from actual corpus rows and their counts are noted), normalized/ascii
Akkadian (the `text_maximal`-style strings and the toy fixture), and the
English literal gloss (patterns from standard RINAP renderings — no gloss
column exists locally, so these serve future eng tiers and the fixtures).

PRECISION OVER RECALL. Every rule is anchored to token boundaries
(lookahead `(?=\s|$)`) or to the start of the text; fuzzy classes are kept
to the single token they must absorb (a royal name, a number word). Curses
in transliteration are precative verb morphology, too variable to match
precisely, so only one tight curse rule is kept on that side; the English
"may the god(s) ..." clause rule carries the recall there. The min-words
guard below means stripping can never empty a document: callers get the
ORIGINAL text back, flagged, when fewer than MIN_WORDS_RETAINED words
would survive.
"""
from __future__ import annotations

import re
from collections import namedtuple

MIN_WORDS_RETAINED = 5

# name, register ('tier0'|'akk'|'eng'), compiled regex, one real example.
Rule = namedtuple("Rule", "name lang rx example")


def _r(name, lang, pattern, example, flags=0):
    return Rule(name, lang, re.compile(pattern, flags), example)


RULES = (
    # --- tier-0 transliteration (derived from orcc_corpus.text_tier0) ---
    _r("t0_palace_opening", "tier0",
       r"^E2-GAL\s+\S+(?=\s|$)",
       "E2-GAL m-d-aš-šur-ŠEŠ-SUM-NA  ('Palace of Esarhaddon', 94 docs)"),
    _r("t0_great_king", "tier0",
       r"LUGAL GAL(?:-u2?)?(?=\s|$)",
       "LUGAL GAL-u2  ('great king', 63 docs)"),
    _r("t0_mighty_king", "tier0",
       r"LUGAL dan-nu(?=\s|$)",
       "LUGAL dan-nu  ('mighty king', 126 docs)"),
    _r("t0_king_of_world", "tier0",
       r"LUGAL kiš-ša2?-ti(?=\s|$)",
       "LUGAL kiš-ša2-ti  ('king of the world', 38 docs)"),
    _r("t0_king_of_assyria", "tier0",
       r"LUGAL KUR aš-šur(?:-ki)?(?=\s|$)",
       "LUGAL KUR aš-šur-ki  ('king of Assyria', 108 docs)"),
    _r("t0_king_four_quarters", "tier0",
       r"LUGAL kib-rat \S+(?=\s|$)",
       "LUGAL kib-rat LIMMU2-ti  ('king of the four quarters', 74 docs)"),
    _r("t0_viceroy_babylon", "tier0",
       r"GIR3-NITA2 KA2-DINGIR-RA-ki(?=\s|$)",
       "GIR3-NITA2 KA2-DINGIR-RA-ki  ('viceroy of Babylon', 84 docs)"),
    _r("t0_king_sumer_akkad", "tier0",
       r"LUGAL KUR EME-GI7 u URI-ki(?=\s|$)",
       "LUGAL KUR EME-GI7 u URI-ki  ('king of Sumer and Akkad', 23 docs)"),
    _r("t0_royal_genealogy", "tier0",
       r"DUMU m-\S+ LUGAL(?: KUR aš-šur(?:-ki)?)?(?=\s|$)",
       "DUMU m-d-30-PAP-MEŠ-SU LUGAL KUR aš-šur-ki  ('son of Sennacherib, "
       "king of Assyria', 76 docs)"),
    _r("t0_favorite_of_gods", "tier0",
       r"mi-gir DINGIR-MEŠ GAL-MEŠ(?=\s|$)",
       "mi-gir DINGIR-MEŠ GAL-MEŠ  ('favorite of the great gods', 51 docs)"),
    _r("t0_curse_angrily", "tier0",
       r"ag-giš li-ru-ur(?:-šu2?-ma)?(?=\s|$)",
       "ag-giš li-ru-ur-šu2-ma  ('may (the god) curse him angrily')"),
    # --- normalized / ascii Akkadian (text_maximal register) ---
    _r("akk_sar_mat_assur", "akk",
       r"[šs]ar m[āa]t a[šs][šs]ur(?=\s|$)",
       "sar mat assur  ('king of the land of Assur')"),
    _r("akk_king_of_assyria_max", "akk",
       r"aš-šur-ki(?=\s|$)",
       "aš-šur-ki  (residual Assyria logogram cluster in text_maximal)"),
    _r("akk_mighty_king_max", "akk",
       r"dan-nu kiš-ša2?-ti(?=\s|$)",
       "dan-nu kiš-ša-ti  ('mighty (king) of the world', 24 docs)"),
    # --- English literal gloss (RINAP-style renderings) ---
    _r("eng_palace_opening", "eng",
       r"^Palace of \S+(?=\s|$)",
       "Palace of Ashurbanipal", re.IGNORECASE),
    _r("eng_great_mighty_king", "eng",
       r"\bgreat king,? mighty king\b",
       "great king, mighty king", re.IGNORECASE),
    _r("eng_king_of_assyria", "eng",
       r"\bking of Assyria\b",
       "Ashurbanipal king of Assyria", re.IGNORECASE),
    _r("eng_king_of_world", "eng",
       r"\bking of the (?:world|universe|four quarters"
       r"(?: of the world)?)\b",
       "king of the four quarters of the world", re.IGNORECASE),
    _r("eng_king_sumer_akkad", "eng",
       r"\bking of (?:the land of )?Sumer and Akkad\b",
       "king of Sumer and Akkad", re.IGNORECASE),
    _r("eng_governor_of", "eng",
       r"\b(?:governor|viceroy) of \S+(?=\s|$)",
       "viceroy of Babylon", re.IGNORECASE),
    _r("eng_royal_genealogy", "eng",
       r"\bson of \S+,? (?:great )?king of Assyria\b",
       "son of Sennacherib, king of Assyria", re.IGNORECASE),
    _r("eng_true_shepherd", "eng",
       r"\b(?:true|faithful) shepherd\b",
       "true shepherd, favorite of the great gods", re.IGNORECASE),
    _r("eng_curse_may_god", "eng",
       r"\bmay the (?:great )?god(?:s|dess(?:es)?)?\b[^.;:]{0,120}",
       "may the great gods curse him with an evil curse", re.IGNORECASE),
)


def merge_spans(spans):
    """Merge overlapping-or-touching [start, end) spans; sorted output."""
    out = []
    for a, b in sorted([int(a), int(b)] for a, b in spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def find_formula_spans(text):
    """All rule matches as (start, end, rule_name), sorted, may overlap."""
    hits = []
    for rule in RULES:
        for m in rule.rx.finditer(text):
            if m.end() > m.start():
                hits.append((m.start(), m.end(), rule.name))
    return sorted(hits)


def removal_spans(text):
    """Merged char spans to delete, each widened over adjacent whitespace
    (trailing normally; leading when the span touches the text's end) so a
    deletion never leaves doubled spaces behind."""
    spans = merge_spans([[a, b] for a, b, _ in find_formula_spans(text)])
    n = len(text)
    for sp in spans:
        while sp[1] < n and text[sp[1]].isspace():
            sp[1] += 1
        if sp[1] == n:
            while sp[0] > 0 and text[sp[0] - 1].isspace():
                sp[0] -= 1
    return merge_spans(spans)


def strip_formulae(text, min_words=MIN_WORDS_RETAINED):
    """Delete every formula span. Returns (text, removed_spans, flagged).

    flagged=True means the min-words guard fired: fewer than `min_words`
    words would have survived, so `text` is returned UNCHANGED and
    removed_spans reports what would have been cut.
    """
    spans = removal_spans(text)
    if not spans:
        return text, [], False
    kept, cur = [], 0
    for a, b in spans:
        kept.append(text[cur:a])
        cur = b
    kept.append(text[cur:])
    stripped = "".join(kept)
    if len(stripped.split()) < min_words:
        return text, spans, True
    return stripped, spans, False
