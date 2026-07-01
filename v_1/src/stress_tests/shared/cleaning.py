"""Cleaning variants for the stress-test extractors — torch-free (unlike
linear_probing/utils.py, which imports torch at module load).

Three cleanings:
  * tier0            — minimal ORACC markup strip (royal names intact, logographic).
  * maximal          — tier0 + the 11 bias-check filters (strips logograms/digits/
                       case → royal names DESTROYED; this is why king_* were tier0-only).
  * maximal_keepking — the "maximal-with-kings" config: locate the commissioning
                       ruler's name span on tier0, FREEZE it, run full `maximal` on
                       everything else, then restore the name verbatim. Gives true
                       maximal *context* with the king token intact, so all three
                       pooling sites (mean / king_last / king_mean) live on ONE
                       cleaning and king coverage = the tier0 ceiling.

                       maximal's `truncate 30 tokens` filter would otherwise drop the
                       name whenever its first occurrence sits past token 30 (it does
                       for ~15-30% of fragments, esp. Ashurbanipal / Sîn-šarru-iškun),
                       collapsing king coverage below tier0. So the truncation here is
                       NAME-AWARE: it keeps the first 30 tokens but always includes the
                       frozen name token, so coverage matches tier0.

The filter set mirrors linear_probing/utils.py (bias-check cell 16); kept in sync
by hand. `find_name_word` comes from king_token (word-level, no model).
"""
from __future__ import annotations

import re

from king_token import find_name_word

# Placeholder that survives every `maximal` filter unchanged (all lowercase, no
# w/y, no case-ending suffix, no digits, not a determinative prefix) so we can
# restore the frozen king name after cleaning the surrounding context.
_PLACEHOLDER = "zzkingnamezz"
_MAX_TOKENS = 30


def clean_tier0(t: str) -> str:
    """Minimal: strip ORACC @v markup, non-breaking space, subscript-x."""
    t = re.sub(r"@[a-z0-9]+", "", t)
    t = t.replace("\xa0", " ")
    t = t.replace("ₓ", "")
    return t


def _truncate(t: str, cap: int = _MAX_TOKENS, force_token: str | None = None) -> str:
    """Keep the first `cap` whitespace tokens; if `force_token` is present in the
    text but outside that window, drop the last kept token to make room for it (so
    the frozen king name is never truncated away)."""
    toks = t.split()
    kept = toks[:cap]
    if force_token is not None and force_token in toks and force_token not in kept:
        kept = toks[: cap - 1] + [force_token]
    return " ".join(kept)


def _maximal_pipeline(t: str, force_token: str | None = None) -> str:
    """tier0 + the 11 maximal filters, in order. `force_token`, when given, makes
    the truncate step name-aware (see _truncate)."""
    t = clean_tier0(t)
    t = re.sub(r"[0-9]", "", t)                                            # strip digits
    t = _truncate(t, _MAX_TOKENS, force_token)                            # truncate 30 (name-aware)
    t = re.sub(r"-(am|im|um|tam|tim|šum)\b", "", t)                       # case endings
    t = t.replace("w", "").replace("y", "")                              # strip w/y
    t = re.sub(r"\b[A-ZŠṢṬḪ][A-ZŠṢṬḪ0-9]+-?", "", t)                     # remove logograms (ALL CAPS)
    t = re.sub(r"\b(I|d|lu2|uru|giš|tug2)-", "", t)                       # strip determinatives
    t = " ".join(re.findall(r"[a-zšṣṭḫāīūē][a-zšṣṭḫāīūē0-9-]*", t))       # keep syllabic
    t = t.translate(str.maketrans("āīūēĀĪŪĒ", "aiueAIUE"))                # normalize long vowels
    t = re.sub(r"([a-zšṣṭḫ])([2-9])", r"\1", t)                          # strip subscript digits
    t = t.lower()                                                         # lowercase
    t = re.sub(r"-meš\b", "", t)                                          # strip -meš plural
    return t


def clean_maximal(t: str) -> str:
    """tier0 + all 11 filters stacked (royal names are destroyed here)."""
    return _maximal_pipeline(t)


def clean_maximal_keepking(text: str, spellings) -> tuple[str, str | None]:
    """Name-protected maximal ("maximal-with-kings").

    Returns (cleaned_text, matched_name). If the ruler's name is not found on the
    tier0 text, returns (plain clean_maximal, None) — the fragment then has no king
    token (found=False), exactly like the plain-maximal case.
    """
    t0 = clean_tier0(text)
    hit = find_name_word(t0, spellings) if spellings else None
    if hit is None:
        return clean_maximal(text), None
    cs, ce, name = hit
    frozen = t0[:cs] + " " + _PLACEHOLDER + " " + t0[ce:]
    cleaned = _maximal_pipeline(frozen, force_token=_PLACEHOLDER)
    cleaned = cleaned.replace(_PLACEHOLDER, name)
    return cleaned, name
