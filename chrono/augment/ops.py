"""Confound-removal ops: the augmentation registry (SLA section 4).

WHAT. Pure functions `(text, spans_dict, rng) -> (text, spans_dict)`,
registered under canonical names in OPS and composed left-to-right by
augment.engine. spans_dict maps a span type (today only "ruler") to
[start, end) char spans; every op returns spans REMAPPED through its own
edits, so a later op (e.g. mask_ruler after a crop) still points at the
right characters. Inputs are never mutated.

WHY A SINGLE EDIT KERNEL. Every op is expressed as a list of
non-overlapping (start, end, replacement) edits fed to _apply_edits, which
rewrites the text once and remaps all spans through one piecewise offset
function. That is the only place offsets are computed, so masking,
stripping, cropping, orthographic normalization and span dropping cannot
disagree about where a ruler name ended up.

Determinism: ops draw only from the numpy Generator handed in; ops that
need no randomness ignore it but keep the uniform signature.
"""
from __future__ import annotations

import re
import unicodedata

from chrono.augment import formulae

MASK_TOKEN = "<RULER>"
_WORD = re.compile(r"\S+")
# determinative prefixes as they look AFTER lowercasing + diacritic
# folding but BEFORE digit stripping ("lu2-", "na4-" stay distinctive);
# conservative on purpose — a false strip corrupts a real word.
_DETS = ("m", "f", "d", "uru", "kur", "lu2", "gis", "iti", "na4",
         "anse", "munus", "id2", "tug2")
_DET_SUFFIXES = ("-ki", "-mes")
_MASK_RX = re.compile(r"^<[A-Z]+>$")


def _copy_spans(spans_dict):
    return {k: [[int(a), int(b)] for a, b in v]
            for k, v in spans_dict.items()}


def _map_pos(i, meta, end):
    """Old char index -> new, through edit meta (s, e, new_s, new_e).
    Positions inside an edited region clamp to the replacement; `end`
    picks the clamp side so [a, b) spans clip rather than bleed."""
    off = 0
    for s, e, ns, ne in meta:
        if i <= s:
            break
        if i < e or (end and i == e):
            return ne if end else ns
        off = ne - e
    return i + off


def _apply_edits(text, spans_dict, edits):
    """Apply sorted non-overlapping (start, end, repl) edits; remap every
    span, dropping spans that collapse to zero length."""
    if not edits:
        return text, _copy_spans(spans_dict)
    edits = sorted((int(s), int(e), r) for s, e, r in edits)
    for (_, e1, _), (s2, _, _) in zip(edits, edits[1:]):
        if e1 > s2:
            raise ValueError("overlapping edits")
    parts, meta, cur, pos = [], [], 0, 0
    for s, e, r in edits:
        parts.append(text[cur:s])
        pos += s - cur
        meta.append((s, e, pos, pos + len(r)))
        parts.append(r)
        pos += len(r)
        cur = e
    parts.append(text[cur:])
    new_spans = {}
    for key, sp in spans_dict.items():
        kept = []
        for a, b in sp:
            na = _map_pos(int(a), meta, end=False)
            nb = _map_pos(int(b), meta, end=True)
            if nb > na:
                kept.append([na, nb])
        new_spans[key] = kept
    return "".join(parts), new_spans


def _clean_spans(spans, n):
    """Clip to [0, n), drop empties, merge overlaps."""
    clipped = [[max(0, int(a)), min(n, int(b))] for a, b in spans]
    return formulae.merge_spans([s for s in clipped if s[1] > s[0]])


def mask_ruler(text, spans_dict, rng):
    """Replace every ruler span with the typed token <RULER>. Idempotent:
    masked spans point at the token itself, so a second pass rewrites
    <RULER> with <RULER>."""
    spans = _clean_spans(spans_dict.get("ruler", []), len(text))
    edits = [(a, b, MASK_TOKEN) for a, b in spans]
    out, new_spans = _apply_edits(text, spans_dict, edits)
    if "ruler" in new_spans:
        new_spans["ruler"] = formulae.merge_spans(new_spans["ruler"])
    return out, new_spans


def strip_formula(text, spans_dict, rng):
    """Delete formula spans via the formulae rule library. The min-words
    guard is enforced there: when it fires the text (and spans) come back
    unchanged rather than gutted."""
    stripped, removed, flagged = formulae.strip_formulae(text)
    if flagged or not removed:
        return text, _copy_spans(spans_dict)
    return _apply_edits(text, spans_dict, [(a, b, "") for a, b in removed])


def _delete_words(text, spans_dict, w_lo, w_hi, words):
    """Delete words[w_lo:w_hi] plus one flank of whitespace."""
    a, b = words[w_lo].start(), words[w_hi - 1].end()
    while b < len(text) and text[b].isspace():
        b += 1
    if b == len(text):
        while a > 0 and text[a - 1].isspace():
            a -= 1
    return _apply_edits(text, spans_dict, [(a, b, "")])


def _make_crop(n):
    def crop(text, spans_dict, rng):
        words = list(_WORD.finditer(text))
        if len(words) <= n:
            return text, _copy_spans(spans_dict)
        start = int(rng.integers(0, len(words) - n + 1))
        a = words[start].start()
        b = words[start + n - 1].end()
        return _apply_edits(text, spans_dict,
                            [(0, a, ""), (b, len(text), "")])
    crop.__name__ = crop.__qualname__ = f"crop{n}"
    crop.__doc__ = (f"Contiguous {n}-word crop at an rng-drawn offset; "
                    f"texts of <= {n} words pass through whole.")
    return crop


crop8 = _make_crop(8)
crop16 = _make_crop(16)
crop32 = _make_crop(32)
crop64 = _make_crop(64)


def _norm_token(tok):
    """Orthographic collapse for one token: lowercase, fold diacritics
    (š->s, ṣ->s, ṭ->t via NFD), drop aleph marks and sign-index digits,
    strip determinative prefixes/suffixes. Typed mask tokens (<RULER>)
    pass through untouched. Idempotent by construction."""
    if _MASK_RX.match(tok):
        return tok
    t = unicodedata.normalize("NFD", tok.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("ʾ", "").replace("’", "").replace("'", "")
    changed = True
    while changed:
        changed = False
        for p in _DETS:
            if t.startswith(p + "-") and len(t) > len(p) + 1:
                t = t[len(p) + 1:]
                changed = True
    t = re.sub(r"(?<=[a-z])[0-9]+", "", t)
    for sfx in _DET_SUFFIXES:
        if t.endswith(sfx) and len(t) > len(sfx):
            t = t[:-len(sfx)]
    return t if t else tok.lower()


def orthonorm(text, spans_dict, rng):
    """Collapse orthographic variation (case, diacritics, sign indices,
    determinatives) so views cannot be dated by spelling convention."""
    edits = []
    for m in _WORD.finditer(text):
        norm = _norm_token(m.group())
        if norm != m.group():
            edits.append((m.start(), m.end(), norm))
    return _apply_edits(text, spans_dict, edits)


def drop_span(text, spans_dict, rng):
    """Delete one contiguous ~10%-of-words span at an rng-drawn offset."""
    words = list(_WORD.finditer(text))
    n = len(words)
    if n < 2:
        return text, _copy_spans(spans_dict)
    k = min(n - 1, max(1, int(round(0.10 * n))))
    start = int(rng.integers(0, n - k + 1))
    return _delete_words(text, spans_dict, start, start + k, words)


OPS = {
    "mask_ruler": mask_ruler,
    "strip_formula": strip_formula,
    "crop8": crop8,
    "crop16": crop16,
    "crop32": crop32,
    "crop64": crop64,
    "orthonorm": orthonorm,
    "drop_span": drop_span,
}
