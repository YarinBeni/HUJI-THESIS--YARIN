"""King-name token locator (P1 / T10 king_last + king_mean pooling sites).

A royal name in tier0 transliteration is a single hyphen-joined whitespace word,
e.g. Esarhaddon = ``m-aš-šur-PAP-AŠ``. We key on each fragment's KNOWN ``ruler``
label (so we never confuse the commissioner with an ancestor named in the
genealogy or an enemy king in a campaign narrative) and locate that ruler's own
spelling, preferring the EARLIEST occurrence (the opening titulary
``E2-GAL m-<KING> LUGAL GAL-u2 ...``).

Two layers:
  * ``find_name_word``      — char-span of the name in the raw text (CPU, no model).
                              Used for the J1 coverage report.
  * ``name_token_span``     — maps that char-span to tokenizer token indices via
                              offset mapping. Used at extraction time (J4) to pool
                              ``king_last`` (last token) / ``king_mean`` (mean) over
                              the name's subword tokens.

king_* sites are TIER0-ONLY: maximal cleaning strips the logograms/determinatives
that spell royal names, so the name is unrecoverable there.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
SPELLINGS_CSV = HERE / "ruler_spellings.csv"


def load_spellings(csv_path: Path = SPELLINGS_CSV) -> dict[str, list[str]]:
    """ruler -> list of transliteration spelling variants (status != 'review' kept,
    but review/low_coverage rows are still loaded; the locator just won't match if
    the spelling is wrong)."""
    out: dict[str, list[str]] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sp = [s for s in row["spellings"].split("|") if s]
            if sp:
                out[row["ruler"]] = sp
    return out


def find_name_word(text: str, spellings: list[str]) -> Optional[tuple[int, int, str]]:
    """Return (char_start, char_end, matched_spelling) of the earliest occurrence
    of any spelling as a standalone whitespace token, else None."""
    best: Optional[tuple[int, int, str]] = None
    for sp in spellings:
        # standalone token: bounded by start/whitespace on both sides
        for m in re.finditer(r"(?:^|\s)(" + re.escape(sp) + r")(?:\s|$)", text):
            cs, ce = m.start(1), m.end(1)
            if best is None or cs < best[0]:
                best = (cs, ce, sp)
            break  # earliest occurrence of THIS spelling is enough
    return best


def name_token_span(text: str, tokenizer, spellings: list[str]) -> Optional[tuple[int, int]]:
    """Map the located name char-span to inclusive token indices using the model
    tokenizer's offset mapping. Returns (tok_start, tok_end) or None.

    Caller pools hidden states at these positions:
      king_last -> hidden[tok_end]
      king_mean -> hidden[tok_start : tok_end + 1].mean(0)
    Note: indices are into the tokenization of ``text`` alone; if the extractor
    prepends a prompt, offset the result by the prompt's token length (the T10
    redo passes the fragment span and adds the prefix length).
    """
    hit = find_name_word(text, spellings)
    if hit is None:
        return None
    cs, ce, _ = hit
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    toks = [i for i, (a, b) in enumerate(offsets) if b > a and a < ce and b > cs]
    if not toks:
        return None
    return (min(toks), max(toks))


def name_span_by_ids(input_ids, tokenizer, spellings: list[str]) -> Optional[tuple[int, int]]:
    """Tokenizer-agnostic fallback for tokenizers without offset mapping
    (e.g. some sentencepiece models). Tokenize each spelling alone and find the
    EARLIEST contiguous matching subsequence of token ids in ``input_ids``.
    Returns inclusive (tok_start, tok_end) or None."""
    ids = list(int(x) for x in input_ids)
    best = None
    for sp in spellings:
        sub = tokenizer(sp, add_special_tokens=False)["input_ids"]
        if not sub:
            continue
        L = len(sub)
        for i in range(len(ids) - L + 1):
            if ids[i:i + L] == sub:
                if best is None or i < best[0]:
                    best = (i, i + L - 1)
                break
    return best


def locate_king_tokens(text, tokenizer, spellings, input_ids=None):
    """Unified entry: try char-offset mapping first, fall back to id-subsequence.
    `input_ids` (the actual model input, possibly with special tokens) is used for
    the fallback; pass it from the extractor."""
    try:
        span = name_token_span(text, tokenizer, spellings)
        if span is not None:
            return span
    except Exception:
        pass
    if input_ids is not None:
        return name_span_by_ids(input_ids, tokenizer, spellings)
    return None


def coverage_report(df, spellings_map: dict[str, list[str]], text_col: str = "text_tier0"):
    """Per-ruler word-level coverage (fraction of fragments where the ruler's own
    name was located). Returns (per_ruler_rows, overall_dict)."""
    rows = []
    cov_n = tot_n = 0
    for ruler, sp in spellings_map.items():
        sub = df[df["ruler"] == ruler]
        if len(sub) == 0:
            continue
        hits = sub[text_col].apply(lambda t: find_name_word(str(t), sp) is not None).sum()
        rows.append({"ruler": ruler, "n": int(len(sub)), "hits": int(hits),
                     "coverage": round(hits / len(sub), 3)})
        cov_n += hits
        tot_n += len(sub)
    overall = {
        "mapped_rulers": len(rows),
        "rows_with_mapped_ruler": int(tot_n),
        "rows_with_name_found": int(cov_n),
        "coverage_within_mapped": round(cov_n / tot_n, 3) if tot_n else 0.0,
        "share_of_corpus_mapped": round(tot_n / len(df), 3),
    }
    return rows, overall
