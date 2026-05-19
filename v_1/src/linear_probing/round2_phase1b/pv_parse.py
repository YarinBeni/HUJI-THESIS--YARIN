"""pv_parse.py — Shared parsing utilities for Phase 1b prompt variants.

Implements:
 - Canonical ruler-name normalization table (from pv0.md §"Canonical ruler name
   normalization table").
 - JSON-first then regex-fallback parser for model outputs.
 - Prompt frontmatter (YAML-lite) + body extraction from approved .md files.

These functions are pure — no model, no network. Safe to unit-test.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Canonical ruler name normalization
# ---------------------------------------------------------------------------
# Maps free-text mentions to the EXACT corpus label used in
# v_1/data/evaluation/corpora/orcc_corpus.parquet column `ruler`.
# Note: the corpus spells Sîn-šarru-iškun with the proper diacritics — we
# canonicalize free-text spellings (Sin-sarru-iskun, Sinsharishkun) to that
# corpus form so scoring downstream compares equal strings.
RULER_CANONICAL: dict[str, str] = {
    # Ashurbanipal
    "ashurbanipal": "Ashurbanipal",
    "assurbanipal": "Ashurbanipal",
    "assur-bani-apli": "Ashurbanipal",
    # Sennacherib
    "sennacherib": "Sennacherib",
    "sin-ahhe-eriba": "Sennacherib",
    # Esarhaddon
    "esarhaddon": "Esarhaddon",
    "asarhaddon": "Esarhaddon",
    "esarra-haddon": "Esarhaddon",
    # Sargon II
    "sargon": "Sargon II",
    "sargon ii": "Sargon II",
    "sargon 2": "Sargon II",
    # Nebuchadnezzar II
    "nebuchadnezzar": "Nebuchadnezzar II",
    "nebuchadrezzar": "Nebuchadnezzar II",
    "nebuchadnezzar ii": "Nebuchadnezzar II",
    "nebuchadnezzar 2": "Nebuchadnezzar II",
    # Tiglath-pileser III
    "tiglath-pileser": "Tiglath-pileser III",
    "tiglathpileser": "Tiglath-pileser III",
    "tiglath pileser": "Tiglath-pileser III",
    "tiglath-pileser iii": "Tiglath-pileser III",
    "tiglath-pileser 3": "Tiglath-pileser III",
    # Nabonidus
    "nabonidus": "Nabonidus",
    "nabu-naid": "Nabonidus",
    # Sîn-šarru-iškun  (corpus spelling uses diacritics)
    "sin-sarru-iskun": "Sîn-šarru-iškun",
    "sin-šarru-iškun": "Sîn-šarru-iškun",
    "sîn-šarru-iškun": "Sîn-šarru-iškun",
    "sinsharishkun": "Sîn-šarru-iškun",
    "sin-sharru-ishkun": "Sîn-šarru-iškun",
}

# Canonical corpus forms (output values).
RULER_CORPUS_LABELS = sorted(set(RULER_CANONICAL.values()))


def normalize_ruler(raw: Any) -> str | None:
    """Map free-text ruler mention -> canonical corpus label, or None."""
    if raw is None:
        return None
    s = str(raw).strip().strip("'\"")
    if not s:
        return None
    key = s.lower()
    # exact match first
    if key in RULER_CANONICAL:
        return RULER_CANONICAL[key]
    # try simple prefix collapse: remove trailing ", king of ..." etc
    key2 = re.split(r"[,;(]", key)[0].strip()
    if key2 in RULER_CANONICAL:
        return RULER_CANONICAL[key2]
    # last-resort: scan for any canonical alias appearing as substring
    for alias, canon in RULER_CANONICAL.items():
        if alias in key:
            return canon
    return None


# ---------------------------------------------------------------------------
# Year parsing
# ---------------------------------------------------------------------------
_YEAR_BCE_RE = re.compile(
    r"(\d{3,4})\s*(?:BCE|B\.C\.E\.|BC|B\.C\.)\b",
    re.IGNORECASE,
)


def normalize_year(raw: Any) -> int | None:
    """Coerce model year output to positive integer (corpus convention)."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        v = int(abs(raw))
        return v if 1 <= v <= 5000 else None
    s = str(raw).strip()
    if not s:
        return None
    # plain integer (may be negative)
    try:
        return int(abs(int(s)))
    except ValueError:
        pass
    # "672 BCE"
    m = _YEAR_BCE_RE.search(s)
    if m:
        return int(m.group(1))
    # bare 3-4 digit number in text
    m = re.search(r"\b(\d{3,4})\b", s)
    if m:
        v = int(m.group(1))
        if 100 <= v <= 5000:
            return v
    return None


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _strip_fences(s: str) -> str:
    """Pull out content of the first ```...``` code fence if present."""
    m = _FENCE_RE.search(s)
    if m:
        return m.group(1)
    return s


def _extract_json_block(output: str, prefer_last: bool = False) -> dict | None:
    """Return parsed JSON dict from output, or None.

    If prefer_last=True (used by pv3 CoT), choose the LAST {...} match
    (the answer that follows the reasoning).
    """
    # First try entire stripped output / fenced block
    candidates: list[str] = []
    stripped = _strip_fences(output).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    matches = list(_JSON_OBJ_RE.finditer(output))
    if matches:
        if prefer_last:
            candidates.append(matches[-1].group())
        else:
            candidates.append(matches[0].group())
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------
def parse_raw_output(
    output: str,
    variant: str,
) -> dict:
    """Parse one model output. Returns dict with fields:
       parsed_ruler, parsed_year, parsed_confidence, parse_error, reasoning_text.

    variant: one of {pv0, pv1, pv2, pv3}.
    """
    result: dict[str, Any] = {
        "parsed_ruler": None,
        "parsed_year": None,
        "parsed_confidence": None,
        "parse_error": None,
        "reasoning_text": None,
    }
    if output is None:
        result["parse_error"] = "empty_output"
        return result

    prefer_last = variant == "pv3"

    # ---- JSON first ----
    obj = _extract_json_block(output, prefer_last=prefer_last)
    if obj is not None:
        result["parsed_ruler"] = normalize_ruler(obj.get("ruler"))
        result["parsed_year"] = normalize_year(obj.get("year_bce"))
        conf = obj.get("confidence")
        if conf is not None:
            try:
                c = float(conf)
                result["parsed_confidence"] = max(0.0, min(1.0, c))
            except (TypeError, ValueError):
                result["parsed_confidence"] = None
        # If both ruler and year parsed cleanly, we are done.
        if result["parsed_ruler"] is not None and result["parsed_year"] is not None:
            if prefer_last:
                # store reasoning text = everything before the final JSON block
                matches = list(_JSON_OBJ_RE.finditer(output))
                if matches:
                    result["reasoning_text"] = output[: matches[-1].start()].strip()
            return result
        # otherwise: fall through to regex to fill remaining nulls

    # ---- Regex fallback ----
    if result["parsed_ruler"] is None:
        # Scan for any canonical alias in the raw text
        text_lc = output.lower()
        best_match: tuple[int, str] | None = None
        for alias, canon in RULER_CANONICAL.items():
            idx = text_lc.find(alias)
            if idx >= 0 and (best_match is None or idx < best_match[0]):
                best_match = (idx, canon)
        if best_match is not None:
            result["parsed_ruler"] = best_match[1]
    if result["parsed_year"] is None:
        m = _YEAR_BCE_RE.search(output)
        if m:
            result["parsed_year"] = int(m.group(1))

    if result["parsed_ruler"] is None and result["parsed_year"] is None:
        result["parse_error"] = "no_match"

    if prefer_last:
        matches = list(_JSON_OBJ_RE.finditer(output))
        if matches:
            result["reasoning_text"] = output[: matches[-1].start()].strip()
        else:
            result["reasoning_text"] = output.strip()

    return result


# ---------------------------------------------------------------------------
# Prompt .md frontmatter + body extraction
# ---------------------------------------------------------------------------
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", re.DOTALL)


def parse_prompt_md(path: str) -> dict:
    """Read a prompt .md file and return dict with:
       variant, frontmatter (raw lines), sections (name -> body text),
       system_prompt, user_template.

    For pv0 the System prompt section says "(none ...)" — we detect that and
    set system_prompt = None (caller decides whether to send empty string).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    m = _FRONTMATTER_RE.match(raw)
    if m:
        frontmatter_block, body = m.group(1), m.group(2)
    else:
        frontmatter_block, body = "", raw

    # Cheap "YAML-lite" parse: key: value pairs, ignore lists.
    fm: dict[str, str] = {}
    for line in frontmatter_block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, _, v = line.partition(":")
        fm[k.strip()] = v.strip()

    # Split body into sections (## headings).
    sections: dict[str, str] = {}
    headings = list(_SECTION_RE.finditer(body))
    for i, h in enumerate(headings):
        name = h.group(1).strip()
        start = h.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        sections[name] = body[start:end].strip()

    # System prompt
    sys_section = sections.get("System prompt", "")
    sys_block = _CODE_BLOCK_RE.search(sys_section)
    if sys_block:
        system_prompt: str | None = sys_block.group(1).strip()
    else:
        # pv0: "(none — no system message is sent for this variant)"
        if "none" in sys_section.lower():
            system_prompt = None
        else:
            system_prompt = None

    # User prompt template
    user_section = sections.get("User prompt template", "")
    user_block = _CODE_BLOCK_RE.search(user_section)
    if user_block:
        user_template = user_block.group(1)  # preserve trailing newlines
    else:
        user_template = user_section

    return {
        "variant": fm.get("variant", ""),
        "frontmatter": fm,
        "sections": sections,
        "system_prompt": system_prompt,
        "user_template": user_template,
    }


# ---------------------------------------------------------------------------
# Span location
# ---------------------------------------------------------------------------
def locate_target_span_chars(prompt_text: str) -> tuple[int, int]:
    """Find char-offsets [start, end) of the TARGET fragment content within
    the rendered prompt string (the text between `<<FRAG>>\n` and `\n<</FRAG>>`).

    For pv2 (few-shot) the few-shot examples use `<<FRAG_EX*>>` delimiters,
    so a bare `<<FRAG>>` lookup correctly picks the target. We pick the LAST
    occurrence to be defensive.

    Returns (char_start, char_end) — slicing prompt_text[start:end] yields
    the fragment content (with surrounding whitespace).
    """
    # Find every <<FRAG>> that is NOT followed by _EX
    starts = []
    i = 0
    while True:
        j = prompt_text.find("<<FRAG>>", i)
        if j < 0:
            break
        starts.append(j)
        i = j + len("<<FRAG>>")
    if not starts:
        raise ValueError("No <<FRAG>> delimiter found in prompt")
    start_open = starts[-1]
    content_start = start_open + len("<<FRAG>>")
    end_close = prompt_text.find("<</FRAG>>", content_start)
    if end_close < 0:
        raise ValueError("No <</FRAG>> delimiter found after <<FRAG>>")
    return content_start, end_close
