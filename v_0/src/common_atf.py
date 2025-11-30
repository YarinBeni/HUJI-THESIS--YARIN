"""Shared ATF cleaning utilities."""
import re

BRACKET_PATTERNS = [
    re.compile(r"\[[^\]]*\]"),
    re.compile(r"\([^\)]*\)"),
    re.compile(r"\{[^\}]*\}"),
]
MULTISPACE_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"[0-9₀₁₂₃₄₅₆₇₈₉]")


def strip_atf(text: str) -> str:
    if not text:
        return ""
    cleaned = text
    for pattern in BRACKET_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = DIGIT_RE.sub("", cleaned)
    cleaned = cleaned.replace("…", " ").replace("×", " ")
    cleaned = MULTISPACE_RE.sub(" ", cleaned)
    return cleaned.strip()
