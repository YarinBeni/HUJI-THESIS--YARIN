"""sanity_check.py — Local-only checks for Phase 1b harness.

NO real Qwen inference. Verifies:
  (1) Prompt frontmatter+body parsing matches each pvN.md.
  (2) Rendered pv0 prompt matches the literal `<<FRAG>>...</FRAG>>` form.
  (3) Char-level span location works on rendered prompts.
  (4) Parser handles {good JSON, fenced JSON, malformed} synthetic outputs.
  (5) Token-level span location works against the actual Qwen tokenizer
      IF transformers is installed; otherwise skipped with a notice.

Run:
    python sanity_check.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import traceback

_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from pv_parse import (
    locate_target_span_chars,
    normalize_ruler,
    normalize_year,
    parse_prompt_md,
    parse_raw_output,
)
from run_pv import render_user_prompt

# _THIS_DIR = v_1/src/linear_probing/round2_phase1b/
# prompts live at      v_1/src/linear_probing/results/orcc_round2_phase1b/prompts/
PROMPTS_DIR = _THIS_DIR.parent / "results/orcc_round2_phase1b/prompts"


def banner(s: str) -> None:
    print(f"\n=== {s} ===", flush=True)


def test_prompt_parsing() -> None:
    banner("1. parse_prompt_md on all 4 variants")
    for v in ["pv0", "pv1", "pv2", "pv3"]:
        p = parse_prompt_md(str(PROMPTS_DIR / f"{v}.md"))
        sys_repr = "<none>" if p["system_prompt"] is None else f"<len={len(p['system_prompt'])}>"
        utpl_len = len(p["user_template"])
        print(f"  {v}: variant={p['variant']!r}  system={sys_repr}  user_template_len={utpl_len}")
        assert p["variant"].startswith(v), f"variant frontmatter mismatch for {v}"
        if v == "pv0":
            assert p["system_prompt"] is None, "pv0 must have no system prompt"
        else:
            assert p["system_prompt"] is not None and "Assyriologist" in p["system_prompt"], \
                f"{v} system prompt missing/wrong"
        assert "{{fragment_text}}" in p["user_template"], f"{v} missing {{{{fragment_text}}}}"


def test_render_and_span() -> None:
    banner("2. render_user_prompt + locate_target_span_chars")
    p0 = parse_prompt_md(str(PROMPTS_DIR / "pv0.md"))
    frag = "a-na be-li-ia qi2-bi2-ma um-ma I-pi2-iq-an-num"
    rendered = render_user_prompt(p0["user_template"], frag)
    print("  pv0 rendered (first 200 chars):")
    print("  ", rendered[:200].replace("\n", "\\n"))
    assert "<<FRAG>>" in rendered and "<</FRAG>>" in rendered
    assert frag in rendered
    assert "Who wrote this and when?" in rendered, "pv0 must contain the canned question"
    cs, ce = locate_target_span_chars(rendered)
    span_text = rendered[cs:ce]
    print(f"  span chars=({cs},{ce})  -> {span_text.strip()[:80]!r}")
    assert frag in span_text, "Fragment text not inside located span"

    # pv2: ensure <<FRAG_EX*>> delimiters do NOT confuse the target locator.
    p2 = parse_prompt_md(str(PROMPTS_DIR / "pv2.md"))
    ex = [
        {"fragment_id": "EX1", "ruler": "Ashurbanipal", "year": 645, "text": "EXAMPLE_TEXT_1"},
        {"fragment_id": "EX2", "ruler": "Sennacherib",  "year": 700, "text": "EXAMPLE_TEXT_2"},
        {"fragment_id": "EX3", "ruler": "Esarhaddon",   "year": 675, "text": "EXAMPLE_TEXT_3"},
        {"fragment_id": "EX4", "ruler": "Sargon II",    "year": 715, "text": "EXAMPLE_TEXT_4"},
        {"fragment_id": "EX5", "ruler": "Tiglath-pileser III", "year": 740, "text": "EXAMPLE_TEXT_5"},
    ]
    rendered2 = render_user_prompt(p2["user_template"], frag, ex)
    cs2, ce2 = locate_target_span_chars(rendered2)
    span2 = rendered2[cs2:ce2]
    print(f"  pv2 target span -> {span2.strip()[:80]!r}")
    assert frag in span2, "pv2 target span doesn't contain TARGET fragment"
    # Make sure none of the example texts are inside the target span
    for e in ex:
        assert e["text"] not in span2, f"pv2 target span LEAKED example {e['fragment_id']}"


def test_pv2_placeholder_fill() -> None:
    banner("2b. pv2 fewshot placeholder substitution (all {{example_N_*}})")
    p2 = parse_prompt_md(str(PROMPTS_DIR / "pv2.md"))
    ex = [
        {"fragment_id": "Q1", "ruler": "Ashurbanipal",       "year": 645, "text": "EX_TEXT_1"},
        {"fragment_id": "Q2", "ruler": "Sennacherib",        "year": 700, "text": "EX_TEXT_2"},
        {"fragment_id": "Q3", "ruler": "Esarhaddon",         "year": 675, "text": "EX_TEXT_3"},
        {"fragment_id": "Q4", "ruler": "Sargon II",          "year": 715, "text": "EX_TEXT_4"},
        {"fragment_id": "Q5", "ruler": "Tiglath-pileser III", "year": 740, "text": "EX_TEXT_5"},
    ]
    out = render_user_prompt(p2["user_template"], "TARGET_AKKADIAN_TEXT", ex)
    import re as _re
    unfilled = _re.findall(r"\{\{[^}]+\}\}", out)
    print(f"  unfilled placeholders: {unfilled}")
    assert not unfilled, f"pv2 left unfilled placeholders: {unfilled}"
    assert "TARGET_AKKADIAN_TEXT" in out
    for i in range(1, 6):
        assert f"EX_TEXT_{i}" in out, f"missing EX_TEXT_{i}"
    n_frag = out.count("<<FRAG>>")
    n_frag_ex = sum(out.count(f"<<FRAG_EX{i}>>") for i in range(1, 6))
    print(f"  <<FRAG>> occurrences (target only)={n_frag}  <<FRAG_EX*>>={n_frag_ex}")
    assert n_frag == 1, "pv2 should have exactly ONE <<FRAG>> (target)"
    assert n_frag_ex == 5, "pv2 should have 5 <<FRAG_EX*>> wrappers (one per example)"


def test_parser() -> None:
    banner("3. parse_raw_output on synthetic outputs")
    samples = [
        ("good_json",
         '{"ruler": "Ashurbanipal", "year_bce": 645, "confidence": 0.9}',
         {"parsed_ruler": "Ashurbanipal", "parsed_year": 645}),
        ("fenced_json",
         "Here is my answer:\n```json\n{\"ruler\": \"Sennacherib\", \"year_bce\": 700}\n```\n",
         {"parsed_ruler": "Sennacherib", "parsed_year": 700}),
        ("malformed_with_year_phrase",
         "This appears to be from Sargon II, written around 715 BCE.",
         {"parsed_ruler": "Sargon II", "parsed_year": 715}),
        ("pv3_cot_last_json",
         "Reasoning: titulary suggests... {ruler: ignored}\n\nFinal:\n"
         '{"ruler": "Nabu-naid", "year_bce": 555, "confidence": 0.8}',
         {"parsed_ruler": "Nabonidus", "parsed_year": 555}),
        ("totally_unparseable",
         "I'm sorry, I cannot determine the answer.",
         {"parsed_ruler": None, "parsed_year": None}),
    ]
    for name, raw, expected in samples:
        variant = "pv3" if "pv3" in name else "pv1"
        got = parse_raw_output(raw, variant)
        ok = all(got.get(k) == v for k, v in expected.items())
        marker = "OK" if ok else "FAIL"
        print(f"  [{marker}] {name}: ruler={got['parsed_ruler']!r}  year={got['parsed_year']!r}  err={got['parse_error']!r}")
        assert ok, f"{name} expected {expected}, got {got}"


def test_normalize_helpers() -> None:
    banner("4. ruler + year normalization helpers")
    assert normalize_ruler("Assurbanipal") == "Ashurbanipal"
    assert normalize_ruler("nebuchadrezzar") == "Nebuchadnezzar II"
    assert normalize_ruler("Sin-sarru-iskun") == "Sîn-šarru-iškun"
    assert normalize_ruler("Sîn-šarru-iškun") == "Sîn-šarru-iškun"
    assert normalize_ruler("Sargon, king of Assyria") == "Sargon II"
    assert normalize_ruler(None) is None
    assert normalize_ruler("") is None
    assert normalize_year(-645) == 645
    assert normalize_year("645 BCE") == 645
    assert normalize_year("704") == 704
    assert normalize_year(None) is None
    print("  ruler/year normalization OK")


def test_tokenizer_span_optional() -> None:
    banner("5. tokenizer span location (optional, requires transformers)")
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        print(f"  SKIP: transformers not importable ({e})")
        return
    # Try loading Qwen tokenizer — may not be cached locally; fall back gracefully.
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    except Exception as e:
        print(f"  SKIP: Qwen tokenizer not available locally ({type(e).__name__}: {e})")
        return
    from run_pv import build_chat_prompt, compute_span_token_indices
    p0 = parse_prompt_md(str(PROMPTS_DIR / "pv0.md"))
    frag = "a-na be-li-ia qi2-bi2-ma um-ma I-pi2-iq-an-num"
    rendered = render_user_prompt(p0["user_template"], frag)
    prompt_str, input_ids = build_chat_prompt(tok, p0["system_prompt"], rendered, "pv0")
    s, e = compute_span_token_indices(tok, prompt_str, input_ids)
    decoded_span = tok.decode(input_ids[0, s:e + 1])
    print(f"  span_start_token={s}  span_end_token={e}  prompt_n_tokens={input_ids.shape[1]}")
    print(f"  decoded span text: {decoded_span!r}")
    assert "a-na" in decoded_span and ("an-num" in decoded_span or "num" in decoded_span), \
        "Tokenizer span did not contain fragment text"


def main() -> int:
    ok = True
    for fn in [
        test_prompt_parsing,
        test_render_and_span,
        test_pv2_placeholder_fill,
        test_parser,
        test_normalize_helpers,
        test_tokenizer_span_optional,
    ]:
        try:
            fn()
        except Exception:
            ok = False
            print(f"\nFAILED in {fn.__name__}:")
            traceback.print_exc()
    banner("SUMMARY")
    print("ALL OK" if ok else "FAILED — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
