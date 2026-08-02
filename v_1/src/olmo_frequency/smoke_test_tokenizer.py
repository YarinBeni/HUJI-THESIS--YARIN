#!/usr/bin/env python3
"""STEP 1 of the OLMo frequency experiment: does OLMo's tokenizer work with our
entity-span logic, before we spend any GPU time?

The whole cell-B pipeline depends on locating the entity's tokens inside a carrier
sentence — "The king [Ashurbanipal] ruled over the land." — so that `ent_last` and
`ent_mean` pool over the NAME rather than the sentence. `extract_entity.encode_with_spans`
does that from the tokenizer's offset mapping, with a prefix-retokenisation fallback for
tokenizers that do not provide one.

If OLMo's tokenizer breaks that, every cell-B number for the arm would be silently
pooled over the wrong tokens, and we would not notice from the scores alone. So this is
checked FIRST, on CPU, in about a minute — no model weights are downloaded, only the
tokenizer.

Checks, per entity type:
  1. the tokenizer loads at all
  2. it exposes an offset mapping (else: fallback path, which is fine but worth knowing)
  3. every located span is non-empty
  4. decoding the located span round-trips to the entity string (whitespace/case-insensitive)

Exit code 0 = safe to proceed to step 2. Non-zero = stop, do not book GPU time.

    python smoke_test_tokenizer.py
    python smoke_test_tokenizer.py --model olmo2_7b --n 10
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WM = os.path.join(os.path.dirname(_HERE), "world_models")
sys.path.insert(0, _WM)
sys.path.insert(0, os.path.join(_WM, "akkadian"))

# This has to import the REAL span logic — testing a copy would prove nothing — and
# extract_entity pulls numpy/pandas, so the conda env must be active. Fail with the
# fix rather than a bare ModuleNotFoundError.
try:
    from wm_lib.registry import MODELS                 # noqa: E402
    import extract_entity as EE                        # noqa: E402
except ModuleNotFoundError as _e:                      # noqa: BLE001
    sys.exit(f"missing dependency ({_e.name}) — activate the env first:\n"
             "    source ~/miniconda3/etc/profile.d/conda.sh && conda activate thesis")

DATA = os.path.join(_WM, "data", "entity_datasets")
ENTITY_TYPES = ["assyrian_ruler", "mesopotamian_place"]


def norm(s):
    return "".join(s.split()).lower()


def check(model_key, n_per_type, max_tokens):
    from transformers import AutoTokenizer

    spec = MODELS[model_key]
    hfid = spec["hfid"]
    print(f"=== {model_key}  ({hfid}) ===", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(spec.get("tokenizer") or hfid,
                                            trust_remote_code=True)
    except Exception as e:                                          # noqa: BLE001
        print(f"  FAIL: tokenizer would not load: {type(e).__name__}: {e}")
        print("        (transformers >= 4.47 is required for OLMo 2)")
        return False

    print(f"  tokenizer   : {type(tok).__name__}")
    print(f"  is_fast     : {getattr(tok, 'is_fast', False)}"
          f"   -> {'offset mapping available' if getattr(tok, 'is_fast', False) else 'PREFIX FALLBACK will be used'}")

    ok = True
    for et in ENTITY_TYPES:
        path = os.path.join(DATA, f"{et}.csv")
        if not os.path.exists(path):
            print(f"  [skip] no dataset at {path}")
            continue
        # stdlib csv, not pandas: this runs before any GPU is booked and should work
        # on a bare login node without the conda env activated.
        rows_csv = list(csv.DictReader(open(path)))
        bare_rows = [r for r in rows_csv if r["template"] == "bare"][:n_per_type]
        sent_rows = [r for r in rows_csv if r["template"] != "bare"][:n_per_type]
        # cover both the bare name and a carrier sentence — the carrier is where span
        # location can realistically go wrong
        sub = bare_rows + sent_rows
        strings = [str(r["entity_string"]) for r in sub]
        spans = [(int(r["ent_start"]), int(r["ent_end"])) for r in sub]
        names = [str(r["name"]) for r in sub]

        try:
            # returns (all_ids, ent_t0, ent_t1, n_truncated); t1 is INCLUSIVE
            all_ids, t0s, t1s, n_trunc = EE.encode_with_spans(
                tok, strings, spans, max_tokens)
        except Exception as e:                                      # noqa: BLE001
            print(f"  {et}: FAIL during encode_with_spans: {type(e).__name__}: {e}")
            ok = False
            continue

        empty, mismatch = 0, []
        for ids, i0, i1, s, (c0, c1), nm in zip(all_ids, t0s, t1s, strings,
                                                spans, names):
            if i1 < i0:
                empty += 1
                continue
            got = tok.decode(ids[int(i0):int(i1) + 1]).strip()
            if norm(got) != norm(s[c0:c1]):
                mismatch.append((nm, s[c0:c1], got))

        n = len(all_ids)
        print(f"  {et}: {n} rows | truncated: {n_trunc} | empty spans: {empty} | "
              f"round-trip mismatches: {len(mismatch)}")
        for nm, want, got in mismatch[:4]:
            print(f"      {nm}: want {want!r}  got {got!r}")
        if empty or mismatch:
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmo2_7b")
    ap.add_argument("--n", type=int, default=10, help="rows per template group")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    good = check(args.model, args.n, args.max_tokens)
    print()
    if good:
        print("PASS — span logic works. Safe to proceed to step 2 (extract + probe).")
        return 0
    print("FAIL — do NOT book GPU time. Fix span handling for this tokenizer first.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
