"""
Phase 1a — Parser (W2.B).

Reads raw outputs from `{out_dir}/raw/{variant}_*.json` written by run_kp.py,
applies the prompt-defined parse instructions, and writes a consolidated
`{out_dir}/parsed/{variant}.json`.

Parse strategy (all 3 variants use the same skeleton):
  1. Strip whitespace.
  2. Strip leading/trailing markdown code fences (```json ... ``` or ``` ... ```).
  3. Try json.loads. On failure, try to extract the first {...} block.
  4. Validate required fields based on variant; if missing, parse_error=True.

CLI:
  python parse_kp.py --variant {kp0|kp1|kp2} --out_dir <path>
"""

import argparse
import json
import re
import sys
from pathlib import Path

KP0_FIELDS = {'start_year', 'end_year', 'confidence', 'declined'}
KP1_FIELDS = {'period', 'rulers', 'confidence'}
KP2_FIELDS = KP0_FIELDS  # identical schema to kp0


def strip_code_fences(s: str) -> str:
    """Remove a single surrounding ```...``` block, optionally tagged ```json."""
    s = s.strip()
    fence_match = re.match(r'^```(?:json)?\s*\n?(.*?)\n?```\s*$', s, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return s


def extract_first_json_object(s: str) -> str | None:
    """Find the first balanced {...} substring. Returns None if not found."""
    start = s.find('{')
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def parse_raw_output(raw: str, required_fields: set[str]) -> tuple[dict | None, bool, str]:
    """Returns (parsed_dict_or_None, parse_error, reason)."""
    if raw is None:
        return None, True, 'raw is None'
    cleaned = strip_code_fences(raw)
    candidates = [cleaned]
    # Fallback: extract first balanced JSON object if direct parse fails.
    fallback = extract_first_json_object(cleaned)
    if fallback is not None and fallback != cleaned:
        candidates.append(fallback)

    last_err = ''
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as e:
            last_err = f'json decode error: {e}'
            continue
        if not isinstance(obj, dict):
            last_err = f'parsed non-dict: {type(obj).__name__}'
            continue
        missing = required_fields - set(obj.keys())
        if missing:
            last_err = f'missing required fields: {sorted(missing)}'
            # Still return — we record parse_error but keep the partial dict.
            return obj, True, last_err
        return obj, False, ''
    return None, True, last_err or 'no valid JSON found'


def required_fields_for(variant: str) -> set[str]:
    if variant == 'kp0':
        return KP0_FIELDS
    if variant == 'kp1':
        return KP1_FIELDS
    if variant == 'kp2':
        return KP2_FIELDS
    raise ValueError(f"Unknown variant: {variant}")


def parse_all(variant: str, out_dir: Path) -> dict:
    raw_dir = out_dir / 'raw'
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw dir does not exist: {raw_dir}")
    files = sorted(raw_dir.glob(f'{variant}_*.json'))
    if not files:
        raise FileNotFoundError(f"No raw files matching {variant}_*.json in {raw_dir}")

    req = required_fields_for(variant)
    results = []
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            rec = json.load(f)
        raw_output = rec.get('raw_output', '')
        parsed, parse_error, reason = parse_raw_output(raw_output, req)
        results.append({
            'input_idx': rec.get('input_idx'),
            'input_value': rec.get('input_value'),
            'fill': rec.get('fill'),
            'raw_output': raw_output,
            'parsed': parsed,
            'parse_error': parse_error,
            'parse_error_reason': reason if parse_error else '',
        })

    consolidated = {
        'variant': variant,
        'n_inputs': len(results),
        'n_parse_errors': sum(1 for r in results if r['parse_error']),
        'results': results,
    }
    parsed_dir = out_dir / 'parsed'
    parsed_dir.mkdir(parents=True, exist_ok=True)
    out_path = parsed_dir / f'{variant}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)
    print(f"[parse_kp] {variant}: {len(results)} records, "
          f"{consolidated['n_parse_errors']} parse errors -> {out_path}")
    return consolidated


def parse_args():
    p = argparse.ArgumentParser(description='Phase 1a parser')
    p.add_argument('--variant', required=True, choices=['kp0', 'kp1', 'kp2'])
    p.add_argument('--out_dir', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    parse_all(args.variant, Path(args.out_dir))


if __name__ == '__main__':
    main()
