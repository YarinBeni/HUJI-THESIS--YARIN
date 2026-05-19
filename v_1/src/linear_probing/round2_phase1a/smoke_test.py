"""
Smoke test for Phase 1a parser + scorer.

NO real Qwen inference. We hand-craft a handful of raw outputs that exercise:
  - clean JSON
  - JSON wrapped in ```json fences
  - JSON wrapped in plain ``` fences
  - malformed JSON (parse_error case)
  - declined=true response
  - kp1 with diacritic-normalized matches
  - kp2 hallucination vs correct decline

Then we write them into a temp out_dir as if run_kp.py had produced them, and
run parse_kp + score_kp end-to-end. Verifies the resulting metrics match
expected values.

Run:
  python v_1/src/linear_probing/round2_phase1a/smoke_test.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from parse_kp import parse_all  # noqa: E402
from score_kp import score_all  # noqa: E402


def write_raw(out_dir: Path, variant: str, idx: int, input_value: str, raw_output: str,
              fill_key: str = 'ruler'):
    raw_dir = out_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        'variant': variant,
        'input_idx': idx,
        'input_value': input_value,
        'fill': {fill_key: input_value},
        'system_prompt': '(stub)',
        'user_message': '(stub)',
        'raw_output': raw_output,
        'model_path': 'STUB',
        'max_new_tokens': 512,
        'timestamp': '2026-05-19T00:00:00',
    }
    with open(raw_dir / f'{variant}_{idx:02d}.json', 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)


def test_kp0(tmp: Path):
    print('\n=== kp0 smoke test ===')
    out = tmp / 'kp0_run'
    # Case 0: Ashurbanipal — clean JSON, correct (true=668/631, pred=669/630).
    write_raw(out, 'kp0', 0, 'Ashurbanipal',
              '{"start_year": 669, "end_year": 630, "confidence": "high", "declined": false}')
    # Case 1: Sennacherib — fenced JSON, way off (pred 1500s) → incorrect.
    write_raw(out, 'kp0', 1, 'Sennacherib',
              '```json\n{"start_year": 1500, "end_year": 1480, "confidence": "low", "declined": false}\n```')
    # Case 2: Esarhaddon — model declined (null years) → not correct.
    write_raw(out, 'kp0', 2, 'Esarhaddon',
              '{"start_year": null, "end_year": null, "confidence": "low", "declined": true}')
    # Case 3: Sargon II — malformed JSON → parse_error.
    write_raw(out, 'kp0', 3, 'Sargon II',
              'I think Sargon II reigned around 722 BCE but I am not sure.')
    # Case 4: Nebuchadnezzar II — correct (true 605/562, pred 600/560, both within ±50).
    write_raw(out, 'kp0', 4, 'Nebuchadnezzar II',
              '{"start_year": 600, "end_year": 560, "confidence": "high", "declined": false}')
    # Case 5: Tiglath-pileser III — has prose AFTER the JSON; first-object extractor should rescue.
    write_raw(out, 'kp0', 5, 'Tiglath-pileser III',
              '{"start_year": 745, "end_year": 727, "confidence": "high", "declined": false}\n\nNotes: ...')
    # Case 6: Nabonidus — fenced without language tag.
    write_raw(out, 'kp0', 6, 'Nabonidus',
              '```\n{"start_year": 556, "end_year": 539, "confidence": "high", "declined": false}\n```')
    # Case 7: Sîn-šarru-iškun — pred 700/650, range hi=700 hits gt_start=627? No: lo=600, hi=700 → 627 ∈ [600,700] → CORRECT.
    write_raw(out, 'kp0', 7, 'Sîn-šarru-iškun',
              '{"start_year": 700, "end_year": 650, "confidence": "low", "declined": false}')

    parse_all('kp0', out)
    metrics = score_all('kp0', out)
    # Expected:
    #   Correct: 0 (Ashurbanipal), 4 (Neb II), 5 (TP III), 6 (Nabonidus), 7 (Sin-sarru) = 5
    #   Wrong: 1 (Sennacherib far off), 2 (declined)
    #   Parse error: 3 (Sargon)
    assert metrics['total'] == 8, metrics['total']
    assert metrics['parse_errors'] == 1, metrics['parse_errors']
    assert metrics['correct'] == 5, f"expected 5 correct, got {metrics['correct']}"
    print(f"  kp0 OK: correct={metrics['correct']}/8, parse_errors={metrics['parse_errors']}, "
          f"accuracy={metrics['accuracy']:.3f}")


def test_kp1(tmp: Path):
    print('\n=== kp1 smoke test ===')
    out = tmp / 'kp1_run'
    # Period 0: Neo-Assyrian — list includes 4/6 of the targets with assorted diacritic stripping.
    write_raw(out, 'kp1', 0, 'Neo-Assyrian',
              '{"period": "Neo-Assyrian", "rulers": ["Ashurbanipal", "sennacherib", '
              '"Esarhaddon", "Sin-sarru-iskun", "Some Other King"], "confidence": "high"}',
              fill_key='period')
    # Period 1: Neo-Babylonian — list includes both NB targets.
    write_raw(out, 'kp1', 1, 'Neo-Babylonian',
              '```json\n{"period": "Neo-Babylonian", "rulers": ["Nabopolassar", '
              '"Nebuchadnezzar II", "Nabonidus"], "confidence": "high"}\n```',
              fill_key='period')

    parse_all('kp1', out)
    metrics = score_all('kp1', out)
    # NA targets = 6, found = 4 (Ashurbanipal, Sennacherib, Esarhaddon, Sin-sarru). Sargon II, TP III missing.
    # NB targets = 2, found = 2.
    # aggregate = 6 / 8 = 0.75
    assert metrics['parse_errors'] == 0
    assert metrics['total_targets'] == 8, metrics['total_targets']
    assert metrics['total_hits'] == 6, metrics['total_hits']
    assert abs(metrics['aggregate_recall'] - 0.75) < 1e-9
    print(f"  kp1 OK: aggregate_recall={metrics['aggregate_recall']:.3f} ({metrics['total_hits']}/{metrics['total_targets']})")
    for per in metrics['per_period']:
        print(f"    {per['period']}: recall={per['recall']:.3f} found={per['found_targets']} missed={per['missed_targets']}")


def test_kp2(tmp: Path):
    print('\n=== kp2 smoke test ===')
    out = tmp / 'kp2_run'
    # 6 correct declines, 1 hallucination, 1 parse error → hallucination_rate = 1/7 ≈ 0.143 → gate passes.
    correct_decline = '{"start_year": null, "end_year": null, "confidence": "low", "declined": true}'
    hallucination = '{"start_year": 700, "end_year": 670, "confidence": "low", "declined": false}'
    malformed = 'Hmm, this name is unfamiliar to me.'
    for i in range(6):
        write_raw(out, 'kp2', i, f'Fake-{i}', correct_decline)
    write_raw(out, 'kp2', 6, 'Fake-6', hallucination)
    write_raw(out, 'kp2', 7, 'Fake-7', malformed)

    parse_all('kp2', out)
    metrics = score_all('kp2', out)
    assert metrics['total'] == 8
    assert metrics['parse_errors'] == 1, metrics['parse_errors']
    assert metrics['scoreable'] == 7
    assert metrics['hallucinations'] == 1, metrics['hallucinations']
    assert metrics['declined_correctly'] == 6
    expected_rate = 1 / 7
    assert abs(metrics['hallucination_rate'] - expected_rate) < 1e-9
    assert metrics['gate_pass'] is True
    print(f"  kp2 OK: hallucination_rate={metrics['hallucination_rate']:.3f}, "
          f"gate_pass={metrics['gate_pass']}, parse_errors={metrics['parse_errors']}")


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_kp0(tmp)
        test_kp1(tmp)
        test_kp2(tmp)
    print('\n[smoke_test] ALL TESTS PASSED')


if __name__ == '__main__':
    main()
