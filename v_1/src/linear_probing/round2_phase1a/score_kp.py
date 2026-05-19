"""
Phase 1a — Scorer (W2.B).

Applies the Phase 1a scoring rules per variant:
  - kp0: a ruler scores CORRECT if either ground-truth (start_year, end_year)
         falls within [min(pred_start, pred_end) - 50, max(pred_start, pred_end) + 50].
  - kp1: per period, recall = |Phase-0 rulers found in returned list (diacritic-
         normalized case-insensitive)| / |Phase-0 rulers belonging to that period|.
  - kp2: hallucination_rate = fraction where declined==False OR start_year is not None.
         Parse errors are counted separately. Phase 1a gate: hallucination_rate < 0.30.

CLI:
  python score_kp.py --variant {kp0|kp1|kp2} --out_dir <path>
"""

import argparse
import json
import unicodedata
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[3]
RULER_REIGNS_PATH = REPO_ROOT / 'v_1' / 'src' / 'linear_probing' / 'results' / 'orcc_round2_phase1a' / 'ruler_reigns.json'

KP0_TOLERANCE_YEARS = 50

# Phase-0 rulers split by period (per task spec).
NA_RULERS = ['Ashurbanipal', 'Sennacherib', 'Esarhaddon', 'Sargon II',
             'Tiglath-pileser III', 'Sîn-šarru-iškun']
NB_RULERS = ['Nebuchadnezzar II', 'Nabonidus']

PERIOD_TO_RULERS = {
    'Neo-Assyrian': NA_RULERS,
    'Neo-Babylonian': NB_RULERS,
}


def normalize_name(s: str) -> str:
    """Diacritic-stripped, lowercased, whitespace-collapsed name for matching."""
    if s is None:
        return ''
    nfkd = unicodedata.normalize('NFKD', s)
    stripped = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return ' '.join(stripped.lower().split())


def load_reigns() -> dict:
    with open(RULER_REIGNS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def score_kp0(parsed: dict) -> dict:
    reigns = load_reigns()
    per_ruler = []
    n_correct = 0
    n_parse_error = 0
    n_total = 0
    for rec in parsed['results']:
        n_total += 1
        ruler = rec['input_value']
        if rec['parse_error'] or rec['parsed'] is None:
            n_parse_error += 1
            per_ruler.append({
                'ruler': ruler,
                'raw_output': rec['raw_output'],
                'parsed': rec['parsed'],
                'parse_error': True,
                'parse_error_reason': rec.get('parse_error_reason', ''),
                'correct': False,
                'reason': 'parse_error',
            })
            continue
        parsed_obj = rec['parsed']
        gt = reigns.get(ruler, {})
        gt_start = gt.get('reign_start_bce')
        gt_end = gt.get('reign_end_bce')
        gt_end_alt = gt.get('reign_end_bce_alt')
        pred_start = parsed_obj.get('start_year')
        pred_end = parsed_obj.get('end_year')
        # If model declined or returned null, that's not "correct" under ±50.
        if pred_start is None and pred_end is None:
            per_ruler.append({
                'ruler': ruler,
                'raw_output': rec['raw_output'],
                'parsed': parsed_obj,
                'parse_error': False,
                'correct': False,
                'reason': 'model_declined_or_null',
                'gt_start': gt_start,
                'gt_end': gt_end,
            })
            continue
        # Defensive: coerce to int when possible.
        try:
            ps = int(pred_start) if pred_start is not None else None
            pe = int(pred_end) if pred_end is not None else None
        except (TypeError, ValueError):
            per_ruler.append({
                'ruler': ruler,
                'raw_output': rec['raw_output'],
                'parsed': parsed_obj,
                'parse_error': False,
                'correct': False,
                'reason': 'non_integer_year',
                'gt_start': gt_start,
                'gt_end': gt_end,
            })
            continue
        years = [v for v in (ps, pe) if v is not None]
        lo = min(years) - KP0_TOLERANCE_YEARS
        hi = max(years) + KP0_TOLERANCE_YEARS
        gt_candidates = [v for v in (gt_start, gt_end, gt_end_alt) if v is not None]
        hits = [v for v in gt_candidates if lo <= v <= hi]
        is_correct = len(hits) > 0
        if is_correct:
            n_correct += 1
        per_ruler.append({
            'ruler': ruler,
            'raw_output': rec['raw_output'],
            'parsed': parsed_obj,
            'parse_error': False,
            'correct': is_correct,
            'pred_window': [lo, hi],
            'gt_start': gt_start,
            'gt_end': gt_end,
            'gt_end_alt': gt_end_alt,
            'hits': hits,
            'reason': 'within_tolerance' if is_correct else 'outside_tolerance',
        })
    n_scoreable = n_total - n_parse_error
    accuracy = (n_correct / n_total) if n_total else 0.0
    accuracy_scoreable = (n_correct / n_scoreable) if n_scoreable else 0.0
    return {
        'variant': 'kp0',
        'tolerance_years': KP0_TOLERANCE_YEARS,
        'total': n_total,
        'correct': n_correct,
        'parse_errors': n_parse_error,
        'error_rate': n_parse_error / n_total if n_total else 0.0,
        'accuracy': accuracy,
        'accuracy_scoreable': accuracy_scoreable,
        'per_ruler': per_ruler,
    }


def score_kp1(parsed: dict) -> dict:
    per_period = []
    total_hits = 0
    total_targets = 0
    n_parse_error = 0
    for rec in parsed['results']:
        period = rec['input_value']
        targets = PERIOD_TO_RULERS.get(period, [])
        target_norm = {normalize_name(r): r for r in targets}
        if rec['parse_error'] or rec['parsed'] is None:
            n_parse_error += 1
            per_period.append({
                'period': period,
                'raw_output': rec['raw_output'],
                'parsed': rec['parsed'],
                'parse_error': True,
                'parse_error_reason': rec.get('parse_error_reason', ''),
                'recall': 0.0,
                'found_targets': [],
                'missed_targets': targets,
                'extras': [],
            })
            total_targets += len(targets)
            continue
        parsed_obj = rec['parsed']
        rulers_list = parsed_obj.get('rulers', []) or []
        if not isinstance(rulers_list, list):
            rulers_list = []
        rulers_norm = [normalize_name(r) for r in rulers_list if isinstance(r, str)]
        found = []
        for tn, original in target_norm.items():
            if tn in rulers_norm:
                found.append(original)
        missed = [r for r in targets if r not in found]
        extras = [r for r, rn in zip(rulers_list, rulers_norm) if rn not in target_norm]
        recall = (len(found) / len(targets)) if targets else 0.0
        total_hits += len(found)
        total_targets += len(targets)
        per_period.append({
            'period': period,
            'raw_output': rec['raw_output'],
            'parsed': parsed_obj,
            'parse_error': False,
            'recall': recall,
            'found_targets': found,
            'missed_targets': missed,
            'extras': extras,
        })
    aggregate_recall = (total_hits / total_targets) if total_targets else 0.0
    return {
        'variant': 'kp1',
        'total_periods': len(parsed['results']),
        'parse_errors': n_parse_error,
        'aggregate_recall': aggregate_recall,
        'total_targets': total_targets,
        'total_hits': total_hits,
        'per_period': per_period,
    }


def score_kp2(parsed: dict) -> dict:
    per_name = []
    n_total = 0
    n_declined_correctly = 0
    n_hallucination = 0
    n_parse_error = 0
    for rec in parsed['results']:
        n_total += 1
        name = rec['input_value']
        if rec['parse_error'] or rec['parsed'] is None:
            n_parse_error += 1
            per_name.append({
                'name': name,
                'raw_output': rec['raw_output'],
                'parsed': rec['parsed'],
                'parse_error': True,
                'parse_error_reason': rec.get('parse_error_reason', ''),
                'declined': None,
                'hallucination': None,
                'status': 'parse_error',
            })
            continue
        po = rec['parsed']
        declined = bool(po.get('declined', False))
        start_year = po.get('start_year')
        end_year = po.get('end_year')
        # Hallucination: model claims dates for a fake ruler.
        is_hallucination = (declined is False) or (start_year is not None)
        if not is_hallucination and declined is True and start_year is None and end_year is None:
            n_declined_correctly += 1
            status = 'declined_correctly'
        else:
            n_hallucination += 1
            status = 'hallucination'
        per_name.append({
            'name': name,
            'raw_output': rec['raw_output'],
            'parsed': po,
            'parse_error': False,
            'declined': declined,
            'start_year': start_year,
            'end_year': end_year,
            'hallucination': is_hallucination,
            'status': status,
        })
    # Hallucination rate is over non-parse-error responses, per kp2.md:76-78
    # ("Parse errors on fake rulers are NOT counted as hallucinations").
    scoreable = n_total - n_parse_error
    hallucination_rate = (n_hallucination / scoreable) if scoreable else 0.0
    gate_pass = hallucination_rate < 0.30
    return {
        'variant': 'kp2',
        'total': n_total,
        'parse_errors': n_parse_error,
        'scoreable': scoreable,
        'declined_correctly': n_declined_correctly,
        'hallucinations': n_hallucination,
        'hallucination_rate': hallucination_rate,
        'gate_threshold': 0.30,
        'gate_pass': gate_pass,
        'per_name': per_name,
    }


def score_all(variant: str, out_dir: Path) -> dict:
    parsed_path = out_dir / 'parsed' / f'{variant}.json'
    if not parsed_path.exists():
        raise FileNotFoundError(f"Parsed file not found: {parsed_path}. "
                                f"Run parse_kp.py first.")
    with open(parsed_path, 'r', encoding='utf-8') as f:
        parsed = json.load(f)

    if variant == 'kp0':
        metrics = score_kp0(parsed)
        one_line = (f"[score_kp] kp0: accuracy={metrics['accuracy']:.3f} "
                    f"({metrics['correct']}/{metrics['total']}), "
                    f"parse_errors={metrics['parse_errors']}")
    elif variant == 'kp1':
        metrics = score_kp1(parsed)
        one_line = (f"[score_kp] kp1: aggregate_recall={metrics['aggregate_recall']:.3f} "
                    f"({metrics['total_hits']}/{metrics['total_targets']}), "
                    f"parse_errors={metrics['parse_errors']}")
    elif variant == 'kp2':
        metrics = score_kp2(parsed)
        one_line = (f"[score_kp] kp2: hallucination_rate={metrics['hallucination_rate']:.3f} "
                    f"({metrics['hallucinations']}/{metrics['scoreable']}), "
                    f"gate_pass={metrics['gate_pass']}, "
                    f"parse_errors={metrics['parse_errors']}")
    else:
        raise ValueError(f"Unknown variant: {variant}")

    scores_dir = out_dir / 'scores'
    scores_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = scores_dir / f'{variant}_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(one_line)
    print(f"[score_kp] Wrote {metrics_path}")
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description='Phase 1a scorer')
    p.add_argument('--variant', required=True, choices=['kp0', 'kp1', 'kp2'])
    p.add_argument('--out_dir', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    score_all(args.variant, Path(args.out_dir))


if __name__ == '__main__':
    main()
