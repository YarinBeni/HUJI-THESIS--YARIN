"""
Phase 1a — Aggregator (W3.B).

Merges the three per-variant metric JSONs (kp0/kp1/kp2) produced by
`score_kp.py` into a single human-readable markdown report and a machine-
readable summary JSON consumable by Wave-4 phase-synthesis.

Inputs (per-variant schemas — see score_kp.py for exact keys):
  - kp0_metrics.json  (score_kp.py:133-143)
        keys: variant, tolerance_years, total, correct, parse_errors,
              error_rate, accuracy, accuracy_scoreable, per_ruler[]
        per_ruler: ruler, raw_output, parsed, parse_error, correct, reason,
                   (pred_window, gt_start, gt_end, gt_end_alt, hits) when no error
  - kp1_metrics.json  (score_kp.py:195-203)
        keys: variant, total_periods, parse_errors, aggregate_recall,
              total_targets, total_hits, per_period[]
        per_period: period, raw_output, parsed, parse_error, recall,
                    found_targets, missed_targets, extras
  - kp2_metrics.json  (score_kp.py:256-267)
        keys: variant, total, parse_errors, scoreable, declined_correctly,
              hallucinations, hallucination_rate, gate_threshold, gate_pass,
              per_name[]
        per_name: name, raw_output, parsed, parse_error, declined,
                  start_year, end_year, hallucination, status

Gate thresholds (per PLAN_round2_qwen_diagnosis.md Phase 1a):
  knows_rulers  := kp0_accuracy >= 0.625 AND
                   kp1_aggregate_recall >= 0.50 AND
                   kp2_hallucination_rate < 0.30

CLI:
  python aggregate_p1a.py \
      --scores_dir v_1/src/linear_probing/results/orcc_round2_phase1a/direct_kp/scores \
      --out_dir    v_1/src/linear_probing/results/orcc_round2_phase1a/aggregated
"""

import argparse
import json
from pathlib import Path

KP0_ACC_THRESHOLD = 0.625        # 5 / 8
KP1_RECALL_THRESHOLD = 0.50
KP2_HALLU_THRESHOLD = 0.30       # strict <


# ---------- IO helpers ----------

def load_metrics(scores_dir: Path) -> dict:
    """Load the three per-variant metric files. Missing files raise."""
    out = {}
    for v in ('kp0', 'kp1', 'kp2'):
        p = scores_dir / f'{v}_metrics.json'
        if not p.exists():
            raise FileNotFoundError(f"Missing metrics file: {p}")
        with open(p, 'r', encoding='utf-8') as f:
            out[v] = json.load(f)
    return out


# ---------- formatters ----------

def _fmt_year(y):
    if y is None:
        return ''
    try:
        return str(int(y))
    except (TypeError, ValueError):
        return str(y)


def _fmt_range_gt(rec: dict) -> str:
    """Ground-truth range string for kp0 per_ruler entry."""
    s = rec.get('gt_start')
    e = rec.get('gt_end')
    alt = rec.get('gt_end_alt')
    if s is None and e is None:
        return 'unknown'
    base = f"{_fmt_year(s)}-{_fmt_year(e)} BCE"
    if alt is not None:
        base += f" (alt end {_fmt_year(alt)})"
    return base


def _fmt_range_pred(rec: dict) -> str:
    """Predicted range string for kp0 per_ruler entry."""
    if rec.get('parse_error'):
        return '(parse_error)'
    parsed = rec.get('parsed') or {}
    ps = parsed.get('start_year')
    pe = parsed.get('end_year')
    if ps is None and pe is None:
        if parsed.get('declined'):
            return '(declined)'
        return '(null)'
    return f"{_fmt_year(ps)}-{_fmt_year(pe)} BCE"


# ---------- markdown section builders ----------

def _md_table(headers, rows):
    """Plain pipe-table that renders OK in a terminal (no GFM-only widgets)."""
    out = ['| ' + ' | '.join(headers) + ' |',
           '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join('' if c is None else str(c) for c in r) + ' |')
    return '\n'.join(out)


def render_kp0_section(m: dict) -> str:
    lines = ['## kp0 — "When did ruler X reign?" (8 real rulers)\n']
    lines.append(f"- Tolerance: +/- {m['tolerance_years']} years")
    lines.append(f"- Total: {m['total']}, Correct: {m['correct']}, "
                 f"Parse errors: {m['parse_errors']}")
    lines.append(f"- Accuracy (overall): {m['accuracy']:.3f}")
    lines.append(f"- Accuracy (scoreable only): {m['accuracy_scoreable']:.3f}")
    lines.append(f"- Gate: accuracy >= {KP0_ACC_THRESHOLD:.3f}? "
                 f"{'PASS' if m['accuracy'] >= KP0_ACC_THRESHOLD else 'FAIL'}\n")
    rows = []
    for rec in m['per_ruler']:
        if rec.get('parse_error'):
            verdict = 'parse_error'
        elif rec.get('correct'):
            verdict = 'HIT'
        else:
            verdict = f"MISS ({rec.get('reason', '')})"
        rows.append([
            rec['ruler'],
            _fmt_range_gt(rec),
            _fmt_range_pred(rec),
            verdict,
            'yes' if rec.get('parse_error') else 'no',
        ])
    lines.append(_md_table(
        ['ruler', 'true range', 'predicted range', 'hit/miss', 'parse_error'],
        rows,
    ))
    return '\n'.join(lines) + '\n'


def render_kp1_section(m: dict) -> str:
    lines = ['## kp1 — "Which rulers reigned during period Y?" (2 periods)\n']
    lines.append(f"- Total periods: {m['total_periods']}, "
                 f"Parse errors: {m['parse_errors']}")
    lines.append(f"- Aggregate recall over Phase-0 rulers: "
                 f"{m['aggregate_recall']:.3f} "
                 f"({m['total_hits']}/{m['total_targets']})")
    lines.append(f"- Gate: aggregate_recall >= {KP1_RECALL_THRESHOLD:.2f}? "
                 f"{'PASS' if m['aggregate_recall'] >= KP1_RECALL_THRESHOLD else 'FAIL'}\n")
    rows = []
    for rec in m['per_period']:
        if rec.get('parse_error'):
            returned = '(parse_error)'
        else:
            parsed = rec.get('parsed') or {}
            returned_list = parsed.get('rulers', []) or []
            returned = '; '.join(str(r) for r in returned_list) if returned_list else '(empty)'
        expected = '; '.join(rec.get('missed_targets', []) + rec.get('found_targets', []))
        # Re-sort to canonical (period definition order) — use found + missed.
        all_targets = rec.get('found_targets', []) + rec.get('missed_targets', [])
        # Note: order is found-first then missed; for display, just join both lists.
        rows.append([
            rec['period'],
            '; '.join(all_targets) if all_targets else '(none)',
            returned,
            f"{rec.get('recall', 0.0):.3f}",
        ])
    lines.append(_md_table(
        ['period', 'expected Phase-0 rulers', 'model-returned set', 'recall'],
        rows,
    ))

    # Informational: extras (correctly-named rulers outside our Phase-0 8).
    any_extras = any(rec.get('extras') for rec in m['per_period'])
    if any_extras:
        lines.append('\n### kp1 extras (informational — rulers returned outside Phase-0 set)\n')
        ex_rows = []
        for rec in m['per_period']:
            extras = rec.get('extras', []) or []
            if extras:
                ex_rows.append([rec['period'], '; '.join(str(e) for e in extras)])
        lines.append(_md_table(['period', 'extras'], ex_rows))
    return '\n'.join(lines) + '\n'


def render_kp2_section(m: dict) -> str:
    lines = ['## kp2 — Hallucination probe (8 fake names)\n']
    lines.append(f"- Total: {m['total']}, Scoreable: {m['scoreable']}, "
                 f"Parse errors: {m['parse_errors']}")
    lines.append(f"- Declined correctly: {m['declined_correctly']}, "
                 f"Hallucinations: {m['hallucinations']}")
    lines.append(f"- Hallucination rate (over scoreable): "
                 f"{m['hallucination_rate']:.3f}")
    lines.append(f"- Gate (< {m['gate_threshold']}): "
                 f"{'PASS' if m['gate_pass'] else 'FAIL'}\n")
    rows = []
    for rec in m['per_name']:
        if rec.get('parse_error'):
            rows.append([rec['name'], 'parse_error', '', 'parse_error'])
            continue
        declined = rec.get('declined')
        ps = rec.get('start_year')
        pe = rec.get('end_year')
        hallu_year = ''
        if ps is not None or pe is not None:
            hallu_year = f"{_fmt_year(ps)}-{_fmt_year(pe)}"
        rows.append([
            rec['name'],
            'yes' if declined else 'no',
            hallu_year,
            rec.get('status', ''),
        ])
    lines.append(_md_table(
        ['fake_name', 'declined', 'hallucinated_year_if_any', 'status'],
        rows,
    ))
    return '\n'.join(lines) + '\n'


# ---------- verdict ----------

def compute_verdict(metrics: dict) -> dict:
    kp0 = metrics['kp0']
    kp1 = metrics['kp1']
    kp2 = metrics['kp2']

    kp0_pass = kp0['accuracy'] >= KP0_ACC_THRESHOLD
    kp1_pass = kp1['aggregate_recall'] >= KP1_RECALL_THRESHOLD
    kp2_pass = kp2['hallucination_rate'] < KP2_HALLU_THRESHOLD

    knows_rulers = kp0_pass and kp1_pass and kp2_pass
    failed = []
    if not kp0_pass:
        failed.append(
            f"kp0 accuracy {kp0['accuracy']:.3f} < {KP0_ACC_THRESHOLD:.3f}"
        )
    if not kp1_pass:
        failed.append(
            f"kp1 aggregate_recall {kp1['aggregate_recall']:.3f} "
            f"< {KP1_RECALL_THRESHOLD:.2f}"
        )
    if not kp2_pass:
        failed.append(
            f"kp2 hallucination_rate {kp2['hallucination_rate']:.3f} "
            f">= {KP2_HALLU_THRESHOLD:.2f}"
        )
    return {
        'knows_rulers': knows_rulers,
        'kp0_pass': kp0_pass,
        'kp1_pass': kp1_pass,
        'kp2_pass': kp2_pass,
        'failed_checks': failed,
        'thresholds': {
            'kp0_accuracy_min': KP0_ACC_THRESHOLD,
            'kp1_recall_min': KP1_RECALL_THRESHOLD,
            'kp2_hallucination_max': KP2_HALLU_THRESHOLD,
        },
    }


def render_verdict_block(verdict: dict, metrics: dict) -> str:
    lines = ['# Phase 1a — Knowledge-Probe Aggregated Report\n']
    if verdict['knows_rulers']:
        lines.append('## VERDICT: Qwen knows the rulers — Phase 1a is NOT the bottleneck.\n')
    else:
        lines.append('## VERDICT: Phase 1a is a candidate bottleneck (one or more sub-checks failed).\n')
        lines.append('Failed sub-checks:')
        for f in verdict['failed_checks']:
            lines.append(f"  - {f}")
        lines.append('')
    lines.append('### Headline metrics')
    lines.append(f"  - kp0 accuracy:            {metrics['kp0']['accuracy']:.3f} "
                 f"(threshold >= {KP0_ACC_THRESHOLD:.3f}, "
                 f"{'PASS' if verdict['kp0_pass'] else 'FAIL'})")
    lines.append(f"  - kp1 aggregate_recall:    {metrics['kp1']['aggregate_recall']:.3f} "
                 f"(threshold >= {KP1_RECALL_THRESHOLD:.2f}, "
                 f"{'PASS' if verdict['kp1_pass'] else 'FAIL'})")
    lines.append(f"  - kp2 hallucination_rate:  {metrics['kp2']['hallucination_rate']:.3f} "
                 f"(threshold < {KP2_HALLU_THRESHOLD:.2f}, "
                 f"{'PASS' if verdict['kp2_pass'] else 'FAIL'})")
    lines.append('')
    return '\n'.join(lines) + '\n'


# ---------- top-level ----------

def build_summary(metrics: dict, verdict: dict) -> dict:
    """Machine-readable summary for Wave-4 consumption."""
    return {
        'verdict': verdict,
        'kp0': {
            'tolerance_years': metrics['kp0']['tolerance_years'],
            'total': metrics['kp0']['total'],
            'correct': metrics['kp0']['correct'],
            'parse_errors': metrics['kp0']['parse_errors'],
            'accuracy': metrics['kp0']['accuracy'],
            'accuracy_scoreable': metrics['kp0']['accuracy_scoreable'],
        },
        'kp1': {
            'total_periods': metrics['kp1']['total_periods'],
            'parse_errors': metrics['kp1']['parse_errors'],
            'total_targets': metrics['kp1']['total_targets'],
            'total_hits': metrics['kp1']['total_hits'],
            'aggregate_recall': metrics['kp1']['aggregate_recall'],
            'per_period_recall': {
                rec['period']: rec['recall']
                for rec in metrics['kp1']['per_period']
            },
        },
        'kp2': {
            'total': metrics['kp2']['total'],
            'scoreable': metrics['kp2']['scoreable'],
            'parse_errors': metrics['kp2']['parse_errors'],
            'declined_correctly': metrics['kp2']['declined_correctly'],
            'hallucinations': metrics['kp2']['hallucinations'],
            'hallucination_rate': metrics['kp2']['hallucination_rate'],
            'gate_threshold': metrics['kp2']['gate_threshold'],
            'gate_pass': metrics['kp2']['gate_pass'],
        },
    }


def aggregate(scores_dir: Path, out_dir: Path) -> dict:
    metrics = load_metrics(scores_dir)
    verdict = compute_verdict(metrics)
    summary = build_summary(metrics, verdict)

    md_parts = [
        render_verdict_block(verdict, metrics),
        render_kp0_section(metrics['kp0']),
        render_kp1_section(metrics['kp1']),
        render_kp2_section(metrics['kp2']),
    ]
    md = '\n'.join(md_parts)

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / 'phase1a_report.md'
    json_path = out_dir / 'phase1a_summary.json'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[aggregate_p1a] Wrote {md_path}")
    print(f"[aggregate_p1a] Wrote {json_path}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description='Phase 1a aggregator')
    p.add_argument('--scores_dir', required=True,
                   help='Directory containing kp0_metrics.json, '
                        'kp1_metrics.json, kp2_metrics.json')
    p.add_argument('--out_dir', required=True,
                   help='Output directory for phase1a_report.md and '
                        'phase1a_summary.json')
    return p.parse_args()


def main():
    args = parse_args()
    aggregate(Path(args.scores_dir), Path(args.out_dir))


if __name__ == '__main__':
    main()
