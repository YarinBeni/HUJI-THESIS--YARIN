"""T12 scoring — parse the FORCED answers (year + ruler) and score with the
same balanced-MC protocol as T11/the probes, plus ruler-identification
accuracy (the pv1-pv3 templates ask for the ruler by name).

Year parsing = T11's ladder (score_gen_dating.parse_year: think-strip, JSON,
range, "N BCE", century, bare-int; plausibility gate [1,2000]). Ruler parsing
= phase-1b's parse_raw_output/normalize_ruler (canonical 8-name table) with a
diacritic-folded substring fallback for non-canonical rulers.

Also rescoreable: the committed phase-1b Qwen2.5-7B direct answers (UNforced,
tier0) as reference rows:  python score_forced.py --phase1b

Usage:
  python score_forced.py --model qwen3_8b --cleaning tier0 --variant pv1
  python score_forced.py --all          # every raw/*.jsonl
  python score_forced.py --phase1b     # rescore round2_phase1b direct answers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]
sys.path.insert(0, str(_THIS.parents[1] / "t11_gen_dating"))
sys.path.insert(0, str(_REPO / "v_1/src/linear_probing/round2_phase1b"))

from score_gen_dating import (parse_year, mentions_ruler, _sp, _fold,          # noqa: E402
                              PLAUS_LO, PLAUS_HI, DEFAULT_CORPUS,
                              DEFAULT_DRAWS, DEFAULT_ORDER)
from pv_parse import parse_raw_output, normalize_ruler                          # noqa: E402

P1B = (_REPO / "v_1/src/linear_probing/results/orcc_round2_phase1b/direct_answers")


def ruler_correct(parsed_ruler, true_ruler: str) -> bool:
    if not parsed_ruler:
        return False
    canon_p = normalize_ruler(parsed_ruler)
    canon_t = normalize_ruler(true_ruler)
    if canon_p and canon_t:
        return canon_p == canon_t
    return _fold(str(parsed_ruler)).strip() == _fold(true_ruler).strip()


def score_records(recs, model, cleaning, variant, corpus, draws, order,
                  out_dir, tag=""):
    df = pd.read_parquet(corpus)
    fids = json.load(open(order))
    assert fids == df["fragment_id"].astype(str).tolist(), "corpus order drift"
    pos = {f: i for i, f in enumerate(fids)}

    n = len(fids)
    pred = np.full(n, np.nan)
    named = np.zeros(n, dtype=bool)
    ruler_ok = np.zeros(n, dtype=bool)
    ruler_answered = np.zeros(n, dtype=bool)
    status_counts: dict[str, int] = {}
    for r in recs:
        i = pos.get(str(r["fragment_id"]))
        if i is None:
            continue
        raw = r["raw_output"]
        y, status = parse_year(raw)
        status_counts[status] = status_counts.get(status, 0) + 1
        if y is not None and PLAUS_LO <= y <= PLAUS_HI:
            pred[i] = y
        named[i] = mentions_ruler(raw, str(r["ruler"]))
        pr = parse_raw_output(raw, variant)
        if pr.get("parsed_ruler"):
            ruler_answered[i] = True
            ruler_ok[i] = ruler_correct(pr["parsed_ruler"], str(r["ruler"]))
    true = df["year"].to_numpy(dtype=float)
    rulers = df["ruler"].astype(str).to_numpy()

    ok = np.isfinite(pred) & np.isfinite(true)
    err = np.abs(pred[ok] - true[ok])
    full = {
        "n_generated": len(recs), "n_scoreable": int(ok.sum()),
        "parse_status_counts": status_counts,
        "spearman_all": _sp(pred[ok], true[ok]),
        "mae": float(err.mean()) if ok.any() else float("nan"),
        "acc@25yr": float((err <= 25).mean()) if ok.any() else float("nan"),
        "acc@50yr": float((err <= 50).mean()) if ok.any() else float("nan"),
        "named_true_ruler_rate": float(named[ok].mean()) if ok.any() else float("nan"),
        "spearman_when_named": _sp(pred[ok & named], true[ok & named]),
        "spearman_when_unnamed": _sp(pred[ok & ~named], true[ok & ~named]),
        "n_named": int((ok & named).sum()), "n_unnamed": int((ok & ~named).sum()),
        "ruler_answer_rate": float(ruler_answered.mean()),
        "ruler_acc_when_answered": (float(ruler_ok[ruler_answered].mean())
                                    if ruler_answered.any() else float("nan")),
        "ruler_acc_overall": float(ruler_ok.mean()),
    }

    dm = np.load(draws)
    sps, taken = [], []
    for d in range(dm.shape[0]):
        rows = np.where(dm[d])[0]
        rows = rows[np.isfinite(pred[rows]) & np.isfinite(true[rows])]
        if len(rows) < 10 or len(set(rulers[rows].tolist())) < 2:
            continue
        sps.append(_sp(pred[rows], true[rows]))
        taken.append(len(rows))
    sps = [s for s in sps if s == s]
    mc = {"n_draws_used": len(sps),
          "mean_frags_per_draw": float(np.mean(taken)) if taken else 0.0,
          "spearman_mean": float(np.mean(sps)) if sps else float("nan"),
          "spearman_std": float(np.std(sps)) if sps else float("nan")}

    out = {"model": model, "cleaning": cleaning, "variant": variant,
           "forced": not tag, "protocol":
           f"forced generated-answer dating ({variant}); MC = same 200 draws{tag}",
           "full_corpus": full, "mc_balanced": mc}
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "t12ref" if tag else "t12f"
    fp = out_dir / f"{stem}__{model}__{cleaning}__{variant}.json"
    fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[{model} x {cleaning} x {variant}] scoreable "
          f"{full['n_scoreable']}/{full['n_generated']}"
          f"  MC={mc['spearman_mean']:.3f}+-{mc['spearman_std']:.3f}"
          f"  acc@50={full['acc@50yr']:.2f}"
          f"  ruler_acc={full['ruler_acc_when_answered']:.2f}"
          f"@{full['ruler_answer_rate']:.2f}", flush=True)
    return fp


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--cleaning")
    p.add_argument("--variant")
    p.add_argument("--all", action="store_true")
    p.add_argument("--phase1b", action="store_true",
                   help="rescore round2_phase1b Qwen2.5-7B direct answers (unforced ref)")
    p.add_argument("--raw_dir", default=str(_THIS.parent / "raw"))
    p.add_argument("--out_dir", default=str(_THIS.parent / "results"))
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--draws", default=str(DEFAULT_DRAWS))
    p.add_argument("--order", default=str(DEFAULT_ORDER))
    a = p.parse_args()
    out_dir = Path(a.out_dir)
    if a.phase1b:
        for pv in ("pv0", "pv1", "pv2", "pv3"):
            recs = []
            for f in sorted((P1B / pv).glob("*.json")):
                d = json.loads(f.read_text())
                recs.append({"fragment_id": d["fragment_id"],
                             "ruler": d["ruler_gt"], "raw_output": d["raw_output"]})
            score_records(recs, "qwen25_7b", "tier0", pv, Path(a.corpus),
                          Path(a.draws), Path(a.order), out_dir,
                          tag="; UNforced phase-1b reference")
        return
    raw_dir = Path(a.raw_dir)
    if a.all:
        runs = sorted(raw_dir.glob("*__*__pv*.jsonl"))
        assert runs, f"no raw jsonl under {raw_dir}"
        for f in runs:
            model, cleaning, pv = f.stem.split("__")
            score_records(load_jsonl(f), model, cleaning, pv, Path(a.corpus),
                          Path(a.draws), Path(a.order), out_dir)
    else:
        assert a.model and a.cleaning and a.variant
        f = raw_dir / f"{a.model}__{a.cleaning}__{a.variant}.jsonl"
        score_records(load_jsonl(f), a.model, a.cleaning, a.variant,
                      Path(a.corpus), Path(a.draws), Path(a.order), out_dir)


if __name__ == "__main__":
    main()
