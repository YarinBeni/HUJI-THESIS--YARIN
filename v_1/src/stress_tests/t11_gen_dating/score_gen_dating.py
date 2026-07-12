"""T11 scoring — parse the generated year answers and score them with the SAME
balanced-MC protocol as the activation probes (200 draws x 8 rulers x 21 frags,
Spearman per draw, mean +- std), so the behavioral number sits next to the
probe number apples-to-apples.

Parsing ladder (see --selftest):
  1. strip Qwen3 thinking  (text after the last '</think>')
  2. strip gpt-oss harmony (text after the last 'assistantfinal' / 'final' channel)
  3. JSON object with "year_bce" (or "year") -> int; null -> declined
  4. fallbacks on the stripped text: "669-631 BCE" range -> midpoint;
     "650 BCE"; "7th century BCE" -> 650; last bare 1-4 digit integer
  5. answers explicitly CE/AD (and not BCE) are counted parsed-but-implausible.
Predictions outside [1, 2000] BCE are kept in the parse count but excluded from
correlation (plausibility gate; corpus years span 7-1132 BCE).

Extra process signal: does the answer text mention the fragment's true ruler
(diacritic-folded substring)? Spearman is also reported conditioned on that,
to separate name-lookup dating from style dating.

Usage:
  python score_gen_dating.py --model qwen3_8b --cleaning tier0   # one run
  python score_gen_dating.py --all                               # every raw jsonl
  python score_gen_dating.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]

DEFAULT_CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
BAL = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"
DEFAULT_DRAWS = BAL / "draws_matrix.npy"
DEFAULT_ORDER = BAL / "corpus_fragment_order.json"

PLAUS_LO, PLAUS_HI = 1, 2000   # BCE; corpus spans 7-1132


# ---------------------------------------------------------------- parsing ---
def strip_reasoning(raw: str) -> str:
    """Drop Qwen3 <think> blocks and gpt-oss analysis channels."""
    s = raw
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[1]
    low = s.lower()
    for marker in ("assistantfinal", "channelfinal"):   # harmony decodings
        if marker in low:
            i = low.rfind(marker)
            s = s[i + len(marker):]
            low = s.lower()
    return s.strip()


_RANGE = re.compile(r"(\d{1,4})\s*(?:-|–|—|\bto\b)\s*(\d{1,4})\s*(?:BCE?|B\.C\.)", re.I)
_YEAR_BCE = re.compile(r"(\d{1,4})\s*(?:BCE?|B\.C\.)", re.I)
_CENTURY = re.compile(r"(\d{1,2})(?:st|nd|rd|th)\s+century\s*(?:BCE?|B\.C\.)", re.I)
_CE_ONLY = re.compile(r"\d{1,4}\s*(?:CE|A\.D\.|AD)\b")
_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.S)
_INT = re.compile(r"\b(\d{1,4})\b")


def parse_year(raw: str):
    """-> (year_bce or None, status). status in
    {json, range, bce, century, bare_int, declined, ce_only, unparsed}."""
    s = strip_reasoning(raw)

    for m in _JSON_OBJ.finditer(s):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        for key in ("year_bce", "year"):
            if key in obj:
                v = obj[key]
                if v is None:
                    return None, "declined"
                if isinstance(v, str):
                    mm = _INT.search(v)
                    v = int(mm.group(1)) if mm else None
                if isinstance(v, (int, float)):
                    return float(v), "json"
                return None, "declined"

    m = _RANGE.search(s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (a + b) / 2.0, "range"
    m = _YEAR_BCE.search(s)
    if m:
        return float(m.group(1)), "bce"
    m = _CENTURY.search(s)
    if m:
        return float(m.group(1)) * 100 - 50, "century"
    if _CE_ONLY.search(s):
        return None, "ce_only"
    ints = [int(x) for x in _INT.findall(s)]
    ints = [x for x in ints if PLAUS_LO <= x <= PLAUS_HI]
    if ints:
        return float(ints[0]), "bare_int"
    return None, "unparsed"


def _fold(s: str) -> str:
    """Lowercase + strip diacritics (Assur-etel-ilani matches Aššur-etel-ilāni)."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().replace("ʾ", "'").replace("ʿ", "'")


def mentions_ruler(raw: str, ruler: str) -> bool:
    base = re.sub(r"\s+(?:I{1,3}|IV|V|VI{0,3}|IX|X)$", "", ruler.strip())
    return len(base) >= 4 and _fold(base) in _fold(raw)


# ---------------------------------------------------------------- scoring ---
def _sp(a, b):
    if len(a) < 3 or len(set(b)) < 2 or len(set(a)) < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def score_run(model: str, cleaning: str, raw_dir: Path, corpus: Path,
              draws: Path, order: Path, out_dir: Path):
    path = raw_dir / f"{model}__{cleaning}.jsonl"
    recs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    df = pd.read_parquet(corpus)
    fids = json.load(open(order))
    assert fids == df["fragment_id"].astype(str).tolist(), "corpus order drift"
    pos = {f: i for i, f in enumerate(fids)}

    n = len(fids)
    pred = np.full(n, np.nan)
    named = np.zeros(n, dtype=bool)
    status_counts: dict[str, int] = {}
    for r in recs:
        i = pos.get(str(r["fragment_id"]))
        if i is None:
            continue
        y, status = parse_year(r["raw_output"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if y is not None and PLAUS_LO <= y <= PLAUS_HI:
            pred[i] = y
        named[i] = mentions_ruler(r["raw_output"], str(r["ruler"]))
    true = df["year"].to_numpy(dtype=float)
    rulers = df["ruler"].astype(str).to_numpy()

    ok = np.isfinite(pred) & np.isfinite(true)
    err = np.abs(pred[ok] - true[ok])
    full = {
        "n_generated": len(recs), "n_scoreable": int(ok.sum()),
        "parse_status_counts": status_counts,
        "spearman_all": _sp(pred[ok], true[ok]),
        "mae": float(err.mean()) if ok.any() else float("nan"),
        "acc@10yr": float((err <= 10).mean()) if ok.any() else float("nan"),
        "acc@25yr": float((err <= 25).mean()) if ok.any() else float("nan"),
        "acc@50yr": float((err <= 50).mean()) if ok.any() else float("nan"),
        "named_true_ruler_rate": float(named[ok].mean()) if ok.any() else float("nan"),
        "spearman_when_named": _sp(pred[ok & named], true[ok & named]),
        "spearman_when_unnamed": _sp(pred[ok & ~named], true[ok & ~named]),
        "n_named": int((ok & named).sum()), "n_unnamed": int((ok & ~named).sum()),
    }

    # balanced MC — the SAME 200 draws as every activation probe
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
          "spearman_std": float(np.std(sps)) if sps else float("nan"),
          "per_draw_spearman": [round(float(x), 4) for x in sps]}

    out = {"model": model, "cleaning": cleaning, "protocol":
           "generated-answer dating; MC = same 200 balanced 8x21 draws as probes",
           "full_corpus": full, "mc_balanced": mc}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"t11_gen__{model}__{cleaning}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[{model} x {cleaning}] scoreable {full['n_scoreable']}/{full['n_generated']}"
          f"  sp_all={full['spearman_all']:.3f}  MC={mc['spearman_mean']:.3f}"
          f"+-{mc['spearman_std']:.3f} ({mc['n_draws_used']} draws)"
          f"  acc@25={full['acc@25yr']:.2f}  named={full['named_true_ruler_rate']:.2f}",
          flush=True)
    return out_path


# --------------------------------------------------------------- selftest ---
_SELFTEST = [
    # qwen3 thinking + json
    ('<think>\nHmm, Ashurbanipal... 669-631.\n</think>\n\n{"year_bce": 650, "basis": "Ashurbanipal titulary"}', 650.0, "json"),
    # gpt-oss harmony
    ('analysisUser wants a date. Esarhaddon reigned 680-669 BCE.assistantfinal{"year_bce": 675, "basis": "Esarhaddon"}', 675.0, "json"),
    # declined json
    ('{"year_bce": null, "basis": "cannot estimate"}', None, "declined"),
    # year as string in json
    ('{"year_bce": "c. 640 BCE", "basis": "style"}', 640.0, "json"),
    # loose range
    ("This fragment likely dates to 669–631 BCE, the reign of Ashurbanipal.", 650.0, "range"),
    # plain BCE
    ("Probably composed around 604 BC under Nebuchadnezzar II.", 604.0, "bce"),
    # century
    ("Hard to say; 7th century BCE at best.", 650.0, "century"),
    # CE-only answer -> not scoreable
    ("This looks medieval, around 1200 CE.", None, "ce_only"),
    # bare integer fallback
    ("year_bce 883", 883.0, "bare_int"),
    # garbage
    ("I am sorry, I cannot help with that.", None, "unparsed"),
    # thinking never closed + number inside would be WRONG to use; but our
    # strip only acts on a closed tag, so the number is still recovered:
    ("<think>maybe 650 BCE", 650.0, "bce"),
]


def selftest():
    bad = 0
    for raw, want_y, want_s in _SELFTEST:
        y, s = parse_year(raw)
        okk = (y == want_y) and (s == want_s)
        bad += not okk
        print(("OK " if okk else "FAIL") + f"  ({y}, {s})  <- {raw[:60]!r}")
    assert mentions_ruler("reign of Assur-etel-ilani probably", "Aššur-etel-ilāni")
    assert mentions_ruler("Nebuchadnezzar, king of Babylon", "Nebuchadnezzar II")
    assert not mentions_ruler("some Assyrian king", "Esarhaddon")
    print("ruler-mention checks OK")
    if bad:
        sys.exit(f"{bad} selftest failures")
    print("selftest PASSED")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--cleaning")
    p.add_argument("--all", action="store_true", help="score every raw/*.jsonl")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--raw_dir", default=str(_THIS.parent / "raw"))
    p.add_argument("--out_dir", default=str(_THIS.parent / "results"))
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--draws", default=str(DEFAULT_DRAWS))
    p.add_argument("--order", default=str(DEFAULT_ORDER))
    a = p.parse_args()
    if a.selftest:
        selftest(); return
    raw_dir, out_dir = Path(a.raw_dir), Path(a.out_dir)
    if a.all:
        runs = sorted(raw_dir.glob("*__*.jsonl"))
        assert runs, f"no raw jsonl under {raw_dir}"
        for f in runs:
            model, cleaning = f.stem.rsplit("__", 1)
            score_run(model, cleaning, raw_dir, Path(a.corpus),
                      Path(a.draws), Path(a.order), out_dir)
    else:
        assert a.model and a.cleaning, "--model and --cleaning (or --all/--selftest)"
        score_run(a.model, a.cleaning, raw_dir, Path(a.corpus),
                  Path(a.draws), Path(a.order), out_dir)


if __name__ == "__main__":
    main()
