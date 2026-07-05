"""rescore_t9_kp1.py — kp1 (period->rulers recall) rescored by RAW-TEXT SCAN.

The original scorer (round2_phase1a/score_kp.py::score_kp1) requires (a) a parseable
JSON object and (b) an exact diacritic-normalized name match inside its "rulers"
list. That under-counts in two documented ways:
  * JSON parse failures: qwen3_32b (Neo-Assyrian) and qwen3_8b (Neo-Babylonian)
    returned malformed/truncated JSON -> scored 0 for the whole period; gpt-oss-120b
    emits its reasoning ("analysis") channel first and hit the max_new_tokens=512
    budget (run_kp.py) BEFORE printing the final JSON -> scored 0 on both periods.
  * exact-match: a name variant ("Assurbanipal" vs "Ashurbanipal",
    "Sin-shar-ishkun" vs "Sîn-šarru-iškun") counts as a miss.

Rescore rule here: normalize the WHOLE raw output (NFKD diacritic-strip, lowercase)
and count a target ruler as found if its normalized name — or a listed common
variant — appears as a substring. This recovers knowledge lost to formatting; it
CANNOT recover text lost to truncation (gpt-oss's Neo-Assyrian list is cut mid-way
at "Shalmaneser V (727-722)," so later kings are genuinely absent from the output).

Writes results/t9_kp1_rescored.json and prints a comparison table.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

ST = Path(__file__).resolve().parents[1]

NA = ["Ashurbanipal", "Sennacherib", "Esarhaddon", "Sargon II",
      "Tiglath-pileser III", "Sîn-šarru-iškun"]
NB = ["Nebuchadnezzar II", "Nabonidus"]
TARGETS = {"Neo-Assyrian": NA, "Neo-Babylonian": NB}

# common romanization variants (Assyriological literature) checked IN ADDITION to
# the canonical normalized name; kept explicit so the rescore is auditable.
VARIANTS = {
    "Ashurbanipal": ["assurbanipal", "asurbanipal", "ashur-bani-pal", "assur-bani-pal"],
    "Sennacherib": ["sin-ahhe-eriba", "sin-ahhi-eriba"],
    "Esarhaddon": ["ashur-aha-iddina", "assur-aha-iddina", "asarhaddon"],
    "Sargon II": ["sharru-kin", "sarru-kin", "sargon"],
    "Tiglath-pileser III": ["tiglathpileser", "tukulti-apil-esharra", "tiglat-pileser"],
    "Sîn-šarru-iškun": ["sin-shar-ishkun", "sin-sharra-ishkun", "sin-sar-iskun",
                        "sinsharishkun", "sin-shar-ishkun"],
    "Nebuchadnezzar II": ["nebuchadrezzar", "nabu-kudurri-usur", "nebuchadnezzar"],
    "Nabonidus": ["nabu-naid", "nabonid"],
}


def norm(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return " ".join("".join(c for c in nfkd if not unicodedata.combining(c)).lower().split())


def scan(raw: str, ruler: str) -> bool:
    r = norm(raw)
    if norm(ruler) in r:
        return True
    return any(v in r for v in VARIANTS.get(ruler, []))


def main():
    out = {"rule": "normalized substring scan over RAW output (+ listed variants)",
           "note": "recovers format/parse losses; cannot recover truncation "
                   "(gpt-oss NA list cut at max_new_tokens=512)",
           "models": {}}
    print(f"{'model':14s} {'period':14s} {'orig':>6} {'rescored':>9}  found / missed")
    for m in ["qwen3_1b7", "qwen3_8b", "qwen3_32b", "gpt_oss_120b"]:
        base = ST / f"redo_t9_knowledge/direct_kp_{m}"
        parsed = json.loads((base / "parsed/kp1.json").read_text())
        orig = json.loads((base / "scores/kp1_metrics.json").read_text())
        hits = tot = 0
        per = {}
        for rec in parsed["results"]:
            period = rec["input_value"]
            targets = TARGETS[period]
            found = [t for t in targets if scan(rec["raw_output"], t)]
            missed = [t for t in targets if t not in found]
            hits += len(found); tot += len(targets)
            per[period] = {"found": found, "missed": missed,
                           "recall": len(found) / len(targets)}
            op = next(p for p in orig["per_period"] if p["period"] == period)
            print(f"{m:14s} {period:14s} {op['recall']:>6.2f} {per[period]['recall']:>9.2f}  "
                  f"{found} / {missed}")
        out["models"][m] = {"orig_recall": orig["aggregate_recall"],
                            "rescored_recall": hits / tot, "per_period": per}
        print(f"{m:14s} {'AGGREGATE':14s} {orig['aggregate_recall']:>6.2f} {hits/tot:>9.2f}")
    fp = ST / "results" / "t9_kp1_rescored.json"
    fp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", fp)


if __name__ == "__main__":
    main()
