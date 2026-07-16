#!/usr/bin/env python3
"""05_merge_stress_coords.py — fold the stress-test embeddings into the seal_eda GUI.

Data side: re-key stress_coords.json entries (<cleaning>__<method>__LNN__<proj> ->
<method>__<cleaning>__LNN__<proj>, the GUI's buildKey() format), pad with
[None,None] for the SEAL rows, validate ORCC row alignment, merge into
seal_viz_data.json. Covers cleanings maximal (fills the existing button for the
follow-up methods) + engtier0 + engmaximal (new buttons).

GUI side (seal_eda.html): adds two cleaning buttons after Max+Kings —
"English t0" (translations of tier0; the translator restores real king names)
and "English max" (translations of maximal; CAVEAT: hallucinated king names —
kept for inspection, excluded from all quantitative slides). Layer snapping is
already data-driven (04's patch), so no JS changes are needed.

Idempotent: safe to re-run. Usage:  python v_1/src/viz/05_merge_stress_coords.py
Then rebuild the standalone: python v_1/src/viz/03_build_standalone_html.py
"""
from __future__ import annotations

import json
from pathlib import Path

VIZ = Path(__file__).resolve().parent
BASE = VIZ / "seal_viz_data.json"
SC = VIZ / "stress_coords.json"
HTML = VIZ / "seal_eda.html"


def merge_data():
    base = json.loads(BASE.read_text())
    sc = json.loads(SC.read_text())
    frs = base["fragments"]
    n_seal = sum(1 for f in frs if f.get("corpus") != "orcc")
    orcc_ids = [f["fragment_id"] for f in frs if f.get("corpus") == "orcc"]
    assert orcc_ids == sc["fragment_ids"], "ORCC row order mismatch — do not merge"
    added = skipped = 0
    for key, coords in sc["embeddings"].items():
        cleaning, method, layer, proj = key.split("__")
        new_key = f"{method}__{cleaning}__{layer}__{proj}"
        if new_key in base["embeddings"]:
            skipped += 1
            continue
        assert len(coords) == len(orcc_ids), new_key
        base["embeddings"][new_key] = [[None, None]] * n_seal + coords
        added += 1
    BASE.write_text(json.dumps(base), encoding="utf-8")
    print(f"merged {added} keys (skipped {skipped} already present) into {BASE.name} "
          f"({BASE.stat().st_size/1e6:.1f} MB, {len(base['embeddings'])} total keys)")


def patch_html():
    h = HTML.read_text(encoding="utf-8")
    if 'data-val="engtier0"' in h:
        print("HTML already patched")
        return

    anchor = """layers each: L00 and that model's best maxking layer.
            </span>
          </span>
        </button>"""
    assert h.count(anchor) == 1
    btns = anchor + """
        <button data-val="engtier0">
          English t0
          <span class="tip">
            <span class="ico" tabindex="0">i</span>
            <span class="body">
              <strong>English (tier0)</strong> — full-fragment machine translations of
              the tier0 texts (Thalesian/cuneiformBase-400m), embedded per model,
              mean pool. The translator restores REAL king names, so this variant is
              name-carrying by construction. Two layers per model: L00 + that
              model's best translation year layer.
            </span>
          </span>
        </button>
        <button data-val="engmaximal">
          English max
          <span class="tip">
            <span class="ico" tabindex="0">i</span>
            <span class="body">
              <strong>English (maximal) — CAVEAT</strong>: translations of the
              name-stripped maximal texts. Fed name-stripped input, the translator
              HALLUCINATES king names, so this variant is excluded from every
              quantitative slide; it is kept here for visual inspection only.
            </span>
          </span>
        </button>"""
    h = h.replace(anchor, btns)
    HTML.write_text(h, encoding="utf-8")
    print("HTML patched: engtier0 + engmaximal cleaning buttons added")


if __name__ == "__main__":
    merge_data()
    patch_html()
