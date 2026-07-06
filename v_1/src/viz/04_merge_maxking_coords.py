#!/usr/bin/env python3
"""04_merge_maxking_coords.py — fold the maxking embeddings into the seal_eda GUI.

Data side: re-key maxking_coords.json entries (maxking__<method>__LNN__<proj> ->
<method>__maxking__LNN__<proj>, matching the GUI's buildKey() format), pad with
[None,None] for the 384 SEAL rows, validate row alignment against the ORCC
fragment order, and merge into seal_viz_data.json.

GUI side (seal_eda.html): adds the 7 new method options + a "Maximal+Kings"
cleaning button + data-driven layer snapping (the maxking dump has 2 layers per
model: L0 + that model's best maxking layer), + method tooltips.

Idempotent: safe to re-run. Usage:  python v_1/src/viz/04_merge_maxking_coords.py
"""
from __future__ import annotations

import json
from pathlib import Path

VIZ = Path(__file__).resolve().parent
BASE = VIZ / "seal_viz_data.json"
MK = VIZ / "maxking_coords.json"
HTML = VIZ / "seal_eda.html"

NEW_METHODS = {
    "qwen3_1b7": "Qwen3-1.7B", "qwen3_8b": "Qwen3-8B", "qwen3_32b": "Qwen3-32B",
    "gpt_oss_120b": "gpt-oss-120B", "thalesian_akk300m": "Thalesian-AKK-300m",
    "thalesian_cunei400m": "Thalesian-cunei-400m", "umt5_base": "uMT5-base",
}


def merge_data():
    base = json.loads(BASE.read_text())
    mk = json.loads(MK.read_text())
    frs = base["fragments"]
    n_seal = sum(1 for f in frs if f.get("corpus") != "orcc")
    orcc_ids = [f["fragment_id"] for f in frs if f.get("corpus") == "orcc"]
    assert orcc_ids == mk["fragment_ids"], "ORCC row order mismatch — do not merge"
    added = 0
    for key, coords in mk["embeddings"].items():
        _, method, layer, proj = key.split("__")
        new_key = f"{method}__maxking__{layer}__{proj}"
        assert len(coords) == len(orcc_ids)
        base["embeddings"][new_key] = [[None, None]] * n_seal + coords
        added += 1
    BASE.write_text(json.dumps(base), encoding="utf-8")
    print(f"merged {added} keys into {BASE.name} "
          f"({BASE.stat().st_size/1e6:.1f} MB, {len(base['embeddings'])} total keys)")


def patch_html():
    h = HTML.read_text(encoding="utf-8")
    if 'data-val="maxking"' in h:
        print("HTML already patched"); return

    # 1) method options
    old = '<option value="mlm">Yarin MLM</option>'
    opts = old + "".join(f'\n        <option value="{v}">{lbl} (follow-up)</option>'
                         for v, lbl in NEW_METHODS.items())
    assert h.count(old) == 1
    h = h.replace(old, opts)

    # 2) cleaning button (after the maximal button)
    anchor = """most writing-convention signals.
            </span>
          </span>
        </button>"""
    btn = anchor + """
        <button data-val="maxking">
          Max+Kings
          <span class="tip">
            <span class="ico" tabindex="0">i</span>
            <span class="body">
              <strong>maximal-with-kings</strong> — the full maximal cleaning, but the
              commissioning ruler's name span is located first and frozen intact
              (name-aware truncation keeps it). All three pooling sites live on one
              text. Available for the follow-up methods (and Random Qwen) at two
              layers each: L00 and that model's best maxking layer.
            </span>
          </span>
        </button>"""
    assert h.count(anchor) == 1
    h = h.replace(anchor, btn)

    # 3) data-driven layer snapping
    old_clamp = """function clampLayerForMethod(method, rawVal) {
  if (method === "mlm") {
    return MLM_LAYERS.reduce((prev, cur) =>
      Math.abs(cur - rawVal) < Math.abs(prev - rawVal) ? cur : prev
    );
  }
  return rawVal;
}"""
    new_clamp = """function availableLayers(method, cleaning) {
  if (!DATA) return null;
  const re = new RegExp("^" + method + "__" + cleaning + "__L(\\\\d+)__");
  const found = new Set();
  Object.keys(DATA.embeddings).forEach(k => {
    const m = k.match(re);
    if (m) found.add(parseInt(m[1], 10));
  });
  return found.size ? [...found].sort((a, b) => a - b) : null;
}

function clampLayerForMethod(method, rawVal) {
  const avail = availableLayers(method, state.cleaning);
  if (avail && avail.length) {
    return avail.reduce((prev, cur) =>
      Math.abs(cur - rawVal) < Math.abs(prev - rawVal) ? cur : prev
    );
  }
  if (method === "mlm") {
    return MLM_LAYERS.reduce((prev, cur) =>
      Math.abs(cur - rawVal) < Math.abs(prev - rawVal) ? cur : prev
    );
  }
  return rawVal;
}"""
    assert h.count(old_clamp) == 1
    h = h.replace(old_clamp, new_clamp)

    # 4) configureSlider: use discovered max for non-tfidf methods
    old_else = """  } else {
    slider.disabled = false;
    slider.min = 0; slider.max = 28; slider.step = 1;
    slider.value = state.layer;
    valEl.textContent = layerLabel(state.layer);
    valEl.style.opacity = "1";
  }
}"""
    new_else = """  } else {
    slider.disabled = false;
    const avail = availableLayers(method, state.cleaning);
    slider.min = 0;
    slider.max = (avail && avail.length) ? avail[avail.length - 1] : 28;
    slider.step = 1;
    const snapped = clampLayerForMethod(method, state.layer);
    state.layer = snapped;
    slider.value = snapped;
    valEl.textContent = layerLabel(snapped);
    valEl.style.opacity = "1";
  }
}"""
    assert h.count(old_else) == 1
    h = h.replace(old_else, new_else)

    # 5) reconfigure slider when the cleaning changes
    old_wire = """      state[stateKey] = btn.dataset.val;
      render();"""
    new_wire = """      state[stateKey] = btn.dataset.val;
      if (stateKey === "cleaning") configureSlider(state.method);
      render();"""
    assert h.count(old_wire) == 1
    h = h.replace(old_wire, new_wire)

    # 6) method tooltips
    descs = "".join(
        f"""  {v}: `
    <strong>{lbl}</strong> — follow-up stress-test embedding (maximal-with-kings
    cleaning, mean pool). Two layers bundled: L00 and this model's best maxking
    layer. See v_1/src/stress_tests/ADVISOR_WALKTHROUGH.md.
  `,\n""" for v, lbl in NEW_METHODS.items())
    old_end = "};"
    # insert before the METHOD_DESC closing brace (first occurrence after its start)
    i = h.index("const METHOD_DESC = {")
    j = h.index("\n};", i)
    h = h[:j] + ",\n" + descs.rstrip().rstrip(",") + h[j:]
    HTML.write_text(h, encoding="utf-8")
    print("patched seal_eda.html")


if __name__ == "__main__":
    merge_data()
    patch_html()
