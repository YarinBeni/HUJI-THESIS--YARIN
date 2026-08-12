"""F25b-L18 — the labeled peek: hunt year features in the ONLY layer
Neuronpedia hosts (karvonen layer 18), with the FVU caveat welded on.

Layer 18 fails the pre-registered reconstruction gate by a mile (best 65k
config FVU >> .35), so NOTHING here enters the main evidence chain. But
Neuronpedia hosts third-party autointerp labels ONLY for layer 18 — so this
run buys the one thing layer 9 cannot: independent labels for
year-correlated features. Read as corroborating anecdote, never as a result;
the FVU of the instrument is stored next to every label.

Steps: pick the 65k layer-18 file whose trainer k matches the hosted source
id convention (l0-80 -> k=80), encode the cell-A entity activations, hunt
(fire >= 2%, Spearman rho with death year on held-out entities, top-50 by
|rho|), then probe the Neuronpedia source grid and fetch labels for the
top-50.

    python l18_peek.py
Writes results/l18_peek.csv + results/l18_peek.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import karvonen as K                                     # noqa: E402
from fetch_labels import classify_label, fetch, probe_source  # noqa: E402
_SAE1 = os.path.abspath(os.path.join(_HERE, "..", "sae"))
sys.path.insert(0, _SAE1)
from fvu_gate import ENT_ACTS, METHOD, load_layer_acts   # noqa: E402

RESULTS = os.path.join(_HERE, "results")
_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
ENT_CSV = os.path.join(_WM, "data", "entity_datasets", "historical_figure.csv")
LAYER, OFFSET = 18, 1                    # offset per the step-0 scan
CANDIDATES = [f"saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_18/"
              f"trainer_{t}/ae.pt" for t in (2, 3)]      # the 65k trainers


def main():
    repo = json.load(open(os.path.join(RESULTS, "pipeline.json")))[
        "step0"]["repo"]
    fn, kk = None, None
    for c in CANDIDATES:
        try:
            cfg = K.trainer_config(repo, c)
            k = cfg["trainer"].get("k")
            print(f"[cfg] {c}: k={k}", flush=True)
            if k == 80:                  # matches the hosted l0-80 source
                fn, kk = c, k
                break
            if fn is None:
                fn, kk = c, k
        except Exception as e:                            # noqa: BLE001
            print(f"[cfg] {c}: {e}", flush=True)
    if fn is None:
        sys.exit("no 65k layer-18 file reachable")
    sae = K.load(repo, fn)
    print(f"[sae] {fn} (k={kk})", flush=True)

    ent = pd.read_csv(ENT_CSV)
    Xa = load_layer_acts(os.path.join(ENT_ACTS, METHOD, "historical_figure"),
                         LAYER + OFFSET)
    fvu_val = float(K.fvu(Xa, sae))
    print(f"[gate] FVU on cell A = {fvu_val:.3f}  (gate is .35 — this run is"
          " an anecdote by construction)", flush=True)

    Za = K.encode(Xa, sae).numpy()
    yr = ent["death_year"].values.astype(float)
    ok = ent["is_test"].astype(bool).values & np.isfinite(yr)
    fire = (Za[ok] > 0).mean(0)
    cand = np.where(fire >= 0.02)[0]
    rho = np.array([stats.spearmanr(Za[ok, f], yr[ok]).correlation
                    for f in cand])
    tab = (pd.DataFrame({"feature": cand, "fire_cellA": fire[cand],
                         "rho_year": rho})
           .reindex(np.abs(rho).argsort()[::-1]).head(50)
           .reset_index(drop=True))
    os.makedirs(RESULTS, exist_ok=True)
    tab.to_csv(os.path.join(RESULTS, "l18_peek.csv"), index=False)
    print(f"[hunt] {len(cand)} candidates; top |rho| = "
          f"{tab.rho_year.abs().max():.2f}", flush=True)

    model, src, tried = probe_source(LAYER,
                                     tab.feature.astype(int).head(5).tolist())
    out = {"layer": LAYER, "offset": OFFSET, "file": fn, "k": kk,
           "fvu_cellA": fvu_val, "gate": 0.35, "flagged_anecdote": True,
           "source_probe": tried, "model": model, "source": src,
           "labels": []}
    if src:
        for _, r in tab.iterrows():
            j = fetch(model, src, int(r.feature))
            expl = ""
            if isinstance(j, dict):
                ex = j.get("explanations") or []
                expl = (ex[0].get("description", "") if ex
                        else j.get("description", "") or "")
            out["labels"].append({
                "feature": int(r.feature), "rho_year": float(r.rho_year),
                "fire_cellA": float(r.fire_cellA), "label": expl,
                "label_class": classify_label(expl)})
            print(f"  {int(r.feature)} rho={r.rho_year:+.2f}: "
                  f"{expl[:70]}", flush=True)
            time.sleep(0.4)
    else:
        print("[labels] no working Neuronpedia source found — hunt CSV "
              "stands alone", flush=True)

    with open(os.path.join(RESULTS, "l18_peek.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] -> {RESULTS}/l18_peek.json", flush=True)


if __name__ == "__main__":
    main()
