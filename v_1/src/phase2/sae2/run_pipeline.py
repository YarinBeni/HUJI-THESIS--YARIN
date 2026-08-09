"""SAE2 pipeline (job F22): steps 0-2 + 4 of the handoff plan, on the
Neuronpedia-labeled Karvonen dictionary — an independent replication of
F7/F8/F11 in a second, independently trained SAE, now including CELL B.

  step 0  discover the release, pick the layer closest to 24 among covered
          layers, settle the empirical file-index offset (never assume
          Qwen-Scope's convention transfers);
  step 1  FVU gate on FOUR populations of last-token vectors: cell-A entities,
          cell-B Assyrian-ruler names (omitted in F7 — new), English glosses,
          Akkadian;
  step 2  feature hunt: candidate screen on entities, rho(strength, year),
          firing table across all four populations, cos(decoder row, ridge);
  step 4  token-level fired-anywhere per population, same pre-registered rules
          as F11 (eng >= 10% replicates propagation-firing; akk < 2%
          replicates non-engagement) + per-document firing fraction and a
          coarse position profile (early/middle/late thirds).

Replication verdicts are emitted explicitly per claim. Interpretation labels
are fetched separately (fetch_labels.py) so a Neuronpedia outage cannot kill
the compute run.

    python run_pipeline.py            # full
    python run_pipeline.py --steps 0 1

Writes sae2/results/pipeline.json + feature_hunt2.layer{L}.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import karvonen as K                                     # noqa: E402
_SAE1 = os.path.abspath(os.path.join(_HERE, "..", "sae"))
sys.path.insert(0, _SAE1)
from fvu_gate import AKK_ACTS, ENT_ACTS, METHOD, load_layer_acts  # noqa: E402
_PAIRS = os.path.abspath(os.path.join(_HERE, "..", "pairs"))
sys.path.insert(0, _PAIRS)
import pairs_data as P                                   # noqa: E402
_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
DIRS_A = os.path.join(_WM, "results", "directions")
ENT_CSV = os.path.join(_WM, "data", "entity_datasets", "historical_figure.csv")
RESULTS = os.path.join(_HERE, "results")

POPULATIONS = {
    "cellA_entities": os.path.join(ENT_ACTS, METHOD, "historical_figure"),
    "cellB_rulers": os.path.join(ENT_ACTS, METHOD, "assyrian_ruler"),
    "eng_tier0_frags": os.path.join(AKK_ACTS, METHOD, "eng_tier0"),
    "akk_maximal_frags": os.path.join(AKK_ACTS, METHOD, "akk_maximal"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 1, 2, 4])
    ap.add_argument("--target-layer", type=int, default=24)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--min-fire", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    out = {}

    # ---- step 0: discover, then FVU-SCAN every (layer, file, offset) -----
    # The first run taught us the repo has many configs per layer; the gate
    # itself now selects the instrument: every candidate is scored on cell-A
    # activations and the global best (layer, file, offset) wins.
    repo, layers_avail, notes = K.discover()
    scan = []
    for L in sorted(layers_avail):
        for fn in layers_avail[L]:
            try:
                cand = K.load(repo, fn)
            except SystemExit:
                scan.append({"layer": L, "file": fn, "error": "keys"})
                continue
            for off in (0, 1):
                X = load_layer_acts(POPULATIONS["cellA_entities"], L + off)
                if X is None:
                    continue
                v = round(K.fvu(X, cand), 4)
                scan.append({"layer": L, "file": fn, "offset": off,
                             "d_sae": cand["d_sae"], "mode": cand["mode"],
                             "fvu_cellA": v})
                print(f"  scan L{L} off{off} {fn}: d_sae={cand['d_sae']} "
                      f"fvu={v}", flush=True)
            del cand
    scored = [r for r in scan if "fvu_cellA" in r]
    if not scored:
        sys.exit("no candidate SAE could be scored")
    scored.sort(key=lambda r: r["fvu_cellA"])
    best_raw = scored[0]
    # LESSON FROM THE SECOND F22 RUN (23760): the global-min FVU landed on a
    # 16k trainer whose feature indices Neuronpedia does not label — the whole
    # point of this dictionary. Among near-ties (within FVU_TOL of the best)
    # prefer the 65k width, which is the Neuronpedia-covered release.
    FVU_TOL = 0.02
    labeled = [r for r in scored if r["d_sae"] == 65536
               and r["fvu_cellA"] <= best_raw["fvu_cellA"] + FVU_TOL]
    pick = labeled[0] if labeled else best_raw
    L, fn, off = pick["layer"], pick["file"], pick["offset"]
    sae = K.load(repo, fn)
    out["step0"] = {"repo": repo,
                    "layers_available": sorted(layers_avail),
                    "layer_used": L, "file_used": fn, "offset": off,
                    "sae_mode": sae["mode"], "d_sae": sae["d_sae"],
                    "fvu_best_raw": best_raw, "labeled_width_preferred":
                    bool(labeled) and pick is not best_raw,
                    "trainer_config": K.trainer_config(repo, fn),
                    "scan": scan, "notes": notes}
    print(f"[step0] PICK: L{L} off{off} {fn} d_sae={sae['d_sae']} "
          f"fvu={pick['fvu_cellA']} (raw best {best_raw['fvu_cellA']} "
          f"d_sae={best_raw['d_sae']})", flush=True)

    # ---- step 1: FVU gate, four populations ------------------------------
    if 1 in args.steps:
        row = {}
        for name, d in POPULATIONS.items():
            X = load_layer_acts(d, L + off)
            row[name] = None if X is None else round(K.fvu(X, sae), 4)
        out["step1_fvu"] = row
        print(f"[step1] fvu: {row}", flush=True)
        ca = row.get("cellA_entities")
        if ca is None or ca > 0.35:
            out["step1_verdict"] = "GATE FAILED — do not interpret features"
            print(out["step1_verdict"], flush=True)
            json.dump(out, open(os.path.join(RESULTS, "pipeline.json"), "w"),
                      indent=2)
            return

    # ---- step 2: feature hunt --------------------------------------------
    if 2 in args.steps:
        import torch
        ent = pd.read_csv(ENT_CSV)
        Xa = load_layer_acts(POPULATIONS["cellA_entities"], L + off)
        Z = K.encode(Xa, sae).numpy()
        year = ent["death_year"].values.astype(float)
        okm = ent["is_test"].astype(bool).values & np.isfinite(year)
        fire = (Z[okm] > 0).mean(0)
        cand = np.where(fire >= args.min_fire)[0]
        rho = np.array([stats.spearmanr(Z[okm, f], year[okm]).correlation
                        for f in cand])
        g = sorted(__import__("glob").glob(os.path.join(
            DIRS_A, METHOD, "historical_figure.*.layer*.npz")))
        coef = np.load(g[0])["coef"].astype(np.float32).ravel() if g else None
        W_dec = sae["W_dec"].numpy()
        cosd = ((W_dec[cand] @ coef)
                / (np.linalg.norm(W_dec[cand], axis=1)
                   * np.linalg.norm(coef) + 1e-8)) if coef is not None else \
            np.full(len(cand), np.nan)
        tab = pd.DataFrame({"feature": cand, "fire_cellA": fire[cand],
                            "rho_year": rho, "cos_ridge": cosd})
        tab = tab.reindex(tab.rho_year.abs().sort_values(ascending=False)
                          .index).head(args.top)
        for name in ("cellB_rulers", "eng_tier0_frags", "akk_maximal_frags"):
            Xp = load_layer_acts(POPULATIONS[name], L + off)
            if Xp is None:
                tab[f"fire_{name}"] = np.nan
                continue
            Zp = K.encode(Xp, sae).numpy()
            tab[f"fire_{name}"] = (Zp[:, tab.feature.values] > 0).mean(0)
        csv = os.path.join(RESULTS, f"feature_hunt2.layer{L}.csv")
        tab.to_csv(csv, index=False)
        out["step2"] = {
            "n_candidates": int(len(cand)),
            "top_abs_rho": float(tab.rho_year.abs().iloc[0]),
            "max_cos_ridge": float(np.nanmax(np.abs(tab.cos_ridge))),
            "median_fire": {c.replace("fire_", ""): float(tab[c].median())
                            for c in tab.columns if c.startswith("fire_")},
            "table": csv}
        print(f"[step2] {out['step2']}", flush=True)
        print(tab.head(10).to_string(index=False), flush=True)

    # ---- step 4: token-level firing --------------------------------------
    if 4 in args.steps:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        sys.path.insert(0, _WM)
        from wm_lib import registry
        tab = pd.read_csv(out["step2"]["table"]) if "step2" in out else \
            pd.read_csv(sorted(__import__("glob").glob(
                os.path.join(RESULTS, "feature_hunt2.layer*.csv")))[-1])
        feats = tab.feature.astype(int).tolist()
        hfid = registry.MODELS[METHOD]["hfid"]
        tok = AutoTokenizer.from_pretrained(hfid)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            hfid, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
        dev = model.device
        W_enc = sae["W_enc"].to(dev)
        b_enc = sae["b_enc"].to(dev)
        theta = sae["theta"].to(dev) if sae["theta"] is not None else None
        fidx = torch.tensor(feats, device=dev)

        df = P.load_eligible()
        ent = pd.read_csv(ENT_CSV)
        sets = {
            "cellA_entities": ent[ent.is_test.astype(bool)].name.astype(str)
                                 .sample(400, random_state=0).tolist(),
            "eng_tier0_frags": df.text_eng_tier0.fillna("").astype(str).tolist(),
            "akk_maximal_frags": df.text_akk.fillna("").astype(str).tolist(),
        }
        out["step4"] = {}
        for name, texts in sets.items():
            texts = [t for t in texts if t.strip()]
            fired, frac, thirds = [], [], np.zeros(3)
            nseen = 0
            with torch.no_grad():
                for i in range(0, len(texts), args.batch):
                    bt = texts[i:i + args.batch]
                    enc = tok(bt, return_tensors="pt", padding=True,
                              truncation=True, max_length=512).to(dev)
                    hs = model(**enc, output_hidden_states=True).hidden_states
                    # our file layer (L+off) equals hidden_states[L+off] —
                    # files label hs[1:] as 1..N; use the EMPIRICAL offset
                    h = hs[L + off].float()
                    pre = h @ W_enc.T + b_enc
                    if theta is not None:
                        z = torch.relu(pre) * (pre > theta)
                    else:
                        val, idx = torch.topk(pre, 80, dim=-1)
                        z = torch.zeros_like(pre).scatter_(
                            -1, idx, torch.relu(val))
                    zf = z[..., fidx] * enc.attention_mask.unsqueeze(-1)
                    fired.append((zf > 0).any(1).cpu().numpy())
                    Tlen = enc.attention_mask.sum(1, keepdim=True).clamp(min=1)
                    frac.append(((zf > 0).sum(1) / Tlen).cpu().numpy())
                    # position profile: which third of the text fires
                    pos = torch.arange(zf.shape[1], device=dev)[None, :, None]
                    third = (3 * pos / Tlen.unsqueeze(-1)).clamp(max=2.999).long()
                    for t3 in range(3):
                        thirds[t3] += ((zf > 0) & (third == t3)).sum().item()
                    nseen += len(bt)
            fired = np.concatenate(fired)
            out["step4"][name] = {
                "n_texts": int(nseen),
                "median_fired_anywhere": float(np.median(fired.mean(0))),
                "median_fire_fraction": float(np.median(
                    np.concatenate(frac).mean(0))),
                "position_thirds": (thirds / max(thirds.sum(), 1)).round(3)
                .tolist()}
            print(f"[step4] {name}: {out['step4'][name]}", flush=True)
        eng = out["step4"].get("eng_tier0_frags", {}).get(
            "median_fired_anywhere", 0)
        akk = out["step4"].get("akk_maximal_frags", {}).get(
            "median_fired_anywhere", 1)
        out["replication_verdicts"] = {
            "eng_midtext_firing_replicates": bool(eng >= 0.10),
            "akk_non_engagement_replicates": bool(akk < 0.02)}
        print(f"[verdicts] {out['replication_verdicts']}", flush=True)

    with open(os.path.join(RESULTS, "pipeline.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {RESULTS}/pipeline.json", flush=True)


if __name__ == "__main__":
    main()
