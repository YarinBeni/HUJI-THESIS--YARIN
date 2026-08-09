"""SAE2 step 5 (job F23) — feature-level interventions on the labeled dictionary.

Three runs, all with the non-surgicality discipline (Feldman et al. 2026):
every treated condition has a firing-rate-matched random-feature control, and
the claim is treated-minus-control, never treated alone.

  1. AMPLIFY/SUPPRESS on cell-A entity prompts: add alpha * act95 * d_i at all
     positions of the hook layer; read the frozen cell-A ridge probe at the
     last token. A causal time feature should move the read-out monotonically
     in alpha where the matched control does not.
  2. ABLATE: h <- h - z_i(h) * d_i (remove the feature's own contribution)
     on cell-A prompts; same read-out, same control.
  3. THE BRIDGE: on English gloss fragments, clamp the top temporal features
     ON at mid-text positions and ask (a) does the last-token ridge read-out
     move, and (b) do the features now FIRE at the last token — the direct
     causal test of whether the F11 mid-text firing can be made to propagate.

Caution inherited from F12, verbatim from the plan: direction-steering was
null in this model; feature-steering may be too. A null here WITH the control
is a publishable mechanistic result (transient firing, not causally
recoverable at readout); report it against the control, don't retry until
significant.

    python feature_steer.py            # top-5 temporal + 5 matched controls

Writes results/steer.layer{L}.json. GPU.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import karvonen as K                                     # noqa: E402
_SAE1 = os.path.abspath(os.path.join(_HERE, "..", "sae"))
sys.path.insert(0, _SAE1)
from fvu_gate import ENT_ACTS, METHOD, load_layer_acts   # noqa: E402
_PAIRS = os.path.abspath(os.path.join(_HERE, "..", "pairs"))
sys.path.insert(0, _PAIRS)
import pairs_data as P                                   # noqa: E402
_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
sys.path.insert(0, _WM)
from wm_lib import registry                              # noqa: E402

DIRS_A = os.path.join(_WM, "results", "directions")
ENT_CSV = os.path.join(_WM, "data", "entity_datasets", "historical_figure.csv")
RESULTS = os.path.join(_HERE, "results")
ALPHAS = [-8, -4, -2, 0, 2, 4, 8]


def pick_features(tab, n=5):
    """Top-|rho| temporal candidates + firing-rate-matched near-zero-rho
    controls."""
    t = tab.reindex(tab.rho_year.abs().sort_values(ascending=False).index)
    treat = t.head(n)
    pool = tab[tab.rho_year.abs() < 0.05]
    ctrl_rows = []
    for _, r in treat.iterrows():
        if len(pool) == 0:
            break
        j = (pool.fire_cellA - r.fire_cellA).abs().idxmin()
        ctrl_rows.append(pool.loc[j])
        pool = pool.drop(j)
    return treat, pd.DataFrame(ctrl_rows)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-entities", type=int, default=200)
    ap.add_argument("--n-frags", type=int, default=300)
    ap.add_argument("--n-feats", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    pipe = json.load(open(os.path.join(RESULTS, "pipeline.json")))
    repo, L = pipe["step0"]["repo"], pipe["step0"]["layer_used"]
    off = pipe["step0"]["offset"]
    _, files, _ = K.discover()
    sae = K.load(repo, files[L])
    tab = pd.read_csv(sorted(glob.glob(os.path.join(
        RESULTS, "feature_hunt2.layer*.csv")))[-1])
    treat, ctrl = pick_features(tab, args.n_feats)
    print(f"[feats] treat={treat.feature.tolist()} "
          f"ctrl={ctrl.feature.tolist()}", flush=True)

    # per-feature activation scale (act95 on cell A) for clamp units
    Xa = load_layer_acts(os.path.join(ENT_ACTS, METHOD, "historical_figure"),
                         L + pipe["step0"]["offset"])
    Za = K.encode(Xa, sae).numpy()
    scale = {int(f): max(float(np.quantile(Za[:, int(f)][Za[:, int(f)] > 0], .95))
                         if (Za[:, int(f)] > 0).any() else 1.0, 1e-3)
             for f in pd.concat([treat, ctrl]).feature}

    # frozen cell-A ridge read-out
    g = sorted(glob.glob(os.path.join(DIRS_A, METHOD,
                                      "historical_figure.*.layer*.npz")))
    coef = np.load(g[0])["coef"].astype(np.float32).ravel()
    import re
    LA = int(re.search(r"layer(\d+)\.npz$", g[0]).group(1))

    hfid = registry.MODELS[METHOD]["hfid"]
    tok = AutoTokenizer.from_pretrained(hfid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        hfid, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    dev = model.device
    coef_t = torch.from_numpy(coef).to(dev)
    W_dec = sae["W_dec"]

    ent = pd.read_csv(ENT_CSV)
    ent = ent[ent.is_test.astype(bool) & ent.death_year.notna()].sample(
        args.n_entities, random_state=0)
    df = P.load_eligible()
    frags = df.text_eng_tier0.fillna("").astype(str)
    frags = [t for t in frags if t.strip()][:args.n_frags]

    def run(texts, feat, alpha, mode, exclude_last=False):
        """Mean frozen-probe score at last token + feature fire-rate at last
        token, under the intervention. mode: 'add' (alpha*act95*d) or
        'ablate' (subtract the feature's own contribution)."""
        d_vec = torch.from_numpy(W_dec[int(feat)].numpy()).to(dev,
                                                              torch.bfloat16)
        w_enc_i = sae["W_enc"][int(feat)].to(dev)
        b_i = float(sae["b_enc"][int(feat)])
        th_i = (float(sae["theta"][int(feat)])
                if sae["theta"] is not None and sae["theta"].ndim else
                (float(sae["theta"]) if sae["theta"] is not None else None))
        scores, fires = [], []
        # hidden_states[L+off] is what the SAE reads; that is the OUTPUT of
        # transformer block (L+off-1) — hook there, per the empirical offset
        blk = model.model.layers[L + off - 1]

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            T = h.shape[1]
            sl = slice(0, T - 1) if exclude_last else slice(0, T)
            if mode == "add" and alpha != 0:
                h[:, sl] = h[:, sl] + alpha * d_vec
            elif mode == "ablate":
                z = torch.relu(h[:, sl].float() @ w_enc_i + b_i)
                if th_i is not None:
                    z = z * (z > th_i)
                h[:, sl] = h[:, sl] - (z.unsqueeze(-1)
                                       * d_vec.float()).to(h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h

        with torch.no_grad():
            for i in range(0, len(texts), args.batch):
                bt = texts[i:i + args.batch]
                enc = tok(bt, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(dev)
                hd = blk.register_forward_hook(hook) \
                    if (mode == "ablate" or alpha != 0) else None
                res = model(**enc, output_hidden_states=True)
                if hd:
                    hd.remove()
                last = enc.attention_mask.sum(1) - 1
                bidx = torch.arange(len(bt), device=dev)
                hA = res.hidden_states[LA][bidx, last].float()
                scores.extend((hA @ coef_t).cpu().tolist())
                hL = res.hidden_states[L + off][bidx, last].float()
                zlast = torch.relu(hL @ w_enc_i + b_i)
                fires.extend((zlast > (th_i or 0)).float().cpu().tolist())
        return float(np.mean(scores)), float(np.mean(fires))

    out = {"layer": L, "read_layer": LA,
           "treat": treat.feature.astype(int).tolist(),
           "ctrl": ctrl.feature.astype(int).tolist(), "runs": {}}
    names = ent.name.astype(str).tolist()
    for group, feats in (("treat", out["treat"]), ("ctrl", out["ctrl"])):
        for f in feats:
            rec = {"amplify": {}, "ablate": None, "bridge": {}}
            for a in ALPHAS:
                s, _ = run(names, f, a * scale[f], "add")
                rec["amplify"][str(a)] = s
            s, _ = run(names, f, 0, "ablate")
            rec["ablate"] = s
            for a in (0, 4):
                s, fr = run(frags, f, a * scale[f], "add", exclude_last=True)
                rec["bridge"][str(a)] = {"probe": s, "fire_last": fr}
            out["runs"][f"{group}:{f}"] = rec
            print(f"[{group}:{f}] amp0={rec['amplify']['0']:+.1f} "
                  f"amp8={rec['amplify']['8']:+.1f} abl={rec['ablate']:+.1f} "
                  f"bridge fire {rec['bridge']['0']['fire_last']:.2f}->"
                  f"{rec['bridge']['4']['fire_last']:.2f}", flush=True)

    with open(os.path.join(RESULTS, f"steer.layer{L}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {RESULTS}/steer.layer{L}.json", flush=True)


if __name__ == "__main__":
    main()
