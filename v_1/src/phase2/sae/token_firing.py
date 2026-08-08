"""E5.3 — token-level firing of the year features: closing F8's biggest gap.

F8 measured feature firing on the LAST-TOKEN vector of each fragment and found
the year features silent on documents. But a feature could fire mid-text (on a
date word, an era phrase) and simply not survive to the final position — in
which case the "time features never engage on documents" claim would be an
artifact of where we looked. This script re-runs the fragments (and a sample of
entity mentions) through Qwen3-8B, encodes EVERY token position through the
Scope SAE at the F8 layer, and reports per feature:

  * fired-anywhere rate: fraction of texts where the feature exceeds 0 at ANY
    token — the honest version of F8's firing rate;
  * max activation over tokens (95th percentile across texts);
  * token-level firing rate.

If fired-anywhere stays near zero on documents, F8's conclusion survives its
own audit. If it jumps, the story changes to "fires locally, does not
propagate" — which is a different (and testable) mechanism.

    python token_firing.py                # F8's layer, its top-50 features
    python token_firing.py --n-entities 400 --max-frags 600

GPU job (F11). Downloads nothing new beyond the SAE layer already cached by F7.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fvu_gate import METHOD, RESULTS, load_sae            # noqa: E402

_PAIRS = os.path.abspath(os.path.join(_HERE, "..", "pairs"))
sys.path.insert(0, _PAIRS)
import pairs_data as P                                    # noqa: E402
_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
sys.path.insert(0, _WM)
from wm_lib import registry                               # noqa: E402

ENT_CSV = os.path.join(_WM, "data", "entity_datasets", "historical_figure.csv")


def token_features(texts, sae_layer_block, feats, batch, max_len):
    """Run texts through the model; return per-text (fired_any, max_act) for the
    requested features, plus token-level fire counts."""
    import torch
    hfid = registry.MODELS[METHOD]["hfid"]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hfid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        hfid, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    W_enc, b_enc, W_dec, b_dec = load_sae(sae_layer_block)
    dev = model.device
    W_enc, b_enc = W_enc.to(dev), b_enc.to(dev)
    fidx = torch.tensor(feats, device=dev)

    fired, maxact, tokfire, ntok = [], [], np.zeros(len(feats)), 0
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            bt = [str(t) for t in texts[i:i + batch]]
            enc = tok(bt, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(dev)
            hs = model(**enc, output_hidden_states=True).hidden_states
            # hidden_states[k] = resid after block k-1 -> SAE block L is hs[L+1]
            h = hs[sae_layer_block + 1].float()
            pre = h @ W_enc.T + b_enc                    # (b, t, d_sae)
            k = 100
            val, idx = torch.topk(pre, k, dim=-1)
            z = torch.zeros_like(pre).scatter_(-1, idx, torch.relu(val))
            zf = z[..., fidx]                            # (b, t, n_feats)
            mask = enc.attention_mask.bool().unsqueeze(-1)
            zf = zf * mask
            fired.append((zf > 0).any(dim=1).cpu().numpy())
            maxact.append(zf.max(dim=1).values.cpu().numpy())
            tokfire += (zf > 0).sum(dim=(0, 1)).cpu().numpy()
            ntok += int(enc.attention_mask.sum())
            if (i // batch) % 20 == 0:
                print(f"  {i + len(bt)}/{len(texts)}", flush=True)
    return (np.concatenate(fired), np.concatenate(maxact),
            tokfire / max(ntok, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-entities", type=int, default=400)
    ap.add_argument("--max-frags", type=int, default=1187)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    hunts = sorted(f for f in os.listdir(RESULTS)
                   if f.startswith("feature_hunt.layer") and f.endswith(".csv"))
    if not hunts:
        sys.exit("run feature_hunt first — the feature list comes from it")
    hunt = pd.read_csv(os.path.join(RESULTS, hunts[-1]))
    L = int(hunts[-1].split("layer")[1].split(".")[0])
    feats = hunt.feature.astype(int).tolist()
    print(f"[feats] {len(feats)} year-features from {hunts[-1]} (SAE block {L})",
          flush=True)

    df = P.load_eligible()
    ent = pd.read_csv(ENT_CSV)
    ent = ent[ent.is_test.astype(bool)].sample(
        n=min(args.n_entities, len(ent)), random_state=0)
    sets = {
        "cellA_entities": ent.name.astype(str).tolist(),
        "eng_tier0_frags": df.text_eng_tier0.fillna("").astype(str)
                             .tolist()[:args.max_frags],
        "akk_maximal_frags": df.text_akk.fillna("").astype(str)
                               .tolist()[:args.max_frags],
    }

    out = {"sae_block": L, "features": feats, "sets": {}}
    for name, texts in sets.items():
        texts = [t for t in texts if t.strip()]
        print(f"[run] {name}: {len(texts)} texts", flush=True)
        fired, maxact, tokrate = token_features(texts, L, feats,
                                                args.batch, args.max_len)
        out["sets"][name] = {
            "n_texts": len(texts),
            "fired_anywhere_rate": fired.mean(axis=0).round(4).tolist(),
            "p95_max_act": np.quantile(maxact, .95, axis=0).round(3).tolist(),
            "token_fire_rate": tokrate.round(6).tolist(),
        }
        print(f"  median fired-anywhere over features: "
              f"{np.median(fired.mean(axis=0)):.4f}", flush=True)

    # the one-line verdict F8 needs
    med = {n: float(np.median(s["fired_anywhere_rate"]))
           for n, s in out["sets"].items()}
    out["median_fired_anywhere"] = med
    print(f"[verdict] median fired-anywhere: {med}", flush=True)

    with open(os.path.join(RESULTS, f"token_firing.layer{L}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {RESULTS}/token_firing.layer{L}.json", flush=True)


if __name__ == "__main__":
    main()
