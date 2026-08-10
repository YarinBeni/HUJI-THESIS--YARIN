"""SAE2 step 3'' (job F25) — interpret features WITHOUT the logit lens.

The user's (correct) objection to F24: logit lens at layer 9 of 36 is a weak
instrument — early-layer decoder rows need not write directly to the
vocabulary, and early layers carry many context-dependent features the lens
misses. The standard practice (Neuronpedia dashboards, EleutherAI autointerp)
interprets a feature by its MAX-ACTIVATING EXAMPLES; Anthropic's Golden Gate
demo interprets it CAUSALLY by clamping it during generation and reading what
the model starts talking about. This job does both:

  1. max-activating contexts: for each top feature, the 20 strongest
     activations across cell-A entity prompts, English glosses and Akkadian
     fragments, each with a +-8-token context window and the firing token
     marked >>like this<<;
  2. Golden-Gate generation: clamp the feature at alpha * act95 at every
     position of the hook layer while the chat model answers a neutral
     open prompt; the drift of the generations (vs an unclamped baseline)
     is the feature's behavioural label.

    python feature_interp.py                  # top-10 hunt features
    python feature_interp.py --features 9763 56768

Writes results/feature_interp.layer{L}.json. GPU.
"""
from __future__ import annotations

import argparse
import glob
import heapq
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

ENT_CSV = os.path.join(_WM, "data", "entity_datasets", "historical_figure.csv")
RESULTS = os.path.join(_HERE, "results")

GEN_PROMPTS = [
    "Tell me about yourself.",
    "Write a short paragraph about anything you like.",
    "What is on your mind today?",
]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=int, nargs="*", default=None)
    ap.add_argument("--top-feats", type=int, default=10)
    ap.add_argument("--top-examples", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--gen-tokens", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    pipe = json.load(open(os.path.join(RESULTS, "pipeline.json")))
    repo, L = pipe["step0"]["repo"], pipe["step0"]["layer_used"]
    off = pipe["step0"]["offset"]
    sae = K.load(repo, pipe["step0"]["file_used"])
    tab = pd.read_csv(sorted(glob.glob(os.path.join(
        RESULTS, "feature_hunt2.layer*.csv")))[-1])
    feats = (args.features if args.features
             else tab.feature.astype(int).head(args.top_feats).tolist())

    # act95 per feature for the clamp scale (cell-A encodings, CPU)
    Xa = load_layer_acts(os.path.join(ENT_ACTS, METHOD, "historical_figure"),
                         L + off)
    Za = K.encode(Xa, sae).numpy()
    scale = {int(f): max(float(np.quantile(Za[:, int(f)][Za[:, int(f)] > 0],
                                           .95))
                         if (Za[:, int(f)] > 0).any() else 1.0, 1e-3)
             for f in feats}

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
    fidx = torch.tensor([int(f) for f in feats], device=dev)

    # ---- part 1: max-activating contexts ---------------------------------
    ent = pd.read_csv(ENT_CSV)
    df = P.load_eligible()
    sets = {
        "cellA_entities": ent[ent.is_test.astype(bool)].name.astype(str)
                             .sample(400, random_state=0).tolist(),
        "eng_tier0_frags": df.text_eng_tier0.fillna("").astype(str).tolist(),
        "akk_maximal_frags": df.text_akk.fillna("").astype(str).tolist(),
    }
    heaps = {int(f): [] for f in feats}    # (value, pop, text_i, tok_i, ctx)
    push_id = 0
    with torch.no_grad():
        for pop, texts in sets.items():
            texts = [t for t in texts if t.strip()]
            for i in range(0, len(texts), args.batch):
                bt = texts[i:i + args.batch]
                enc = tok(bt, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(dev)
                hs = model(**enc, output_hidden_states=True).hidden_states
                h = hs[L + off].float()
                pre = h @ W_enc.T + b_enc
                if theta is not None:
                    z = torch.relu(pre) * (pre > theta)
                else:
                    val, idx = torch.topk(pre, 80, dim=-1)
                    z = torch.zeros_like(pre).scatter_(
                        -1, idx, torch.relu(val))
                zf = (z[..., fidx] * enc.attention_mask.unsqueeze(-1)).cpu()
                ids = enc.input_ids.cpu()
                am = enc.attention_mask.cpu()
                for fi, f in enumerate(feats):
                    zz = zf[..., fi]
                    flat = zz.flatten()
                    k = min(args.top_examples, int((flat > 0).sum()))
                    if k == 0:
                        continue
                    topv, topi = torch.topk(flat, k)
                    T = zz.shape[1]
                    for v, ix in zip(topv.tolist(), topi.tolist()):
                        b, t = divmod(ix, T)
                        real = int(am[b].sum())   # don't decode pad tokens
                        lo, hi = max(0, t - 8), min(real, t + 9)
                        left = tok.decode(ids[b, lo:t])
                        mid = tok.decode(ids[b, t:t + 1])
                        right = tok.decode(ids[b, t + 1:hi])
                        ctx = f"{left} >>{mid}<< {right}".strip()
                        push_id += 1
                        item = (v, push_id, pop, ctx)
                        hp = heaps[int(f)]
                        if len(hp) < args.top_examples:
                            heapq.heappush(hp, item)
                        else:
                            heapq.heappushpop(hp, item)

    # ---- part 2: Golden-Gate generation ----------------------------------
    def generate(feat=None, alpha=0.0):
        hd = None
        if feat is not None and alpha:
            import torch as _t
            d_vec = _t.from_numpy(
                sae["W_dec"][int(feat)].numpy()).to(dev, _t.bfloat16)
            blk = model.model.layers[L + off - 1]

            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h += alpha * d_vec
                return (h,) + out[1:] if isinstance(out, tuple) else h
            hd = blk.register_forward_hook(hook)
        outs = []
        try:
            for p in GEN_PROMPTS:
                msgs = [{"role": "user", "content": p}]
                text = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
                enc = tok(text, return_tensors="pt",
                          add_special_tokens=False).to(dev)
                gen = model.generate(**enc, max_new_tokens=args.gen_tokens,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
                outs.append(tok.decode(gen[0, enc.input_ids.shape[1]:],
                                       skip_special_tokens=True))
        finally:
            if hd:
                hd.remove()
        return outs

    out = {"layer": L, "alpha": args.alpha, "features": {}}
    baseline = generate()
    out["baseline_generations"] = baseline
    for f in feats:
        ex = sorted(heaps[int(f)], reverse=True)
        rec = {
            "rho_year": float(tab[tab.feature == f].rho_year.iloc[0])
            if (tab.feature == f).any() else None,
            "act95": scale[int(f)],
            "max_activating": [
                {"value": round(v, 2), "population": pop, "context": ctx}
                for v, _, pop, ctx in ex],
            "generations_clamped": generate(f, args.alpha * scale[int(f)]),
        }
        out["features"][int(f)] = rec
        pops = pd.Series([e["population"] for e in rec["max_activating"]])
        print(f"[{f}] top-ex pops: {pops.value_counts().to_dict()} | "
              f"ex1: {rec['max_activating'][0]['context'][:70] if rec['max_activating'] else '-'}",
              flush=True)
        print(f"     gen: {rec['generations_clamped'][0][:100]}", flush=True)

    path = os.path.join(RESULTS, f"feature_interp.layer{L}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"[done] -> {path}", flush=True)


if __name__ == "__main__":
    main()
