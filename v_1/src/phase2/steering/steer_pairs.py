"""E2 — causal steering along the fitted year direction (NAACL'25 recipe).

El-Shangiti et al. showed the first PLS component of entity-token activations is
USED, not just decodable: adding alpha * w at the second entity's name token flips
the model's earlier/later answers, with an equal-norm random direction as control.
This transplants their protocol onto our ladder with our frozen cell-A ridge
direction, in a three-rung ladder:

  cell A: "Did {X} die before {Y}?"  — famous entities   (expected: flips)
  cell B: "Did {X} rule before {Y}?" — Assyrian rulers    (localizes degradation)
  (cell C — steering inside Akkadian fragments — needs the king-token span
   machinery and is deliberately deferred; see README.)

TWO READ-OUTS, because F2 showed these models' Yes/No calibration is fragile:
  * flip rate: answers changed vs the alpha=0 run (the paper's metric);
  * logit shift: mean change of logit(Yes)-logit(No) — a dose-response that
    exists even under a No-bias.
Controls: equal-norm random direction (per the paper) and alpha=0.

The intervention: forward hook on the residual stream after block L, adding
(alpha/||w||) * w ONLY at the last token of the second entity's name span
(char-span -> token-span via offset_mapping).

    python steer_pairs.py --method qwen3_8b --cell A
    python steer_pairs.py --method qwen3_8b --cell B --n-pairs 80

Writes results/{method}.cell{cell}.json. GPU job (F9).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
sys.path.insert(0, _WM)
from wm_lib import registry                              # noqa: E402

DIRS_A = os.path.join(_WM, "results", "directions")
ENT_CSV = os.path.join(_WM, "data", "entity_datasets", "historical_figure.csv")
CORPUS = os.path.join(os.path.dirname(_WM), "..",
                      "data/evaluation/corpora/orcc_corpus.parquet")
RESULTS = os.path.join(_HERE, "results")

PROMPT = {"A": "Did {x} die before {y}? Answer only Yes or No.\nAnswer:",
          "B": ("{x} and {y} were ancient Mesopotamian kings. "
                "Did {x} rule before {y}? Answer only Yes or No.\nAnswer:")}


def sample_pairs(cell, n_pairs, seed):
    rng = np.random.default_rng(seed)
    if cell == "A":
        df = pd.read_csv(ENT_CSV)
        df = df[df.is_test.astype(bool) & df.death_year.notna()]
        names, years = df.name.values, df.death_year.values.astype(float)
    else:
        df = pd.read_parquet(os.path.abspath(CORPUS))
        df = df[df.year.notna()][["ruler", "year"]].drop_duplicates("ruler")
        names, years = df.ruler.values, -df.year.values.astype(float)
        # corpus years are BCE magnitudes; negate so "earlier" sorts first
    rows, seen = [], set()
    while len(rows) < n_pairs and len(seen) < 50 * n_pairs:
        i, j = rng.integers(0, len(names), 2)
        if i == j or years[i] == years[j] or (i, j) in seen:
            seen.add((i, j))
            continue
        seen.add((i, j))
        rows.append({"x": names[i], "y": names[j],
                     "label": int(years[i] < years[j])})   # 1 = Yes, x earlier
    d = pd.DataFrame(rows)
    ones = d.label.sum()
    print(f"[pairs] cell {cell}: {len(d)} pairs, {ones} Yes / {len(d) - ones} No",
          flush=True)
    return d


def find_direction(method):
    g = sorted(glob.glob(os.path.join(DIRS_A, method,
                                      "historical_figure.*.layer*.npz")))
    if not g:
        sys.exit(f"no cell-A direction for {method} — run probe_wm first")
    L = int(re.search(r"layer(\d+)\.npz$", g[0]).group(1))
    w = np.load(g[0])["coef"].astype(np.float32).ravel()
    return w / (np.linalg.norm(w) + 1e-8), L, os.path.basename(g[0])


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--cell", required=True, choices=["A", "B"])
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[-24, -16, -8, 0, 8, 16, 24])
    ap.add_argument("--layer-span", type=int, default=4,
                    help="also steer at bestlayer +- this")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    w, LA, src = find_direction(args.method)
    pairs = sample_pairs(args.cell, args.n_pairs, args.seed)
    hfid = registry.MODELS[args.method]["hfid"]
    tok = AutoTokenizer.from_pretrained(hfid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        hfid, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    n_blocks = len(model.model.layers)

    def ids_of(words):
        out = set()
        for word in words:
            for v in (word, " " + word):
                t = tok.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    out.add(t[0])
        return sorted(out)
    yes_ids, no_ids = ids_of(["Yes", "yes"]), ids_of(["No", "no"])

    # prompts + the token position of the SECOND entity's last name token
    prompts, positions = [], []
    for r in pairs.itertuples():
        p = PROMPT[args.cell].format(x=r.x, y=r.y)
        enc = tok(p, return_offsets_mapping=True, add_special_tokens=True)
        c0 = p.rfind(str(r.y))
        c1 = c0 + len(str(r.y))
        toks = [i for i, (a, b) in enumerate(enc.offset_mapping)
                if a < c1 and b > c0]
        if not toks:
            continue
        prompts.append(p)
        positions.append(toks[-1])
    labels = pairs.label.values[:len(prompts)]
    print(f"[prompts] {len(prompts)} usable", flush=True)

    rng = np.random.default_rng(args.seed)
    w_rand = rng.standard_normal(len(w)).astype(np.float32)
    w_rand /= np.linalg.norm(w_rand)

    def run(layer_block, alpha, vec):
        """Score all prompts with (alpha * vec) added at each prompt's position
        on the residual stream after block layer_block."""
        shifts = np.zeros(len(prompts))
        vec_t = torch.from_numpy(vec).to(model.device, torch.bfloat16)

        for i in range(0, len(prompts), 16):
            bp = prompts[i:i + 16]
            bpos = positions[i:i + 16]
            enc = tok(bp, return_tensors="pt", padding=True).to(model.device)

            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                for bi, pos in enumerate(bpos):
                    h[bi, pos] = h[bi, pos] + alpha * vec_t
                return (h,) + out[1:] if isinstance(out, tuple) else h
            hd = (model.model.layers[layer_block].register_forward_hook(hook)
                  if alpha != 0 else None)
            with torch.no_grad():
                lg = model(**enc).logits
            if hd:
                hd.remove()
            last = enc.attention_mask.sum(1) - 1
            row = lg[torch.arange(len(bp)), last].float()
            s = (row[:, yes_ids].logsumexp(-1)
                 - row[:, no_ids].logsumexp(-1)).cpu().numpy()
            shifts[i:i + 16] = s
        return shifts

    # our file layers label hidden_states[1:]; block index = file layer - 1
    blocks = sorted({max(0, min(n_blocks - 1, LA - 1 + d))
                     for d in (-args.layer_span, 0, args.layer_span)})
    out = {"method": args.method, "cell": args.cell, "direction": src,
           "cellA_layer": LA, "blocks": blocks, "alphas": args.alphas,
           "n_prompts": len(prompts), "runs": {}}
    for blk in blocks:
        base = run(blk, 0.0, w)
        base_pred = base > 0
        acc0 = float((base_pred.astype(int) == labels).mean())
        rec = {"alpha0_acc": acc0, "alpha0_yes_rate": float(base_pred.mean()),
               "steer": {}, "random_control": {}}
        for alpha in args.alphas:
            if alpha == 0:
                continue
            for name, vec in (("steer", w), ("random_control", w_rand)):
                s = run(blk, alpha, vec)
                rec[name][str(alpha)] = {
                    "flip_rate": float((s > 0).astype(int).__ne__(
                        base_pred.astype(int)).mean()),
                    "mean_logit_shift": float((s - base).mean()),
                    "acc": float(((s > 0).astype(int) == labels).mean()),
                }
            st = rec["steer"][str(alpha)]
            rc = rec["random_control"][str(alpha)]
            print(f"[blk {blk} a={alpha:+.0f}] steer flip={st['flip_rate']:.2f} "
                  f"dlogit={st['mean_logit_shift']:+.2f} | "
                  f"rand flip={rc['flip_rate']:.2f} "
                  f"dlogit={rc['mean_logit_shift']:+.2f}", flush=True)
        out["runs"][str(blk)] = rec

    os.makedirs(RESULTS, exist_ok=True)
    pth = os.path.join(RESULTS, f"{args.method}.cell{args.cell}.json")
    with open(pth, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {pth}", flush=True)


if __name__ == "__main__":
    main()
