"""E1 behavioural probe: ask the MODEL (not a probe) which text is earlier.

Protocol adapted from "The Geometry of Numerical Reasoning" (El-Shangiti et al.,
NAACL 2025), transplanted from entity names to whole inscriptions: balanced pairs,
a single-token Yes/No read-out from the first-token logits, and the P(A,B) vs
P(B,A) consistency check — every pair is asked in BOTH presentation orders, and a
model only "knows" the answer when the two orders agree.

Balancing is the same engine as the probe (quota per ruler-pair, macro metrics);
one deterministic draw, because behavioural querying at 100 draws would be GPU
suicide for no statistical gain — the uncertainty that matters here is across
ruler-pairs, not across resamples of the big grids.

Truncation: each text is clipped to --max-words words (default 120) so a pair
fits comfortably in context and the two texts get equal room.

Base-model caveat, stated rather than hidden: Llama-2-7B and OLMo-2 are base
models with no instruction tuning; their Yes/No calibration is expected to be
poor, and chance-level results for them are uninformative about representation
content (that is what the E1 probe is for). The interesting arms here are the
instruct-capable ones (qwen3 family, gpt_oss_120b).

    python behavioral_pairs.py --method qwen3_8b --variant eng_tier0
    python behavioral_pairs.py --method qwen3_8b --variant akk_maximal

Writes results/behavioral/{method}.{variant}.json
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
import pairs_data as P                                  # noqa: E402
_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
sys.path.insert(0, _WM)
from wm_lib import registry                             # noqa: E402

RESULTS = os.path.join(_HERE, "results", "behavioral")
TEXT_COL = {"akk_maximal": "text_akk", "eng_maximal": "text_eng",
            "eng_tier0": "text_eng_tier0"}

PROMPT = (
    "Below are two ancient royal inscriptions.\n\n"
    "Text A:\n{a}\n\n"
    "Text B:\n{b}\n\n"
    "Question: Was Text A composed earlier in history than Text B? "
    "Answer only Yes or No.\nAnswer:"
)


def clip(text, max_words):
    return " ".join(str(text).split()[:max_words])


def build_prompts(df, m, max_words, variant, seed):
    rng = np.random.default_rng(seed)
    pairs = P.draw_pairs(df, m, rng, P.eligible_ruler_pairs(df))
    col = TEXT_COL[variant]
    txt = df.set_index("pos")[col].fillna("").astype(str)
    rows = []
    for r in pairs.itertuples():
        a, b = clip(txt[r.pos_a], max_words), clip(txt[r.pos_b], max_words)
        if not a.strip() or not b.strip():
            continue
        # both presentation orders of the SAME pair -> consistency is measurable
        rows.append({"prompt": PROMPT.format(a=a, b=b), "label": r.label,
                     "pair_id": r.Index, "order": 0,
                     "rp": f"{min(r.ruler_a, r.ruler_b)}|{max(r.ruler_a, r.ruler_b)}",
                     "dyear": r.dyear})
        rows.append({"prompt": PROMPT.format(a=b, b=a), "label": 1 - r.label,
                     "pair_id": r.Index, "order": 1,
                     "rp": rows[-1]["rp"], "dyear": r.dyear})
    return pd.DataFrame(rows)


def yes_no_scores(method, prompts, batch_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = registry.MODELS[method]
    hfid = spec["hfid"] if isinstance(spec, dict) else spec.hfid
    tok = AutoTokenizer.from_pretrained(hfid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"       # last-token index below assumes right padding
    model = AutoModelForCausalLM.from_pretrained(
        hfid, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    # every leading-space/case variant of Yes and No that maps to one token
    def ids_of(words):
        out = set()
        for w in words:
            for v in (w, " " + w):
                t = tok.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    out.add(t[0])
        return sorted(out)
    yes_ids, no_ids = ids_of(["Yes", "yes"]), ids_of(["No", "no"])
    assert yes_ids and no_ids, "tokenizer has no single-token Yes/No"

    scores = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to(model.device)
            logits = model(**enc).logits
            last = enc.attention_mask.sum(1) - 1
            lg = logits[torch.arange(len(batch)), last]
            y = lg[:, yes_ids].logsumexp(-1)
            n = lg[:, no_ids].logsumexp(-1)
            scores.extend((y - n).float().cpu().tolist())
            if (i // batch_size) % 20 == 0:
                print(f"  {i + len(batch)}/{len(prompts)}", flush=True)
    return np.array(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--variant", default="eng_tier0", choices=list(TEXT_COL))
    ap.add_argument("--m", type=int, default=3,
                    help="pairs per ruler-pair (each asked in both orders)")
    ap.add_argument("--max-words", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=P.SEED)
    args = ap.parse_args()

    df = P.load_eligible()
    prompts = build_prompts(df, args.m, args.max_words, args.variant, args.seed)
    print(f"[data] {len(prompts)} prompts ({len(prompts) // 2} pairs, "
          f"{prompts.rp.nunique()} ruler-pairs)", flush=True)

    prompts["score"] = yes_no_scores(args.method, prompts.prompt.tolist(),
                                     args.batch_size)
    prompts["pred"] = (prompts.score > 0).astype(int)
    prompts["correct"] = (prompts.pred == prompts.label).astype(float)

    # consistency: the two orders of one pair must give OPPOSITE answers
    wide = prompts.pivot_table(index="pair_id", columns="order",
                               values=["pred", "correct", "dyear"],
                               aggfunc="first")
    consistent = (wide["pred"][0] != wide["pred"][1]).astype(float)
    rp_of = prompts.drop_duplicates("pair_id").set_index("pair_id")["rp"]

    per_rp = prompts.groupby("rp")["correct"].mean()
    cut = pd.cut(prompts.dyear, [0, 25, 75, 200, np.inf], right=False)
    out = {
        "method": args.method, "variant": args.variant, "m": args.m,
        "max_words": args.max_words, "n_pairs": int(len(wide)),
        "n_ruler_pairs": int(prompts.rp.nunique()),
        "macro_acc": float(per_rp.mean()),
        "micro_acc": float(prompts.correct.mean()),
        "order_consistency": float(consistent.mean()),
        "macro_acc_consistent_only": float(
            prompts[prompts.pair_id.isin(consistent[consistent == 1].index)]
            .groupby("rp")["correct"].mean().mean()) if consistent.sum() else None,
        "acc_by_dyear": {str(iv): float(g.correct.mean())
                         for iv, g in prompts.groupby(cut, observed=True)
                         if len(g) >= 20},
        "yes_rate": float(prompts.pred.mean()),   # a Yes-bias detector
    }
    os.makedirs(RESULTS, exist_ok=True)
    pth = os.path.join(RESULTS, f"{args.method}.{args.variant}.json")
    with open(pth, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] macro_acc={out['macro_acc']:.3f} "
          f"consistency={out['order_consistency']:.3f} "
          f"yes_rate={out['yes_rate']:.3f} -> {pth}", flush=True)
    _ = rp_of  # kept for future per-ruler-pair dumps


if __name__ == "__main__":
    main()
