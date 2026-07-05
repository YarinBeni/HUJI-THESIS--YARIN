"""J3 — prompted-activation extractor for the T10 redo, pooling mean + king_last
+ king_mean (whole-sentence last-token dropped per Yarin).

Adapts round2_phase1b/extract_prompted_acts.py: same prompt construction
(pv0-pv3, reusing run_pv helpers) but pools at our three sites:
  * mean      — masked mean over the target <<FRAG>> span tokens.
  * king_last — last token of the commissioning ruler's name (inside the span).
  * king_mean — mean over the ruler-name tokens.
Causal models only (Qwen3, gpt-oss), exactly like the original T10.

Output: {out_dir}/prompted_king/{variant}/L{LL}.npz with keys
  mean, king_last, king_mean (N,D float32; king_* NaN where name not found),
  fragment_ids, rulers, years, found.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]
sys.path.insert(0, str(_REPO / "v_1/src/linear_probing/round2_phase1b"))
sys.path.insert(0, str(_THIS.parents[1] / "shared"))

from pv_parse import parse_prompt_md          # noqa: E402
from run_pv import (                          # noqa: E402
    DEFAULT_CORPUS, DEFAULT_DRAWS_MATRIX, DEFAULT_FRAGMENT_ORDER, DEFAULT_PROMPTS_DIR,
    FEWSHOT_VARIANTS, SEED, build_chat_prompt, compute_span_token_indices,
    render_user_prompt, select_fewshot_examples,
)
import king_token as kt                        # noqa: E402
from cleaning import clean_maximal_keepking    # noqa: E402


def _fragment_text(row, cleaning, spellings):
    """The fragment text to embed inside the prompt, per --cleaning.
    tier0 (default, original behavior) / maximal (names destroyed -> king sites all
    NaN, mean-only) / maxking (maximal context, king name frozen in)."""
    if cleaning == "tier0":
        return str(row.text_tier0)
    if cleaning == "maximal":
        return str(row.text_maximal)
    sp = spellings.get(getattr(row, "ruler", None), [])
    return clean_maximal_keepking(str(row.text_tier0), sp)[0]


def run(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    sub = "prompted_king" if args.cleaning == "tier0" else f"prompted_king_{args.cleaning}"
    out_dir = Path(args.out_dir) / sub / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)

    prompt = parse_prompt_md(str(Path(args.prompts_dir) / f"{args.variant}.md"))
    df = pd.read_parquet(args.corpus)
    spellings = kt.load_spellings()

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
            output_hidden_states=True, attn_implementation="sdpa")  # avoid OOM on big attention
    except Exception as e:  # noqa: BLE001
        print(f"[load] sdpa failed ({type(e).__name__}); default attention", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto", output_hidden_states=True)
    model.eval()
    core = getattr(model, "model", model)

    fewshot = None
    if args.variant in FEWSHOT_VARIANTS:
        fewshot = select_fewshot_examples(
            df=df, draws_matrix_path=Path(args.draws_matrix),
            fragment_order_path=Path(args.fragment_order), tokenizer=tok,
            n_examples=5, truncate_tokens=150, seed=SEED)

    layers = [int(x) for x in str(args.layers).split(",")] if args.layers != "all" else None
    buf = {"mean": {}, "king_last": {}, "king_mean": {}}
    found = np.zeros(len(df), dtype=bool)
    fids, rulers, years = [], [], []
    t0 = time.time()

    for i, row in enumerate(df.itertuples(index=False)):
        text = _fragment_text(row, args.cleaning, spellings)
        if not text.strip():
            text = "..."   # ~6 all-logogram frags empty under maximal; keep seq non-empty
        up = render_user_prompt(prompt["user_template"], text, fewshot)
        pstr, input_ids = build_chat_prompt(tok, prompt.get("system_prompt"), up, args.variant)
        try:
            s_start, s_end = compute_span_token_indices(tok, pstr, input_ids)
        except Exception as e:
            print(f"  [span-err] {row.fragment_id}: {e}", flush=True); continue

        # locate king-name tokens INSIDE the target fragment span (avoids few-shot)
        sp = spellings.get(getattr(row, "ruler", None), [])
        span = None
        if sp:
            sub = kt.name_span_by_ids(input_ids[0, s_start:s_end + 1].tolist(), tok, sp)
            if sub is not None:
                span = (s_start + sub[0], s_start + sub[1])
        found[i] = span is not None
        fids.append(str(row.fragment_id)); rulers.append(str(row.ruler))
        years.append(np.nan if pd.isna(row.year) else float(row.year))  # NA -> NaN (int32 can't hold NA)

        ids = input_ids.to(model.device)
        attn = torch.ones_like(ids)
        with torch.no_grad():
            hs = core(input_ids=ids, attention_mask=attn, output_hidden_states=True,
                      use_cache=False).hidden_states
        Ls = layers if layers is not None else list(range(len(hs)))
        for L in Ls:
            h = hs[L][0].float().cpu().numpy()          # (seq, D)
            seg = h[s_start:s_end + 1]
            buf["mean"].setdefault(L, []).append(seg.mean(0).astype(np.float32))
            if span is None:
                nan = np.full(h.shape[-1], np.nan, np.float32)
                buf["king_last"].setdefault(L, []).append(nan)
                buf["king_mean"].setdefault(L, []).append(nan)
            else:
                buf["king_last"].setdefault(L, []).append(h[span[1]].astype(np.float32))
                buf["king_mean"].setdefault(L, []).append(h[span[0]:span[1] + 1].mean(0).astype(np.float32))
        if i % 200 == 0:
            print(f"[{args.variant}] {i}/{len(df)} found={found[:i+1].sum()} ({time.time()-t0:.0f}s)", flush=True)

    for L in buf["mean"]:
        np.savez_compressed(
            out_dir / f"L{L:02d}.npz",
            mean=np.vstack(buf["mean"][L]), king_last=np.vstack(buf["king_last"][L]),
            king_mean=np.vstack(buf["king_mean"][L]),
            fragment_ids=np.array(fids), rulers=np.array(rulers),
            years=np.array(years, dtype=np.float32), found=found)
    print(f"[{args.variant}] DONE coverage={found.mean():.3f} layers={len(buf['mean'])}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["pv0", "pv1", "pv2", "pv3"])
    p.add_argument("--cleaning", default="tier0", choices=["tier0", "maximal", "maxking"],
                   help="fragment text cleaning inside the prompt (default tier0 = original)")
    p.add_argument("--model_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--layers", default="all", help="'all' or comma list of layer indices")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--prompts_dir", default=str(DEFAULT_PROMPTS_DIR))
    p.add_argument("--draws_matrix", default=str(DEFAULT_DRAWS_MATRIX))
    p.add_argument("--fragment_order", default=str(DEFAULT_FRAGMENT_ORDER))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
