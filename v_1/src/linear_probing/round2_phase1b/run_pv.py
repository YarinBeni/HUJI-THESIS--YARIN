"""run_pv.py — Phase 1b direct-answer inference driver for one prompt variant.

Runs Qwen 2.5-7B-Instruct on all ORCC fragments wrapped in a prompt template
(pv0 / pv1 / pv2 / pv3). Writes one JSON file per fragment with:
  fragment_id, ruler_gt, year_gt, raw_output, parsed_ruler, parsed_year,
  parsed_confidence, parse_error, span_token_indices, reasoning_text (pv3).

Reuses model-loading patterns from v_1/src/linear_probing/03_extract_seal_activations.py
(lines 42-58: AutoTokenizer + AutoModelForCausalLM with bfloat16 + device_map="auto").

CLI:
    python run_pv.py --variant pv0 --out_dir <path>
    python run_pv.py --variant pv2 --model_path <override> --out_dir <path>

Constraints:
  - pv0 sends NO system message (system_prompt suppressed in chat template).
  - pv2 selects 5 held-out fragments from ORCC at startup using draws_matrix.npy.
  - Generation is deterministic: do_sample=False, fixed seed.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

_THIS_FILE = pathlib.Path(__file__).resolve()
_THIS_DIR = _THIS_FILE.parent
_REPO_ROOT = _THIS_FILE.parents[4]  # lititure-review/
sys.path.insert(0, str(_THIS_DIR))

from pv_parse import (  # noqa: E402
    parse_prompt_md,
    parse_raw_output,
    locate_target_span_chars,
)

# ---------------------------------------------------------------------------
# Default paths (override via CLI / env)
# ---------------------------------------------------------------------------
DEFAULT_CORPUS = _REPO_ROOT / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
DEFAULT_PROMPTS_DIR = (
    _REPO_ROOT / "v_1/src/linear_probing/results/orcc_round2_phase1b/prompts"
)
DEFAULT_DRAWS_MATRIX = (
    _REPO_ROOT
    / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/draws_matrix.npy"
)
DEFAULT_FRAGMENT_ORDER = (
    _REPO_ROOT
    / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/corpus_fragment_order.json"
)
DEFAULT_OUT_ROOT = (
    _REPO_ROOT / "v_1/src/linear_probing/results/orcc_round2_phase1b"
)

DEFAULT_MODEL = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
SEED = 42

# Variants that require few-shot example pool selection at startup.
FEWSHOT_VARIANTS = {"pv2"}

# Ruler set for few-shot pool selection — must match RULERS_8 in
# v_1/src/linear_probing/round2_phase0/build_balanced_subset.py
RULERS_8 = [
    "Ashurbanipal",
    "Sennacherib",
    "Esarhaddon",
    "Sargon II",
    "Nebuchadnezzar II",
    "Tiglath-pileser III",
    "Nabonidus",
    "Sîn-šarru-iškun",
]
FEWSHOT_PREFERRED = [
    "Ashurbanipal",
    "Sennacherib",
    "Esarhaddon",
    "Sargon II",
    "Tiglath-pileser III",
]


# ---------------------------------------------------------------------------
# Few-shot pool selection (pv2 only)
# ---------------------------------------------------------------------------
def select_fewshot_examples(
    df: pd.DataFrame,
    draws_matrix_path: pathlib.Path,
    fragment_order_path: pathlib.Path,
    tokenizer,
    n_examples: int = 5,
    truncate_tokens: int = 150,
    seed: int = SEED,
) -> list[dict]:
    """Pick `n_examples` ORCC fragments (1 per ruler) that are NOT in any MC draw.

    Returns list of dicts with keys: fragment_id, ruler, year, text.
    `text` is the tier-0 text truncated at `truncate_tokens` tokens.

    Raises FileNotFoundError if draws_matrix.npy is missing (Phase 0 prerequisite).
    """
    if not draws_matrix_path.exists():
        raise FileNotFoundError(
            f"draws_matrix.npy missing at {draws_matrix_path}. "
            "Run Phase 0 build_balanced_subset.py first."
        )
    if not fragment_order_path.exists():
        raise FileNotFoundError(
            f"corpus_fragment_order.json missing at {fragment_order_path}."
        )
    draws_matrix = np.load(draws_matrix_path)  # (n_draws, n_frags)
    with open(fragment_order_path, "r", encoding="utf-8") as f:
        order: list[str] = json.load(f)

    # Fragment is "holdout" if NEVER appears in any draw row.
    ever_used = draws_matrix.any(axis=0)  # (n_frags,) bool
    holdout_ids = {fid for fid, used in zip(order, ever_used) if not used}

    rng = random.Random(seed)
    selected: list[dict] = []
    used_rulers: set[str] = set()

    # Greedy: walk preferred ruler order first, fall back to remaining 8.
    ruler_order = FEWSHOT_PREFERRED + [r for r in RULERS_8 if r not in FEWSHOT_PREFERRED]

    for ruler in ruler_order:
        if len(selected) >= n_examples:
            break
        if ruler in used_rulers:
            continue
        candidates = df[(df["ruler"] == ruler) & (df["fragment_id"].isin(holdout_ids))]
        if len(candidates) == 0:
            print(f"  [pv2 fewshot] WARN: no holdout fragments for ruler {ruler!r}", flush=True)
            continue
        # prefer longer fragments (more formulaic signal) — sort by word_count desc, pick top-5, then random
        cand_sorted = candidates.sort_values("word_count", ascending=False).head(max(5, len(candidates) // 2))
        idx = rng.randrange(len(cand_sorted))
        row = cand_sorted.iloc[idx]
        text_full = row["text_tier0"]
        # truncate to truncate_tokens using the actual tokenizer
        tok_ids = tokenizer.encode(text_full, add_special_tokens=False)
        if len(tok_ids) > truncate_tokens:
            text_trunc = tokenizer.decode(tok_ids[:truncate_tokens], skip_special_tokens=True)
        else:
            text_trunc = text_full
        selected.append({
            "fragment_id": str(row["fragment_id"]),
            "ruler": ruler,
            "year": int(row["year"]),
            "text": text_trunc,
        })
        used_rulers.add(ruler)

    if len(selected) < n_examples:
        raise RuntimeError(
            f"pv2 fewshot: only found {len(selected)}/{n_examples} examples. "
            f"Used rulers: {used_rulers}"
        )

    return selected


def fill_fewshot_template(template: str, examples: list[dict]) -> str:
    """Replace {{example_N_text/ruler/year}} placeholders. N is 1-indexed."""
    out = template
    for i, ex in enumerate(examples, start=1):
        out = out.replace(f"{{{{example_{i}_text}}}}", ex["text"])
        out = out.replace(f"{{{{example_{i}_ruler}}}}", ex["ruler"])
        out = out.replace(f"{{{{example_{i}_year}}}}", str(ex["year"]))
    return out


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------
def render_user_prompt(
    user_template: str,
    fragment_text: str,
    fewshot_examples: list[dict] | None = None,
) -> str:
    """Substitute {{fragment_text}} (and few-shot placeholders) into template."""
    t = user_template
    if fewshot_examples:
        t = fill_fewshot_template(t, fewshot_examples)
    t = t.replace("{{fragment_text}}", fragment_text)
    return t


def build_chat_prompt(
    tokenizer,
    system_prompt: str | None,
    user_prompt: str,
    variant: str,
):
    """Apply Qwen chat template. For pv0 we send NO system message (no default
    Qwen system prompt is injected because we omit the system turn entirely).

    Returns (prompt_string, input_ids tensor of shape (1, T)).
    """
    if variant == "pv0":
        # Explicitly omit system turn to suppress Qwen's default system message.
        messages = [{"role": "user", "content": user_prompt}]
    else:
        messages = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt},
        ]
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
    return prompt_str, enc.input_ids


def compute_span_token_indices(
    tokenizer,
    prompt_str: str,
    input_ids,
) -> tuple[int, int]:
    """Return (span_start_token, span_end_token) — token indices (in input_ids[0])
    such that span_end_token is the LAST token of the target fragment content,
    inclusive (the token used for activation pooling).

    Strategy:
      1. Find char offsets of fragment content via locate_target_span_chars().
      2. Re-tokenize prompt with return_offsets_mapping=True (fast tokenizer
         required). Map char offsets -> token indices.
    """
    char_start, char_end = locate_target_span_chars(prompt_str)
    enc = tokenizer(
        prompt_str,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors=None,
    )
    offsets = enc["offset_mapping"]  # list of (start, end)
    span_start_token = None
    span_end_token = None
    for tok_idx, (s, e) in enumerate(offsets):
        if s == e:  # special token / boundary
            continue
        # first token whose start >= char_start AND that lies before char_end
        if span_start_token is None and s >= char_start and s < char_end:
            span_start_token = tok_idx
        if s < char_end and e <= char_end:
            span_end_token = tok_idx
    if span_start_token is None or span_end_token is None:
        # Fallback: search by re-decoding tokens
        # (Should not happen with Qwen fast tokenizer.)
        raise RuntimeError(
            f"Failed to locate fragment span tokens. char range=({char_start},{char_end})"
        )
    return span_start_token, span_end_token


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir) / "direct_answers" / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Seed ----
    random.seed(SEED)
    np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ---- Load prompt ----
    prompt_path = pathlib.Path(args.prompts_dir) / f"{args.variant}.md"
    prompt = parse_prompt_md(str(prompt_path))
    print(f"[prompt] variant={args.variant}  system={'<empty>' if prompt['system_prompt'] is None else 'present'}", flush=True)

    # ---- Load corpus ----
    df = pd.read_parquet(args.corpus)
    print(f"[corpus] loaded {len(df)} fragments from {args.corpus}", flush=True)
    assert "fragment_id" in df.columns and "text_tier0" in df.columns

    # ---- Load model ----
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[model] loading {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"[model] loaded. device={next(model.parameters()).device}", flush=True)

    # ---- Few-shot pool (pv2 only) ----
    fewshot_examples: list[dict] | None = None
    if args.variant in FEWSHOT_VARIANTS:
        fewshot_examples = select_fewshot_examples(
            df=df,
            draws_matrix_path=pathlib.Path(args.draws_matrix),
            fragment_order_path=pathlib.Path(args.fragment_order),
            tokenizer=tokenizer,
            n_examples=5,
            truncate_tokens=150,
            seed=SEED,
        )
        pool_path = out_dir.parent.parent / f"fewshot_pool_{args.variant}.json"
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pool_path, "w", encoding="utf-8") as f:
            json.dump(fewshot_examples, f, indent=2, ensure_ascii=False)
        print(f"[pv2] few-shot pool saved to {pool_path}", flush=True)
        for ex in fewshot_examples:
            print(f"  ex: {ex['fragment_id']}  {ex['ruler']}  {ex['year']} BCE", flush=True)

    # ---- Generation loop ----
    n = len(df)
    max_new = args.max_new_tokens
    print(f"[gen] starting; n={n}  max_new_tokens={max_new}  do_sample=False", flush=True)
    for i, row in enumerate(df.itertuples(index=False)):
        fragment_id = str(row.fragment_id)
        out_path = out_dir / f"{fragment_id}.json"
        if out_path.exists() and not args.overwrite:
            continue

        fragment_text = str(row.text_tier0)
        user_prompt = render_user_prompt(prompt["user_template"], fragment_text, fewshot_examples)
        prompt_str, input_ids = build_chat_prompt(
            tokenizer, prompt["system_prompt"], user_prompt, args.variant,
        )
        try:
            span_start_token, span_end_token = compute_span_token_indices(
                tokenizer, prompt_str, input_ids,
            )
        except Exception as e:
            span_start_token, span_end_token = -1, -1
            print(f"  [span-err] {fragment_id}: {e}", flush=True)

        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            gen_out = model.generate(
                input_ids,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=1.0,  # ignored when do_sample=False
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_tokens = gen_out[0, input_ids.shape[1]:]
        raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        parsed = parse_raw_output(raw_output, args.variant)
        record: dict[str, Any] = {
            "fragment_id": fragment_id,
            "variant": args.variant,
            "ruler_gt": str(row.ruler),
            "year_gt": int(row.year) if row.year is not None else None,
            "raw_output": raw_output,
            "parsed_ruler": parsed["parsed_ruler"],
            "parsed_year": parsed["parsed_year"],
            "parsed_confidence": parsed["parsed_confidence"],
            "parse_error": parsed["parse_error"],
            "reasoning_text": parsed["reasoning_text"],
            "span_token_indices": {
                "span_start_token": int(span_start_token),
                "span_end_token": int(span_end_token),
                "prompt_n_tokens": int(input_ids.shape[1]),
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        if (i + 1) % 50 == 0 or i + 1 == n:
            elapsed = (time.time() - t0) / 60
            print(f"  [gen] {i + 1}/{n}  ({elapsed:.1f} min)", flush=True)

    elapsed = (time.time() - t0) / 60
    print(f"[done] {args.variant}  {n} fragments  {elapsed:.1f} min  -> {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1b direct-answer driver")
    p.add_argument("--variant", required=True, choices=["pv0", "pv1", "pv2", "pv3"])
    p.add_argument("--model_path", default=DEFAULT_MODEL,
                   help=f"HF model path or local dir (default env QWEN_MODEL_PATH or {DEFAULT_MODEL})")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_ROOT),
                   help="Output root (writes to {out_dir}/direct_answers/{variant}/)")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--prompts_dir", default=str(DEFAULT_PROMPTS_DIR))
    p.add_argument("--draws_matrix", default=str(DEFAULT_DRAWS_MATRIX),
                   help="Path to Phase 0 draws_matrix.npy (required for pv2)")
    p.add_argument("--fragment_order", default=str(DEFAULT_FRAGMENT_ORDER),
                   help="Path to Phase 0 corpus_fragment_order.json (required for pv2)")
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="Generation budget per fragment (pv3 CoT may need >=512)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-generate even if per-fragment JSON already exists")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
