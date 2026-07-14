"""T12 — FORCED-answer dating across the T10 prompt variants (pv0-pv3).

T11 let the model answer {"year_bce": null} ("cannot estimate") and the small
models took that exit for 87-100% of Akkadian fragments. T12 removes the exit:
every variant's null-escape sentence is replaced by a forced-commitment
instruction, so we observe the model's BEST GUESS under each prompting style
(bare / expert-framed / few-shot / chain-of-thought) — can it answer correctly
even where the signal is not linearly decodable from activations?

Prompt templates = the T10/phase-1b pv0-pv3 markdown specs, reused verbatim
via pv_parse.parse_prompt_md; pv2's five in-context examples come from the
same leakage-free non-eval-ruler pool (run_pv.select_fewshot_examples, seed
42). For cleaning=engtier0 the fragment text (and the pv2 example texts) are
the committed Thalesian English translations and "in transliteration" wording
becomes "in English translation".

Output: raw/{model}__{cleaning}__{pv}.jsonl (gitignored). Scoring lives in
score_forced.py.

Usage:
  python generate_forced.py --model qwen3_8b --model_path Qwen/Qwen3-8B \
      --cleaning tier0 --variant pv1 [--batch-size 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]
sys.path.insert(0, str(_THIS.parents[1] / "shared"))
sys.path.insert(0, str(_THIS.parents[1] / "t11_gen_dating"))
sys.path.insert(0, str(_REPO / "v_1/src/linear_probing/round2_phase1b"))

from generate_dates import fragment_texts, DEFAULT_CORPUS, DEFAULT_TRANSLATIONS  # noqa: E402
from pv_parse import parse_prompt_md                                             # noqa: E402
from run_pv import select_fewshot_examples, fill_fewshot_template               # noqa: E402

PROMPTS_DIR = (_REPO / "v_1/src/linear_probing/results/orcc_round2_phase1b/prompts")
BAL = _REPO / "v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset"
VARIANTS = ["pv0", "pv1", "pv2", "pv3"]
CLEANINGS = ["tier0", "maximal", "maxking", "engtier0", "engmaximal"]

NULL_ESCAPE = ("If you cannot determine the ruler or year, use null for that "
               "field.")
FORCED = ("You must commit to your single best estimate for BOTH fields even "
          "if uncertain — never output null; a specific best guess is required.")

PV0_TAIL = "Who wrote this and when?"
PV0_FORCED_TAIL = (
    "Who wrote this and when? Commit to your single best guess — do not say "
    "you cannot tell. End your answer with a JSON object on its own line: "
    '{"ruler": "<name>", "year_bce": <positive integer, years BCE>}')


def load_forced_template(variant: str, english: bool):
    spec = parse_prompt_md(str(PROMPTS_DIR / f"{variant}.md"))
    user, system = spec["user_template"], spec["system_prompt"]
    if variant == "pv0":
        assert PV0_TAIL in user, "pv0 tail drifted"
        user = user.replace(PV0_TAIL, PV0_FORCED_TAIL)
    else:
        assert NULL_ESCAPE in user, f"{variant}: null-escape sentence drifted"
        user = user.replace(NULL_ESCAPE, FORCED)
    if english:
        repl = [("Akkadian royal inscription in transliteration",
                 "Mesopotamian royal inscription in English translation"),
                ("transliterated cuneiform texts in standard romanized notation",
                 "English translations of cuneiform royal inscriptions")]
        for a, b in repl:
            user = user.replace(a, b)
            if system:
                system = system.replace(a, b)
    return system, user


def engify_examples(examples, translations_path, col="eng_tier0"):
    tr = pd.read_parquet(translations_path).set_index("fragment_id")[col]
    out = []
    for ex in examples:
        e = dict(ex)
        eng = tr.get(str(ex["fragment_id"]))
        if isinstance(eng, str) and eng.strip():
            words = eng.split()
            e["text"] = " ".join(words[:150])
        out.append(e)
    return out


def run(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    out_dir = Path(args.out_dir) / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}__{args.cleaning}__{args.variant}.jsonl"

    df = pd.read_parquet(args.corpus)
    texts, _ = fragment_texts(df, args.cleaning, Path(args.translations))
    english = args.cleaning in ("engtier0", "engmaximal")
    system, user_tpl = load_forced_template(args.variant, english)

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    fewshot = None
    if args.variant == "pv2":
        fewshot = select_fewshot_examples(
            df, BAL / "draws_matrix.npy", BAL / "corpus_fragment_order.json",
            tok, n_examples=5, truncate_tokens=150, seed=42)
        if english:
            col = "eng_tier0" if args.cleaning == "engtier0" else "eng_maximal"
            fewshot = engify_examples(fewshot, args.translations, col)
        user_tpl = fill_fewshot_template(user_tpl, fewshot)

    def render(t):
        t = t.strip() or "..."
        words = t.split()
        if args.max_frag_words and len(words) > args.max_frag_words:
            t = " ".join(words[:args.max_frag_words])
        user = user_tpl.replace("{{fragment_text}}", t)
        msgs = ([{"role": "user", "content": user}] if args.variant == "pv0"
                else [{"role": "system", "content": system or ""},
                      {"role": "user", "content": user}])
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False,
                                           reasoning_effort="low")
        except (TypeError, ValueError):
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="sdpa")
    except Exception as e:  # noqa: BLE001  (gpt-oss: no sdpa)
        print(f"[load] sdpa failed ({type(e).__name__}); default attention", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    prompts = [render(t) for t in texts]
    done = set()
    if out_path.exists() and not args.overwrite:
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["fragment_id"] for line in f if line.strip()}
        print(f"[resume] {len(done)} fragments already in {out_path.name}", flush=True)

    fids = df["fragment_id"].astype(str).tolist()
    rulers = df["ruler"].astype(str).tolist()
    years = [None if pd.isna(y) else float(y) for y in df["year"]]
    todo = [i for i in range(len(df)) if fids[i] not in done]

    max_new = args.max_new_tokens or (768 if args.variant == "pv3" else 512)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fout:
        for b in range(0, len(todo), args.batch_size):
            idx = todo[b:b + args.batch_size]
            import torch
            enc = tok([prompts[i] for i in idx], return_tensors="pt",
                      padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=False,
                                     max_new_tokens=max_new,
                                     pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx):
                raw = tok.decode(gen[j, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
                fout.write(json.dumps({
                    "fragment_id": fids[i], "ruler": rulers[i], "year": years[i],
                    "cleaning": args.cleaning, "model": args.model,
                    "variant": args.variant, "forced": True,
                    "raw_output": raw}, ensure_ascii=False) + "\n")
            fout.flush()
            if (b // args.batch_size) % 10 == 0:
                print(f"[{args.model} x {args.cleaning} x {args.variant}] "
                      f"{b + len(idx)}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{args.model} x {args.cleaning} x {args.variant}] DONE "
          f"{len(todo)} -> {out_path} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--cleaning", required=True, choices=CLEANINGS)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--max-frag-words", type=int, default=300)
    p.add_argument("--max-new-tokens", type=int, default=0,
                   help="0 = per-variant default (512; 768 for pv3)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--translations", default=str(DEFAULT_TRANSLATIONS))
    p.add_argument("--out_dir", default=str(_THIS.parent))
    run(p.parse_args())
