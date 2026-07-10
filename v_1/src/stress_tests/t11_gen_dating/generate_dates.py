"""T11 — generated-answer dating (behavioral counterpart of T10).

T10 read the ACTIVATIONS under dating prompts and probed them linearly; this
experiment reads the model's actual ANSWER. Each fragment is shown to a chat
LLM which must output a JSON year estimate; score_gen_dating.py then computes
the same balanced-MC Spearman used for the activation probes, so behavior and
linear-recoverability land on one comparable axis:
  * probe > answer  -> linearly-recoverable signal the model can't verbalize
  * answer > probe  -> the model "knows" in a way a linear year-probe misses
  * both ~ random   -> strengthens "no timeline" overall.

Inputs per --cleaning: tier0 / maximal (names stripped) / maxking (maximal
context, king name kept) from the corpus; engtier0 = the committed Thalesian
English translation (translation/translations.parquet). Fragment text is capped
at --max-frag-words (default 300, matching the T10 gpt-oss extraction) for ALL
models so every model x cleaning sees identical inputs.

Output: raw/{model}__{cleaning}.jsonl (gitignored; one record per fragment with
the raw generation). Scoring/parsing lives in score_gen_dating.py.

Usage:
  python generate_dates.py --model qwen3_8b --model_path Qwen/Qwen3-8B \
      --cleaning tier0 [--batch-size 8]
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

import king_token as kt                        # noqa: E402
from cleaning import clean_maximal_keepking    # noqa: E402

DEFAULT_CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
DEFAULT_TRANSLATIONS = _THIS.parents[1] / "translation/translations.parquet"

CLEANINGS = ["tier0", "maximal", "maxking", "engtier0"]

SYSTEM_PROMPT = (
    "You are an expert Assyriologist specializing in Mesopotamian royal "
    "inscriptions. You will be shown one fragment of an ancient Akkadian royal "
    "inscription, either in transliteration or in English translation. Estimate "
    "the year it was composed. Respond with a single JSON object and nothing "
    'else: {"year_bce": <positive integer, years BCE>, "basis": "<at most 10 '
    'words: what you dated it by>"}. If you cannot estimate a year, respond '
    '{"year_bce": null, "basis": "cannot estimate"}.'
)

USER_TEMPLATE = (
    "Fragment ({notation}):\n\n{text}\n\nJSON answer:"
)


def fragment_texts(df: pd.DataFrame, cleaning: str, translations_path: Path):
    """Aligned list of texts (corpus order) + the notation label for the prompt."""
    if cleaning == "tier0":
        return df["text_tier0"].astype(str).tolist(), "Akkadian transliteration"
    if cleaning == "maximal":
        return df["text_maximal"].astype(str).tolist(), "Akkadian transliteration"
    if cleaning == "maxking":
        spellings = kt.load_spellings()
        out = []
        for row in df.itertuples(index=False):
            sp = spellings.get(getattr(row, "ruler", None), [])
            out.append(clean_maximal_keepking(str(row.text_tier0), sp)[0])
        return out, "Akkadian transliteration"
    if cleaning == "engtier0":
        tr = pd.read_parquet(translations_path).set_index("fragment_id")
        eng = tr["eng_tier0"].reindex(df["fragment_id"].astype(str)).fillna("")
        return eng.astype(str).tolist(), "English translation"
    raise ValueError(cleaning)


def build_chat_texts(tok, texts, notation, max_frag_words):
    """Rendered chat prompts (thinking disabled / low effort where supported)."""
    out = []
    for t in texts:
        t = t.strip() or "..."
        words = t.split()
        if max_frag_words and len(words) > max_frag_words:
            t = " ".join(words[:max_frag_words])   # keep the opening titulary
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": USER_TEMPLATE.format(notation=notation, text=t)}]
        try:
            # extra kwargs land in the jinja context; each template uses its own
            # (Qwen3: enable_thinking=False skips <think>; gpt-oss harmony:
            # reasoning_effort shortens the analysis channel) and ignores the rest
            s = tok.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=False,
                                        reasoning_effort="low")
        except (TypeError, ValueError):
            s = tok.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True)
        out.append(s)
    return out


def run(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    out_dir = Path(args.out_dir) / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}__{args.cleaning}.jsonl"

    df = pd.read_parquet(args.corpus)
    texts, notation = fragment_texts(df, args.cleaning, Path(args.translations))
    prompts = None   # built after tokenizer loads

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # batched greedy generation
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="sdpa")
    except Exception as e:  # noqa: BLE001  (gpt-oss: no sdpa)
        print(f"[load] sdpa failed ({type(e).__name__}); default attention", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    prompts = build_chat_texts(tok, texts, notation, args.max_frag_words)

    # resume support: skip fragments already generated (re-queued jobs)
    done = set()
    if out_path.exists() and not args.overwrite:
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["fragment_id"] for line in f if line.strip()}
        print(f"[resume] {len(done)} fragments already in {out_path.name}", flush=True)

    fids = df["fragment_id"].astype(str).tolist()
    rulers = df["ruler"].astype(str).tolist()
    years = [None if pd.isna(y) else float(y) for y in df["year"]]
    todo = [i for i in range(len(df)) if fids[i] not in done]

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fout:
        for b in range(0, len(todo), args.batch_size):
            idx = todo[b:b + args.batch_size]
            enc = tok([prompts[i] for i in idx], return_tensors="pt",
                      padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=False,
                                     max_new_tokens=args.max_new_tokens,
                                     pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx):
                raw = tok.decode(gen[j, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
                fout.write(json.dumps({
                    "fragment_id": fids[i], "ruler": rulers[i], "year": years[i],
                    "cleaning": args.cleaning, "model": args.model,
                    "raw_output": raw}, ensure_ascii=False) + "\n")
            fout.flush()
            if (b // args.batch_size) % 10 == 0:
                d = b + len(idx)
                print(f"[{args.model} x {args.cleaning}] {d}/{len(todo)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[{args.model} x {args.cleaning}] DONE {len(todo)} generated "
          f"-> {out_path} ({time.time()-t0:.0f}s)", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="short name used in filenames")
    p.add_argument("--model_path", required=True, help="HF id or local path")
    p.add_argument("--cleaning", required=True, choices=CLEANINGS)
    p.add_argument("--max-frag-words", type=int, default=300,
                   help="word cap for ALL models (identical inputs; T10 parity)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate even if the output JSONL already has rows")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--translations", default=str(DEFAULT_TRANSLATIONS))
    p.add_argument("--out_dir", default=str(_THIS.parent))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
