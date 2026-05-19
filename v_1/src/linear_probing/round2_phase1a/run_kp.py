"""
Phase 1a — Knowledge-Probing Inference Driver (W2.B).

Runs Qwen2.5-7B-Instruct on the approved kp0/kp1/kp2 prompt variants WITHOUT
any Akkadian fragment context. The goal is to find out whether Qwen knows the
8 Phase-0 rulers (kp0), can list rulers by period (kp1), and correctly declines
on plausible-but-fake names (kp2).

Reuses the model-loading idiom from:
  v_1/src/linear_probing/03_extract_seal_activations.py:44-58

CLI:
  python run_kp.py --variant {kp0|kp1|kp2} --model_path <path> --out_dir <path>

Default model path resolution order:
  1. --model_path arg if provided
  2. env var QWEN_MODEL_PATH if set
  3. fallback to HF hub ID 'Qwen/Qwen2.5-7B-Instruct'

Output layout:
  {out_dir}/raw/{variant}_{input_idx:02d}.json   — one file per input
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Repo-root resolution: this file is at v_1/src/linear_probing/round2_phase1a/run_kp.py
_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[3]
PROMPTS_DIR = REPO_ROOT / 'v_1' / 'src' / 'linear_probing' / 'results' / 'orcc_round2_phase1a' / 'prompts'
RULER_REIGNS_PATH = REPO_ROOT / 'v_1' / 'src' / 'linear_probing' / 'results' / 'orcc_round2_phase1a' / 'ruler_reigns.json'

DEFAULT_HF_ID = 'Qwen/Qwen2.5-7B-Instruct'

NB_PERIODS = ['Neo-Assyrian', 'Neo-Babylonian']


def parse_prompt_md(md_path: Path) -> dict:
    """Parse YAML-frontmatter + section-headed markdown prompt file.

    Returns dict with keys: meta (frontmatter as raw text), system, user_template,
    fake_names (kp2 only), parse_instructions.
    """
    text = md_path.read_text(encoding='utf-8')

    # Strip frontmatter
    fm_match = re.match(r'^---\n(.*?)\n---\n(.*)$', text, re.DOTALL)
    if not fm_match:
        raise ValueError(f"{md_path}: no YAML frontmatter found")
    frontmatter_raw = fm_match.group(1)
    body = fm_match.group(2)

    # Section regex — capture from "## NAME" to next "## " or EOF
    def extract_section(name: str) -> str:
        pattern = rf'## {re.escape(name)}\s*\n(.*?)(?=\n## |\Z)'
        m = re.search(pattern, body, re.DOTALL)
        if not m:
            return ''
        return m.group(1).strip()

    system = extract_section('System prompt')
    user_template = extract_section('User prompt template')
    parse_instructions = extract_section('Parse instructions')
    fake_names_section = extract_section('Fake ruler names (eval inputs for kp2)')

    # Fake names: parse the numbered list "1. Name"
    fake_names = []
    if fake_names_section:
        for line in fake_names_section.splitlines():
            m = re.match(r'^\s*\d+\.\s+(.+?)\s*$', line)
            if m:
                fake_names.append(m.group(1))

    if not system or not user_template:
        raise ValueError(f"{md_path}: missing required section(s) (system/user)")

    return {
        'frontmatter': frontmatter_raw,
        'system': system,
        'user_template': user_template,
        'parse_instructions': parse_instructions,
        'fake_names': fake_names,
    }


def get_eval_inputs(variant: str, prompt_data: dict) -> list[tuple[str, dict]]:
    """Returns list of (placeholder_value, fill_dict) tuples."""
    if variant == 'kp0':
        with open(RULER_REIGNS_PATH, 'r', encoding='utf-8') as f:
            reigns = json.load(f)
        rulers = [k for k in reigns.keys() if not k.startswith('_')]
        assert len(rulers) == 8, f"Expected 8 Phase-0 rulers, got {len(rulers)}"
        return [(r, {'ruler': r}) for r in rulers]
    elif variant == 'kp1':
        return [(p, {'period': p}) for p in NB_PERIODS]
    elif variant == 'kp2':
        names = prompt_data['fake_names']
        assert len(names) == 8, f"Expected 8 fake names in kp2.md, got {len(names)}"
        return [(n, {'ruler': n}) for n in names]
    else:
        raise ValueError(f"Unknown variant: {variant}")


def fill_template(template: str, fill: dict) -> str:
    out = template
    for k, v in fill.items():
        out = out.replace('{' + k + '}', v)
    return out


def resolve_model_path(arg: str | None) -> str:
    if arg:
        return arg
    env = os.environ.get('QWEN_MODEL_PATH')
    if env:
        return env
    return DEFAULT_HF_ID


def run_inference(variant: str, model_path: str, out_dir: Path, max_new_tokens: int = 512):
    md_path = PROMPTS_DIR / f'{variant}.md'
    if not md_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {md_path}")
    prompt_data = parse_prompt_md(md_path)
    eval_inputs = get_eval_inputs(variant, prompt_data)

    raw_dir = out_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import so smoke tests that only exercise parsing logic don't need torch.
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[run_kp] Loading model: {model_path}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"[run_kp] Model loaded on {device} in {time.time()-t0:.1f}s", flush=True)

    system_prompt = prompt_data['system']
    user_template = prompt_data['user_template']

    for idx, (placeholder_value, fill) in enumerate(eval_inputs):
        user_msg = fill_template(user_template, fill)
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_msg},
        ]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Slice off the prompt
        generated = output_ids[0, inputs['input_ids'].shape[1]:]
        raw_output = tokenizer.decode(generated, skip_special_tokens=True)

        record = {
            'variant': variant,
            'input_idx': idx,
            'input_value': placeholder_value,
            'fill': fill,
            'system_prompt': system_prompt,
            'user_message': user_msg,
            'raw_output': raw_output,
            'model_path': model_path,
            'max_new_tokens': max_new_tokens,
            'timestamp': datetime.now().isoformat(),
        }
        out_path = raw_dir / f'{variant}_{idx:02d}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"[run_kp] {variant}[{idx}] '{placeholder_value}' -> {out_path}", flush=True)

    print(f"[run_kp] Done: {len(eval_inputs)} raw outputs in {raw_dir}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description='Phase 1a knowledge-probe inference driver')
    p.add_argument('--variant', required=True, choices=['kp0', 'kp1', 'kp2'])
    p.add_argument('--model_path', default=None,
                   help='Local path or HF id for Qwen 2.5-7B-Instruct. '
                        'Falls back to env $QWEN_MODEL_PATH, then HF hub id.')
    p.add_argument('--out_dir', required=True, help='Output directory root for this variant run')
    p.add_argument('--max_new_tokens', type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model_path)
    out_dir = Path(args.out_dir)
    run_inference(args.variant, model_path, out_dir, max_new_tokens=args.max_new_tokens)


if __name__ == '__main__':
    main()
