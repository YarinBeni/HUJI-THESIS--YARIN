#!/usr/bin/env python3
"""
Step 2: LLM Baseline Predictions via OpenRouter

Sends texts to LLMs via OpenRouter API and collects predictions.
Supports caching, resume, rate limiting, and multiple models.

Usage:
    python 02_llm_baseline.py --model gpt-oss-20b               # Full run
    python 02_llm_baseline.py --model gpt-oss-20b --sample 100  # Test run
    python 02_llm_baseline.py --model gpt-oss-20b --resume      # Resume from cache
    python 02_llm_baseline.py --model gpt-oss-20b --dry-run     # Estimate cost only
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ALL_MODELS,
    CACHE_DIR,
    TEXTS_JSONL,
    TOKEN_STATS_JSON,
    PROMPT_TEMPLATE,
    MAX_COMPLETION_TOKENS,
    DEFAULT_RATE_LIMIT_SLEEP,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
)

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_cache_path(model_name: str) -> Path:
    """Get cache file path for a model."""
    safe_name = model_name.replace('/', '_').replace('.', '_')
    return CACHE_DIR / f"{safe_name}.jsonl"


def load_cached_ids(cache_path: Path) -> set:
    """Load already-processed fragment IDs from cache."""
    if not cache_path.exists():
        return set()

    cached_ids = set()
    with open(cache_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                cached_ids.add(record['fragment_id'])
            except (json.JSONDecodeError, KeyError):
                continue
    return cached_ids


def load_texts(jsonl_path: Path, sample: Optional[int] = None) -> list:
    """
    Load texts from JSONL file.

    Args:
        jsonl_path: Path to JSONL file
        sample: If set, only load first N texts

    Returns:
        List of text records
    """
    texts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if sample and i >= sample:
                break
            texts.append(json.loads(line))
    return texts


def create_prompt(text: str) -> str:
    """Create the full prompt for a text."""
    return PROMPT_TEMPLATE.format(full_text=text)


def parse_llm_response(response_text: str) -> dict:
    """
    Parse LLM response in markdown field format.

    Expected format:
        **Reasoning**: Multi-line reasoning about period and century...
        **Period**: Old Babylonian
        **Century**: 1800 BCE
        **Place**: Mari
        **Catalog ID**: ARM 10 33

    Falls back to JSON parsing if markdown fields not found.

    Args:
        response_text: Raw response from LLM

    Returns:
        Parsed prediction dict, or error dict if parsing fails
    """
    import re

    text = response_text.strip()

    # Field patterns: **Label**: value (case-insensitive)
    # Reasoning may span multiple lines, so we capture until the next **Field**
    field_map = {
        'period': r'\*\*Period\*\*\s*:\s*(.+)',
        'century_estimate': r'\*\*Century\*\*\s*:\s*(.+)',
        'place_discovery': r'\*\*Place\*\*\s*:\s*(.+)',
        'catalog_id': r'\*\*Catalog\s*ID\*\*\s*:\s*(.+)',
    }

    result = {}

    # Parse reasoning separately (may span multiple lines)
    reasoning_match = re.search(
        r'\*\*Reasoning\*\*\s*:\s*(.*?)(?=\*\*Period\*\*|\*\*Century\*\*|\*\*Place\*\*|\*\*Catalog|$)',
        text, re.IGNORECASE | re.DOTALL
    )
    if reasoning_match:
        result['reasoning'] = reasoning_match.group(1).strip()

    # Parse single-line fields
    for field, pattern in field_map.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result[field] = match.group(1).strip()

    # If we got at least period, consider it a success
    if 'period' in result:
        all_fields = ['reasoning', 'period', 'century_estimate', 'place_discovery', 'catalog_id']
        for field in all_fields:
            if field not in result:
                result[field] = 'Unknown'
        result['raw_response'] = text[:500]
        return result

    # Fallback: try JSON parsing
    try:
        json_text = text
        if '```json' in json_text:
            start = json_text.find('```json') + 7
            end = json_text.find('```', start)
            json_text = json_text[start:end].strip()
        elif '```' in json_text:
            start = json_text.find('```') + 3
            end = json_text.find('```', start)
            json_text = json_text[start:end].strip()
        if '{' in json_text:
            start = json_text.find('{')
            end = json_text.rfind('}') + 1
            json_text = json_text[start:end]

        parsed = json.loads(json_text)
        for field in ['period', 'place_discovery', 'catalog_id']:
            if field not in parsed:
                parsed[field] = 'Unknown'
        return parsed

    except (json.JSONDecodeError, ValueError):
        pass

    # Nothing worked
    return {
        'reasoning': 'Could not parse response',
        'period': 'Parse Error',
        'century_estimate': 'Parse Error',
        'place_discovery': 'Parse Error',
        'catalog_id': 'Parse Error',
        'raw_response': text[:500],
    }


def call_openrouter(
    api_key: str,
    model_id: str,
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[dict, dict]:
    """
    Call OpenRouter API using requests.

    Args:
        api_key: OpenRouter API key
        model_id: OpenRouter model ID
        prompt: Full prompt text
        timeout: Request timeout in seconds

    Returns:
        Tuple of (prediction_dict, usage_dict)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/akkadian-text-classification",
        "X-Title": "Akkadian Text Classification Research",
    }

    data = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": MAX_COMPLETION_TOKENS,
    }

    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=data,
        timeout=timeout,
    )
    response.raise_for_status()

    result = response.json()

    # Extract content (handle reasoning models where content may be empty)
    message = result['choices'][0]['message']
    content = message.get('content', '') or ''

    # For reasoning models (GPT-OSS, o1, etc.): if content is empty,
    # try to extract the answer from reasoning_details
    if not content.strip() and 'reasoning' in message:
        content = message['reasoning']
    if not content.strip() and 'reasoning_details' in message:
        # reasoning_details is a list of dicts with 'text' field
        parts = [d.get('text', '') for d in message['reasoning_details'] if d.get('text')]
        content = '\n'.join(parts)

    prediction = parse_llm_response(content)

    # Extract usage
    usage_data = result.get('usage', {})
    usage = {
        'prompt_tokens': usage_data.get('prompt_tokens', 0),
        'completion_tokens': usage_data.get('completion_tokens', 0),
        'total_tokens': usage_data.get('total_tokens', 0),
    }

    return prediction, usage


def run_predictions(
    model_name: str,
    sample: Optional[int] = None,
    resume: bool = True,
    dry_run: bool = False,
    rate_limit: float = DEFAULT_RATE_LIMIT_SLEEP,
    max_retries: int = DEFAULT_MAX_RETRIES,
):
    """
    Run predictions for all texts using specified model.

    Args:
        model_name: Model name from config
        sample: Only process first N texts
        resume: Resume from cache (skip already-processed)
        dry_run: Only estimate costs, don't call API
        rate_limit: Sleep between API calls
        max_retries: Max retries per text
    """
    # Validate model
    if model_name not in ALL_MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {list(ALL_MODELS.keys())}")
        sys.exit(1)

    model_id = ALL_MODELS[model_name]
    cache_path = get_cache_path(model_name)

    print("=" * 60)
    print(f"LLM Baseline Predictions: {model_name}")
    print("=" * 60)
    print(f"  Model ID: {model_id}")
    print(f"  Cache: {cache_path}")

    # Load texts
    print(f"\nLoading texts from {TEXTS_JSONL}...")
    texts = load_texts(TEXTS_JSONL, sample=sample)
    print(f"  Loaded {len(texts):,} texts")

    # Load cached IDs
    cached_ids = set()
    if resume:
        cached_ids = load_cached_ids(cache_path)
        print(f"  Found {len(cached_ids):,} cached predictions")

    # Filter to unprocessed texts
    texts_to_process = [t for t in texts if t['fragment_id'] not in cached_ids]
    print(f"  Texts to process: {len(texts_to_process):,}")

    if not texts_to_process:
        print("\nAll texts already processed!")
        return

    # Dry run - estimate costs
    if dry_run:
        print("\n[DRY RUN] Estimating costs...")

        # Load token stats
        if TOKEN_STATS_JSON.exists():
            with open(TOKEN_STATS_JSON, 'r') as f:
                stats = json.load(f)
            avg_tokens = stats['avg_tokens_per_text']
            prompt_tokens = stats['prompt_template_tokens']
        else:
            avg_tokens = 100  # fallback estimate
            prompt_tokens = 500

        # Estimate total tokens
        n_texts = len(texts_to_process)
        est_input_tokens = n_texts * (avg_tokens + prompt_tokens)
        est_output_tokens = n_texts * 150  # ~150 tokens per response

        print(f"\n  Texts to process:     {n_texts:,}")
        print(f"  Est. input tokens:    {est_input_tokens:,}")
        print(f"  Est. output tokens:   {est_output_tokens:,}")
        print(f"  Est. total tokens:    {est_input_tokens + est_output_tokens:,}")

        print("\n  Cost depends on model pricing (check OpenRouter)")
        return

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("\nError: OPENROUTER_API_KEY environment variable not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key'")
        sys.exit(1)

    print(f"  API key: {api_key[:12]}...{api_key[-4:]}")

    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Process texts
    print(f"\nProcessing {len(texts_to_process):,} texts...")

    total_input_tokens = 0
    total_output_tokens = 0
    errors = 0

    with open(cache_path, 'a', encoding='utf-8') as cache_file:
        for text_record in tqdm(texts_to_process, desc=f"[{model_name}]"):
            fragment_id = text_record['fragment_id']
            full_text = text_record['full_text']

            # Create prompt
            prompt = create_prompt(full_text)

            # Call API with retries
            prediction = None
            usage = None

            for attempt in range(max_retries):
                try:
                    prediction, usage = call_openrouter(
                        api_key=api_key,
                        model_id=model_id,
                        prompt=prompt,
                    )
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        print(f"\n  Retry {attempt + 1}/{max_retries} for {fragment_id}: {e}")
                        time.sleep(wait_time)
                    else:
                        print(f"\n  Failed after {max_retries} attempts: {fragment_id}")
                        prediction = {
                            'period': 'API Error',
                            'century_estimate': 'API Error',
                            'place_discovery': 'API Error',
                            'catalog_id': 'API Error',
                            'reasoning': f'API error: {str(e)}',
                        }
                        usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                        errors += 1

            # Track tokens
            if usage:
                total_input_tokens += usage.get('prompt_tokens', 0)
                total_output_tokens += usage.get('completion_tokens', 0)

            # Save to cache
            cache_record = {
                'fragment_id': fragment_id,
                'model': model_name,
                'model_id': model_id,
                'prediction': prediction,
                'usage': usage,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            cache_file.write(json.dumps(cache_record, ensure_ascii=False) + '\n')
            cache_file.flush()

            # Rate limiting
            time.sleep(rate_limit)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Texts processed:      {len(texts_to_process):,}")
    print(f"  Errors:               {errors:,}")
    print(f"  Total input tokens:   {total_input_tokens:,}")
    print(f"  Total output tokens:  {total_output_tokens:,}")
    print(f"  Total tokens:         {total_input_tokens + total_output_tokens:,}")
    print(f"  Cache saved to:       {cache_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM baseline predictions via OpenRouter"
    )
    parser.add_argument(
        '--model', '-m',
        required=True,
        help=f"Model name. Available: {list(ALL_MODELS.keys())}"
    )
    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=None,
        help="Only process first N texts (for testing)"
    )
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        default=True,
        help="Resume from cache (default: True)"
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help="Don't resume from cache, start fresh"
    )
    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help="Estimate costs without calling API"
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=DEFAULT_RATE_LIMIT_SLEEP,
        help=f"Sleep between API calls (default: {DEFAULT_RATE_LIMIT_SLEEP}s)"
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help="List available models and exit"
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        print("\nFree (Phase A):")
        for name, mid in sorted(ALL_MODELS.items())[:1]:
            print(f"  {name}: {mid}")
        print("\nOpen Source (Phase C):")
        for name in ['qwen-2.5-7b', 'qwen-2.5-32b', 'qwen-2.5-72b',
                     'mixtral-8x7b', 'mistral-small', 'mistral-large',
                     'deepseek-chat', 'deepseek-v3', 'deepseek-r1']:
            if name in ALL_MODELS:
                print(f"  {name}: {ALL_MODELS[name]}")
        print("\nPaid (Phase D):")
        for name in ['gemini-2.0-flash', 'gpt-4o', 'gpt-4o-mini', 'sonnet-3.5', 'grok-2']:
            if name in ALL_MODELS:
                print(f"  {name}: {ALL_MODELS[name]}")
        return

    resume = args.resume and not args.no_resume

    run_predictions(
        model_name=args.model,
        sample=args.sample,
        resume=resume,
        dry_run=args.dry_run,
        rate_limit=args.rate_limit,
    )


if __name__ == "__main__":
    main()
