"""
Configuration for LLM Baseline Evaluation Pipeline.
Model registry, API settings, and prompt templates.
"""
import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
EVAL_DIR = DATA_DIR / "evaluation"
CORPORA_DIR = EVAL_DIR / "corpora"
BASELINES_DIR = EVAL_DIR / "baselines"
CACHE_DIR = BASELINES_DIR / "cache"

# Source data
SOURCE_PARQUET = CORPORA_DIR / "unified_3groups_akkadian_letters.parquet"

# Corpus output files (written by 01_create_corpus.py / 02_prepare_texts.py)
TEXTS_PARQUET = CORPORA_DIR / "texts_for_evaluation.parquet"
TEXTS_JSONL = CORPORA_DIR / "texts_for_evaluation.jsonl"
TOKEN_STATS_JSON = CORPORA_DIR / "texts_token_stats.json"

# Baseline output files (written by 04_aggregate_results.py / 05_evaluate_baseline.py)
PREDICTIONS_PARQUET = BASELINES_DIR / "baseline_predictions.parquet"
RESULTS_REPORT_MD = BASELINES_DIR / "baseline_results_report.md"
METRICS_JSON = BASELINES_DIR / "baseline_metrics.json"

# =============================================================================
# Model Registry
# =============================================================================

# Phase A — Free models for pipeline validation (no API cost)
MODELS_FREE = {
    # OpenAI GPT-OSS (open-weight, Apache 2.0 license)
    'gpt-oss-20b': 'openai/gpt-oss-20b:free',      # 21B params, 3.6B active per pass
    'gpt-oss-120b': 'openai/gpt-oss-120b:free',    # 117B params, 5.1B active per pass
    # Meta Llama 4 free tier
    'llama-4-maverick': 'meta-llama/llama-4-maverick:free',
    'llama-4-scout': 'meta-llama/llama-4-scout:free',
    # Google Gemini free tier
    'gemini-2.5-pro-free': 'google/gemini-2.5-pro-exp-03-25:free',
    # Mistral free tier
    'mistral-small-free': 'mistralai/mistral-small-3.1-24b-instruct:free',
    # DeepSeek free tier
    'deepseek-v3-free': 'deepseek/deepseek-chat-v3-0324:free',
    'deepseek-r1-free': 'deepseek/deepseek-r1-zero:free',
}

# Phase C — Open-source models (paid via OpenRouter)
MODELS_OPEN_SOURCE = {
    # Qwen family
    'qwen-2.5-7b': 'qwen/qwen-2.5-7b-instruct',
    'qwen-2.5-32b': 'qwen/qwen-2.5-32b-instruct',
    'qwen-2.5-72b': 'qwen/qwen-2.5-72b-instruct',
    # Mixtral/Mistral family
    'mixtral-8x7b': 'mistralai/mixtral-8x7b-instruct',
    'mistral-small': 'mistralai/mistral-small',
    'mistral-large': 'mistralai/mistral-large',
    # DeepSeek family
    'deepseek-chat': 'deepseek/deepseek-chat',
    'deepseek-v3': 'deepseek/deepseek-chat-v3',
    'deepseek-r1': 'deepseek/deepseek-r1',
}

# Phase D — Paid commercial models
MODELS_PAID = {
    'gemini-2.0-flash': 'google/gemini-2.0-flash-001',
    'gpt-4o': 'openai/gpt-4o',
    'gpt-4o-mini': 'openai/gpt-4o-mini',
    'sonnet-3.5': 'anthropic/claude-3.5-sonnet',
    'grok-2': 'x-ai/grok-2',
}

# Combined registry
ALL_MODELS = {**MODELS_FREE, **MODELS_OPEN_SOURCE, **MODELS_PAID}

# =============================================================================
# API Settings
# =============================================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_RATE_LIMIT_SLEEP = 0.5  # seconds between API calls
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 60  # seconds

# =============================================================================
# Prompt Template
# =============================================================================
PROMPT_TEMPLATE = """You are an expert in ancient Akkadian cuneiform texts and Assyriology.
Analyze the following transliterated Akkadian text and predict its metadata.

=== TEXT ===
{full_text}
=== END TEXT ===

First reason about the text, then provide your predictions.

**Reasoning**: (up to 3 lines) Explain why you assigned this period and century.
**Period**: (e.g. Old Babylonian, Middle Babylonian, Neo-Babylonian, Late Babylonian, Old Assyrian, Middle Assyrian, Neo-Assyrian, Old Akkadian, etc.)
**Century**: (100-year estimate, e.g. "1700 BCE" meaning approximately 1700-1600 BCE)
**Place**: (likely provenance, e.g. Mari, Nineveh, Babylon, Sippar, Assur, Uruk, Nimrud, Larsa)
**Catalog ID**: (publication number or CDLI P-number, e.g. ARM 10 33, P224378, AO 8957)

=== EXAMPLE 1 ===
**Reasoning**: The greeting formula "a-na ... qi2-bi2-ma / um-ma ... -ma" is characteristic of Old Babylonian epistolary conventions. The mention of Hammurabi and administrative land allocation points to the 18th century BCE Larsa region.
**Period**: Old Babylonian
**Century**: 1800 BCE
**Place**: Larsa
**Catalog ID**: AbB 11 166
=== END EXAMPLE 1 ===

=== EXAMPLE 2 ===
**Reasoning**: The phrase "a-bat LUGAL" (word of the king) and the administrative tone with date formulae using month names (iti-GAN) are typical Neo-Assyrian royal correspondence. The mention of BAD3-MAN-GIN (Dur-Sharrukin) situates this in the Sargonid period.
**Period**: Neo-Assyrian
**Century**: 700 BCE
**Place**: Nimrud (Kalhu)
**Catalog ID**: P224403
=== END EXAMPLE 2 ===

=== EXAMPLE 3 ===
**Reasoning**: The text uses Late Babylonian orthographic conventions and the blessing formula invoking Bel and Nabu (d-EN u d-AG) is characteristic of Late Babylonian private correspondence from the Uruk region.
**Period**: Late Babylonian
**Century**: 500 BCE
**Place**: Uruk
**Catalog ID**: AOAT 25 46
=== END EXAMPLE 3 ===

Respond using EXACTLY the format above. Do NOT add any extra text before or after."""

# Max tokens for LLM response (includes reasoning + answer for thinking models)
# GPT-OSS-20B uses ~1500-2500 reasoning tokens + ~200 answer tokens
MAX_COMPLETION_TOKENS = 4096

# Ground truth column mappings
GROUND_TRUTH_COLUMNS = {
    'period': 'period',
    'temporal_group': 'temporal_group',
    'place': 'place_discovery',
    'catalog_id': 'fragment_id',
}

# Valid period labels (ground truth in our corpus — models may predict others)
VALID_PERIODS = ['Old Babylonian', 'Neo-Assyrian', 'Late Babylonian']
