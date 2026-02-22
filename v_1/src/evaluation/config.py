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
EVAL_DIR = DATA_DIR / "evaluation_corpora"
CACHE_DIR = EVAL_DIR / "cache"

# Source data
SOURCE_PARQUET = EVAL_DIR / "unified_3groups_akkadian_letters.parquet"

# Output files
TEXTS_PARQUET = EVAL_DIR / "texts_for_evaluation.parquet"
TEXTS_JSONL = EVAL_DIR / "texts_for_evaluation.jsonl"
TOKEN_STATS_JSON = EVAL_DIR / "texts_token_stats.json"
PREDICTIONS_PARQUET = EVAL_DIR / "baseline_predictions.parquet"
RESULTS_REPORT_MD = EVAL_DIR / "baseline_results_report.md"
METRICS_JSON = EVAL_DIR / "baseline_metrics.json"

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
PROMPT_TEMPLATE = """
You are an expert in ancient Akkadian cuneiform texts and Assyriology.
Analyze the following transliterated Akkadian text and predict its metadata.

=== TEXT ===
{full_text}
=== END TEXT ===

Predict the following metadata fields. Keep your reasoning to 1-2 sentences.

**Period**: (choose one) Old Babylonian | Neo-Assyrian | Late Babylonian
**Century**: (specific range, e.g. "18th century BCE" or "7th century BCE")
**Domain**: (choose one) Administrative Letter | Political Letter | Private Letter | Diplomatic Letter | Neo-Assyrian Letter | Late Babylonian Letter | Unknown
**Place**: (likely provenance) e.g. Mari, Nineveh, Babylon, Sippar, Assur, Uruk, or Unknown
**Confidence**: high | medium | low
**Reasoning**: (1-2 sentences only)

=== EXAMPLE RESPONSE ===
**Period**: Old Babylonian
**Century**: 18th century BCE
**Domain**: Administrative Letter
**Place**: Mari
**Confidence**: high
**Reasoning**: The text uses typical Old Babylonian epistolary formulae and references Mari administrative officials.
=== END EXAMPLE ===

Respond using EXACTLY the format above. Do NOT add any extra text before or after."""

# Max tokens for LLM response (includes reasoning + answer for thinking models)
# GPT-OSS-20B uses ~1500-2500 reasoning tokens + ~200 answer tokens
MAX_COMPLETION_TOKENS = 4096

# Ground truth column mappings
GROUND_TRUTH_COLUMNS = {
    'period': 'period',
    'temporal_group': 'temporal_group',
    'domain': 'domain_standard',
    'place': 'place_discovery',
}

# Valid period labels
VALID_PERIODS = ['Old Babylonian', 'Neo-Assyrian', 'Late Babylonian']

# Valid domain labels
VALID_DOMAINS = [
    'Administrative Letter',
    'Political Letter',
    'Private Letter',
    'Diplomatic Letter',
    'Unknown',
]
