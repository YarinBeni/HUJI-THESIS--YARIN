# Phase 2: LLM Baseline - Final Model Selection

**Date**: January 28, 2026
**Strategy**: Aligned tiers (Small, Mid, Large/SOTA) across Open-Source and Commercial providers

---

## Selection Strategy

Test **3 tiers** per provider family to compare performance vs. cost:
- **Small (7B-14B)**: Fast, cheap, baseline performance
- **Mid (30B-70B)**: Balanced performance/cost
- **Large/SOTA (70B+)**: Best performance, highest cost

---

## Open Source Models (via OpenRouter)

### Family 1: Qwen (Alibaba)
| Tier | Model ID | Params | Input $/M | Output $/M | Context | Est. Cost |
|------|----------|--------|-----------|------------|---------|-----------|
| **Small** | `qwen/qwen-2.5-7b-instruct` | 7B | $0.04 | $0.10 | 32K | $0.20 |
| **Mid** | `qwen/qwen3-32b` | 32B | $0.08 | $0.24 | 40K | $0.43 |
| **Large** | `qwen/qwen-2.5-72b-instruct` | 72B | $0.12 | $0.39 | 32K | $0.66 |

### Family 2: Mistral (Mistral AI)
| Tier | Model ID | Params | Input $/M | Output $/M | Context | Est. Cost |
|------|----------|--------|-----------|------------|---------|-----------|
| **Small** | `mistralai/mistral-nemo` | 12B | $0.02 | $0.04 | 131K | $0.09 |
| **Mid** | `mistralai/mixtral-8x7b-instruct` | 47B (8x7B MoE) | $0.54 | $0.54 | 32K | $2.07 |
| **Large** | `mistralai/mistral-large-2512` | ~250B | $0.50 | $1.50 | 262K | $2.67 |

### Family 3: DeepSeek (DeepSeek AI)
| Tier | Model ID | Params | Input $/M | Output $/M | Context | Est. Cost |
|------|----------|--------|-----------|------------|---------|-----------|
| **Small** | `deepseek/deepseek-chat-v3.1` | ~14B | $0.15 | $0.75 | 32K | $1.02 |
| **Mid** | `deepseek/deepseek-v3.2` | ~671B (MoE) | $0.25 | $0.38 | 163K | $1.06 |
| **Large** | `deepseek/deepseek-r1-distill-llama-70b` | 70B | $0.03 | $0.11 | 131K | $1.19* |

*Note: R1-distill is a reasoning model, uses ~10M output tokens instead of 0.75M

### Family 4: Llama (Meta) - BONUS
| Tier | Model ID | Params | Input $/M | Output $/M | Context | Est. Cost |
|------|----------|--------|-----------|------------|---------|-----------|
| **Small** | `meta-llama/llama-3.1-8b-instruct` | 8B | $0.02 | $0.05 | 16K | $0.10 |
| **Mid** | `meta-llama/llama-3.3-70b-instruct` | 70B | $0.10 | $0.32 | 131K | $0.55 |
| **Large** | `nousresearch/hermes-3-llama-3.1-405b` | 405B | $1.00 | $1.00 | 131K | $3.83 |

**Open Source Subtotal**: ~$14 (all 12 models)

---

## Commercial Models (Paid APIs)

### Family 1: OpenAI GPT
| Tier | Model ID | Input $/M | Output $/M | Context | Est. Cost |
|------|----------|-----------|------------|---------|-----------|
| **Small/Fast** | `openai/gpt-4o-mini` | $0.15 | $0.60 | 128K | $0.91 |
| **Mid** | `openai/gpt-4o` | $2.50 | $10.00 | 128K | $15.20 |
| **Large/SOTA** | `openai/gpt-5.1` | $1.25 | $10.00 | 400K | $11.36 |

### Family 2: Anthropic Claude
| Tier | Model ID | Input $/M | Output $/M | Context | Est. Cost |
|------|----------|-----------|------------|---------|-----------|
| **Small/Fast** | `anthropic/claude-3-haiku` | $0.25 | $1.25 | 200K | $1.71 |
| **Mid** | `anthropic/claude-sonnet-4.5` | $3.00 | $15.00 | 1000K | $20.50 |
| **Large/SOTA** | `anthropic/claude-opus-4.6` | $5.00 | $25.00 | 1000K | $34.17 |

### Family 3: Google Gemini
| Tier | Model ID | Input $/M | Output $/M | Context | Est. Cost |
|------|----------|-----------|------------|---------|-----------|
| **Small/Fast** | `google/gemini-2.0-flash-001` | $0.10 | $0.40 | 1048K | $0.61 |
| **Mid** | `google/gemini-3-flash-preview` | $0.50 | $3.00 | 1048K | $3.79 |
| **Large/SOTA** | `google/gemini-3-pro-preview` | $2.00 | $12.00 | 1048K | $15.16 |

### Family 4: xAI Grok
| Tier | Model ID | Input $/M | Output $/M | Context | Est. Cost |
|------|----------|-----------|------------|---------|-----------|
| **Small/Fast** | `x-ai/grok-4-fast` | $0.20 | $0.50 | 2000K | $0.99 |
| **Mid** | `x-ai/grok-3-mini` | $0.30 | $0.50 | 131K | $1.30 |
| **Large/SOTA** | `x-ai/grok-4` | $3.00 | $15.00 | 256K | $20.50 |

**Commercial Subtotal**: ~$126 (all 12 models)

---

## Free Tier Models (Phase A - Testing)

| Model ID | Family | Params | Context | Notes |
|----------|--------|--------|---------|-------|
| `openai/gpt-oss-20b:free` | GPT OSS | 21B | 131K | Reasoning model (already tested) |
| `openai/gpt-oss-120b:free` | GPT OSS | 117B | 131K | Reasoning model |
| `deepseek/deepseek-r1-0528:free` | DeepSeek | ~671B | 163K | Reasoning model |
| `mistralai/mistral-small-3.1-24b-instruct:free` | Mistral | 24B | 128K | Standard model |
| `meta-llama/llama-3.3-70b-instruct:free` | Llama | 70B | 128K | Standard model |
| `qwen/qwen3-next-80b-a3b-instruct:free` | Qwen | 80B | 262K | Standard model |

**Free Tier Subtotal**: $0 (6 models)

---

## Final Recommended Configuration

### Phase A: Free Tier (6 models) - $0
Test pipeline with free models, refine prompt with mentors.

### Phase B: Open Source Small+Mid (8 models) - ~$6
- Qwen: 7B, 32B
- Mistral: 12B (Nemo), Mixtral-8x7B
- DeepSeek: v3.1, v3.2
- Llama: 8B, 70B

### Phase C: Open Source Large (4 models) - ~$8
- Qwen: 72B
- Mistral: Large-2512
- DeepSeek: R1-distill-70B
- Llama: Hermes-405B

### Phase D: Commercial All Tiers (12 models) - ~$126
- All GPT, Claude, Gemini, Grok models (Small, Mid, Large)

---

## Cost Summary

| Phase | Models | Total Cost |
|-------|--------|------------|
| A: Free | 6 | $0 |
| B: Open Small+Mid | 8 | $6 |
| C: Open Large | 4 | $8 |
| D: Commercial | 12 | $126 |
| **TOTAL** | **30 models** | **~$140** |

**With 20% buffer for retries/prompt iterations**: **~$170**

---

## Key Decisions

### ✅ Why This Selection?

1. **Aligned Tiers**: Every family has Small/Mid/Large variants
2. **All Models Verified**: Every model exists on OpenRouter and is accessible
3. **Cost-Effective Progression**: Test cheap models first, scale to expensive ones
4. **Diverse Architectures**:
   - Dense models (Llama, GPT, Claude)
   - MoE models (Mixtral, DeepSeek, Qwen)
   - Reasoning models (R1, GPT-OSS)

### ✅ Replacements Made

**Original Plan Issues**:
- ❌ "Mistral-7B" → doesn't exist, replaced with **Mistral-Nemo (12B)** - closest small Mistral
- ❌ "DeepSeek-7B" → doesn't exist, replaced with **DeepSeek-v3.1 (14B)** - smallest DeepSeek chat model
- ❌ "GPT-5.2-Pro" → replaced with **GPT-5.1** - more widely available, similar performance
- ❌ "Gemini 1.5 Pro-002" → replaced with **Gemini-3-Pro-Preview** - newer, better model
- ❌ "Claude 3 Opus" → replaced with **Claude-Opus-4.6** - latest Opus version
- ❌ "Grok 4 Heavy" → doesn't exist, used **Grok-4** (regular) - SOTA Grok model

### ✅ Bonus Additions

- **Llama family**: Added as 4th open-source family (free 70B model available!)
- **Free tier expansion**: Found 6 high-quality free models including GPT-OSS-120B

---

## Execution Plan

```bash
# Phase A: Free models (refine prompt)
python 02_llm_baseline.py --model gpt-oss-20b --sample 100
python 02_llm_baseline.py --model gpt-oss-120b --sample 100
python 02_llm_baseline.py --model deepseek-r1-0528-free --sample 100
python 02_llm_baseline.py --model mistral-small-3.1-free --sample 100
python 02_llm_baseline.py --model llama-3.3-70b-free --sample 100
python 02_llm_baseline.py --model qwen3-next-80b-free --sample 100

# >> REVIEW WITH MENTORS, FINALIZE PROMPT <<

# Phase B: Open-source Small+Mid (full corpus ~5K texts)
python 02_llm_baseline.py --model qwen-2.5-7b
python 02_llm_baseline.py --model qwen3-32b
python 02_llm_baseline.py --model mistral-nemo
python 02_llm_baseline.py --model mixtral-8x7b
python 02_llm_baseline.py --model deepseek-chat-v3.1
python 02_llm_baseline.py --model deepseek-v3.2
python 02_llm_baseline.py --model llama-3.1-8b
python 02_llm_baseline.py --model llama-3.3-70b

# Phase C: Open-source Large
python 02_llm_baseline.py --model qwen-2.5-72b
python 02_llm_baseline.py --model mistral-large-2512
python 02_llm_baseline.py --model deepseek-r1-distill-llama-70b
python 02_llm_baseline.py --model hermes-3-llama-405b

# Phase D: Commercial
# Small tier
python 02_llm_baseline.py --model gpt-4o-mini
python 02_llm_baseline.py --model claude-3-haiku
python 02_llm_baseline.py --model gemini-2.0-flash
python 02_llm_baseline.py --model grok-4-fast

# Mid tier
python 02_llm_baseline.py --model gpt-4o
python 02_llm_baseline.py --model claude-sonnet-4.5
python 02_llm_baseline.py --model gemini-3-flash
python 02_llm_baseline.py --model grok-3-mini

# Large/SOTA tier
python 02_llm_baseline.py --model gpt-5.1
python 02_llm_baseline.py --model claude-opus-4.6
python 02_llm_baseline.py --model gemini-3-pro
python 02_llm_baseline.py --model grok-4
```

---

## Next Steps

1. **Update `config.py`** with these exact model IDs
2. **Run Phase A** (free tier, 6 models × 100 samples each)
3. **Analyze results**, refine prompt if needed
4. **Get mentor approval** on prompt and approach
5. **Run Phase B+C** (open-source, 12 models × 5K texts)
6. **Request budget** for Phase D (~$130)
7. **Run Phase D** (commercial, 12 models × 5K texts)
8. **Compare all 30 models** in final evaluation report

---

**Total Investment**: ~$170 (with buffer)
**Total Models**: 30
**Total Predictions**: 150,000 (30 models × 5,005 texts)

**Per-model cost range**: $0 (free) to $34 (Opus-4.6)
**Median cost per model**: ~$1.50
