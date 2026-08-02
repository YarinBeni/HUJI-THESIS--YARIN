"""Model registry for the W (world-models replication) section.

Method names follow the thesis ladder conventions (J12 etc.); `random` is the
random-init Qwen3-8B control with the same name it has everywhere else in the repo.

`fallback_hfid` handles the gated meta-llama repos: the NousResearch mirrors host the
same fp16 weights ungated. Override the primary org with WM_LLAMA_ORG (e.g. set it to
NousResearch to skip the gated attempt entirely).

`layer_stride`: keep every stride-th transformer layer (embedding layer always
skipped) to bound activation storage for the deep models.
"""
import os

ARCH_CAUSAL = "causal"
ARCH_ENCODER = "encoder"

# Default to the ungated NousResearch mirrors (identical Llama-2 weights): the
# meta-llama repos are gated and fail with "Cannot access gated repo" unless the
# HF_TOKEN account has been granted access. Set WM_LLAMA_ORG=meta-llama to use the
# official repos when you have access.
_LLAMA_ORG = os.environ.get("WM_LLAMA_ORG", "NousResearch")

# transformers>=5 cannot convert Llama-2's SentencePiece tokenizer.model (it forces
# the tiktoken loader, which crashes on "Error parsing line b'\x0e'") on ANY path
# (fast/slow/LlamaTokenizerFast). Load the Llama tokenizer from a repo that already
# ships a prebuilt tokenizer.json instead: the standard ungated Llama tokenizer,
# same 32k SentencePiece vocab as Llama-2 (token ids match the weights), loads with
# zero conversion. Override with WM_LLAMA_TOK.
_LLAMA_TOK = os.environ.get("WM_LLAMA_TOK", "hf-internal-testing/llama-tokenizer")

# Cluster-local dir for materialized random checkpoints (llama2_70b_random, built by W0).
WM_MODELS_DIR = os.environ.get(
    "WM_MODELS_DIR", os.path.expanduser("~/projects/wm_models"))


def _m(hfid, arch, *, random=False, fallback=None, stride=1, gpus=1,
       sites=None, random_of=None, tokenizer=None):
    return {
        "hfid": hfid,
        "arch": arch,
        "random": random,            # build from config (seed 42) instead of loading weights
        "fallback_hfid": fallback,   # tried when the primary load fails (gated repo etc.)
        "tokenizer_hfid": tokenizer,  # load the tokenizer from here instead of the model dir
        "layer_stride": stride,
        "gpus": gpus,                # documentation only; sbatch files own the real values
        # default pooling sites: paper-faithful `last` for causal, both for encoders
        # (encoders have no causal last-token summary, mean is the thesis-canonical pool)
        "sites": sites or (["last", "mean"] if arch == ARCH_ENCODER else ["last"]),
        "random_of": random_of,      # method whose config/tokenizer the random arm mirrors
    }


MODELS = {
    # ---- thesis ladder -------------------------------------------------------
    "qwen3_1b7":           _m("Qwen/Qwen3-1.7B", ARCH_CAUSAL),
    "qwen3_8b":            _m("Qwen/Qwen3-8B", ARCH_CAUSAL),
    "qwen3_32b":           _m("Qwen/Qwen3-32B", ARCH_CAUSAL, stride=2),
    "gpt_oss_120b":        _m("openai/gpt-oss-120b", ARCH_CAUSAL, gpus=8),
    "thalesian_akk300m":   _m("Thalesian/AKK_300m", ARCH_ENCODER),
    "thalesian_cunei400m": _m("Thalesian/cuneiformBase-400m", ARCH_ENCODER),
    "umt5_base":           _m("google/umt5-base", ARCH_ENCODER),
    "random":              _m("Qwen/Qwen3-8B", ARCH_CAUSAL, random=True,
                              random_of="qwen3_8b"),
    # ---- Gurnee & Tegmark's models, trained ---------------------------------
    "llama2_7b":  _m(f"{_LLAMA_ORG}/Llama-2-7b-hf", ARCH_CAUSAL,
                     fallback="NousResearch/Llama-2-7b-hf", tokenizer=_LLAMA_TOK),
    "llama2_13b": _m(f"{_LLAMA_ORG}/Llama-2-13b-hf", ARCH_CAUSAL,
                     fallback="NousResearch/Llama-2-13b-hf", tokenizer=_LLAMA_TOK),
    "llama2_70b": _m(f"{_LLAMA_ORG}/Llama-2-70b-hf", ARCH_CAUSAL,
                     fallback="NousResearch/Llama-2-70b-hf", stride=2, gpus=4,
                     tokenizer=_LLAMA_TOK),
    # ---- Gurnee & Tegmark's models, random-init (the control they never ran) -
    "llama2_7b_random":  _m(f"{_LLAMA_ORG}/Llama-2-7b-hf", ARCH_CAUSAL, random=True,
                            fallback="NousResearch/Llama-2-7b-hf",
                            tokenizer=_LLAMA_TOK, random_of="llama2_7b"),
    "llama2_13b_random": _m(f"{_LLAMA_ORG}/Llama-2-13b-hf", ARCH_CAUSAL, random=True,
                            fallback="NousResearch/Llama-2-13b-hf",
                            tokenizer=_LLAMA_TOK, random_of="llama2_13b"),
    # ---- OLMo: the one arm whose TRAINING CORPUS is public ------------------
    # Every other arm here can be probed but not audited — we cannot count how often
    # an entity appeared in Llama's or Qwen's training data, because that data is not
    # published. OLMo 2 ships open weights AND the open olmo-mix/Dolma corpus, which
    # is what makes the frequency dose-response experiment possible at all
    # (v_1/src/olmo_frequency/). Needs transformers >= 4.47.
    "olmo2_7b":        _m("allenai/OLMo-2-1124-7B", ARCH_CAUSAL),
    "olmo2_7b_random": _m("allenai/OLMo-2-1124-7B", ARCH_CAUSAL, random=True,
                          random_of="olmo2_7b"),
    # ---- debug arm (not in the ladder; excluded from aggregation tables) ----
    "pythia_70m_test": _m("EleutherAI/pythia-70m", ARCH_CAUSAL),
    # 70B random is materialized once by build_random_llama.py (W0) because
    # from_config needs a 137GB CPU-RAM spike; extraction then loads the saved
    # checkpoint like any other model (random=False here on purpose).
    "llama2_70b_random": _m(os.path.join(WM_MODELS_DIR, "llama2_70b_random"),
                            ARCH_CAUSAL,
                            fallback=None, stride=2, gpus=4,
                            tokenizer=_LLAMA_TOK, random_of="llama2_70b"),
}

RANDOM_SEED = 42
