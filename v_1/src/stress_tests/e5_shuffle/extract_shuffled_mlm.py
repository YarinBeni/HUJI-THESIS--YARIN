"""E5 word-shuffle extraction for the MLM (AeneasForMLM) — the one model the
HF-based extract_shuffled_acts.py cannot load (sign-level tokenizer, bespoke
torso). Reuses that script's twin builder (word-cap FIRST, then a seed-42
permutation — content identical, order differs) and the J4d MLM loader.

Akkadian cleanings only (the sign vocabulary cannot tokenize English).
Writes mlm_{unshuf|shuf}{cleaning}_mean/layer_LL.npz so probe_e5_mc scores it
with --method mlm, exactly like the HF models.

Usage:  python extract_shuffled_mlm.py [--cleanings tier0,maximal]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parents[1] / "shared"))
sys.path.insert(0, str(_REPO / "v_1/src/archive/baseline_mlm"))

from extract_shuffled_acts import build_variants, SEED                  # noqa: E402
from data_utils import load_vocabulary, tokenize_text                   # noqa: E402
from model import AeneasConfig, AeneasForMLM                            # noqa: E402

CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
ACTS = _REPO / "v_1/src/linear_probing/results/orcc__embed/activations"
CHECKPOINT = _REPO / "v_1/models/baseline_retrained/baseline_best.pt"
VOCAB = _REPO / "v_1/data/training_ready/vocab.json"
TRANSLATIONS = _THIS.parents[1] / "translation/translations.parquet"


def load_mlm(device: str):
    import torch
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    config = AeneasConfig.from_dict(ckpt["config"])
    model = AeneasForMLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, config


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--cleanings", default="tier0,maximal")
    p.add_argument("--max-frag-words", type=int, default=120,
                   help="cap BEFORE shuffling (encoder convention, 512-token window)")
    p.add_argument("--max-tokens", type=int, default=512)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(CORPUS)
    sign_to_id, _ = load_vocabulary(str(VOCAB))
    model, config = load_mlm(device)
    layers = list(range(config.num_layers + 1))

    def mean_acts(texts):
        buf = {L: [] for L in layers}
        for t in texts:
            ids, mask = tokenize_text(t.replace("-", " "), sign_to_id,
                                      max_length=a.max_tokens)
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)
            mask_t = torch.tensor([mask], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(ids_t, mask_t, output_hidden_states=True,
                            hidden_states_layers=layers)
            m = mask_t[0].float().unsqueeze(-1)
            for L in layers:
                h = out["hidden_states"][L][0]
                buf[L].append(((h * m).sum(0) / m.sum(0).clamp(min=1))
                              .cpu().float().numpy())
        return buf

    for cleaning in a.cleanings.split(","):
        assert cleaning in ("tier0", "maximal"), f"{cleaning}: Akkadian only for the MLM"
        unshuf, shuf = build_variants(df, cleaning, TRANSLATIONS, a.max_frag_words)
        for var, texts in (("unshuf", unshuf), ("shuf", shuf)):
            t0 = time.time()
            outdir = ACTS / f"mlm_{var}{cleaning}_mean"
            outdir.mkdir(parents=True, exist_ok=True)
            buf = mean_acts(texts)
            for L in layers:
                np.savez_compressed(outdir / f"layer_{L:02d}.npz",
                                    activations=np.stack(buf[L]))
            (outdir / "metadata.json").write_text(json.dumps({
                "experiment": "e5_word_shuffle", "method": "mlm", "seed": SEED,
                "variant": var, "cleaning": cleaning,
                "max_frag_words": a.max_frag_words, "max_tokens": a.max_tokens,
                "n_fragments": len(texts), "layers": layers}, indent=2))
            print(f"[mlm {var}{cleaning}] {len(texts)} frags x {len(layers)} layers "
                  f"-> {outdir.name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
