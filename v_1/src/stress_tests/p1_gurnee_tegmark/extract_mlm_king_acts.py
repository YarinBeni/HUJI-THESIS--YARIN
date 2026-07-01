"""J4d — MLM (AeneasForMLM) activation extraction on the balanced-MC setup.

The custom Akkadian MLM is NOT HF-loadable (sign-level tokenizer, bespoke torso),
so it never went through the shared king pipeline. This script brings it onto the
SAME footing as every other model in the balanced-MC ladder:

  {acts_root}/mlm_tier0_mean/     layer_00..16.npz   (mean over all tokens, tier0)
  {acts_root}/mlm_maximal_mean/   layer_00..16.npz   (mean over all tokens, maximal)
  {acts_root}/mlm_tier0_kinglast/ layer_00..16.npz + king_coverage.json  (tier0 only)
  {acts_root}/mlm_tier0_kingmean/ layer_00..16.npz + king_coverage.json

Dirs are named `<method>_<cleaning>_<pool>` so shared/geo_loader.find_acts_dir and
probe_p1_mc pick them up with method="mlm" exactly like the HF models.

King location is SIGN-LEVEL: a ruler spelling `m-aš-šur-PAP-AŠ` is the sign
sequence [m, aš, šur, PAP, AŠ]; the MLM text is `text.replace('-',' ')` split on
whitespace, so we find the earliest contiguous run of those signs and offset by 1
for the prepended [CLS]. king_* remain tier0-only (maximal strips the logograms).

Run from repo root (GPU or CPU; the model is tiny):
    python v_1/src/stress_tests/p1_gurnee_tegmark/extract_mlm_king_acts.py
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
sys.path.insert(0, str(_THIS.parents[1] / "shared"))
sys.path.insert(0, str(_REPO / "v_1/src/archive/baseline_mlm"))

import king_token as kt                                   # noqa: E402
from data_utils import load_vocabulary, tokenize_text     # noqa: E402
from model import AeneasConfig, AeneasForMLM              # noqa: E402

DEFAULT_CORPUS = _REPO / "v_1/data/evaluation/corpora/orcc_corpus.parquet"
DEFAULT_ACTS = _REPO / "v_1/src/linear_probing/results/orcc__embed/activations"
CHECKPOINT = _REPO / "v_1/models/baseline_retrained/baseline_best.pt"
VOCAB = _REPO / "v_1/data/training_ready/vocab.json"
CLS_OFFSET = 1  # tokenize_text prepends [CLS]


def king_sign_span(text_tier0: str, spelling_variants: list[str]):
    """Earliest contiguous run of a ruler's spelling as SIGN tokens.

    Returns inclusive (start_sign, end_sign) into the whitespace-split sign list of
    `text_tier0.replace('-',' ')`, or None. Mirrors find_name_word's earliest-first
    preference but at sign granularity (the MLM's actual tokenization unit)."""
    signs = text_tier0.replace("-", " ").split()
    best = None
    for variant in spelling_variants:
        vs = variant.split("-")
        L = len(vs)
        if L == 0 or L > len(signs):
            continue
        for i in range(len(signs) - L + 1):
            if signs[i:i + L] == vs:
                if best is None or i < best[0]:
                    best = (i, i + L - 1)
                break  # earliest occurrence of THIS variant
    return best


def load_mlm(device: str):
    import torch
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    config = AeneasConfig.from_dict(ckpt["config"])
    model = AeneasForMLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    print(f"  MLM d_model={config.d_model} num_layers={config.num_layers} "
          f"epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss'):.4f}")
    return model, config


def run(args):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(args.corpus)
    sign_to_id, _ = load_vocabulary(str(VOCAB))
    spellings = kt.load_spellings()
    model, config = load_mlm(device)
    all_layers = list(range(config.num_layers + 1))  # 0..16
    max_content = args.max_tokens - 2                 # room for [CLS]/[SEP]

    acts_root = Path(args.acts_root)
    d_mean_t0 = acts_root / "mlm_tier0_mean"
    d_mean_mx = acts_root / "mlm_maximal_mean"
    d_klast = acts_root / "mlm_tier0_kinglast"
    d_kmean = acts_root / "mlm_tier0_kingmean"
    for d in (d_mean_t0, d_mean_mx, d_klast, d_kmean):
        d.mkdir(parents=True, exist_ok=True)

    n = len(df)
    buf = {name: {L: [] for L in all_layers}
           for name in ("mean_t0", "mean_mx", "klast", "kmean")}
    found = np.zeros(n, dtype=bool)
    t0 = time.time()

    def hidden_states(text_signspace: str):
        ids, mask = tokenize_text(text_signspace, sign_to_id, max_length=args.max_tokens)
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        mask_t = torch.tensor([mask], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids_t, mask_t, output_hidden_states=True,
                        hidden_states_layers=all_layers)
        return out["hidden_states"], mask_t[0]

    for i, row in enumerate(df.itertuples(index=False)):
        tier0 = str(row.text_tier0)
        maximal = str(getattr(row, "text_maximal", tier0))
        ruler = getattr(row, "ruler", None)
        sp = spellings.get(ruler, [])

        # ---- tier0 pass: mean + king ----
        hs_t0, mask_t0 = hidden_states(tier0.replace("-", " "))
        span = king_sign_span(tier0, sp) if sp else None
        if span is not None:
            s_tok, e_tok = span[0] + CLS_OFFSET, span[1] + CLS_OFFSET
            if e_tok >= min(len(mask_t0), max_content + CLS_OFFSET):
                span = None  # name fell outside the truncation window
        found[i] = span is not None
        for L in all_layers:
            h = hs_t0[L][0]                                   # (T, d_model)
            m = mask_t0.float().unsqueeze(-1)
            mean_vec = (h * m).sum(0) / m.sum(0).clamp(min=1)
            buf["mean_t0"][L].append(mean_vec.cpu().float().numpy())
            if span is not None:
                s_tok, e_tok = span[0] + CLS_OFFSET, span[1] + CLS_OFFSET
                buf["klast"][L].append(h[e_tok].cpu().float().numpy())
                buf["kmean"][L].append(h[s_tok:e_tok + 1].mean(0).cpu().float().numpy())
            else:
                nan = np.full(config.d_model, np.nan, dtype=np.float32)
                buf["klast"][L].append(nan)
                buf["kmean"][L].append(nan)

        # ---- maximal pass: mean only ----
        hs_mx, mask_mx = hidden_states(maximal.replace("-", " "))
        for L in all_layers:
            h = hs_mx[L][0]
            m = mask_mx.float().unsqueeze(-1)
            mean_vec = (h * m).sum(0) / m.sum(0).clamp(min=1)
            buf["mean_mx"][L].append(mean_vec.cpu().float().numpy())

        if i % 200 == 0:
            print(f"[mlm] {i}/{n} king_found={found[:i+1].sum()} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    for L in all_layers:
        np.savez_compressed(d_mean_t0 / f"layer_{L:02d}.npz",
                            activations=np.vstack(buf["mean_t0"][L]).astype(np.float32))
        np.savez_compressed(d_mean_mx / f"layer_{L:02d}.npz",
                            activations=np.vstack(buf["mean_mx"][L]).astype(np.float32))
        np.savez_compressed(d_klast / f"layer_{L:02d}.npz",
                            activations=np.vstack(buf["klast"][L]).astype(np.float32))
        np.savez_compressed(d_kmean / f"layer_{L:02d}.npz",
                            activations=np.vstack(buf["kmean"][L]).astype(np.float32))

    cov = dict(
        fragment_ids=df["fragment_id"].astype(str).tolist(),
        years=[None if pd.isna(v) else int(v) for v in df["year"]],
        rulers=df["ruler"].astype(str).tolist(),
        found=found.astype(int).tolist(),
    )
    meta_common = {"method": "mlm", "model_id": "AeneasForMLM",
                   "checkpoint": str(CHECKPOINT), "n": int(n),
                   "n_layers": len(all_layers), "hidden_dim": int(config.d_model)}
    for d, pool in ((d_klast, "kinglast"), (d_kmean, "kingmean")):
        (d / "king_coverage.json").write_text(json.dumps(cov), encoding="utf-8")
        (d / "metadata.json").write_text(json.dumps(
            {**meta_common, "cleaning": "tier0", "pool": pool,
             "n_found": int(found.sum()),
             "coverage": round(float(found.mean()), 3)}, indent=2), encoding="utf-8")
    for d, clean in ((d_mean_t0, "tier0"), (d_mean_mx, "maximal")):
        (d / "metadata.json").write_text(json.dumps(
            {**meta_common, "cleaning": clean, "pool": "mean"}, indent=2), encoding="utf-8")

    print(f"[mlm] DONE king_coverage={found.mean():.3f} ({found.sum()}/{n}) "
          f"layers={len(all_layers)}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--acts-root", default=str(DEFAULT_ACTS))
    p.add_argument("--max-tokens", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
