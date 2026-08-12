"""F29 — mid-layer-faithful lens: tuned-lens translators for ALL directions.

THE REVIEWER OBJECTION THIS ANSWERS. The raw logit lens (F6/F21) is only
faithful near the final layers, but several of our directions live mid-stack
(the E1 document directions at L9-L25, the document-year regression axes).
"Junk under the raw lens" is therefore not yet evidence of meaninglessness.

THE FIX, following the tuned lens (Belrose et al. 2023, arXiv 2303.08112):
train, per (model, layer), an affine translator A_L that maps the layer-L
residual to the final-layer representation by minimizing
KL(final logits || lens logits) on held-in text. A DIRECTION d then reads out
as W_U(gamma (.) A_L d_hat) — the translator bias cancels for directions.
Translators are trained on our own corpus (English glosses + Akkadian
transliterations + entity name prompts), no external downloads, and cached.

Directions read (whatever exists on disk is picked up):
  cellA        historical_figure ridge   (probe_wm npz, its best layer)
  cellB        assyrian_ruler ridge      (cached by e3_transfer --entity-set)
  docreg_eng / docreg_akk  ridge on FRAGMENT YEAR, fitted here at the
               pairs-probes best layer on all fragments — a READING
               instrument only, never an evaluation probe
  bt_eng / bt_akk          E1 pairwise directions (std -> raw via sd)

Per direction, both the raw lens and the tuned lens are reported: top/bottom
token ends + the F21 spectroscopy (decile composition, z vs N_NULL random
directions pushed through the SAME translator).

    python lens_tuned.py --method olmo2_7b
    python lens_tuned.py --method olmo2_7b --steps 0    # reuse cached A_L

Writes results/tuned.{method}.json; translators cached under
results/translators/{method}.layer{L}.npz.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_PAIRS = os.path.abspath(os.path.join(_HERE, "..", "pairs"))
sys.path.insert(0, _PAIRS)
_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
sys.path.insert(0, _WM)

from lens_spectroscopy import B, CATS, N_NULL, classify, spectrum  # noqa: E402
from logit_lens import load_unembed                                # noqa: E402
import pairs_data as P                                             # noqa: E402

DIRS_A = os.path.join(_WM, "results", "directions")
DIRS_PAIR = os.path.join(_PAIRS, "results", "directions")
AKK_ACTS = os.path.join(_WM, "akkadian", "activations")
RESULTS = os.path.join(_HERE, "results")
TR_DIR = os.path.join(RESULTS, "translators")
RIDGE_ALPHAS = np.logspace(-1, 5, 13)


def frag_acts(method, variant, L, df):
    p = os.path.join(AKK_ACTS, method, variant, f"mean.layer{L}.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["acts"].astype(np.float32)[df.pos.values]


def collect_directions(method, df):
    """{name: (vector_raw_coords, layer)} for everything on disk."""
    dirs = {}
    for ent, tag in (("historical_figure", "cellA"),
                     ("assyrian_ruler", "cellB")):
        g = sorted(glob.glob(os.path.join(DIRS_A, method,
                                          f"{ent}.*.layer*.npz")))
        if g:
            L = int(re.search(r"layer(\d+)\.npz$", g[0]).group(1))
            dirs[tag] = (np.load(g[0])["coef"].astype(np.float32).ravel(), L)
    for variant, tag in (("eng_tier0", "eng"), ("akk_maximal", "akk")):
        # E1 pairwise direction, standardized -> raw
        for p in sorted(glob.glob(os.path.join(
                DIRS_PAIR, f"{method}.{variant}.mean.layer*.npz"))):
            L = int(re.search(r"layer(\d+)\.npz$", p).group(1))
            X = frag_acts(method, variant, L, df)
            if X is not None:
                w = np.load(p)["w"].astype(np.float32) / (X.std(0) + 1e-8)
                dirs[f"bt_{tag}"] = (w, L)
                break
        # document-year regression direction, fitted here (reading only)
        pj = os.path.join(_PAIRS, "results", "probes",
                          f"{method}.{variant}.mean.json")
        if os.path.exists(pj):
            L = json.load(open(pj))["best_layer"]
            X = frag_acts(method, variant, L, df)
            if X is not None:
                from sklearn.linear_model import RidgeCV
                y = df.year.values.astype(float)
                r = RidgeCV(alphas=RIDGE_ALPHAS).fit(X, (y - y.mean())
                                                     / (y.std() + 1e-9))
                dirs[f"docreg_{tag}"] = (r.coef_.astype(np.float32).ravel(), L)
    return dirs


def build_corpus(df):
    texts = (df.eng_tier0.fillna("").astype(str).tolist()
             + df.text_maximal.fillna("").astype(str).tolist())
    ent_csv = os.path.join(_WM, "data", "entity_datasets",
                           "historical_figure.csv")
    if os.path.exists(ent_csv):
        import pandas as pd
        ent = pd.read_csv(ent_csv)
        col = "name" if "name" in ent else ent.columns[0]
        texts += [f"{n} was a historical figure."
                  for n in ent[col].astype(str).tolist()[:4000]]
    return [t for t in texts if len(t.split()) >= 3]


def train_translators(method, layers, steps, seed=0):
    """One affine A_L per requested layer, trained jointly in single forward
    passes; KL(final || lens) on 64 sampled positions per sequence."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from wm_lib import registry

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hfid = registry.MODELS[method]["hfid"]
    tok = AutoTokenizer.from_pretrained(hfid)
    model = AutoModelForCausalLM.from_pretrained(
        hfid, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(dev)
    model.eval()
    d = model.config.hidden_size
    gamma = model.model.norm.weight.detach().float().to(dev)
    W_U = model.get_output_embeddings().weight.detach().float().to(dev)

    os.makedirs(TR_DIR, exist_ok=True)
    A, opt_params, todo = {}, [], []
    for L in layers:
        cache = os.path.join(TR_DIR, f"{method}.layer{L}.npz")
        if os.path.exists(cache) or steps == 0:
            if os.path.exists(cache):
                A[L] = torch.from_numpy(np.load(cache)["A"]).float().to(dev)
            continue
        M = torch.eye(d, device=dev, requires_grad=True)
        A[L] = M
        opt_params.append(M)
        todo.append(L)
    if not todo:
        del model
        torch.cuda.empty_cache()
        return {L: a.detach().cpu().numpy() if hasattr(a, "detach") else a
                for L, a in A.items()}

    df = P.load_eligible()
    texts = build_corpus(df)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(opt_params, lr=1e-4)
    print(f"[train] {method}: layers {todo}, {len(texts)} docs, "
          f"{steps} steps on {dev}", flush=True)
    for step in range(steps):
        batch = [texts[i] for i in rng.integers(0, len(texts), 4)]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=384).to(dev)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        mask = enc.attention_mask.bool()
        flat = mask.view(-1)
        idx = torch.nonzero(flat).squeeze(1)
        if len(idx) > 256:
            idx = idx[torch.randperm(len(idx), device=dev)[:256]]
        teacher = out.logits.float().view(-1, out.logits.shape[-1])[idx]
        t_logp = F.log_softmax(teacher, dim=-1)
        loss = 0.0
        for L in todo:
            h = out.hidden_states[L].float().view(-1, d)[idx]
            hh = h @ A[L].T
            hh = hh / (hh.norm(dim=-1, keepdim=True)
                       / np.sqrt(d) + 1e-6)          # RMS, then gamma
            s_logits = (gamma * hh) @ W_U.T
            s_logp = F.log_softmax(s_logits, dim=-1)
            loss = loss + F.kl_div(s_logp, t_logp, log_target=True,
                                   reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  step {step}: KL {float(loss):.4f}", flush=True)
    for L in todo:
        np.savez_compressed(os.path.join(TR_DIR, f"{method}.layer{L}.npz"),
                            A=A[L].detach().cpu().float().numpy())
    del model
    torch.cuda.empty_cache()
    return {L: (a.detach().cpu().numpy() if hasattr(a, "detach") else a)
            for L, a in A.items()}


def spectro(scores_fn, vecs, cats, u_norms, seed=0):
    """Composition + z for a direction under a scoring function, against
    N_NULL random directions pushed through the SAME scoring function."""
    out = {}
    rng = np.random.default_rng(seed)
    for name, v in vecs:
        raw = scores_fn(v)
        cosv = raw / u_norms
        rec = {}
        for tag, sc in (("raw", raw), ("cos", cosv)):
            comp, _ = spectrum(sc, cats, np.zeros(len(sc), bool),
                               np.zeros(len(sc)))
            rec[tag] = comp
        out[name] = rec
    # shared null
    dim = len(vecs[0][1])
    nulls = {"raw": [], "cos": []}
    for _ in range(N_NULL):
        r = rng.standard_normal(dim).astype(np.float32)
        sr = scores_fn(r)
        for tag, sc in (("raw", sr), ("cos", sr / u_norms)):
            comp, _ = spectrum(sc, cats, np.zeros(len(sc), bool),
                               np.zeros(len(sc)))
            nulls[tag].append(comp)
    res = {}
    for name in out:
        res[name] = {}
        for tag in ("raw", "cos"):
            nl = np.stack(nulls[tag])
            mu, sd = nl.mean(0), nl.std(0) + 1e-9
            z = (out[name][tag] - mu) / sd
            res[name][tag] = {"composition": out[name][tag].tolist(),
                              "z_scores": z.tolist()}
    return res


def token_ends(scores, tokens, k=25):
    order = np.argsort(scores)
    return {"negative_end": [{"token": tokens[i], "score": float(scores[i])}
                             for i in order[:k]],
            "positive_end": [{"token": tokens[i], "score": float(scores[i])}
                             for i in order[::-1][:k]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--steps", type=int, default=1200,
                    help="translator training steps (0 = cached only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = P.load_eligible()
    dirs = collect_directions(args.method, df)
    if not dirs:
        sys.exit("no directions found on disk")
    layers = sorted({L for _, L in dirs.values()})
    print(f"[dirs] {[(k, L) for k, (_, L) in dirs.items()]}", flush=True)

    A = train_translators(args.method, layers, args.steps, args.seed)

    tok, W_U, gamma = load_unembed(args.method)
    tokens = tok.convert_ids_to_tokens(list(range(W_U.shape[0])))
    tokens = ["" if t is None else t for t in tokens]
    cats = np.array([CATS.index(classify(t)) for t in tokens])
    u_norms = np.linalg.norm(W_U, axis=1) + 1e-8

    out = {"method": args.method, "buckets": B, "n_null": N_NULL,
           "cats": CATS, "directions": {}}
    for name, (v, L) in dirs.items():
        vh = v / (np.linalg.norm(v) + 1e-8)
        variants = {"rawlens": vh}
        if L in A:
            tv = A[L] @ vh
            variants["tuned"] = tv / (np.linalg.norm(tv) + 1e-8)
        rec = {"layer": int(L), "has_translator": L in A}
        score = lambda u: W_U @ (gamma * (u / (np.linalg.norm(u) + 1e-8)))
        specs = spectro(score, list(variants.items()), cats, u_norms,
                        args.seed)
        for vt, u in variants.items():
            rec[vt] = {"ends": token_ends(score(u), tokens),
                       "spectroscopy": specs[vt]}
        out["directions"][name] = rec
        for vt in variants:
            ci = CATS.index("temporal_ancient")
            z1 = np.array(rec[vt]["spectroscopy"]["cos"]["z_scores"])[0, ci]
            print(f"  [{name}/{vt}] L{L} decile-1 ancient z={z1:+.2f}",
                  flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    pth = os.path.join(RESULTS, f"tuned.{args.method}.json")
    with open(pth, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[done] -> {pth}", flush=True)


if __name__ == "__main__":
    main()
