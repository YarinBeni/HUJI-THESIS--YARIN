"""F26 — igniting the anchor inside documents (the last unrun experiment).

THE QUESTION. The program's causal picture so far: the entity-time machinery
exists (F8/F22), is fed by onomastic features (F25), and amplifying those
features at an ENTITY PROMPT moves the frozen year read-out (F23) — but
mid-text firings in documents never reach the document-level read-out (F23
bridge). What was never tested: can the anchor be LIT deliberately inside a
DOCUMENT context — at the position where the document mentions its king?
That is the original "steering in cell C" plan (skipped after F12's
direction-steering null), now re-justified because FEATURE-level
interventions do work (F23 amplify).

DESIGN (pre-registered before the run):
  * Arm FEAT — clamp each of the top-5 onomastic features (F22 hunt) at
    alpha * act95 on the intervention span; controls = firing-rate-matched
    |rho|<.05 features (F23's discipline). Hook at the SAE layer.
  * Arm DIR — add alpha_rel * ||h_pos|| * w_hat (the frozen cell-A ridge
    direction, unit-normalized) on the span, at blocks {10, 20, 28};
    control = a fixed random unit direction, same norms (F12's discipline).
  * Spans:  eng_tier0 fragments that literally contain their ruler's name
    (~46%) -> the NAME token span only (offset-mapped);
            akk fragments -> ALL positions except the last token (no name
    mapping exists in transliteration; all-but-last upper-bounds any
    span-specific effect and leaves the read-out position untouched).
  * Read-outs at the LAST token: (a) frozen cell-A ridge probe score,
    (b) does the clamped feature itself fire (FEAT arm).

DECISION RULES (verbatim):
  1. eng name-span moves the probe beyond the control band while akk stays
     flat -> "the anchor lights in the gloss language; in Akkadian there is
     nothing to light".
  2. everything flat -> the final causal seal: the anchor cannot be ignited
     from document context at all, even where F23 proved the lever works.
  3. akk moves beyond control -> first evidence of a latent Akkadian anchor
     (would be the surprise of the program; report with the control math).
A null WITH controls is publishable (F12 caution, verbatim) — no
retry-until-significant.

    python ignite_anchor.py            # both arms, both languages

Writes steering/results/ignite.json. GPU.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_SAE2 = os.path.abspath(os.path.join(_HERE, "..", "sae2"))
sys.path.insert(0, _SAE2)
import karvonen as K                                     # noqa: E402
_SAE1 = os.path.abspath(os.path.join(_HERE, "..", "sae"))
sys.path.insert(0, _SAE1)
from fvu_gate import ENT_ACTS, METHOD, load_layer_acts   # noqa: E402
_PAIRS = os.path.abspath(os.path.join(_HERE, "..", "pairs"))
sys.path.insert(0, _PAIRS)
import pairs_data as P                                   # noqa: E402
_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
sys.path.insert(0, _WM)
from wm_lib import registry                              # noqa: E402

DIRS_A = os.path.join(_WM, "results", "directions")
ENT_CSV = os.path.join(_WM, "data", "entity_datasets", "historical_figure.csv")
RESULTS = os.path.join(_HERE, "results")

FEAT_ALPHAS = [0, 4, 8]
DIR_ALPHAS = [0, 2, 4]
# blocks with >=5 layers between the hook and the layer-29 read-out; a
# block-28 hook could not causally reach the last token of hs[29] at all
DIR_BLOCKS = [8, 16, 24]


def name_variants(ruler):
    """Strings to search for in the English gloss (royal-name span)."""
    base = str(ruler)
    outs = {base}
    outs.add(base.split("(")[0].strip())
    outs.add(re.sub(r"\s+[IVX]+$", "", base).strip())     # drop ordinal
    return sorted({o for o in outs if len(o) >= 4}, key=len, reverse=True)


def find_span(text, ruler):
    """(char_start, char_end) of the first ruler-name occurrence, or None."""
    low = text.lower()
    for v in name_variants(ruler):
        i = low.find(v.lower())
        if i >= 0:
            return i, i + len(v)
    return None


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frags", type=int, default=400)
    ap.add_argument("--n-feats", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    # ---- instrument + features (same as F23) -----------------------------
    pipe = json.load(open(os.path.join(_SAE2, "results", "pipeline.json")))
    L, off = pipe["step0"]["layer_used"], pipe["step0"]["offset"]
    sae = K.load(pipe["step0"]["repo"], pipe["step0"]["file_used"])
    tab = pd.read_csv(sorted(glob.glob(os.path.join(
        _SAE2, "results", "feature_hunt2.layer*.csv")))[-1])

    from scipy import stats as sstats
    ent_full = pd.read_csv(ENT_CSV)
    Xa = load_layer_acts(os.path.join(ENT_ACTS, METHOD, "historical_figure"),
                         L + off)
    Za = K.encode(Xa, sae).numpy()
    yr = ent_full["death_year"].values.astype(float)
    okm = ent_full["is_test"].astype(bool).values & np.isfinite(yr)
    fire_all = (Za[okm] > 0).mean(0)
    cand = np.where(fire_all >= 0.02)[0]
    rho_all = np.array([sstats.spearmanr(Za[okm, f], yr[okm]).correlation
                        for f in cand])
    pool = pd.DataFrame({"feature": cand, "fire_cellA": fire_all[cand],
                         "rho_year": rho_all})
    t = tab.reindex(tab.rho_year.abs().sort_values(ascending=False).index)
    treat = t.head(args.n_feats)
    pl = pool[pool.rho_year.abs() < 0.05].copy()
    ctrl_rows = []
    for _, r in treat.iterrows():
        if len(pl) == 0:
            break
        j = (pl.fire_cellA - r.fire_cellA).abs().idxmin()
        ctrl_rows.append(pl.loc[j])
        pl = pl.drop(j)
    ctrl = pd.DataFrame(ctrl_rows)
    if len(ctrl) == 0:
        sys.exit("no rate-matched controls")
    feats_treat = treat.feature.astype(int).tolist()
    feats_ctrl = ctrl.feature.astype(int).tolist()
    scale = {int(f): max(float(np.quantile(Za[:, int(f)][Za[:, int(f)] > 0],
                                           .95))
                         if (Za[:, int(f)] > 0).any() else 1.0, 1e-3)
             for f in feats_treat + feats_ctrl}
    print(f"[feats] treat={feats_treat} ctrl={feats_ctrl}", flush=True)

    # frozen cell-A ridge read-out + unit direction for the DIR arm
    g = sorted(glob.glob(os.path.join(DIRS_A, METHOD,
                                      "historical_figure.*.layer*.npz")))
    coef = np.load(g[0])["coef"].astype(np.float32).ravel()
    LA = int(re.search(r"layer(\d+)\.npz$", g[0]).group(1))
    w_hat = coef / (np.linalg.norm(coef) + 1e-8)
    rng = np.random.default_rng(0)
    r_hat = rng.standard_normal(w_hat.shape[0]).astype(np.float32)
    r_hat /= np.linalg.norm(r_hat) + 1e-8

    hfid = registry.MODELS[METHOD]["hfid"]
    tok = AutoTokenizer.from_pretrained(hfid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        hfid, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    dev = model.device
    coef_t = torch.from_numpy(coef).to(dev)
    w_t = torch.from_numpy(w_hat).to(dev, torch.bfloat16)
    r_t = torch.from_numpy(r_hat).to(dev, torch.bfloat16)
    W_enc, b_enc = sae["W_enc"].to(dev), sae["b_enc"].to(dev)
    theta = sae["theta"].to(dev) if sae["theta"] is not None else None

    # ---- texts + spans ---------------------------------------------------
    df = P.load_eligible()
    cand_e, cand_s = [], []
    for _, r in df.iterrows():
        txt = str(r.text_eng_tier0)
        if not txt.strip():
            continue
        sp = find_span(txt, r.ruler)
        if sp:
            cand_e.append(txt)
            cand_s.append(sp)
    sel = np.random.default_rng(1).choice(
        len(cand_e), min(args.n_frags, len(cand_e)), replace=False)
    eng = [cand_e[i] for i in sel]
    eng_spans = [cand_s[i] for i in sel]
    akk_all = [t for t in df.text_akk.fillna("").astype(str) if t.strip()]
    ka = np.random.default_rng(2).choice(
        len(akk_all), min(args.n_frags, len(akk_all)), replace=False)
    akk = [akk_all[i] for i in ka]
    print(f"[texts] eng-with-name-span={len(eng)}/{len(cand_e)} "
          f"akk={len(akk)}", flush=True)

    def token_mask(enc, offmap, spans, mode):
        """(B, T) bool mask of intervention positions. mode 'span' uses the
        char spans via offset mapping; 'all_but_last' covers every real token
        except the sequence's own last one."""
        B, T = enc.input_ids.shape
        m = torch.zeros(B, T, dtype=torch.bool)
        lens = enc.attention_mask.sum(1)
        if mode == "all_but_last":
            pos = torch.arange(T)[None, :]
            m = pos < (lens[:, None].cpu() - 1)
            return m
        offs = offmap.cpu()                      # (B, T, 2)
        for b, (cs, ce) in enumerate(spans):
            hit = (offs[b, :, 0] < ce) & (offs[b, :, 1] > cs) \
                & (enc.attention_mask[b].cpu() > 0)
            # never touch the read-out position
            hit[int(lens[b]) - 1] = False
            m[b] = hit
        return m

    def run(texts, spans, span_mode, kind, ident, alpha, block=None):
        """One condition -> mean probe score at last token (+ fire rate for
        FEAT). kind: 'feat'|'dir'|'none'."""
        if kind == "feat":
            d_vec = torch.from_numpy(
                sae["W_dec"][int(ident)].numpy()).to(dev, torch.bfloat16)
            w_enc_i = sae["W_enc"][int(ident)].to(dev)
            b_i = float(sae["b_enc"][int(ident)])
            th = sae["theta"]
            th_i = None if th is None else float(th.reshape(-1)[int(ident)]
                                                 if th.numel() > 1 else
                                                 th.reshape(-1)[0])
            blk = model.model.layers[L + off - 1]
        elif kind == "dir":
            d_vec = w_t if ident == "ridge" else r_t
            blk = model.model.layers[block]
        state = {}

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            m = state["mask"].to(h.device).unsqueeze(-1)
            if kind == "feat":
                h += m * (alpha * d_vec)
            else:   # dir: alpha_rel * per-position residual norm
                nrm = h.float().norm(dim=-1, keepdim=True).to(h.dtype)
                h += m * (alpha * nrm * d_vec)
            return (h,) + out[1:] if isinstance(out, tuple) else h

        scores, fires = [], []
        with torch.no_grad():
            for i in range(0, len(texts), args.batch):
                bt = texts[i:i + args.batch]
                bs = spans[i:i + args.batch] if spans else None
                enc = tok(bt, return_tensors="pt", padding=True,
                          truncation=True, max_length=512,
                          return_offsets_mapping=(span_mode == "span"))
                offmap = enc.pop("offset_mapping", None)
                enc = enc.to(dev)
                state["mask"] = token_mask(enc, offmap, bs, span_mode)
                hd = blk.register_forward_hook(hook) \
                    if (kind != "none" and alpha != 0) else None
                res = model(input_ids=enc.input_ids,
                            attention_mask=enc.attention_mask,
                            output_hidden_states=True)
                if hd:
                    hd.remove()
                last = enc.attention_mask.sum(1) - 1
                bidx = torch.arange(len(bt), device=dev)
                hA = res.hidden_states[LA][bidx, last].float()
                scores.extend((hA @ coef_t).cpu().tolist())
                if kind == "feat":
                    hL = res.hidden_states[L + off][bidx, last].float()
                    z = torch.relu(hL @ w_enc_i + b_i)
                    fires.extend((z > (th_i or 0)).float().cpu().tolist())
        return (float(np.mean(scores)),
                float(np.mean(fires)) if fires else None)

    # diagnostic: how many name-span tokens actually fall inside the
    # 512-token window (a truncated-away span makes that fragment a no-op)
    span_counts = []
    for i in range(0, len(eng), args.batch):
        bt, bs = eng[i:i + args.batch], eng_spans[i:i + args.batch]
        enc = tok(bt, return_tensors="pt", padding=True, truncation=True,
                  max_length=512, return_offsets_mapping=True)
        offmap = enc.pop("offset_mapping")
        mm = token_mask(enc, offmap, bs, "span")
        span_counts.extend(mm.sum(1).tolist())
    span_counts = np.array(span_counts)
    print(f"[spans] fragments with >=1 span token: "
          f"{(span_counts > 0).mean():.3f}, mean tokens "
          f"{span_counts.mean():.1f}", flush=True)

    out = {"la": LA, "sae_layer": L, "offset": off,
           "treat": feats_treat, "ctrl": feats_ctrl,
           "n_eng": len(eng), "n_akk": len(akk),
           "eng_span_coverage": float((span_counts > 0).mean()),
           "eng_span_tokens_mean": float(span_counts.mean()), "arms": {}}
    langs = {"eng_namespan": (eng, eng_spans, "span"),
             "akk_allbutlast": (akk, None, "all_but_last")}

    for lname, (texts, spans, smode) in langs.items():
        arm = {}
        for group, feats in (("treat", feats_treat), ("ctrl", feats_ctrl)):
            for f in feats:
                for a in FEAT_ALPHAS:
                    s, fr = run(texts, spans, smode, "feat", f,
                                a * scale[int(f)])
                    arm[f"feat:{group}:{f}:a{a}"] = {"probe": s,
                                                     "fire_last": fr}
        for ident in ("ridge", "randdir"):
            for block in DIR_BLOCKS:
                for a in DIR_ALPHAS:
                    if a == 0 and block != DIR_BLOCKS[0]:
                        continue
                    s, _ = run(texts, spans, smode, "dir", ident, a,
                               block=block)
                    arm[f"dir:{ident}:b{block}:a{a}"] = {"probe": s}
        out["arms"][lname] = arm
        base = arm[f"feat:treat:{feats_treat[0]}:a0"]["probe"]
        print(f"[{lname}] baseline probe {base:+.2f}", flush=True)
        for k, v in arm.items():
            if k.endswith(":a0"):
                continue
            print(f"  {k}: probe {v['probe']:+.2f}"
                  + (f" fire {v['fire_last']:.2f}"
                     if v.get("fire_last") is not None else ""), flush=True)

    with open(os.path.join(RESULTS, "ignite.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {RESULTS}/ignite.json", flush=True)


if __name__ == "__main__":
    main()
