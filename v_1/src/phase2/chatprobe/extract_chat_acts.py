"""E-chat — probe the representations a CHAT model builds in conversation mode.

THE OBJECTION THIS ANSWERS (the user's, and it is fair): every probing result
so far fed bare text into the model. For base models (Llama-2, OLMo-2) that is
exactly their training-time operation. For chat-tuned models (Qwen3, gpt-oss)
bare text is an off-distribution way to run them — maybe the fragment
representations look different when the model is addressed the way it was
trained to be addressed. Cell A working without a template (rho=.88) proves the
representations exist; it does not prove document representations are unchanged.

WHAT THIS DOES. Re-extracts fragment activations for a chat model with each
fragment wrapped in its chat template:

    user: "When was the following ancient royal inscription composed?\n\n{text}"
    -> activations captured with the generation prompt appended

Pooling: `mean` over the FRAGMENT's own tokens only (char-span -> token-span, so
template tokens don't dilute it) and `last` = the final prompt token — the
state from which the model would start answering, i.e. its conversational
summary of the document.

Output goes to world_models/akkadian/activations/{method}_chat/{variant}/ in
the standard {site}.layer{L}.npz format, so the ENTIRE existing tool chain
(probe_pairs, e3, seriation...) works on the new arm by name: qwen3_8b_chat.

    python extract_chat_acts.py --method qwen3_8b --variant akk_maximal

GPU job (F20), which then runs the E1 pairwise probe on the new arm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAIRS = os.path.abspath(os.path.join(_HERE, "..", "pairs"))
sys.path.insert(0, _PAIRS)
import pairs_data as P                                   # noqa: E402
import probe_pairs as PP                                 # noqa: E402
_WM = os.path.abspath(os.path.join(_HERE, "..", "..", "world_models"))
sys.path.insert(0, _WM)
from wm_lib import registry                              # noqa: E402

QUESTION = "When was the following ancient royal inscription composed?\n\n"


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="qwen3_8b")
    ap.add_argument("--variant", default="akk_maximal", choices=list(PP.TEXT_COL))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--layer-stride", type=int, default=1)
    args = ap.parse_args()

    df = P.load_eligible()
    texts = df[PP.TEXT_COL[args.variant]].fillna("").astype(str).tolist()
    hfid = registry.MODELS[args.method]["hfid"]
    tok = AutoTokenizer.from_pretrained(hfid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        hfid, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    n_layers = model.config.num_hidden_layers

    def render(t):
        msgs = [{"role": "user", "content": QUESTION + t}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    kept = list(range(1, n_layers + 1, args.layer_stride))
    d = model.config.hidden_size
    out = {s: {L: np.zeros((len(df), d), np.float16) for L in kept}
           for s in ("mean", "last")}

    with torch.no_grad():
        for i in range(0, len(texts), args.batch):
            bt = texts[i:i + args.batch]
            rendered = [render(t) for t in bt]
            enc = tok(rendered, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_len,
                      add_special_tokens=False,
                      return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            hs = model(**enc, output_hidden_states=True).hidden_states
            attn = enc["attention_mask"]
            last_ix = attn.sum(1) - 1
            # fragment token mask: tokens whose char span overlaps the
            # fragment's char span inside the rendered string
            fmask = torch.zeros_like(attn, dtype=torch.bool)
            for bi, (r, t) in enumerate(zip(rendered, bt)):
                if not t.strip():
                    continue
                c0 = r.find(t[:80]) if t[:80] else -1
                c1 = c0 + len(t) if c0 >= 0 else -1
                if c0 < 0:            # truncated search string; fall back to
                    c0, c1 = 0, len(r)  # all tokens (rare; flagged by count)
                om = offsets[bi]
                fmask[bi] = ((om[:, 0] < c1) & (om[:, 1] > c0)
                             ).to(fmask.device)
            fmask = fmask & attn.bool()
            for L in kept:
                h = hs[L].float()
                bidx = torch.arange(h.shape[0], device=h.device)
                out["last"][L][i:i + len(bt)] = (
                    h[bidx, last_ix.to(h.device)].cpu().numpy()
                    .astype(np.float16))
                m = fmask.to(h.device).unsqueeze(-1).float()
                out["mean"][L][i:i + len(bt)] = (
                    (h * m).sum(1) / m.sum(1).clamp(min=1.0)
                ).cpu().numpy().astype(np.float16)
            if (i // args.batch) % 20 == 0:
                print(f"  {i + len(bt)}/{len(texts)}", flush=True)

    arm = f"{args.method}_chat"
    dst = os.path.join(_WM, "akkadian", "activations", arm, args.variant)
    os.makedirs(dst, exist_ok=True)
    for site in ("mean", "last"):
        for L in kept:
            np.savez_compressed(os.path.join(dst, f"{site}.layer{L}.npz"),
                                acts=out[site][L])
    with open(os.path.join(dst, "metadata.json"), "w") as f:
        json.dump({"method": arm, "hfid": hfid, "variant": args.variant,
                   "chat_template": True, "question": QUESTION.strip(),
                   "n_frags": len(df), "d": d, "layers": kept,
                   "max_len": args.max_len}, f, indent=2)
    print(f"[done] {arm}/{args.variant}: {len(kept)} layers x 2 sites -> {dst}",
          flush=True)


if __name__ == "__main__":
    main()
