"""train_ssl_e2e.py — from-scratch SSL encoder on Akkadian text, views made ON
THE FLY (PLAN_SCALE_SSL, S2 scaling family: S / M / L / XL).

Every step draws a source-balanced batch of texts and builds two fresh views
per text in the loader (crop / span-drop / orthographic normalisation at word
level, then random sign masking), so the number of distinct pairs is
unbounded and nothing is written to disk. Objectives as in train_ssl.py:
barlow | byol | jepa | infonce. bf16 autocast, AdamW + warmup/cosine,
checkpoint every --ckpt-every steps (resumes from the same path), a WALL-CLOCK
budget (--hours) and, every --eval-every steps, a quick 5-fold linear period
probe on the labelled texts so the learning curve is visible in the log and in
results.parquet. At the end (or when the budget runs out) every clean text is
embedded into the SSL EmbStore under model 'ssl_e2e::<run>' (layer 0, site h)
for the full probe battery.

    python chrono/scripts/train_ssl_e2e.py --size M --objective jepa --hours 5
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chrono import common                                       # noqa: E402
from chrono.models.store import EmbStore                        # noqa: E402
from chrono.losses import bt_loss, variance_loss                # noqa: E402
from chrono.augment.ops import OPS                              # noqa: E402
from chrono.ssl.tokenizer import SignTokenizer                  # noqa: E402
from chrono.ssl.encoder import SignEncoder, HybridEncoder, SIZES  # noqa: E402

WORD_OPS = ["crop16", "crop32", "drop_span", "orthonorm"]


class Views:
    """On-the-fly view generator: word-level op (or none) then sign masking."""
    def __init__(self, tok: SignTokenizer, max_len: int, p_mask: float, p_op: float = 0.8):
        self.tok, self.max_len, self.p_mask, self.p_op = tok, max_len, p_mask, p_op

    def make(self, text: str, rng: np.random.Generator, heavy: bool = False) -> list[int]:
        if heavy or rng.random() < self.p_op:
            op = OPS[rng.choice(["crop16", "crop32", "drop_span"] if heavy else WORD_OPS)]
            text, _ = op(text, {}, rng)
        ids = self.tok.encode(text, self.max_len)
        p = self.p_mask * (2 if heavy else 1)
        if p > 0 and len(ids) > 2:
            m = rng.random(len(ids)) < p; m[0] = False
            ids = [self.tok.mask if k else i for i, k in zip(ids, m)]
        return ids


def pad(batch: list[list[int]], max_len: int) -> torch.Tensor:
    T = min(max(len(b) for b in batch), max_len)
    out = torch.zeros(len(batch), T, dtype=torch.long)
    for i, b in enumerate(batch):
        out[i, :len(b[:T])] = torch.tensor(b[:T])
    return out


class EMA:
    def __init__(self, model, m=0.996):
        import copy
        self.t = copy.deepcopy(model).eval(); self.m = m
        for q in self.t.parameters(): q.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for a, b in zip(self.t.parameters(), model.parameters()):
            a.mul_(self.m).add_(b.detach(), alpha=1 - self.m)


def infonce(pa, pb, temp):
    za, zb = torch.nn.functional.normalize(pa, dim=1), torch.nn.functional.normalize(pb, dim=1)
    logits = za @ zb.T / temp; tgt = torch.arange(len(za), device=za.device)
    return 0.5 * (torch.nn.functional.cross_entropy(logits, tgt) + torch.nn.functional.cross_entropy(logits.T, tgt))


@torch.no_grad()
def embed(model, tok, texts, max_len, dev, bs=512, frozen=None):
    model.eval(); out = []
    if frozen is not None:
        bs = 64
    for lo in range(0, len(texts), bs):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            if frozen is not None:
                st, m = frozen(list(texts[lo:lo + bs])); h, _, _ = model(st, m)
            else:
                ids = pad([tok.encode(t, max_len) for t in texts[lo:lo + bs]], max_len).to(dev)
                h, _, _ = model(ids)
        out.append(h.float().cpu().numpy())
    model.train(); return np.concatenate(out)


def quick_period_probe(H, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    yi = LabelEncoder().fit_transform(y); accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(H, yi):
        sc = StandardScaler().fit(H[tr])
        clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(H[tr]), yi[tr])
        accs.append(balanced_accuracy_score(yi[te], clf.predict(sc.transform(H[te]))))
    return float(np.mean(accs))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S", choices=list(SIZES)); ap.add_argument("--objective", default="barlow")
    ap.add_argument("--corpus", default=os.path.join(common.REPO, "chrono", "artifacts_ssl", "corpus_all.parquet"))
    ap.add_argument("--store-root", default=os.path.join(common.REPO, "chrono", "artifacts_ssl", "emb_store"))
    ap.add_argument("--out-dir", default=os.path.join(common.REPO, "chrono", "artifacts_ssl", "e2e"))
    ap.add_argument("--hours", type=float, default=5.0); ap.add_argument("--max-steps", type=int, default=10**9)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--max-len", type=int, default=192); ap.add_argument("--p-mask", type=float, default=0.15)
    ap.add_argument("--balance-alpha", type=float, default=0.5); ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.996); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=1000); ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N train texts")
    ap.add_argument("--frozen", default=None,
                    help="HYBRID family: registry key of a frozen encoder whose token states are the input "
                         "(e.g. thalesian_cunei400m, llama2_7b); a fresh Transformer of --size trains on top")
    ap.add_argument("--frozen-layer", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(argv)
    fam = "hyb" if args.frozen else "e2e"
    run = args.run_name or (f"hyb_{args.objective}_{args.size}_{args.frozen}-s{args.seed}" if args.frozen
                            else f"e2e_{args.objective}_{args.size}-s{args.seed}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    out_dir = os.path.join(args.out_dir, run); os.makedirs(out_dir, exist_ok=True)

    c = pd.read_parquet(args.corpus, columns=["uid", "source", "text", "split", "period_norm"])
    tr = c[c["split"] == "train"].reset_index(drop=True)
    if args.limit: tr = tr.iloc[:args.limit]
    vpath = os.path.join(args.out_dir, "vocab.json")
    tok = SignTokenizer.load(vpath) if os.path.exists(vpath) else SignTokenizer.fit(tr["text"])
    if not os.path.exists(vpath): tok.save(vpath)
    texts = tr["text"].to_numpy(); src = tr["source"].to_numpy()
    n_src = pd.Series(src).value_counts(); w = (n_src ** args.balance_alpha) / n_src
    p = w.loc[src].to_numpy(); p = p / p.sum()
    lab = c[c["period_norm"].notna() & (c["split"] != "dated")]
    vc = lab["period_norm"].value_counts(); lab = lab[lab["period_norm"].isin(vc[vc >= 30].index)]
    lab = lab.sample(min(len(lab), 4000), random_state=0)

    frozen = None
    if args.frozen:
        from chrono.ssl.frozen import FrozenTokenEncoder
        frozen = FrozenTokenEncoder(args.frozen, args.frozen_layer, args.max_len, dev)
        model = HybridEncoder(frozen.d, args.size).to(dev)
        print(f"[hyb] frozen {args.frozen} L{args.frozen_layer} d={frozen.d} -> trainable {args.size}", flush=True)
    else:
        model = SignEncoder(len(tok), args.size, args.max_len).to(dev)
    ema = EMA(model, args.ema) if args.objective in ("byol", "jepa") else None
    pred = (torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)).to(dev)
            if args.objective in ("byol", "jepa") else None)
    params = list(model.parameters()) + (list(pred.parameters()) if pred else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    views = Views(tok, args.max_len, args.p_mask)
    ck = os.path.join(out_dir, "ckpt.pt"); step = 0; curve = []
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev); model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        step = st["step"]; curve = st["curve"]
        if pred: pred.load_state_dict(st["pred"])
        if ema: ema.t.load_state_dict(st["ema"])
        print(f"[e2e] resumed {run} at step {step}", flush=True)
    n_par = model.n_params(); budget_s = args.hours * 3600; t0 = time.time()
    print(f"[e2e] {run}: size {args.size} params {n_par/1e6:.1f}M vocab {len(tok)} train texts {len(texts):,} "
          f"objective {args.objective} batch {args.batch} budget {args.hours}h dev {dev}", flush=True)

    def lr_at(s):
        # linear warmup, then constant: the horizon is a wall-clock budget,
        # not a step count, so a cosine schedule has nothing to anchor to
        return args.lr * min(1.0, s / max(1, args.warmup))

    def word_view(text, heavy=False):
        """text-level part of a view (for the hybrid family; masking happens in embedding space)"""
        if heavy or rng.random() < 0.8:
            op = OPS[rng.choice(["crop16", "crop32", "drop_span"] if heavy else WORD_OPS)]
            text, _ = op(text, {}, rng)
        return text

    def fwd(net, texts_batch, heavy=False, clean=False):
        """hybrid forward: frozen token states (no grad) -> drop mask -> trainable net"""
        st, m = frozen([t if clean else word_view(t, heavy) for t in texts_batch])
        drop = None
        if not clean:
            pm = args.p_mask * (2 if heavy else 1)
            drop = (torch.rand(m.shape, device=m.device) < pm) & m
        return net(st, m, drop)

    step0 = step; rows = []
    while step < args.max_steps and (time.time() - t0) < budget_s:
        bi = rng.choice(len(texts), size=args.batch, replace=False, p=p)
        if frozen is None:
            if args.objective == "jepa":
                va = [views.make(texts[i], rng, heavy=True) for i in bi]; vb = [tok.encode(texts[i], args.max_len) for i in bi]
            else:
                va = [views.make(texts[i], rng) for i in bi]; vb = [views.make(texts[i], rng) for i in bi]
            xa, xb = pad(va, args.max_len).to(dev), pad(vb, args.max_len).to(dev)
        for g in opt.param_groups: g["lr"] = lr_at(step)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            if frozen is not None:
                tb = [texts[i] for i in bi]
                h_a, s_a, p_a = fwd(model, tb, heavy=(args.objective == "jepa"))
                if ema is not None:
                    with torch.no_grad(): _, _, p_b = fwd(ema.t, tb, clean=(args.objective == "jepa"))
                else:
                    _, _, p_b = fwd(model, tb)
            else:
                h_a, s_a, p_a = model(xa)
                if ema is not None:
                    with torch.no_grad(): _, _, p_b = ema.t(xb)
                else:
                    _, _, p_b = model(xb)
            if args.objective == "barlow":
                loss = bt_loss(p_a.float(), p_b.float(), lambda_offdiag=0.005)
            elif args.objective in ("byol", "jepa"):
                q = torch.nn.functional.normalize(pred(p_a).float(), dim=1); z = torch.nn.functional.normalize(p_b.float(), dim=1)
                loss = 2 - 2 * (q * z).sum(1).mean()
            elif args.objective == "infonce":
                loss = infonce(p_a.float(), p_b.float(), args.temp)
            else:
                raise ValueError(args.objective)
            loss = loss + 0.1 * variance_loss(s_a.float())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if ema is not None: ema.update(model)
        curve.append(float(loss.detach())); step += 1
        if step % 100 == 0:
            print(f"[e2e] step {step} loss {np.mean(curve[-100:]):.4f} lr {lr_at(step):.2e} "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if step % args.eval_every == 0 and len(lab) >= 200:
            H = embed(model, tok, lab["text"].tolist(), args.max_len, dev, frozen=frozen)
            acc = quick_period_probe(H, lab["period_norm"].to_numpy())
            print(f"[e2e] step {step} quick period probe (linear, {lab.period_norm.nunique()} classes, n={len(lab)}) "
                  f"bal.acc {acc:.3f}", flush=True)
            rows.append(dict(run_id=f"ssl_{fam}::{run}", git_sha="", config_sha="", seed=args.seed, split="ssl_cv",
                             metric="quick_period_probe", value=acc, n=len(lab),
                             extra=json.dumps({"step": step, "loss": float(np.mean(curve[-100:])), "size": args.size,
                                               "objective": args.objective, "params": n_par, "frozen": args.frozen})))
        if step % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "curve": curve,
                        "pred": pred.state_dict() if pred else None, "ema": ema.t.state_dict() if ema else None}, ck)

    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.join(out_dir, "final.pt"))
    clean = c.drop_duplicates("uid")
    H = embed(model, tok, clean["text"].tolist(), args.max_len, dev, frozen=frozen)
    store = EmbStore(args.store_root)
    for lo in range(0, len(clean), 4096):
        store.put(f"ssl_{fam}::{run}", 0, "h", ("ssl::" + clean["uid"].iloc[lo:lo + 4096]).tolist(), H[lo:lo + 4096])
    rows.append(dict(run_id=f"ssl_{fam}::{run}", git_sha="", config_sha="", seed=args.seed, split="train", metric="final_loss",
                     value=float(np.mean(curve[-200:])) if curve else float("nan"), n=len(texts),
                     extra=json.dumps({"steps": step, "hours": round((time.time() - t0) / 3600, 2), "size": args.size,
                                       "objective": args.objective, "params": n_par})))
    common.append_results(rows)
    print(f"[e2e] done {run}: {step} steps in {(time.time() - t0) / 3600:.2f}h; embeddings -> store 'ssl_{fam}::{run}' "
          f"({len(clean):,} texts); final.pt in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
