"""train_ssl.py — S2 of PLAN_SCALE_SSL: self-supervised pretraining of an adapter
head on frozen-encoder features of the whole Akkadian corpus. No dates.

Objectives (config `ssl.objective`):
  barlow   Barlow Twins between two views of one text (redundancy reduction),
           online head vs EMA twin -- the E-MIN recipe without the ordering term
  byol     online predictor MLP -> MSE to the EMA twin's projection (BYOL)
  jepa     latent prediction: CONTEXT = a corrupted view (tokmask / crop),
           TARGET = EMA twin's embedding of the CLEAN text; predictor on the
           context side, MSE in latent space. The adapter-space analogue of
           I-JEPA/data2vec: predict the representation of what is missing.
  infonce  NT-Xent between the two views' projections, temperature `ssl.temp`

Views come from views_ssl.parquet (make_ssl_views.py); features from the SSL
EmbStore ('ssl::<view_id>'). Texts are drawn with source-balanced probability
p_s ∝ n_s^alpha (alpha = ssl.balance_alpha, default .5) so ORACC/eBL do not
swamp Archibab. After training the head embeds every text's CLEAN view and
writes h into the SAME store under model 'ssl::<run_name>' (layer 0, site
'h'), so probe_representations.py can score it like any encoder.

    python chrono/scripts/train_ssl.py --config chrono/configs/ssl_barlow_cunei.yaml
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chrono import common                                       # noqa: E402
from chrono.models.store import EmbStore
from chrono.eval.erasure import LeaceEraser                        # noqa: E402
from chrono.models.heads import AdapterHead, EmaTwin            # noqa: E402
from chrono.losses import bt_loss, variance_loss                # noqa: E402

CONTEXT_VIEWS = ("tokmask", "crop16", "crop32", "drop_span")


class Predictor(torch.nn.Module):
    def __init__(self, d, hidden=512):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


def infonce(pa, pb, temp):
    za, zb = torch.nn.functional.normalize(pa, dim=1), torch.nn.functional.normalize(pb, dim=1)
    logits = za @ zb.T / temp
    tgt = torch.arange(len(za), device=za.device)
    return 0.5 * (torch.nn.functional.cross_entropy(logits, tgt) + torch.nn.functional.cross_entropy(logits.T, tgt))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--views", default=os.path.join(common.REPO, "chrono", "artifacts_ssl", "views_ssl.parquet"))
    ap.add_argument("--corpus", default=os.path.join(common.REPO, "chrono", "artifacts_ssl", "corpus_all.parquet"))
    ap.add_argument("--store-root", default=os.path.join(common.REPO, "chrono", "artifacts_ssl", "emb_store"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", default=os.path.join(common.REPO, "chrono", "artifacts_ssl", "ssl_runs"))
    # --- S5b anti-shortcut arms (advisor, 2026-09-04). The 32-run sweep showed
    # every SSL family learns SOURCE (probe .92-.98) and never beats the frozen
    # encoder on dating; these arms forbid the shortcut during training.
    ap.add_argument("--anti", choices=["leace", "adv", "both"], default=None,
                    help="leace: re-fit a LEACE eraser for source on h every "
                         "--refit-every steps and apply it inside the forward "
                         "pass; adv: gradient-reversal source classifier on h; "
                         "both: the two at once — the eraser kills the linear "
                         "trace, the adversary chases what grows back")
    ap.add_argument("--refit-every", type=int, default=500)
    ap.add_argument("--steps", type=int, default=None, help="override train.steps (smoke tests)")
    ap.add_argument("--lambda-adv", type=float, default=1.0)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    feats, scfg, tcfg = cfg["features"], cfg["ssl"], cfg["train"]
    seed = int(args.seed if args.seed is not None else tcfg.get("seed", 0))
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    suffix = f"_{args.anti}" if args.anti else ""
    run = f"{cfg['run_name']}{suffix}-s{seed}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    corpus = pd.read_parquet(args.corpus, columns=["uid", "source", "split"])
    train_uids = corpus.loc[corpus["split"] == "train", "uid"].to_numpy()
    views = pd.read_parquet(args.views, columns=["uid", "source", "augs", "seed", "view_id"])
    views = views[views["uid"].isin(set(train_uids))].reset_index(drop=True)
    store = EmbStore(args.store_root)
    print(f"[ssl] {run}: {len(train_uids):,} train texts, {len(views):,} views; loading features", flush=True)
    X = torch.as_tensor(store.get(feats["model"], feats["layer"], feats["site"],
                                  ("ssl::" + views["view_id"]).tolist())).float()
    d_in = X.shape[1]
    # index: uid -> {aug: [row positions]}
    idx = {}
    for pos, (u, a) in enumerate(zip(views["uid"], views["augs"])):
        idx.setdefault(u, {}).setdefault(a, []).append(pos)
    uids = np.array([u for u in train_uids if u in idx and "" in idx[u]])
    src_of = corpus.set_index("uid")["source"]
    n_src = src_of.loc[uids].value_counts()
    alpha = float(scfg.get("balance_alpha", 0.5))
    w = (n_src ** alpha) / n_src            # per-text weight so that P(source) ∝ n^alpha
    p = w.loc[src_of.loc[uids]].to_numpy(); p = p / p.sum()

    head = AdapterHead(d_in=d_in, d_hidden=int(cfg.get("head", {}).get("d_hidden", 512)),
                       d_proj=int(cfg.get("head", {}).get("d_proj", 128))).to(dev)
    twin = EmaTwin(head, momentum=float(scfg.get("ema", 0.996)))
    obj = scfg["objective"]
    pred = Predictor(head.proj.out_features).to(dev) if obj in ("byol", "jepa") else None
    params = list(head.parameters()) + (list(pred.parameters()) if pred else [])
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("wd", 1e-4)))
    steps, batch = int(args.steps or tcfg["steps"]), int(tcfg["batch"])
    n_par = sum(q.numel() for q in head.parameters())
    print(f"[ssl] objective={obj} d_in={d_in} head_params={n_par:,} steps={steps} batch={batch} "
          f"balance_alpha={alpha} sources={n_src.to_dict()}", flush=True)

    def pick(u, choices):
        avail = [a for a in choices if a in idx[u]]
        a = choices[0] if not avail else avail[rng.integers(len(avail))]
        rows = idx[u].get(a, idx[u][""])
        return rows[rng.integers(len(rows))]

    # ---- anti-shortcut machinery -------------------------------------------
    src_cat = pd.Categorical(src_of.loc[uids])
    code_of = pd.Series(src_cat.codes, index=uids)      # uid -> source class
    n_classes = len(src_cat.categories)
    Z_onehot = np.eye(n_classes, dtype=np.float64)[src_cat.codes]
    eraser_M = eraser_mu = None            # torch constants, refreshed by refit
    adv_clf = None
    if args.anti in ("adv", "both"):
        adv_clf = torch.nn.Linear(head.proj.in_features, n_classes).to(dev)
        opt.add_param_group({"params": adv_clf.parameters()})

    class _GRL(torch.autograd.Function):
        """Gradient reversal (Ganin & Lempitsky): identity forward, -lam on the
        way back, so the classifier learns the source while the head unlearns it."""
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam; return x.view_as(x)
        @staticmethod
        def backward(ctx, g):
            return -ctx.lam * g, None

    def fwd(module, x):
        """mlp -> (optional LEACE) -> axis/proj, so the erasure sits INSIDE the
        computation the SSL loss sees, not applied post-hoc as in P1."""
        h = module.mlp(x)
        if eraser_M is not None:
            h = h - (h - eraser_mu) @ eraser_M.T
        return h, module.axis(h).squeeze(-1), module.proj(h)

    def refit_eraser():
        """LEACE for source on the CURRENT h over a balanced sample; the fitted
        affine map is then a constant inside fwd() until the next refit
        (arXiv:2502.02820 uses the same erase-inside-training pattern)."""
        nonlocal eraser_M, eraser_mu
        take = rng.choice(len(uids), size=min(8192, len(uids)), replace=False, p=p)
        rows = [idx[u][""][0] for u in uids[take]]
        with torch.no_grad():
            hs = []
            for lo in range(0, len(rows), 2048):
                h0 = head.mlp(X[rows[lo:lo + 2048]].to(dev))
                hs.append(h0.cpu().numpy())
        er = LeaceEraser().fit(np.concatenate(hs), Z_onehot[take])
        eraser_M = torch.as_tensor(er.M, dtype=torch.float32, device=dev)
        eraser_mu = torch.as_tensor(er.mu_x, dtype=torch.float32, device=dev)
        return er.rank

    curve, t0 = [], time.time()
    all_views = [a for a in ("", "crop16", "crop32", "drop_span", "orthonorm", "tokmask")]
    for step in range(steps):
        bu = uids[rng.choice(len(uids), size=batch, replace=False, p=p)]
        if obj == "jepa":
            ra = [pick(u, CONTEXT_VIEWS) for u in bu]           # corrupted context
            rb = [idx[u][""][0] for u in bu]                    # clean target
        else:
            ra = [pick(u, all_views) for u in bu]; rb = [pick(u, all_views) for u in bu]
        if args.anti in ("leace", "both") and step % args.refit_every == 0:
            rk = refit_eraser()
            if step % 2000 == 0:
                print(f"[ssl] step {step}: LEACE refit, rank {rk}", flush=True)
        xa, xb = X[ra].to(dev), X[rb].to(dev)
        h_a, s_a, p_a = fwd(head, xa)
        with torch.no_grad():
            _, _, p_b = fwd(twin.target, xb)
        if obj == "barlow":
            loss = bt_loss(p_a, p_b, lambda_offdiag=float(scfg.get("lambda_offdiag", 0.005)))
        elif obj in ("byol", "jepa"):
            q = torch.nn.functional.normalize(pred(p_a), dim=1)
            z = torch.nn.functional.normalize(p_b, dim=1)
            loss = 2 - 2 * (q * z).sum(1).mean()
        elif obj == "infonce":
            _, _, p_b_online = fwd(head, xb)
            loss = infonce(p_a, p_b_online, float(scfg.get("temp", 0.1)))
        else:
            raise ValueError(obj)
        loss = loss + float(scfg.get("lambda_var", 0.0)) * variance_loss(s_a)
        if adv_clf is not None:
            lam = args.lambda_adv * min(1.0, step / max(1, steps // 5))   # warmup
            y_src = torch.as_tensor(code_of.loc[bu].to_numpy(), dtype=torch.long, device=dev)
            adv_logits = adv_clf(_GRL.apply(h_a, lam))
            adv_loss = torch.nn.functional.cross_entropy(adv_logits, y_src)
            loss = loss + adv_loss
            if step % 200 == 0:
                acc = (adv_logits.argmax(1) == y_src).float().mean()
                print(f"[ssl] step {step} adv_acc {acc:.3f} lam {lam:.2f}", flush=True)
        opt.zero_grad(); loss.backward(); opt.step(); twin.update()
        curve.append(float(loss.detach()))
        if step % 200 == 0 or step == steps - 1:
            print(f"[ssl] step {step} loss {np.mean(curve[-200:]):.4f} ({time.time() - t0:.0f}s)", flush=True)

    # write head + embed every CLEAN text (all splits) under model 'ssl::<run>'
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({k: v.cpu() for k, v in head.state_dict().items()}, os.path.join(args.out_dir, f"{run}.pt"))
    clean = pd.read_parquet(args.views, columns=["uid", "augs", "view_id"])
    clean = clean[clean["augs"] == ""].drop_duplicates("uid")
    Xc = torch.as_tensor(store.get(feats["model"], feats["layer"], feats["site"], ("ssl::" + clean["view_id"]).tolist())).float()
    if args.anti in ("leace", "both"):
        refit_eraser()                      # final eraser on the finished head
    head.eval(); H = []
    with torch.no_grad():
        for lo in range(0, len(Xc), 2048):
            h, _, _ = fwd(head, Xc[lo:lo + 2048].to(dev)); H.append(h.cpu().numpy())
    H = np.concatenate(H)
    for lo in range(0, len(clean), 4096):
        store.put(f"ssl::{run}", 0, "h", ("ssl::" + clean["uid"].iloc[lo:lo + 4096]).tolist(), H[lo:lo + 4096])
    common.append_results([dict(run_id=f"ssl::{run}", git_sha="", config_sha=common.config_sha(cfg), seed=seed,
                                split="train", metric=m, value=float(v), n=len(uids),
                                extra=json.dumps({"objective": obj, "features": feats, "head_params": n_par}))
                           for m, v in (("final_loss", np.mean(curve[-200:])), ("first_loss", np.mean(curve[:50])))])
    print(f"[ssl] done {run}: head -> {args.out_dir}/{run}.pt; embeddings -> store model 'ssl::{run}' L0 site h "
          f"({len(clean):,} texts)", flush=True)


if __name__ == "__main__":
    main()
