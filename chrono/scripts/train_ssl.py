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
from chrono.models.store import EmbStore                        # noqa: E402
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
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    feats, scfg, tcfg = cfg["features"], cfg["ssl"], cfg["train"]
    seed = int(args.seed if args.seed is not None else tcfg.get("seed", 0))
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    run = f"{cfg['run_name']}-s{seed}"
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
    steps, batch = int(tcfg["steps"]), int(tcfg["batch"])
    n_par = sum(q.numel() for q in head.parameters())
    print(f"[ssl] objective={obj} d_in={d_in} head_params={n_par:,} steps={steps} batch={batch} "
          f"balance_alpha={alpha} sources={n_src.to_dict()}", flush=True)

    def pick(u, choices):
        avail = [a for a in choices if a in idx[u]]
        a = choices[0] if not avail else avail[rng.integers(len(avail))]
        rows = idx[u].get(a, idx[u][""])
        return rows[rng.integers(len(rows))]

    curve, t0 = [], time.time()
    all_views = [a for a in ("", "crop16", "crop32", "drop_span", "orthonorm", "tokmask")]
    for step in range(steps):
        bu = uids[rng.choice(len(uids), size=batch, replace=False, p=p)]
        if obj == "jepa":
            ra = [pick(u, CONTEXT_VIEWS) for u in bu]           # corrupted context
            rb = [idx[u][""][0] for u in bu]                    # clean target
        else:
            ra = [pick(u, all_views) for u in bu]; rb = [pick(u, all_views) for u in bu]
        xa, xb = X[ra].to(dev), X[rb].to(dev)
        h_a, s_a, p_a = head(xa)
        with torch.no_grad():
            _, _, p_b = twin(xb)
        if obj == "barlow":
            loss = bt_loss(p_a, p_b, lambda_offdiag=float(scfg.get("lambda_offdiag", 0.005)))
        elif obj in ("byol", "jepa"):
            q = torch.nn.functional.normalize(pred(p_a), dim=1)
            z = torch.nn.functional.normalize(p_b, dim=1)
            loss = 2 - 2 * (q * z).sum(1).mean()
        elif obj == "infonce":
            _, _, p_b_online = head(xb)
            loss = infonce(p_a, p_b_online, float(scfg.get("temp", 0.1)))
        else:
            raise ValueError(obj)
        loss = loss + float(scfg.get("lambda_var", 0.0)) * variance_loss(s_a)
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
    head.eval(); H = []
    with torch.no_grad():
        for lo in range(0, len(Xc), 2048):
            h, _, _ = head(Xc[lo:lo + 2048].to(dev)); H.append(h.cpu().numpy())
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
