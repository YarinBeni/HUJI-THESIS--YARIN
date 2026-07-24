"""W probing: per-layer ridge probes over saved activations for one method.

Reads activations/{method}/{entity_type}/{site}.layer{L}.npz, writes
results/probes/{method}/{entity_type}.{site}.{probe}.json (committed) plus the
best-layer projection csv.gz under results/projections/ and the best-layer probe
direction npz under results/directions/ (gitignored, for later neuron work).

    python probe_wm.py --method qwen3_8b
    python probe_wm.py --method llama2_70b --cleanup      # rm npz after success
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wm_lib import entity_data, probing  # noqa: E402
from wm_lib.registry import MODELS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ACTS_DIR = os.path.join(HERE, "activations")
RESULTS_DIR = os.path.join(HERE, "results")


def probe_one(method, entity_type, site, args):
    act_dir = os.path.join(ACTS_DIR, method, entity_type)
    layer_files = sorted(
        glob.glob(os.path.join(act_dir, f"{site}.layer*.npz")),
        key=lambda p: int(re.search(r"layer(\d+)\.npz$", p).group(1)))
    if not layer_files:
        print(f"[skip] no {site} activations for {method}/{entity_type}")
        return None

    df = entity_data.load_entity_df(entity_type)
    target, valid = entity_data.target_values(entity_type, df)
    is_test = df.is_test.values.astype(bool)
    feature, is_place = entity_data.FEATURES[entity_type]

    with open(os.path.join(act_dir, "metadata.json")) as f:
        meta = json.load(f)
    n_rows = meta["n_rows"]
    if n_rows != len(df):  # smoke-test extraction subset
        target, valid, is_test = target[:n_rows], valid[:n_rows], is_test[:n_rows]
        df = df.iloc[:n_rows]

    per_layer, best = {}, (None, -np.inf)
    for path in layer_files:
        li = int(re.search(r"layer(\d+)\.npz$", path).group(1))
        X = np.load(path)["acts"][:n_rows][valid]
        if np.isnan(X).any():
            print(f"[warn] NaN activations, skipping layer {li}")
            continue
        if args.probe == "pls":
            scores, probe, proj = probing.run_pls_probe(
                X, target[valid], is_test[valid], is_place)
        else:
            scores, probe, proj = probing.run_probe(
                X, target[valid], is_test[valid], is_place)
        per_layer[li] = scores
        if scores["test"]["r2"] > best[1]:
            best = ((li, probe, proj), scores["test"]["r2"])
        print(f"[{method}/{entity_type}/{site}] layer {li}: "
              f"test r2={scores['test']['r2']:.3f}", flush=True)

    if not per_layer:
        return None
    (bl, bprobe, bproj), br2 = best

    out = {
        "method": method,
        "entity_type": entity_type,
        "feature": feature,
        "site": site,
        "probe": args.probe,
        "n": int(valid.sum()),
        "n_dropped_nan_target": int((~valid).sum()),
        "n_test": int(is_test[valid].sum()),
        "layers": {str(k): v for k, v in sorted(per_layer.items())},
        "best_layer": bl,
        "best_test_r2": float(br2),
        "best_test_spearman": per_layer[bl]["test"].get(
            "spearman",
            (per_layer[bl]["test"].get("lat_spearman", float("nan"))
             + per_layer[bl]["test"].get("lon_spearman", float("nan"))) / 2),
    }
    pdir = os.path.join(RESULTS_DIR, "probes", method)
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, f"{entity_type}.{site}.{args.probe}.json"), "w") as f:
        json.dump(out, f, indent=2)

    # best-layer projection for map/timeline figures
    jdir = os.path.join(RESULTS_DIR, "projections", method)
    os.makedirs(jdir, exist_ok=True)
    proj_df = pd.DataFrame(
        {"is_test": is_test[valid]},
        index=np.arange(len(df))[valid])
    if is_place:
        proj_df["pred_lon"], proj_df["pred_lat"] = bproj[:, 0], bproj[:, 1]
        proj_df["lon"], proj_df["lat"] = target[valid][:, 0], target[valid][:, 1]
    else:
        proj_df["pred"], proj_df["true"] = bproj, target[valid]
    proj_df.to_csv(os.path.join(
        jdir, f"{entity_type}.{site}.layer{bl}.csv.gz"), index_label="row")

    # best-layer probe direction (for later neuron-alignment work)
    ddir = os.path.join(RESULTS_DIR, "directions", method)
    os.makedirs(ddir, exist_ok=True)
    if hasattr(bprobe, "coef_"):
        np.savez_compressed(
            os.path.join(ddir, f"{entity_type}.{site}.layer{bl}.npz"),
            coef=np.asarray(bprobe.coef_, dtype=np.float32),
            intercept=np.atleast_1d(bprobe.intercept_).astype(np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(MODELS))
    ap.add_argument("--entity-type", default="all",
                    choices=["all"] + entity_data.ENTITY_TYPES)
    ap.add_argument("--probe", default="ridge", choices=["ridge", "pls"])
    ap.add_argument("--sites", default=None,
                    help="comma list to restrict pooling sites (e.g. 'mean' to skip "
                         "re-probing an already-done 'last')")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete the method's npz activations after all probes succeed")
    args = ap.parse_args()
    only = set(args.sites.split(",")) if args.sites else None

    ets = entity_data.ENTITY_TYPES if args.entity_type == "all" else [args.entity_type]
    all_ok = True
    for et in ets:
        sites = MODELS[args.method]["sites"]
        act_dir = os.path.join(ACTS_DIR, args.method, et)
        found_sites = {re.match(r"(\w+)\.layer", os.path.basename(p)).group(1)
                       for p in glob.glob(os.path.join(act_dir, "*.layer*.npz"))}
        run_sites = sorted(set(sites) | found_sites)
        if only:
            run_sites = [s for s in run_sites if s in only]
        for site in run_sites:
            ok = probe_one(args.method, et, site, args)
            all_ok = all_ok and (ok is not None)

    if args.cleanup and all_ok:
        n = 0
        for p in glob.glob(os.path.join(ACTS_DIR, args.method, "*", "*.npz")):
            os.remove(p)
            n += 1
        print(f"[cleanup] removed {n} npz files for {args.method}")
    elif args.cleanup:
        print("[cleanup] SKIPPED: some probes missing/failed; activations kept")


if __name__ == "__main__":
    main()
