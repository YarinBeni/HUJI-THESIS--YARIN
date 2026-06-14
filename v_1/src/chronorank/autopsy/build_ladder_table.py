"""
Pillar 1 — 1b control-ladder table builder (CPU, instant).

Reads the maximal-balanced PLS *summary* JSONs produced by run_mc_probes.py for the
control ladder and emits the factor table that decides T/A/O/F.

For each method it takes the BEST-layer year Spearman (mean over the 200 balanced MC
draws) under a given cleaning, plus the MAE at that layer, and assembles the three
controlled comparisons:

  Thalesian vs vanilla uMT5  -> (F) the cuneiform finetune
  vanilla uMT5 vs Qwen3-8B   -> (A)+(T) enc-dec/bidirectional + tokenizer bundle
  (1a fertility)             -> (T) descriptively, to split (A) from (T)

Run after the 1b sbatch has written summaries into --probes-dir.
Output: results/ladder_table.csv  (+ stdout decision reads)
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# NOTE: qwen3_1b7 / qwen3_32b base maximal+mean PLS are NOT re-probed by P1b; their
# canonical numbers live in v_1/src/geodesic/maximal_figs/tables/T1_year_pls_maximal.csv
# (qwen3_1b7=0.355 L9, qwen3_32b=0.340 L6). Qwen3-1.7B is the *size-matched* (0.4B-class)
# architecture comparator for uMT5 — fairer than the 8B. Listed here so a full re-probe
# (point --probes-dir at a run that includes them) tabulates them too.
LADDER = [
    ("thalesian_cunei400m", "Thalesian (uMT5-base + cuneiform finetune)"),
    ("thalesian_akk300m",   "Thalesian AKK-300m (uMT5 + finetune, variant)"),
    ("umt5_base",           "vanilla google/umt5-base (NO finetune)"),
    ("qwen3_1b7",           "Qwen3-1.7B base (decoder-only, SIZE-MATCHED)"),
    ("qwen3_8b",            "Qwen3-8B base (decoder-only)"),
    ("qwen3_32b",           "Qwen3-32B base (decoder-only)"),
    ("gpt_oss_120b",        "gpt-oss-120b base (decoder-only)"),
    ("random",              "random (same tokenizer as Qwen, untrained)"),
]


def best_year_layer(summary_path: Path):
    """Return (best_spearman_mean, mae_at_best, layer, target) over year configs."""
    if not summary_path.exists():
        return None
    d = json.load(open(summary_path))
    pc = d.get("per_config", {})
    best = None
    for key, m in pc.items():
        if "year" not in key:           # skip ruler configs
            continue
        s = m.get("spearman_mean")
        if not isinstance(s, (int, float)):
            continue
        if best is None or s > best[0]:
            layer = next((p for p in key.split("__") if p.startswith("L")), "L??")
            target = key.split("__")[-1]
            best = (s, m.get("mae_mean"), layer, target, m.get("spearman_std"))
    return best


def collect(probes_dir: Path, method_tag: str):
    rows = []
    for method, label in LADDER:
        sp = probes_dir / f"{method}_pls__{method_tag}__summary.json"
        b = best_year_layer(sp)
        if b is None:
            print(f"  [missing] {sp.name}")
            rows.append({"method": method, "label": label, "spearman": None})
            continue
        rows.append({
            "method": method, "label": label,
            "best_layer": b[2], "target": b[3],
            "spearman": round(b[0], 4), "spearman_std": round(b[4], 4) if b[4] else None,
            "mae_years": round(b[1], 2) if b[1] is not None else None,
        })
    return rows


def decision(rows):
    g = {r["method"]: r["spearman"] for r in rows if r.get("spearman") is not None}
    th = g.get("thalesian_cunei400m")
    um = g.get("umt5_base")
    qw = g.get("qwen3_8b")
    rd = g.get("random")
    lines = ["\n=== DECISION READS ==="]
    if th is not None and um is not None:
        d = th - um
        verdict = ("the cuneiform FINETUNE (F) does the work -> next: finetune big models "
                   "(seq2seq/translation objective, go to 1c)"
                   if d > 0.03 else
                   "the WIN is the BASE model (arch/tokenizer/pretraining), NOT the finetune "
                   "-> next: pick a better enc-dec backbone / scale uMT5")
        lines.append(f"Thalesian({th:.3f}) vs uMT5({um:.3f})  Δ={d:+.3f}  ->  (F): {verdict}")
    if um is not None and qw is not None:
        d = um - qw
        verdict = ("encoder-decoder/bidirectional+multilingual base MATTERS (A/T) "
                   "-> next: take a bigger enc-dec (uMT5-XL), scale within the right arch"
                   if d > 0.03 else
                   "base architecture is NOT it -> the story is the cuneiform finetune (F/O)")
        lines.append(f"uMT5({um:.3f}) vs Qwen3-8B({qw:.3f})  Δ={d:+.3f}  ->  (A)+(T): {verdict}")
    if rd is not None:
        lines.append(f"(sanity) random baseline Spearman = {rd:.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes-dir",
                    default="v_1/src/chronorank/autopsy/results/probes")
    ap.add_argument("--out", default="v_1/src/chronorank/autopsy/results")
    args = ap.parse_args()
    pdir = Path(args.probes_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for tag, clean in [("mc_balanced", "tier0"), ("mc_balanced_maximal", "maximal")]:
        print(f"\n### cleaning={clean} (tag={tag}) ###")
        rows = collect(pdir, tag)
        for r in rows:
            r["cleaning"] = clean
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print(decision(rows))
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    cols = ["cleaning", "method", "label", "best_layer", "target",
            "spearman", "spearman_std", "mae_years"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out / "ladder_table.csv", index=False)
    print(f"\nSaved {out/'ladder_table.csv'}")


if __name__ == "__main__":
    main()
