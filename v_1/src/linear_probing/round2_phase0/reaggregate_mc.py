#!/usr/bin/env python3
"""Re-aggregate every existing balanced-MC summary from its per-draw JSONs.

Job M part C (MASTER_BACKFILL_PLAN §5). The per-draw JSONs already carry the
full year & ruler metric sets — only the old aggregator dropped them. This
re-runs the widened `_aggregate_summary` over the existing `draw*.json` files
and overwrites each `*__summary.json` in place. No cluster re-run.

(MASE/MdAPE/shuffled for Ridge cls_numeric only appear in NEW cluster draws;
old draws legitimately lack them and are skipped via .get.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_mc_probes import _aggregate_summary  # noqa: E402

PROBES_DIR = (
    Path(__file__).resolve().parents[1]
    / "results" / "orcc_round2_phase0" / "probes"
)
METHOD_TAG = "mc_balanced"


def main() -> None:
    summaries = sorted(PROBES_DIR.glob(f"*__{METHOD_TAG}__summary.json"))
    if not summaries:
        print(f"[reaggregate] no summaries found in {PROBES_DIR}")
        return

    print(f"[reaggregate] {len(summaries)} summaries in {PROBES_DIR}\n")
    for sp in summaries:
        probe = sp.name.split(f"__{METHOD_TAG}__summary.json")[0]
        out = _aggregate_summary(PROBES_DIR, probe, METHOD_TAG)
        with open(sp, "w") as f:
            json.dump(out, f, indent=2)
        per_cfg = out.get("per_config", {})
        sample_key = next(iter(per_cfg), None)
        sample_metrics = (
            sorted(per_cfg[sample_key].keys()) if sample_key else []
        )
        print(f"  {probe}: n_draws={out['n_draws']}, n_configs={len(per_cfg)}")
        print(f"    sample cfg [{sample_key}] keys: {sample_metrics}")
    print(f"\n[reaggregate] overwrote {len(summaries)} summaries.")


if __name__ == "__main__":
    main()
