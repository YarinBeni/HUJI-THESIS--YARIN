#!/bin/bash
# Wave 5 — the remaining decided experiments. No interdependencies; F18 needs
# only F7/F8 outputs (already committed). Cell-C steering was SKIPPED per
# F12's pre-registered rule.
#
#   bash v_1/src/phase2/submit_wave5.sh
#
#   F15  E6 Esarhaddon micro-study            (CPU x5)
#   F16  E7 spectral seriation (+Esarhaddon)  (CPU x5)
#   F17  E4 confounder erasure                (CPU x3)
#   F18  E-prop: max-pooled SAE features      (GPU x1)
#   F19  site=last probes + spec curve        (CPU x1)
set -euo pipefail
cd "$(dirname "$0")/../../.."      # repo root

F15=$(sbatch --parsable v_1/src/phase2/esarhaddon/sbatch/F15_esarhaddon.sbatch)
F16=$(sbatch --parsable v_1/src/phase2/seriation/sbatch/F16_seriation.sbatch)
F17=$(sbatch --parsable v_1/src/phase2/erasure/sbatch/F17_confounders.sbatch)
F18=$(sbatch --parsable v_1/src/phase2/sae/sbatch/F18_propagation.sbatch)
F19=$(sbatch --parsable v_1/src/phase2/pairs/sbatch/F19_last_and_spec.sbatch)

echo "submitted wave 5:"
echo "  F15 esarhaddon     ${F15}"
echo "  F16 seriation      ${F16}"
echo "  F17 erasure        ${F17}"
echo "  F18 propagation    ${F18}"
echo "  F19 last + spec    ${F19}"
squeue -u "$USER"
