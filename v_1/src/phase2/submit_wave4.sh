#!/bin/bash
# Wave 4 — the gap-fix wave. One shot, no interdependencies (all fixes read
# only committed inputs). Safe to re-run.
#
#   bash v_1/src/phase2/submit_wave4.sh
#
#   F10  behavioural rerun WITH chat template            (GPU x3)
#   F11  SAE token-level firing audit                    (GPU x1)
#   F12  steering v2: early blocks + norm-relative alpha (GPU x2)
#   F13  E3 last-on-last + positive control + surgical   (CPU x4)
#   F14  llama eng stride-1 + lens random-calibration    (CPU x6)
set -euo pipefail
cd "$(dirname "$0")/../../.."      # repo root

F10=$(sbatch --parsable v_1/src/phase2/pairs/sbatch/F10_behavioral_chat.sbatch)
F11=$(sbatch --parsable v_1/src/phase2/sae/sbatch/F11_token_firing.sbatch)
F12=$(sbatch --parsable v_1/src/phase2/steering/sbatch/F12_steering_v2.sbatch)
F13=$(sbatch --parsable v_1/src/phase2/transfer/sbatch/F13_e3_last.sbatch)
F14=$(sbatch --parsable v_1/src/phase2/pairs/sbatch/F14_llama_stride1_lens.sbatch)

echo "submitted wave 4:"
echo "  F10 behavioral+chat   ${F10}"
echo "  F11 token firing      ${F11}"
echo "  F12 steering v2       ${F12}"
echo "  F13 e3 last+controls  ${F13}"
echo "  F14 stride1 + lens    ${F14}"
squeue -u "$USER"
