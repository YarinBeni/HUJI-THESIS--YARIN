# Sensitivity: duplicates / century docs (condition `orig`)

docs: all 1193 · no_dup 1166 · no_century 1179 · both 1153

| run | subset | mc ρ (mean over seeds) | gkf pooled ρ |
|---|---|---|---|
| `emin2_cunei400m_t0_akk` | all | +0.609 | +0.512 |
| `emin2_cunei400m_t0_akk` | no_dup | +0.613 | +0.513 |
| `emin2_cunei400m_t0_akk` | no_century | +0.609 | +0.517 |
| `emin2_cunei400m_t0_akk` | no_dup_no_century | +0.613 | +0.515 |
| `emin2_llama2_7b_t0_akk` | all | +0.538 | +0.440 |
| `emin2_llama2_7b_t0_akk` | no_dup | +0.538 | +0.439 |
| `emin2_llama2_7b_t0_akk` | no_century | +0.538 | +0.446 |
| `emin2_llama2_7b_t0_akk` | no_dup_no_century | +0.538 | +0.444 |
| `baseline_ridge_L12mean_akk` | all | +0.447 | +0.420 |
| `baseline_ridge_L12mean_akk` | no_dup | +0.448 | +0.419 |
| `baseline_ridge_L12mean_akk` | no_century | +0.447 | +0.417 |
| `baseline_ridge_L12mean_akk` | no_dup_no_century | +0.448 | +0.415 |
| `baseline_ridge_L16mean_akk` | all | +0.353 | +0.310 |
| `baseline_ridge_L16mean_akk` | no_dup | +0.361 | +0.308 |
| `baseline_ridge_L16mean_akk` | no_century | +0.353 | +0.303 |
| `baseline_ridge_L16mean_akk` | no_dup_no_century | +0.361 | +0.302 |
