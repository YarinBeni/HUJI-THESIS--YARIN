# P1 — head ladder (C5): Chrono-Barlow on LEACE-erased features — tier0 Akkadian

mc ρ (mean ± sd over seeds; pooled gkf ρ in brackets). Probe = ridge on the SAME erased features (C4). Δ = head − probe on the same rung.

## cunei400m (L12 mean)

| erased | head | probe (ridge) | Δ head−probe | head retains |
|---|---|---|---|---|
| none | +0.609 ± 0.012 [+0.512] | +0.447 [+0.422] | +0.162 | 100 % |
| provenance | +0.358 ± 0.004 [+0.245] (n=3) | +0.142 [+0.113] | +0.216 | 59% |
| period | +0.382 ± 0.017 [+0.245] (n=3) | +0.246 [+0.158] | +0.136 | 63% |
| subgenre | +0.470 ± 0.021 [+0.425] (n=3) | +0.294 [+0.197] | +0.176 | 77% |
| length | +0.594 ± 0.003 [+0.501] (n=3) | +0.423 [+0.312] | +0.171 | 98% |

## llama2_7b (L16 mean)

| erased | head | probe (ridge) | Δ head−probe | head retains |
|---|---|---|---|---|
| none | +0.538 ± 0.029 [+0.440] | +0.356 [+0.313] | +0.182 | 100 % |
| provenance | +0.378 ± 0.024 [+0.252] (n=3) | -0.001 [-0.004] | +0.379 | 70% |
| period | +0.375 ± 0.029 [+0.308] (n=3) | +0.219 [+0.180] | +0.156 | 70% |
| subgenre | +0.404 ± 0.046 [+0.360] (n=3) | +0.277 [+0.233] | +0.127 | 75% |
| length | +0.508 ± 0.030 [+0.419] (n=3) | +0.255 [+0.191] | +0.253 | 95% |

## qwen3_8b (L18 mean)

| erased | head | probe (ridge) | Δ head−probe | head retains |
|---|---|---|---|---|
| none | +0.432 ± 0.016 [+0.350] | +0.259 [+0.261] | +0.173 | 100 % |
| provenance | +0.288 ± 0.025 [+0.187] (n=3) | -0.198 [-0.140] | +0.486 | 67% |
| period | +0.337 ± 0.017 [+0.239] (n=3) | +0.160 [+0.183] | +0.177 | 78% |
| subgenre | +0.322 ± 0.051 [+0.231] (n=3) | +0.214 [+0.188] | +0.108 | 75% |
| length | +0.444 ± 0.009 [+0.355] (n=3) | +0.193 [+0.158] | +0.251 | 103% |


## Reading (2026-09-02, all 36 runs)

1. **The head's advantage survives every single-variable erasure, and grows
   under provenance erasure.** On the same LEACE-erased features the ridge
   probe keeps .14 (cunei), .00 (Llama) or goes negative (Qwen −.20); the
   head keeps .36 / .38 / .29 — i.e. 59–70 % of its unerased accuracy. The
   head−probe margin after provenance erasure is +.22 / +.38 / +.49.
2. **Provenance and period are the two rungs that cost the head anything
   (−.15 to −.25 each); object type −.03 to −.14; length ≈ 0.** Same ordering as
   the probe ladder, but the head loses a smaller *share* on every rung.
3. **The head does not depend on length at all** (98–103 % retained), where
   the probes lose up to .10 — consistent with the crop views in training.
4. **Interpretation, with its limit.** LEACE removes *linear* readability of
   the concept. Either the head finds chronology that is not site/period, or
   it reconstructs site nonlinearly from what remains. The nonlinear-recovery
   check (C5b + `probe_head_hidden.py`) decides this; until it reports, the
   claim is: *the head's chronological signal is not linearly reducible to
   find-spot, period, object type or length, while the frozen probe's largely
   is.*

Sources: `tier0/ladder/head_scores/*.parquet` (C5), `tier0/ladder/*_ridge.md`
(C4), `scripts/aggregate_head_ladder.py`.
