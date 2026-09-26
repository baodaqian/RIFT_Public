# Training-viewpoint downsampling: time calculation

2026-09-20. What-if analysis continuing [scale_guide.md](scale_guide.md) and
[parallization_guide.md](parallization_guide.md). The user subsequently selected
**2400 collection / 1500 GOTCHA** for Delta; the deterministic selection and
receiving-agent implementation requirements are in
[the NCSA handoff](NCSA_Delta_Production_Handoff.md#0-selected-delta-training-subsets--2400--1500).
Subset integration still requires the handoff's validation gates; this
calculation authorizes no jobs.

The subsequent acquisition decision is **tunable Tx/Rx counts, first collection
1t1r (Tx 0 / Rx 0)**; see the handoff for the pending implementation and GeRaF
one-bank adaptation. This guide isolates viewpoint downsampling at the original
**16t16r collection** acquisition. Its "all samples" and timing statements below
refer to that earlier calculation, not new 1t1r measurements. Native GOTCHA
already has one pair per pulse and retains every pulse/frequency in each sector.

The paired sizes 1600/1000, 2000/1250, 2400/1500 and 2800/1750 retain
50%, 62.5%, 75% and 87.5% of the original collection/GOTCHA training roles.
Keep validation at 1000/440, reserved test at 1000/440, and all samples within
selected views/sectors. Collection figures are for one object; six independent
objects multiply collection GPU-hours by six.

## What actually gets faster

| Method | Effect of retaining fraction f of training viewpoints |
| --- | --- |
| RIFT | Keep 150 epochs: f times the sector/view updates; approximately f times warm training work at the same point count. Validation still runs 150 times. |
| SpINR | Keep 1500 epochs: approximately f times training work. Collection has N/4 updates/epoch; GOTCHA has ceil(selected native pulses/1024). Validation still runs 300 times. |
| SE Stage 1 | Keep 150 outer iterations: approximately f times data-dependent solver work if the occupied groups and line-search behavior are comparable. Validation still runs 15 times. Initialization remains gated. |
| SE Stage 2 | Keep 5000 updates: no proportional speedup from fewer viewpoints. CPU cloud processing may change. |
| Radar Fields | Ten frames/update; round the 800-update target upward to complete epochs. See the budget table below. |
| GeRaF | Keep 50000 updates: warm computation is not reduced proportionally. Training-only accumulated-target preparation and response-bank checkpoint I/O shrink; these can matter substantially to total time. Lazy-target runtime still needs profiling. |
| RadarSplat | Keep 2000 updates: warm rendering/fitting time is approximately unchanged. Training-target conversion shrinks; validation-target conversion and final validation stay fixed. |

Fewer viewpoints do not make an individual spatial/physics tile cheaper or
justify changing the guide's GPU tiles, precision, model size or optimizer batch.
Train-dependent normalization and preparation must be recomputed for each subset.
Angular distribution, adaptive growth, SE line search and data-dependent masks
can change per-update time, so f is a work estimate, not a measured speedup.

## A100 40 GB full-budget scenarios

The original guide's RIFT/SpINR rows are **per epoch**, not full runs.
The following values multiply by 150/1500 epochs and add scheduled validation.
They inherit the existing calculators' **uncalibrated timing assumptions**;
they are not benchmarks, confidence intervals or scheduler wall-time requests.
Preparation, initialization, refinement overhead, checkpoint/I/O and unmodeled
CPU work are additional.

RIFT's two columns hold active points constant throughout the run: initial
G48/G32 or full capacity 262144. A real adaptive trajectory is unknown; these
are scenarios, not guaranteed runtime bounds.

| Collection training views | RIFT initial points, hours | RIFT at capacity, hours | SpINR G96/GL2, hours |
| --- | ---: | ---: | ---: |
| 3200, original | 59–179 | 140–422 | 62724–191427 |
| 1600 | 32–97 | 77–230 | 32941–100536 |
| 2000 | 39–118 | 93–278 | 40387–123258 |
| 2400 | 46–138 | 108–326 | 47833–145981 |
| 2800 | 53–158 | 124–374 | 55278–168704 |

| GOTCHA training pass-sectors | RIFT initial points, hours | RIFT at capacity, hours | SpINR G48/GL2, hours |
| --- | ---: | ---: | ---: |
| 2000, original | 86–258 | 637–1912 | 29718–90894 |
| 1000 | 45–134 | 331–994 | 15044–46016 |
| 1250 | 55–165 | 408–1223 | 18713–57236 |
| 1500 | 65–196 | 484–1453 | 22381–68455 |
| 1750 | 75–227 | 560–1682 | 26049–79675 |

GOTCHA rows use proportional pulse/frequency work from the original 236826
training pulses and 101543576 complex samples. Exact work and pulse-batch
ceilings require the actual subset manifest. Collection SpINR similarly uses
the original average geometry-dependent selected-bin workload.

At half the views, warm training plus validation drops by about **46% for
collection RIFT, 48% for collection SpINR, 48% for GOTCHA RIFT and 49% for GOTCHA
SpINR**. Preparation/I/O could change these end-to-end percentages.
The inherited SpINR scenarios remain many thousands of hours even after halving;
viewpoint reduction alone does not establish practical full-run feasibility.
An authorized short profile is needed before treating those scenarios as real
resource requirements.

Conditional SE Stage-1 totals on A100, including 15 validations:

| Retained fraction | Collection, hours | GOTCHA, hours |
| --- | ---: | ---: |
| 100% | 11.20–34.37 | 500.34–1535.24 |
| 50% | 5.63–17.26 | 250.95–770.02 |
| 62.5% | 7.02–21.54 | 313.30–961.33 |
| 75% | 8.41–25.82 | 375.64–1152.63 |
| 87.5% | 9.81–30.09 | 437.99–1343.93 |

These assume the guide's 72 occupied groups and accepted line-search scenario.
Selecting fewer shared azimuth sectors can empty groups; recompute the partition
from subset geometry. Add unchanged Stage-2 GPU/dispatch scenario of 1.8–5.3
minutes plus unmodeled CPU work, only if both scientific gates permit Stage 2.

## Radar Fields: complete-epoch rounding

For N views, B=N/10, E=ceil(800/B), and updates=B*E. This is the current
source-adapted-v3 rule, not a new fixed-epoch or fixed-960-update policy.

| Collection N | Epochs | Updates | Warm fitting time relative to original |
| --- | ---: | ---: | ---: |
| 3200 | 3 | 960 | 100% |
| 1600 | 5 | 800 | 83.3% |
| 2000 | 4 | 800 | 83.3% |
| 2400 | 4 | 960 | 100% |
| 2800 | 3 | 840 | 87.5% |

| GOTCHA N | Epochs | Updates | Warm fitting time relative to original |
| --- | ---: | ---: | ---: |
| 2000 | 4 | 800 | 100% |
| 1000 | 8 | 800 | 100% |
| 1250 | 7 | 875 | 109.4% |
| 1500 | 6 | 900 | 112.5% |
| 1750 | 5 | 875 | 109.4% |

The baseline A100 warm totals are 2.4–16 minutes for collection RF and
14.7–120 minutes for native RF. Multiply by the last column; preparation and
validation are extra. RF can spend **more** fitting time with fewer GOTCHA
sectors because of the epoch-rounding rule.

The guide describes RF validation every epoch, but code implements explicit
step intervals (collection 320, GOTCHA 200). Those are equivalent only at the
original dataset sizes. A subset implementation must declare whether to keep
the step interval or use N/10 for every-epoch validation. The latter yields the
epoch counts above as validation counts; the former yields ceil(updates/320)
or ceil(updates/200). Do not assume RF validation overhead scales with f.

RadarSplat's fixed 2000-update A100 warm totals remain approximately 3.3–66.7
minutes for collection and 1.7–33.3 minutes for GOTCHA. The original guide's
collection 3200-view epoch extrapolation exceeds that fixed budget. GeRaF has
no qualified total-time estimate for the 101-cubed lazy-target implementation.

## Subset setup implications

The production loaders fix their original registered splits. Every smaller
case needs a distinct subset identity, fresh matching caches/checkpoints and
frontend/readout support; changing a CLI count alone is insufficient. Preserve
the parent validation/test IDs and select only from the parent's training IDs.
Use the same nested, angularly distributed subsets across methods. A trained
full-data checkpoint would leak the omitted viewpoints into a from-scratch
subset comparison.

Eight-pass GOTCHA with identical selected sectors per pass requires N divisible
by eight. Of the requested smaller counts, only 1000 satisfies that rule.
Nearby shared-sector totals are 1248/1256, 1496/1504 and 1744/1752. Keeping RF's
complete ten-frame batches as well requires N divisible by 40: nearby choices
are 1240/1280, 1480/1520 and 1720/1760. Exact requested counts would require an
explicit change to shared training-sector selection across passes. No rounding
or selection policy is applied to the hypothetical tables here. The selected
Delta 1500-sector policy is specified separately in the NCSA handoff: 187 shared
sectors plus sector 307 in passes 3/4/5/8, giving exactly 177605 training pulses
and 76151564 complex samples. The proportional table values above remain
what-if approximations, not those exact metadata counts.

At 150 epochs, collection RIFT makes 240000/300000/360000/420000 updates and
retains its 15 epoch-based refinement events. Native RIFT makes
150000/187500/225000/262500 updates: with refinement every 100 updates, there
are 1500/1875/2250/2625 events instead of 3000. Keeping these schedules can
change the adaptive trajectory; exposure and checkpoint gates need a separate
subset contract rather than changes to the frozen full-data workflow.

GeRaF and RadarSplat revisit the smaller pool more often at fixed update
budgets. Increasing epochs to preserve the original number of RIFT/SpINR
updates would largely remove their training-time savings and increase
validation work. Keep this distinction explicit when interpreting any accuracy
change; quality cannot be calculated from sample count alone.

## Reproduce

The published base assumptions are in
[estimate_parallelization.py](scripts/estimate_parallelization.py) and
[estimate_rf_geraf_radarsplat.py](scripts/estimate_rf_geraf_radarsplat.py).
For RIFT, SpINR and SE Stage 1, compute
`total_seconds = f * epochs_or_iterations * original_training_unit_seconds + validation_events * original_validation_pass_seconds`.
Use 150/1500/150 training units and 150/300/15 validation events respectively.
For RF, use the complete-epoch update formula above and multiply by the assumed
seconds/update. RadarSplat retains 2000 updates.

The local-only helper `scripts/estimate_downsampling.py` is deliberately excluded
from GitHub. In the source workspace, `python3 scripts/estimate_downsampling.py
--gpu A100` prints work, storage and time tables. `--gpu V100`, `A30`, `A40`,
`L40S`, `H100` or `H200` uses the other hardware scenarios; `--json` includes
all seven. New-card RF/RadarSplat timings remain unestimated. This
calculator is arithmetic only and does not read responses, train, convert targets
or submit jobs.

The arithmetic uses the same source rules across all seven GPU scenarios. Current
SE initialization, V100/TCNN and other CUDA qualification gates remain unchanged.
