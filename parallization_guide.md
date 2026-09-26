# GPU parallelization guide

Current spatial settings: [selected scene budget](docs/SCENE_BUDGET.md). **Resource tables below describe the previous scene budgets and are stale for changed RIFT-native, SpINR, GeRaF, RadarSplat and SE settings.** They must not be used as current fit/runtime predictions; RF settings are unchanged.

**2026-09-20: estimates after 1t1r integration, at 2400 / 1500 training
viewpoints.** One full GPU runs one independent object/method. These are
uncalibrated cost and memory scenarios, not measured runtime bounds, fit
certification or permission to run experiments. See [scale_guide.md](scale_guide.md)
for the short A100/storage reference and [the NCSA handoff](NCSA_Delta_Production_Handoff.md)
for setup and remaining method/acquisition gates.

## Workload and epoch definitions

| Quantity | Collection, one object | GOTCHA, Camry HH, passes 1–8 |
| --- | ---: | ---: |
| Train / validation viewpoints | 2400 / 1000 | 1500 / 440 sectors |
| Physical Tx / Rx | Source indices 0 / 0 | Native monostatic pair per pulse |
| Train / validation complex samples | 1440000 / 600000 | 76151564 / 22338864 |
| Train / validation native pulses | Not the collection viewpoint unit | 177605 / 52100 |
| RIFT updates/epoch | 2400 | 1500 |
| SpINR updates/epoch | 600 × four views | 174 × up to 1024 pulses |
| RF updates/epoch | 240 × ten frames | 150 × ten sectors |
| GeRaF exposure-equivalent epoch | 2400 sampled-view updates | 1500 sampled-sector updates |
| RadarSplat nominal epoch | 2400 sampled-view updates | 1500 sampled-sector updates |

Holdout IDs, all selected native frequencies/pulses and scene sizes stay fixed.
GOTCHA does not get an extra 256-fold reduction: it was already one pair per
pulse. Collection responses shrink 256-fold relative to 2400-view 16t16r;
neural, optimizer, raster and fixed CPU work do not.

Budgets: RIFT 150 epochs; SpINR 150 (150-epoch cosine, selected 2026-09-21); RF **960/900 updates**; GeRaF 50000;
RadarSplat 2000. A collection RadarSplat epoch below is an extrapolation beyond
its 2000-update recipe. Native 2000 updates are 1.333 nominal epochs. SE Stage 1
uses 150 outer iterations across its occupied groups (up to 72), not 72 full
passes per iteration; Stage 2 is a separate 5000-update SDF fit. All SE time
estimates are conditional: initialization/convergence gates remain unresolved.

## Parallelization settings

Use 20% VRAM reserve. Candidate tiles preserve model size, samples, precision and
optimizer batches; they are not a throughput guarantee or automatic defaults.
Both guides share the following table. GB is decimal, GiB binary; use actual
available device bytes when qualifying a run.

| Setting / modeled envelope | V100 16 GB | A30 24 GB | A100 40 GB | A40 48 GB | L40S 48 GB | H100 80 GB SXM | H200 141 GB SXM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RIFT collection point / pair tile | 262144 / 1 | 262144 / 1 | 262144 / 1 | 262144 / 1 | 262144 / 1 | 262144 / 1 | 262144 / 1 |
| RIFT collection envelope, GiB | 1.234 | 1.234 | 1.234 | 1.234 | 1.234 | 1.234 | 1.234 |
| RIFT native point tile / envelope, GiB | 262144 / 11.234 | 262144 / 11.234 | 262144 / 11.234 | 262144 / 11.234 | 262144 / 11.234 | 262144 / 11.234 | 262144 / 11.234 |
| SpINR collection neural / physics / pair tiles | 4096 / 65536 / 1 | 4096 / 65536 / 1 | 4096 / 65536 / 1 | 4096 / 65536 / 1 | 4096 / 65536 / 1 | 4096 / 65536 / 1 | 4096 / 65536 / 1 |
| SpINR collection neural envelope, GiB | 1.625 | 1.625 | 1.625 | 1.625 | 1.625 | 1.625 | 1.625 |
| SpINR native neural tile | 368,640 | 577,536 | 884,736 | 884,736 | 884,736 | 884,736 | 884,736 |
| SpINR native neural envelope, GiB | 12.750 | 19.125 | 28.500 | 28.500 | 28.500 | 28.500 | 28.500 |
| GeRaF collection point / pair tile; envelope GiB | 65536 / 1; 8.049 | 65536 / 1; 8.049 | 65536 / 1; 8.049 | 65536 / 1; 8.049 | 65536 / 1; 8.049 | 65536 / 1; 8.049 | 65536 / 1; 8.049 |
| GeRaF native point tile; pair request 120 | 512 | 1,024 | 2,048 | 4,096 | 4,096 | 8,192 | 15,360 |
| GeRaF native working envelope, decimal GB | 12.00 | 15.42 | 22.24 | 35.89 | 35.89 | 63.20 | 110.98 |
| RF collection / native envelope, GiB | Blocked | 2.01 / 4.53 | 2.01 / 4.53 | 2.01 / 4.53 | 2.01 / 4.53 | 2.01 / 4.53 | 2.01 / 4.53 |
| RadarSplat allowed incidences/product, millions | 7.63 | 11.80 | 20.13 | 24.30 | 24.30 | 40.97 | 72.74 |
| SE Stage-1 collection / native envelope, GiB | 1.018 / 1.094 | 1.018 / 1.094 | 1.018 / 1.094 | 1.018 / 1.094 | 1.018 / 1.094 | 1.018 / 1.094 | 1.018 / 1.094 |

Actual work and remaining limits:

- **RIFT:** collection initial G48 = 110592 points; native G32 = 32768; capacity
  262144, SH degree 0→3. One pair lets every listed GPU's modeled collection
  tile cover the full scene in one block. Each range block still performs 20
  serial gridding offsets. Native pulses render sequentially within a sector
  update; capacity is one spatial block per pulse with the suggested tile.
- **SpINR:** collection G96/GL2 = 7077888 points; native G48/GL2 = 884736.
  Keep FP32 neural evaluation and FP64 physics. Collection neural tile 4096
  gives 1728 calls for field evaluation and again for replay/backward.
  Tx0/Rx0 selects 8–13 scene bins/view, so `min(65536,1048576/bin_count)` is
  now **65536**: **108 renderer blocks/view**, 259200/train traversal and
  108000/validation traversal. Its field adjoint repeats the train traversal.
  Four views/update remain serial. Native rendering is also serial; its 4096
  request clamps to 2416–2473 points/pulse, giving **64362635** forward blocks
  per selected train epoch, repeated for the adjoint. The larger neural tiles
  do not bypass this physics cap.
- **RF:** ten frames × 100 profiles × ten rays/bin remain one combined
  TCNN/BatchNorm update. Sampling with replacement repeats the collection's
  sole pair, leaving at most 130000 / 770000 candidate neural queries.
  `query_chunk` affects evaluation only (65536 / 131072 candidates); increasing
  it cannot enlarge training batches. The current V100/SM70 backend remains
  blocked; no substitute implementation is assumed.
- **GeRaF:** one view/update, collection **one bank**, native **two banks**.
  The unchanged source network uses up to 807 rays × 64 = **51648** target
  points. Collection now has one Tx group/pair; point request 65536 covers the
  trace in one block. The **101³ lazy target** may evaluate up to **413184**
  lattice corners/view before deduplication: seven collection blocks at 65536.
  Native uses up to 60 active-bank / 120 full-sector pulses; point tiles also
  apply to lazy target work. The implemented single-bank adaptation has separate
  gradients, empty-bank and recovery gates; see [antenna selection](docs/ANTENNA_SELECTION.md).
- **RadarSplat:** one raster/update, 20000 Gaussians, 16×16 tiles,
  `batch_per_iter=100`, six product graphs. Selected antennas alter MF target
  construction, not the number of Gaussians or target lattice. Warm fitting
  remains overlap-dependent; it receives no assumed 256-fold speedup.
- **SE:** collection Stage 1 now needs **one** 343-point/600-frequency block
  per view, versus 51 for the former multi-pair candidate. Native remains
  30–31 blocks/pulse, **5483833** per train traversal. Keep 4096/1 point/pair
  requests. Stage 2's three 2048-point sampling batches are unchanged.

P allocated full GPUs allow up to P independent runs with separate outputs
and enough host memory/I/O. No same-GPU colocation or automatic multi-GPU
training speedup is included. The adaptive/default engines remain single-process.

### Memory arithmetic

These inherited coefficients are deliberately conservative **hypotheses**, not
measured saved-tensor bounds. Let G = 2³⁰ bytes, T be a point tile, K a neural
tile, F ≤ 434 native frequencies, P ≤ 120 pulses, and Q candidate neural queries:

```text
RIFT collection: G + 960*T*1                  (T <= 262144)
RIFT native:     G + (96*F+256)*T
SpINR neural:    1.5*G + 32768*K              (quadrature/workspace allowance included)
RF:             1.5*G + (4096+128)*Q          (Q <= 130000 / 770000)
GeRaF collection:8*G + 1024*min(T,51648)*1
GeRaF native:   8*G + 128*T*P*F
RadarSplat:     G + 256*sum(incidences across six product graphs)
SE Stage 1:     G + 96*live_phase_entries     (collection 343*600; native <= 1048576)
```

GeRaF retains the 8-GiB higher-order neural-graph allowance rather than shrinking
it by antenna count; its no-grad lazy MF workspace at T=65536 is smaller than
that modeled training envelope. Native GeRaF uses the prior decimal-GB budget
`M <= 0.8*C*10^9`, as does RadarSplat; RIFT/SpINR use nominal GiB. RadarSplat's
1-GiB allowance includes ordinary sort/projection scratch, but extreme overlap
can require more. SE Stage 2 needs separate derivative-graph qualification.

**Host and disk are separate.** Selected collection RIFT/SpINR response caches
are `(2400+1000)*600*8 = 16.32 MB`, not the old 4.81 GiB. GeRaF CPU response
banks are `16*train_complex_samples`: **23.04 MB / 1.218 GB**, plus containers
and checkpoint staging. Its 101³ FP32 accumulator is **4.12 MB**, with an
8.25-MB atomic preparation peak; no per-view cubes. Network/Adam checkpoints,
RF sampled-query graphs, RadarSplat rasters and SE field banks retain their
model-dependent sizes. Full per-run storage and save-temporary budgets are in
[the scale guide](scale_guide.md#host-memory-disk-caches-and-checkpoints).
Memory is not multiplied by epoch count; native SpINR histories grow with the
budget, and GeRaF banks grow as distinct training views are encountered.

### Applying settings

All tile changes require matching new recipe/cache/checkpoint identities.
Inspect generated commands and retain canonical manifests and antenna controls.
Collection dispatchers do not expose every method's tile flag; do not execute
a printed trainer command before its planned manifest exists.

| Method | Collection control | GOTCHA control |
| --- | --- | --- |
| RIFT | Underlying `train.py --point-chunk 262144 --pair-chunk 1` | Shared frontend `--point-chunk 262144` |
| SpINR | Maintained neural/renderer/pair flags: 4096/65536/1 | `--method-config`, key `spinr`; per-card neural tile, renderer 4096 |
| RF | `train_radar_fields.py --query-chunk 65536` | `radar_fields` config: query_chunk 131072 |
| GeRaF | `--geraf-source-config`: point_chunk 65536, pair_chunk 1, qualified single-bank recipe | `geraf` config: per-card point_chunk, pair_chunk 120; two banks |
| RadarSplat | Preparation tiles only; original fitting recipe | `radarsplat` config; no training batch enlargement |
| SE | `--se-config`: point_chunk 4096, pair_chunk 1 | `sugavanam_ertin` config: same; native pulses stay serial |

Native SpINR G48 uses `protocols/gotcha_spinr_g48.json`; combine its model
settings with the chosen execution tiles in one method mapping. Resuming an
old G96 model requires its original grid and matching original identity.

## Estimated training time per epoch

**Warm-core scenarios, not end-to-end job limits.** Preparation, data streaming,
checkpoint writes, refinement and other unmodeled CPU work are extra.
RIFT covers rendering/gradient traversals; SH evaluation and optimizer work are
additional untimed terms. Especially for small collection RIFT/SE workloads,
excluded work can dominate.
GeRaF includes its stated 0.5–6 s/update neural/host term and lazy-corner work;
its entries supersede the old dense-target timing model. SE's row is one
conditional outer iteration's GPU/dispatch component. **—** means no inherited
RF/RadarSplat service-rate scenario exists for that GPU; memory fit alone does
not supply a runtime estimate.

| Model / dataset | V100 16 GB | A30 24 GB | A100 40 GB | A40 48 GB | L40S 48 GB | H100 80 GB SXM | H200 141 GB SXM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RIFT / Collection | 0.51–1.6 min | 0.51–1.6 min | 0.47–1.5 min | 0.61–2.3 min | 0.52–1.7 min | 0.44–1.4 min | 0.43–1.3 min |
| RIFT / GOTCHA | 0.69–2.1 h | 40–120 min | 25–74 min | 3.3–16 h | 1.3–6.7 h | 13–38 min | 9.4–28 min |
| SpINR / Collection | 4.3–13 h | 5.8–17 h | 3.1–9.5 h | 2.1–7 h | 0.98–3.3 h | 1.1–3.3 h | 1.1–3.2 h |
| SpINR / GOTCHA | 20–62 h | 20–61 h | 15–45 h | 73–350 h | 34–160 h | 10–32 h | 9.4–29 h |
| SE Stage 1 / Collection | 2.8–8.6 s | 2.7–8.5 s | 2.5–7.8 s | 5.2–22 s | 3.4–13 s | 2.3–7.2 s | 2.3–7.1 s |
| SE Stage 1 / GOTCHA | 3.3–10 h | 3.2–9.9 h | 2.5–7.7 h | 11–52 h | 5.3–24 h | 1.9–5.8 h | 1.7–5.4 h |
| Radar Fields / Collection | Blocked | — | 0.6–4 min | — | 0.6–4 min | 0.4–3.2 min | — |
| Radar Fields / GOTCHA | Blocked | — | 2.8–22 min | — | 2.8–25 min | 2.5–21 min | — |
| GeRaF / Collection | 0.36–4.1 h | 0.36–4.1 h | 0.36–4.1 h | 0.37–4.1 h | 0.36–4.1 h | 0.36–4.1 h | 0.36–4.1 h |
| GeRaF / GOTCHA | 2.5–9.4 h | 2.4–9.1 h | 1.5–6.4 h | 11–58 h | 4.7–25 h | 0.81–4.3 h | 0.63–3.8 h |
| RadarSplat / Collection | 0.13–2.7 h | — | 4–80 min | — | 3.2–80 min | 2.4–60 min | — |
| RadarSplat / GOTCHA | 2.5–50 min | — | 1.2–25 min | — | 1–25 min | 0.75–20 min | — |

RIFT adaptive point counts change during training. Initial and full-capacity
endpoints below bracket scene-size assumptions, not all possible wall times:

| RIFT point count / dataset | V100 16 GB | A30 24 GB | A100 40 GB | A40 48 GB | L40S 48 GB | H100 80 GB SXM | H200 141 GB SXM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Initial / Collection | 0.51–1.6 min | 0.51–1.6 min | 0.47–1.5 min | 0.61–2.3 min | 0.52–1.7 min | 0.44–1.4 min | 0.43–1.3 min |
| Initial / GOTCHA | 0.69–2.1 h | 40–120 min | 25–74 min | 3.3–16 h | 1.3–6.7 h | 13–38 min | 9.4–28 min |
| Capacity 262144 / Collection | 0.66–2 min | 0.65–2 min | 0.55–1.7 min | 0.88–3.6 min | 0.67–2.2 min | 0.48–1.5 min | 0.46–1.4 min |
| Capacity 262144 / GOTCHA | 5.3–16 h | 5.1–15 h | 3.1–9.2 h | 26–130 h | 11–53 h | 1.4–4.3 h | 1–3 h |

RF retains 100 replacement-sampled profiles even with one pair. RadarSplat
retains original raster service assumptions; changing the target can change
Gaussian overlap and runtime. Neither service model is a measured GPU ranking.

## Validation and SE Stage 2

Add validation separately: RIFT/RF every epoch; SpINR every five epochs;
SE Stage 1 every ten iterations; GeRaF every 1000 updates (two or three events
per 2400-update exposure unit; one or two per 1500-update unit); RadarSplat at
its fixed terminal budget. Include final validation where required.

| Model / dataset | V100 16 GB | A30 24 GB | A100 40 GB | A40 48 GB | L40S 48 GB | H100 80 GB SXM | H200 141 GB SXM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RIFT / Collection | 4–13 s | 4–12 s | 3.7–11 s | 4.8–18 s | 4.1–13 s | 3.4–11 s | 3.4–10 s |
| RIFT / GOTCHA | 2.3–6.9 min | 2.2–6.6 min | 1.4–4.1 min | 11–54 min | 4.5–22 min | 0.69–2.1 min | 0.52–1.6 min |
| SpINR / Collection | 10–31 min | 10–31 min | 9–28 min | 15–59 min | 11–38 min | 8.2–25 min | 8–25 min |
| SpINR / GOTCHA | 1.6–5 h | 1.6–4.9 h | 1.2–3.8 h | 5.5–26 h | 2.6–12 h | 0.93–2.9 h | 0.85–2.6 h |
| SE Stage 1 / Collection | 0.16–0.51 s | 0.16–0.51 s | 0.15–0.46 s | 0.31–1.3 s | 0.2–0.77 s | 0.14–0.43 s | 0.13–0.42 s |
| SE Stage 1 / GOTCHA | 8.3–25 min | 8.2–25 min | 6.3–19 min | 0.46–2.2 h | 13–60 min | 4.7–15 min | 4.4–13 min |
| Radar Fields / Collection | Blocked | — | 0.42–2.5 min | — | 0.42–3 min | 0.33–2 min | — |
| Radar Fields / GOTCHA | Blocked | — | 0.88–7.1 min | — | 1–8.8 min | 0.7–6.2 min | — |
| GeRaF / Collection | 8.9–100 min | 8.9–100 min | 8.9–100 min | 8.9–100 min | 8.9–100 min | 8.8–100 min | 8.8–100 min |
| GeRaF / GOTCHA | 0.65–2.5 h | 0.62–2.4 h | 24–100 min | 2.9–15 h | 1.2–6.5 h | 13–72 min | 10–63 min |
| RadarSplat / Collection | 1.7–33 min | — | 0.83–17 min | — | 0.67–17 min | 0.5–12 min | — |
| RadarSplat / GOTCHA | 0.37–7.3 min | — | 0.18–3.7 min | — | 0.15–3.7 min | 0.11–2.9 min | — |

SpINR collection includes its exact first-128-training-view diagnostic and both
full-response and selected-bin evaluation. RIFT validation uses initial points.
RF collection validation retains the old whole-frame service allowance rather
than assuming that fixed per-frame overhead vanishes with 255 removed pairs.
Native validation workload is unchanged.

SE uses seven forward-equivalents per Stage-1 iteration: two value/gradient
calls at three equivalents each, plus one line-search value call. Each extra
full-role line-search value sweep adds approximately one seventh of its row;
weight partial sweeps by group workload. CPU solver/read costs are excluded.
Stage 2's unchanged 5000-update GPU/dispatch component on A100 is approximately
**1.8–5.3 min** for either dataset, excluding CPU neighbourhood/iso refresh,
I/O and export; there is no Stage-2 dataset epoch and the gate still blocks fitting.

## Preparation and per-epoch I/O

Cold costs must be added to a fresh run:

| Method | Additional work after antenna/subset selection |
| --- | --- |
| RIFT / SpINR / SE | Train-only normalization/initialization, streaming, refinement or solver CPU work |
| RF | 2400 selected one-pair CPU frame transforms / 177605 native training pulse preparations; original neural batch unchanged |
| GeRaF | Accumulate 101³ train-only lattice per head; recompute lazy per-view corners at fitting/validation; cold response-bank initialization |
| RadarSplat | Rebuild MF-power targets, 11-donor occupancy and background/multipath fits from selected training data; retain raster dimensions |

GeRaF preparation arithmetic: collection `101³*2400*1*20` range-tap entries;
native `101³*76151564` direct-phase entries. Its one grid can be rewritten after
each training viewpoint: **9.89 GB / 6.18 GB cumulative grid-write traffic**,
while only ~8.25 MB of grid payload need coexist during preparation.
With populated banks, latest checkpoints every 100 updates can write about
**24*60 MB = 1.44 GB / 15*1.24 GB = 18.6 GB per exposure epoch**, plus best and
terminal saves. Add measured write time; do not count this traffic as retained
storage. RIFT/SpINR/SE/RF/RadarSplat checkpoints and histories also add I/O.

Suggested no-grad collection preparation tiles: GeRaF 65536 points/one pair
(~1.031 GiB under `G+512*T`); RadarSplat full 33³ = 35937 points/one pair
(~1.017 GiB). Native preparation retains ragged frequencies, coherent pulse
sums and the prior per-card limits; no extra antenna speedup is assumed.
GeRaF shares its point tile between preparation and fitting, so choose it before
creating the identity-bound run. No preparation or conversion was executed here.

## Assumptions and reproduction

```bash
python scripts/estimate_1t1r.py
python scripts/estimate_1t1r.py --json
# Requires local data and the existing RIFT environment; headers/geometry only.
python scripts/estimate_1t1r.py --verify-metadata
```

The [calculator](scripts/estimate_1t1r.py) checks selected metadata without radar
response access. All six collection objects share the checked Tx0/Rx0 geometry.
Collection train/validation selected-bin totals are **26371 / 11013**, plus
**1407** bins for the first-128-view train diagnostic. Native per-pass pulse
and frequency counts are explicit in the calculator; no proportional pulse
approximation or frequency resampling is used.

RIFT/SpINR/SE inherit the original arithmetic cost model:

```text
physics = max(entries*bytes_per_entry / effective_bandwidth,
              entries*equivalent_FP64_ops / effective_FP64_rate)
          + block_count*launch_equivalents*launch_seconds
```

Assumptions are 20–60% bandwidth, 5–25% scalar FP64, 20–60% FP32 GEMM use and
8–25 microseconds per launch-equivalent. Range uses 4096 bytes/2048 operations/
400 launches per point-pair/block; direct bins 512/512/60; native phase
96/128/16; native phase-plus-FFT 192/256/24. These are work proxies, not resident
memory. SpINR adds one field evaluation, replay and approximately two-forward
backward work using 3561600 dense weights.

**New GeRaF scenario:** keep at most 51648 trace points and **413184** MF corners
per view, before mask/dedup savings. Use five range forward-equivalents per
collection trace point (single-bank trace/replay/backward plus predicted MF/
adjoint), or 3.5 native equivalents (half-bank trace plus full predicted MF).
Add one full-channel MF per lazy corner; validation uses two prediction
traversals plus those corners. Then add **0.5–6 s/view** for neural, interpolation
and host service on every GPU, inherited as an explicit assumption rather than
a hardware-scaled observation. This deliberately broad upper-work scenario
must be replaced by measured corner counts and complete-update timings.

RF/RadarSplat retain their four-card unmeasured seconds/update assumptions in
[the baseline calculator](scripts/estimate_rf_geraf_radarsplat.py), with the
new update counts. A30/A40/H200 service times remain unestimated. Original
16t16r calculators and [downsampling scenarios](downsampling_guide.md) remain
historical references; their table counts are not the current workload.

Frozen hardware inputs are 900/933/1555/696/864/3350/4800 GB/s bandwidth for
V100/A30/A100/A40/L40S/H100-SXM/H200-SXM; ordinary FP64 assumptions are
7/5.2/9.7/0.584/1.431/34/34 TFLOPS. A40/L40S FP64 is inferred as FP32/64;
Tensor Core peaks do not substitute for these physics kernels. Sources retained
from the original audit: [V100](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf),
[A30](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/products/a30-gpu/pdf/a30-datasheet.pdf),
[A100](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/),
[A40](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a40/proviz-print-nvidia-a40-datasheet-us-nvidia-1469711-r8-web.pdf),
[L40S](https://www.nvidia.com/en-us/data-center/l40s/),
[H100](https://www.nvidia.com/en-us/data-center/h100/),
[H200](https://www.nvidia.com/en-us/data-center/h200/), and the
[CUDA architecture guide](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html#compute-capability-8-x).
H200 NVL's different FP64 rate requires separate inputs. Build/check the original
CUDA extensions for the actual device; free VRAM is not backend qualification.

Before production, verify antenna/model parity, then measure cold/warm update
peaks, preparation, validation and checkpoint I/O under manager authorization.
No training, GPU benchmark or scheduler submission was performed for this guide.
