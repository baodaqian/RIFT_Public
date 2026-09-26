# Experiment scale guide

Current spatial settings: [selected scene budget](docs/SCENE_BUDGET.md). **Resource tables below describe the previous scene budgets and are stale for changed RIFT-native, SpINR, GeRaF, RadarSplat and SE settings.** They must not be used as current fit/runtime predictions; RF settings are unchanged.

**2026-09-20: projected 1t1r workload, one model per GPU.** Collection means
**one object, 2400 train / 1000 validation views**, using source Tx 0 / Rx 0 and
all 600 frequencies. GOTCHA means **eight-pass Camry HH, 1500 train / 440
validation sectors**, retaining all **177605 / 52100 native pulses**. Six
collection objects require six independent runs.

The [antenna integration](docs/ANTENNA_SELECTION.md) is implemented; its resource
use remains unmeasured. These estimates retain that implementation's models and
selected workloads. Use [the NCSA handoff](NCSA_Delta_Production_Handoff.md) for setup,
identities and execution gates. The manager owns profiling and training.

## Epoch meaning and fixed budgets

| Method | Collection / GOTCHA training unit | Maintained budget |
| --- | --- | --- |
| RIFT | 2400 / 1500 optimizer updates | 150 epochs; G48 / G48 initially, adaptive capacity 262144 |
| SpINR | 600 four-view / 174 up-to-1024-pulse updates | 150 epochs/150-epoch cosine; G48 midpoint on both datasets |
| SE Stage 1 | One outer iteration over all occupied groups, up to 72 | 150 iterations; G40 on both datasets; then 5000 SDF updates if gates pass |
| Radar Fields | 240 / 150 ten-frame updates | 4 epochs/960 updates / 6 epochs/900 updates |
| GeRaF | 2400 / 1500 updates per exposure-equivalent epoch | 50000 updates; 48³ lazy targets; one collection bank / two native banks |
| RadarSplat | 2400 / 1500 updates per nominal epoch | 112000 Gaussians; 2000 updates: 0.833 / 1.333 nominal epochs |

GeRaF and RadarSplat sample views; their epoch units do not guarantee every
view was visited exactly once. SE Stage 2 has no dataset epoch. No optimizer
budget, grid or neural network shrinks merely because one antenna pair is used.

## A100 40 GB: time and GPU space per training unit

**Uncalibrated warm-core scenarios**, excluding preparation, validation,
checkpoint I/O and unmodeled CPU/refinement work. RIFT/SpINR/SE estimates count
physics, neural work where applicable and dispatch; SE's seconds are only its
conditional Stage-1 GPU/dispatch component. GeRaF includes an assumed neural/
host service term and worst-case lazy-corner work. Real wall time can fall
outside these ranges. RIFT's estimate covers renderer/gradient traversals;
untimed SH evaluation and optimizer work must also be added. GPU envelopes are
sizing hypotheses, not measured peaks.

| Method | Collection training epoch | GOTCHA training epoch | Modeled GPU space: collection / GOTCHA |
| --- | --- | --- | --- |
| RIFT | 0.47–1.5 min | 25–74 min | 1.23 / 11.23 GiB at full capacity |
| SpINR | 3.1–9.5 h | 15–45 h | 1.63 / 28.50 GiB |
| SE Stage 1 | 2.5–7.8 s | 2.5–7.7 h | 1.02 / 1.09 GiB; Stage-2 peak unestimated |
| Radar Fields | 0.6–4 min | 2.8–22 min | 2.01 / 4.53 GiB |
| GeRaF | 0.36–4.1 h | 1.5–6.4 h | 8.05 GiB / 22.24 GB |
| RadarSplat | 4–80 min | 1.2–25 min | 1 GiB + 256 bytes × total incidences |

RIFT times above use initial point counts. At full capacity, the A100 estimates
are **0.55–1.7 min / 3.1–9.2 h** per epoch.
GeRaF's collection estimate is dominated by its explicitly assumed 0.5–6 seconds
of neural/host work per update; it is not a fitted timing prediction.

**Why savings differ:** SpINR still evaluates its 7.08M-point collection neural
field; RF still samples 100 profiles with replacement, repeating the sole pair;
RadarSplat still fits 20000 Gaussians to the same raster. Their fixed work and
model memory do not shrink 256-fold. GOTCHA was already one physical pair per
pulse: its reduction comes only from the selected training sectors. Full
assumptions, seven-card timings and validation costs are in the
[parallelization guide](parallization_guide.md).

## GPU tile candidates

Reserve 20% of actual usable VRAM and preserve numerical precision and optimizer
batches. These are candidate execution settings, not automatic CLI defaults.
GeRaF native envelopes use decimal GB; other displayed envelopes use binary GiB.
GeRaF/RadarSplat capacity budgets use decimal device GB; RIFT/SpINR use nominal
GiB, following the original models. RadarSplat's incidence ceiling assumes six equally
sized product graphs; actual overlap and sort workspace must be measured.

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

SpINR native rendering still requests 4096 points but clamps to 2416–2473 per
pulse. RIFT/SpINR native pulses remain serial. SE uses point/pair requests
4096/1; Stage 2 retains three 2048-point batches and needs its own memory check.
RF V100 is blocked by the current original backend, not a lack of free memory.
Use the [documented configuration routes](parallization_guide.md#applying-settings)
and new matching recipe/output identities.

## Host memory, disk caches and checkpoints

**Space is peak resident/retained space, not a fresh allocation each epoch.**
Saves overwrite fixed filenames; add one temporary for atomic writes. The table
uses **decimal MB**. Host values count identified caches/banks only; Python,
model/optimizer, metadata, staging and serialization copies are additional.
Zero means streamed/no persistent cache, not zero process RAM.

| Model / dataset | Host payload MB | Disk cache MB | Checkpoint MB × files | Retained MB | Cache + atomic peak MB |
| --- | --- | --- | --- | --- | --- |
| RIFT / Collection | 16.32 | 0.00 | 140.5 × 3 | 421.5 | 562.0 |
| RIFT / GOTCHA | 0.00 | 0.00 | 141.6 × 3 | 424.7 | 566.2 |
| SpINR / Collection | 16.32 | 0.00 | 44.0 × 4 | 177.2 | 221.2 |
| SpINR / GOTCHA | 0.00 | 0.00 | 81.8 × 3 | 389.0 | 532.7 |
| SE / Collection | 0.79 | 0.00 | 28.3 × 2 | 58.7 | 87.0 |
| SE / GOTCHA | 170.70 | 0.00 | 202.4 × 2 | 496.0 | 698.4 |
| Radar Fields / Collection | 0.00 | 0.20 | 210.0 × 3 | 630.0 | 840.2 |
| Radar Fields / GOTCHA | 0.00 | 0.00 | 210.0 × 3 | 630.0 | 840.0 |
| GeRaF / Collection | 23.04 | 4.12 | 60.0 × 2 | 120.0 | 184.1 |
| GeRaF / GOTCHA | 1218.43 | 4.12 | 1240.0 × 2 | 2480.0 | 3724.1 |
| RadarSplat / Collection | 30.00 | 29.00 | 35.0 × 2 | 70.0 | 134.0 |
| RadarSplat / GOTCHA | 90.00 | 47.00 | 35.0 × 2 | 70.0 | 152.0 |

SpINR history-storage estimates retain the historical 1500-epoch assumption
(current training budget is 150 epochs); native history was estimated at 1500 epochs
by scaling the parent serialization inventory to 1500 sectors. SE totals include
its selected Stage-1 artifact and remain conditional on its initialization gate.
RF uses the existing conservative 210 MB/file allowance. GeRaF's banks alone are
**23.04 MB / 1218.43 MB**, with 60 MB / 1.24 GB checkpoint allowances including
model/Adam and metadata. Its one accumulated 101³ grid is **4.12 MB**; preparation
atomic writes temporarily need **8.25 MB**, with no per-view cube files.

Practical artifact reservations, excluding sources/logs/exports: **1 GB/run**
for RIFT, SpINR, RF or SE; **0.25 GB / 4 GB** for collection/native GeRaF;
**0.2 GB** for RadarSplat. Reserve additional host RAM for checkpoint staging.
Original collection archives stay approximately **12.3 GB/object**: selecting
one pair does not reduce transferred source files.

GeRaF can rewrite roughly **1.44 GB / 18.6 GB per exposure epoch** after banks
are populated (latest every 100 updates), plus best/terminal saves. This is I/O
traffic, not accumulating disk usage. Other checkpoint/history I/O is also
excluded from the timing table. Extra polarizations and independent runs add
separate fields/caches; do not pool objects.

## Reproduce and qualify

```bash
python scripts/estimate_1t1r.py
python scripts/estimate_1t1r.py --json
# Optional: existing RIFT environment and local datasets; metadata only.
python scripts/estimate_1t1r.py --verify-metadata
```

The [calculator](scripts/estimate_1t1r.py) uses exact selected metadata counts and
explicit unmeasured cost assumptions. Historical calculators and
[downsampling scenarios](downsampling_guide.md) retain the original 16t16r
reference; do not mix them with this 1t1r table. No training, conversion or
scheduler action was performed for these estimates. Qualify antenna/recipe
parity, complete-update peaks and end-to-end timing before requesting full runs.
