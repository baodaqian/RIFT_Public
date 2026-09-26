# Training recipes (PVC production campaign, September 2026)

This page records the training recipes that the production runs actually used. The values come
from each run's resolved plan (`plans/<task>.json` in its campaign root), its saved
`recipe.json`/checkpoint recipe, and the campaign's `source/` snapshot. The per-method pages
explain the fidelity choices and deviations behind each value: [SpINR](SPINR_ADAPTATION.md),
[GeRaF](GERAF_V1_HARDENING.md), [Radar Fields](RADAR_FIELDS_ADAPTATION.md) /
[PVC](RADAR_FIELDS_PVC_ADAPTATION.md), [RadarSplat](RADARSPLAT_FIDELITY.md) /
[PVC](RADARSPLAT_PVC_ADAPTATION.md) and [Sugavanam–Ertin](SUGAVANAM_ERTIN_PAPER.md).

**Runs versus the current tree.** Every job runs a file-copy snapshot of the tree taken when
its campaign root was prepared, so later commits never reach a running job. Where the current
tree holds a different recipe from the one a run used, this page says so, per scene. State
as of 2026-09-22, 20:00 CDT. Live job states are in the campaign ledgers and
`RIFT_PVC_Adaptation.md`, not here.

## 1. Compute and campaign roots

- **Hardware:** Intel Data Center GPU Max 1100 (ACES `pvc` partition). Each job gets one
  card, 8 CPU cores, 64 GB and a 24 h limit (`--qos=normal`). The one exception is SE
  Stage 1, which runs on `cpu`-partition nodes (32 cores, 80 GB, 3 h per job).
- **Software:** torch 2.12.1+xpu in the `RIFT-PVC` environment. `PYTORCH_DEBUG_XPU_FALLBACK=1`
  is set, and an "XPU to CPU fallback" warning counts as a defect.
- **Model code:** CUDA-only kernels are replaced by PVC re-implementations, recorded as
  separate backends (Radar Fields: tinycudann torch shim; RadarSplat: gsplat and fused-SSIM
  torch twins). Details are in the PVC pages above.

| Root (`/scratch/group/p.cis261724.000/RIFT_pvc_runs/…`) | Snapshot | Contents |
| --- | --- | --- |
| `production_20260921_35jobs` (C1) | 6635e22 + local edits | RIFT, SpINR, GeRaF on the six collection scenes; SE with the repository solver (superseded); Camry tasks superseded |
| `production_20260921_rf_rs_14jobs` (C2) | 6635e22 + local edits | Radar Fields and RadarSplat, all seven scenes |
| `production_20260922_camry_batched_4jobs` (C3) | ae8cf6e + local edits | Camry SpINR (batched lane); Camry RIFT and SE superseded or held |
| `production_20260922_camry_rift_sum2_1job` (C4) | 96afb8d + local edits | Camry RIFT with the sum2 amplitude (cancelled after epoch 1) |
| `production_20260922_camry_geraf_repaired_1job` (C5) | 0f53192 + local edits | Camry GeRaF with the far-range repair (prepared, held) |
| `se_spgl1_20260922` | bef9468 + local edits | SE Stage 1 with SPGL1 on the six collection scenes |

## 2. Data protocols

### RIFT collection (A320, B787, fire truck, loader, race car, X-59)

- **Source:** `rift_dataset_v1`, 10 000 views per object on a Fibonacci sphere at 10 m
  standoff. The source array is 16 × 16 MIMO. Objects are scaled to 0.10 m maximum extent and
  centred in a 0.30 m support cube (half-width 0.15 m).
- **Frequencies:** all 600 native samples, 8.5–11.495 GHz in 5 MHz steps (fc 10 GHz,
  bandwidth 3 GHz), one chirp.
- **Antennas:** physical Tx 0 / Rx 0 only (`rift_ordered_source_antennas_v1`, "1t1r"). The
  selected response per view is 1 × 1 × 600. See [ANTENNA_SELECTION.md](ANTENNA_SELECTION.md).
- **Split:** `fixed_tail_subsampled`, seed 42: 2 400 train (the first 2 400 of the 3 200-view
  parent PCG64 permutation), 1 000 validation, 1 000 reserved test (sealed, never loaded),
  5 600 unused. Every method uses the same role manifest
  (`rift_dataset_<object>_seed42_train2400_val1000_test1000_v1_1t1r_2010b2bbe725`).

### GOTCHA Camry (HH)

- **Passes and region:** HH, passes 1–8, one field shared across passes. The registered
  `camry` region is a 10 m cube (half-extent 5 m) at (20.66, −18.71, 0.02) m in the native
  frame. The source HH autofocus is applied once by the ingress.
- **Viewpoints:** 1 500 / 440 / 440 train/validation/test pass-sectors. Train is nested in a
  2 000-sector parent; test payloads are sealed.
- **Pulses:** a fixed cap of 16 pulses per sector in every role (seed 42, BLAKE2b priority),
  giving 24 000 / 7 040 / 7 040 pulses. See [GOTCHA_PULSE_SELECTION.md](GOTCHA_PULSE_SELECTION.md).
- **Frequencies:** stride 2 with the last endpoint kept, 213–218 of 424–434 native bins per
  pass (9.288–9.910 GHz), in every role. See
  [GOTCHA_FREQUENCY_SELECTION.md](GOTCHA_FREQUENCY_SELECTION.md).
- **Presumming:** none. This recipe was pinned by the user on 2026-09-22.

## 3. Adaptive RIFT

### Collection (all six scenes, C1)

The argument list is built by `train_rift_dataset.py` from
`rift/b7873200_adaptive_fullscale.py:fullscale_train_argv`, with `--num-train 2400` and
1 Tx / 1 Rx. It runs through `train_pvc.py`. The six scenes differ only in their paths.

| Setting | Value |
| --- | --- |
| Representation | `point_sh` with `--adaptive-capacity-v2`; initial grid 48³ over ±0.15 m (110 592 anchors, 6.25 mm pitch); at most 262 144 points (and active points) |
| Spherical harmonics | initial degree 0, maximum degree 3 |
| Forward model | `range` operator, `sum2` amplitude law, float64 compute, phase sign −1; point chunk 65 536, pair chunk 64 |
| Loss | complex MSE per view in raw units (`--loss complex`, `--mag-weight 0`); learnable global complex gain, warm-started by projection |
| Initialization | `--bp-init 100` (scaled backprojection of the first 100 train views), `--init-scale 0` |
| Optimizer | AdamW, lr 3e-3 (coefficients and gain), position lr 3e-3, Adam ε 1e-8, weight decay 0, no gradient clipping |
| Schedule | cosine warm restarts T₀ = 10, T_mult = 2, η_min = 1e-6, stepped per epoch (cycles of 10/20/40/80 epochs; the budget ends with the fourth cycle) |
| Budget | 150 epochs, one update per view (2 400 updates per epoch, 360 000 in total), fixed view order, seed 42 |
| Priors | group L1 3e-7 and SH degree 1e-9, `fixed_initial` normalization |
| Refinement | every 10 epochs (15 events); probe every 16 views; minimum exposures 3 200 spatial / 200 angular; fractions 1/512 spatial, 1/16 angular; split max level 1 |
| Selection | best validation checkpoint (`--checkpoint-metric val`) |

PVC: `train_pvc.py` rebinds only the accelerator hooks (seeding, RNG payload, collectives);
the argument list is the CUDA one. Epochs 1–3 matched the H100 run to printed precision
(jobs 2153287 vs 2150122). Measured cost is 191.5–260.1 s per epoch, about 8.8 h per scene.
**Tree = run** for all six scenes: every RIFT file the runs use is byte-identical in the
current tree.

### GOTCHA Camry

- **Run (C4, `sum2`, cancelled after epoch 1).**
  - Data and schedule: the Camry protocol above, 150 epochs, one AdamW step per pass-sector
    (1 500 per epoch, reshuffled each epoch), seed 42.
  - Model: G48 over the 10 m cube (208.3 mm pitch), at most 262 144 points, SH degree 0 to 3,
    maximum refinement level 3.
  - Optimizer: lr 3e-3 for coefficients and gain, position lr 1e-4, Adam ε 1e-20, weight decay
    0, no scheduler.
  - Refinement: every 100 updates, probe every 10, fractions 0.05/0.05, minimum exposures 2/2.
  - Forward model: `sum2` amplitude law on the physical range; batched lane
    (`batched_pulses_one_backward_v1`), point chunk 16 384.
  - Loss and start: complex MSE on the ROI range-projected signal (half-Rayleigh, two guards,
    SVD 1e-10), divided by the TRAIN mean projected power. Random 1e-3 start, no priors.
  - Selection: best projected validation RelMSE.
  - Outcome: the user cancelled it after epoch 1 (validation projected RelMSE 16.9).
- **Superseded, never ran:** the C3 and C1 plans are the same without `range_model` (that is,
  unit amplitude). C1 also used the per-pulse loop with point chunk 4 096.
- **Current tree, not yet in a production run.** These are the defaults in both lanes,
  committed in 0f53192:
  - `--initialization backprojection --bp-views 100`: a coherent backprojection of the first
    100 epoch-1 pass-sectors in the ROI-projected domain, then the gain warm start, then a
    coefficient gauge to B787's starting size.
  - `--priors rift_dataset`: group L1 and SH-degree priors at B787's dimensionless strengths,
    μ₁ = 0.115 and μ₂ = 1.46e-6.
  - The loss is unchanged. The optimizer and schedule are **not** aligned with the collection
    recipe (still position lr 1e-4, Adam ε 1e-20, no scheduler, refinement as above); that
    is pending a user decision.
  - The only run of this recipe is smoke 2156186 (2 epochs). It was mechanically clean (exit 0,
    no fallback) but did not pass on quality: validation projected RelMSE was 59.4 at epoch 1
    and 12.2 at epoch 2. No production root has been prepared, by the user's decision. See
    [GOTCHA_FORWARD_MODEL_ALIGNMENT.md](GOTCHA_FORWARD_MODEL_ALIGNMENT.md) §§11–13.

## 4. SpINR

### Collection (all six scenes, C1)

The command is `train_spinr_style_pvc.py --recipe budget48-direct`, with recipe identity
`rift_dataset_spinr_v1_passband_direct_bins_g48_midpoint_150_v1`.

| Setting | Value |
| --- | --- |
| Network | normalized XYZ plus 6 sin/cos bands (39 inputs); six Linear+ReLU layers of width 840; signed real scalar head; 3 566 641 parameters, fp32 |
| Initialization | Kaiming-normal hidden weights, zero biases; head N(0, 0.01/√840) |
| Support and quadrature | 0.30 m cube; G48 midpoint (110 592 points, 6.25 mm pitch); physics in float64/complex128 |
| Operator | product spreading, phase sign −1, closed-form selected finite-DFT bins over all 600 frequencies; scene bins from per-pair geometric bounds |
| Loss | \|·\|² (weight 1) + 0.5·complex² over the selected bins, scaled by N_freq / TRAIN mean raw power |
| Normalization | TRAIN mean raw power; fixed output scale 0.1·√(observed/predicted energy) from 32 random TRAIN views; no learned gain |
| Optimizer | Adam, lr 1e-4, betas 0.9/0.999, ε 1e-8, no weight decay; global gradient-norm clip 1.0 |
| Schedule | cosine to 1e-5 over 150 epochs, no restarts, no plateau stop |
| Budget | 4 whole views per update (4 channels × 600 frequencies at 1t1r); 600 updates per epoch, 90 000 in total; seed-42 permutation each epoch |
| Validation | every 5 epochs over all 1 000 views; select the lowest coherent relative MSE (earliest on ties) |

Scenes differ only in paths and in the data-derived initial output scale (A320 4226.0,
B787 2728.3, fire truck 5351.5, loader 5898.2, race car 5110.6, X-59 2525.8). The collection
runs take about 6.3 h each.

**Tree vs run.** The current tree (committed in 9416aa9, not in any run) changes only recipe *text*: the identity now
says 4 measurement channels per update instead of the paper's 1 024, with a legacy alias
kept for resume. The C1 checkpoints still carry the old "1024 measurement channels per
update" wording.

### GOTCHA Camry (C3 plan)

- **Same as collection:** the network, optimizer, cosine schedule, clipping and loss form.
- **Config:** `protocols/gotcha_spinr_g48_midpoint.json` sets grid 48, one node per cell,
  150 epochs, and a 1 024-pulse batch.
- **Field:** one field for HH, shared across passes.
- **Cube:** 10 m, 0.2083 m pitch, 86.5 rad maximum phase change across a cell.
- **Kernel:** `exact_native_point_kernel_DFT` with exp(−i4πf/c (d − r0_eff))/d².
- **Budget:** 24 updates per epoch, 3 600 in total.
- **Normalization and selection:** normalization and initial scale are per polarization.
  Selection is the minimum pooled ROI-projected complex validation error.
- **PVC execution:** `batched_shard_groups_one_vjp_dft_bmm_v2`, the exact DFT as an fp64 complex
  GEMM, at 13.7 s per update. It is in the runtime recipe but missing from the C3 plan JSON;
  that was fixed in c37adf5.
- **Status:** the C3 job has not run (held).

Deviations are listed in [SPINR_ADAPTATION.md](SPINR_ADAPTATION.md):
- 150 epochs instead of the paper's 1 500.
- An unqualified midpoint quadrature.
- 4 channels per update instead of 1 024.
- Local architecture and optimizer choices.

## 5. GeRaF

### Collection (all six scenes, C1)

The command is `train_geraf_pvc.py --implementation source_v1 --source-config
protocols/geraf_mf48_1t1r.json`. The upstream code is GeRaF-SENS 38266cb.

| Setting | Value |
| --- | --- |
| SDF network | 8 layers, width 256, skip at layer 4, 10 positional-encoding levels, geometric initialization (bias 0.5), weight norm |
| Reflectivity | `ReflectivePowerNetwork` (`feats_dim` 0, sigmoid); `light_power` initialized to 0 and trainable; variance initialized to 0.3 |
| Rays | 32 × 32 aperture grid in the unit disk (at most 807 jittered rays per view); 32 coarse + 64 target samples per ray; `extent_m` 0.15 |
| Targets | 48³ matched-filter lattice with inclusive endpoints (6.38 mm); fp32 MF magnitude (frequencies summed coherently, antennas averaged), trilinear; accumulated target = sum over TRAIN views; `trans_power` 1 |
| Masks | 0.04 current, 0.15 accumulated |
| Loss | L2 on the MF magnitude, plus gradient regression with weight 0.1 |
| Optimizer | AdamW, lr 1e-4 (SDF) and 1e-3 (other groups), weight decay 0, clip 35; cosine with **absolute** η_min 5e-4, so the SDF learning rate rises from 1e-4 toward 5e-4 |
| Budget | 50 000 steps, one view per step in cyclic order (≈ 20.8 passes over 2 400 views), bank size 1 at 1t1r |
| Step hooks | `released_runner_zero`: the model step stays at 0, so `anneal_end` (50 000) and `freeze_inv_s` (10 000) never act |
| Validation | every 1 000 steps over 1 000 views, with fresh unmasked fixed-seed rays; select pooled MF-magnitude MSE; checkpoint every 100 steps; seed 42 |
| Precision | fp16 autocast regions as in the release; receiver geometry cast to float32 (release path) |

The recipe is identical across the six scenes. PVC changes: an fp16 autocast twin replaces
the three vendored-renderer sites and `predict_native`, an XPU RNG twin is saved in
checkpoints, and ray sampling runs in host numpy. These are recorded in
`rift_pvc/vendor/geraf_sens/NOTICE.md` and `RIFT_PVC_Adaptation.md` §6.

**Tree vs run.** The current tree (committed in 9416aa9) adds the recipe key `receiver_geometry`,
default `float64_intersection`. The C1 collection runs do not have it. At the collection's
normalized radius (R ≈ 66.7) no ray misses and hit points move by 4 mm or less.

### GOTCHA Camry (C5, prepared and held)

- **Config:** `protocols/geraf_mf48.json` with bank size 2 (the released round-robin over the
  16 pulses of each sector).
- **Scale:** `extent_m` 5.0, so the target lattice spacing is 212.8 mm.
- **Receiver geometry:** `float64_intersection` (the far-range repair), validated on a card by
  smoke 2156223.
- **Data:** HH head, 1 500 sectors (≈ 33.3 passes), 440 validation sectors; the MF is averaged
  over pulses with the reference-range phase.
- **Status:** job 2156225 is prepared and held, blocked on the GOTCHA amplitude-calibration
  decision. GOTCHA provides no transmit power; with `trans_power` 1 and `light_power`
  starting at 0, Camry GeRaF currently predicts ≈ 0 (validation RelMSE 1.0).
- **Superseded:** the earlier C1 Camry plan is identical except that it lacks
  `receiver_geometry`; it was cancelled.

Deviations (the released stage-1 code configured to v1 settings, fallbacks, MF48 instead of
601³, and no radiometric calibration) are listed in [GERAF_V1_HARDENING.md](GERAF_V1_HARDENING.md).

## 6. Radar Fields

### Collection (C2)

The recipe is `source-adapted-v3`, with backend `upstream-tcnn-torchshim` (torchshim-2, fp32,
fixed-order `bmm:256` weight gradients). References: tiny-cuda-nn 749dd70 and Radar Fields
ee76d76. The six plan commands differ only in the object name.

| Setting | Value |
| --- | --- |
| Data | 2 400 train / 1 000 validation at Tx 0/Rx 0; training seed 0 (distinct from split seed 42) |
| Batch | 10 frames × 100 profiles (drawn with replacement; at 1t1r the single pair repeats) × 10 rays |
| Schedule | 240 batches per epoch × 4 epochs = 960 updates; epoch mask 0.55 / 0.916 / 1 / 0.916 |
| Optimizer | Adam over the 5 released parameter groups, lr 1e-3, betas (0.9, 0.99), ε 1e-15; lr × 0.1^min(step/800, 1) |
| Loss | 0.6 FFT + 0.36 occupancy KL (finite part) + 0.03 bimodal; intensity offset 1.0, scaler 1.0, released range law, 60 dB dynamic range |
| Model | hash grid 16 levels × 2 features, resolution 16→512, log₂ table 19; hidden 64, feature 32, SH degree 3, BatchNorm on |
| Occupancy | noise multiplier 1.5, decay 10 bins, offset −0.15, scale 2.0 |
| Support | 0.15 m half-extent cube, 0.05 m range margin; G48 geometry readout (110 592 points) |
| Evaluation | every 240 steps on all 1 000 validation views; keep the best validation relative MSE |

The completed runs (A320, X-59, fire truck, race car; about 13 min each) all picked their
best checkpoint at step 240, and each final step was worse. Loader and B787 are held and
have not run.

**Backend disclosure.** The torch shim is a re-implementation of tinycudann, not the
original. It differs in three ways: fp32 instead of fp16, a different initialization RNG
stream, and fixed-order weight gradients. Strict numerical parity is not certified
(`campaign.json` `pvc_backends`). The completed runs record the parity status as
"diagnostic" (replay 2154358); the current tree records the closed gate (2155072). See
[RADAR_FIELDS_PVC_ADAPTATION.md](RADAR_FIELDS_PVC_ADAPTATION.md).

**Tree vs run.**
- **Source locks (committed in 9416aa9, not in any run):** the current tree adds `SOURCE_LOCKS` to
  `rift/radar_fields_recipe.py`. It locks exactly the values every run used.
- **fp16 loss scaling (commit ae8cf6e):** added after the C2 snapshot, but inactive with
  `RIFT_PVC_TCNN_HALF=0`.

### GOTCHA Camry (C2 plan, held)

- **Schema and constants:** `gotcha_radar_fields_v3`; model, optimizer and loss constants as
  in the collection.
- **Data:** HH, 1 500 / 440 pass-sectors, 16-pulse cap and frequency stride 2.
- **Schedule:** 150 batches × 6 epochs = 900 updates; evaluation and checkpoint every 150;
  epoch mask 0.359 / 0.638 / 0.859 / 1 / 1 / 1.
- **Geometry:** extent 5 m. Profiles are native pulse IDs. The range grid is Rayleigh-spaced
  with 2 guard cells.
- **Normalization:** peak over the selected TRAIN pulses.
- **Status:** not run. Open user decisions: GOTCHA pose refinement and the zero-outside-cube
  field ([GOTCHA_FORWARD_MODEL_ALIGNMENT.md](GOTCHA_FORWARD_MODEL_ALIGNMENT.md) §13).

The released-code choices and dataset adaptations are listed in
[RADAR_FIELDS_ADAPTATION.md](RADAR_FIELDS_ADAPTATION.md).

## 7. RadarSplat

### Collection (C2, `--fidelity-profile budget48`)

The six scenes' `recipe.json` files differ only in the data-derived train peak power.

| Setting | Value |
| --- | --- |
| Gaussians | 112 000 (released default 20 000), planar z = 0 initialization over ±55 units; 333.33 units/m |
| Initialization | opacity and noise 0.5, scale 0.25 units; SH degree min(step // 200, 5), with the kernel capped at 4 |
| Updates | 2 000, one view each, from seeded (42) shuffled cycles (0.83 passes over 2 400 views) |
| Optimizer | Adam per group, betas (0.9, 0.999), ε 1e-15; lr means 1.6e-4 × 110 (decaying to 0.01× over 2 000 steps), scales 5e-3, quaternions 1e-3, opacities and noise 5e-2, SH 2.5e-3 |
| Densification | none (`refine_stop_iter` 0) |
| Loss | 0.8 L1 + 0.2 (1 − SSIM) + 10 occupancy L1 + 100 size + 1 000 probability; multipath weight 0.6; 2.5-unit near-range mask |
| Occupancy labels | 11 nearest training directions, threshold 0.10 |
| Targets | 33 × 33 × 33 cache over the scene support; 10° element HPBW; leakage 0.19986 m; `range_nufft` in fp64; linear peak normalization; 3 400 targets |
| Selection | final step only |

All six collection runs completed (19–21 min each). The `.model_recipe` values
`spectral_leakage_width_m` 2.0 and `max_size_threshold_m` 1.0 are nominal release values.
The leakage width actually executed is the cache value, 0.19986 m.

**Backend disclosure.** `fork_torch_mirror_xpu_v1` replaces the five CUDA ops with torch
mirrors. It uses autograd for the backward pass and IEEE arithmetic instead of fast-math.
SSIM is `fused_ssim_torch_v1`. Both are recorded in `checkpoints/backend.json` (six listed
deviations). The completed runs' `backend.json` records parity job 2153858 with a
"diagnostic" status because of an environment override in the C2 snapshot. That override is
removed in the current tree (committed in 9416aa9); gate 1 was closed by jobs 2154965 and 2155075. See
[RADARSPLAT_PVC_ADAPTATION.md](RADARSPLAT_PVC_ADAPTATION.md).

### GOTCHA Camry (C2 plan, held)

- **Model:** the same `.model_recipe` as the collection; HH head only.
- **Data:** 1 500 / 440 sector images over the selected pulses and frequencies; 2 000
  updates per head.
- **Grid:** 33 × 33 × 145, 0.1203 m range bins, extent 5 m (10 units/m).
- **Chunks:** point chunk 128, frequency chunk 256.
- **Status:** the GOTCHA fitting path has never run on a card (smoke 2154119 reached only
  target conversion). Open user decisions: occupancy-label normalization and Gaussian size
  relative to the image grid ([GOTCHA_FORWARD_MODEL_ALIGNMENT.md](GOTCHA_FORWARD_MODEL_ALIGNMENT.md) §13).

The released-code choices and conversions are listed in
[RADARSPLAT_FIDELITY.md](RADARSPLAT_FIDELITY.md).

## 8. Sugavanam–Ertin

### Collection (six scenes, `se_spgl1_20260922`)

The recipe is `rift_pvc.sugavanam_ertin_spgl1.make_recipe` applied to
[`protocols/se_g40_readout48.json`](../protocols/se_g40_readout48.json). That is the paper-v1
recipe with the Stage-1 solver made explicit.

| Stage | Setting | Value |
| --- | --- | --- |
| 1 | Sub-apertures | 72 azimuth bins of 5° × 1 elevation bin, fitted from TRAIN directions (24–46 views each) |
| 1 | Voxel field | G40 over ±0.15 m (64 000 complex voxels, 7.5 mm pitch), one field per sub-aperture |
| 1 | Operator | first-order bistatic Fourier kernel k(u_tx + u_rx), native reference range, origin spreading; response over TRAIN RMS |
| 1 | Eq. 4 budget | ‖A x − b‖ ≤ σ with σ² = 1 % of the sub-aperture's energy (`residual_relative_energy` 0.01) |
| 1 | Solver | SPGL1 `spg_bpdn` (github.com/drrelyea/spgl1 @ 405ca805), package defaults (`opt_tol` 1e-4, `bp_tol` 1e-6, `ls_tol` 1e-6, `dec_tol` 1e-4, `iter_lim` 10 × rows), complex variables; one solve per sub-aperture |
| 1 | Gate | every sub-aperture must exit `root_found` or `bp_solution_found`; the repository residual (1e-3) and Frank–Wolfe (1e-5) certificates are diagnostics |
| 1→2 | Cloud | Σ over sub-apertures of \|S_m\|; keep voxels ≥ 0.15 × peak (all of them); PCA normals, radius 0.3 m; strongest-aperture fallback |
| 2 | SDF | 8 Softplus layers of width 512, input skip into layer 4, tanh output; raw xyz + 9 Gaussian Fourier bands (scale 2 cycles/m, seed 42) |
| 2 | Initialization | Gaussian weights and biases, std 0.05 (user-selected; std 1 is degenerate) |
| 2 | Optimizer | Adam, lr 1e-4, ε 1e-8; 5 000 steps; 2 048 on-, off- and iso-surface samples per step |
| 2 | Losses | the six Eq. 19–25 terms, all weight 1; off-surface α 100 |
| 2 | Iso-points | 2 048 attempted, refreshed every 100 steps from step 1; 24 Newton projection steps; paper-literal edge weights |
| 2 | Readout | zero level set of the SDF by marching cubes on a 48³ grid |

- **Stage 1 runs** on CPU (about 2 min per sub-aperture on 32 cores).
- **Gate results:** as of 20:00 CDT, B787, fire truck, race car and loader passed with 72/72
  `root_found`; A320 and X-59 were still solving.
- **Stage 2** (the unchanged workflow, on PVC) has not run yet.
- **Superseded:** the C1 SE runs used the repository solver and ended `stage1_unconverged`
  (0/72).

**Tree vs run.** The current tree adds a per-sub-aperture operator check, which does not
change the recipe identity; the running snapshot predates it. Details are in
[SUGAVANAM_ERTIN_PAPER.md](SUGAVANAM_ERTIN_PAPER.md#spgl1-stage-1-production-collection-lane).

### GOTCHA Camry

On hold for the current campaign (user decision, 2026-09-22). The SPGL1 lane can drive it
(`scripts_pvc/se_spgl1_stage1.py --gotcha …`), but no GOTCHA SE job has been submitted.

## 9. Run and tree per scene

| Method | Collection (6 scenes) | GOTCHA Camry |
| --- | --- | --- |
| Adaptive RIFT | C1; tree = run | ran C4 (sum2, random start), cancelled after epoch 1; tree has the backprojection start + priors (smoke only) |
| SpINR | C1; tree differs in recipe text only (4 vs 1024 channels) | C3 plan, held, not run |
| GeRaF | C1; tree adds `receiver_geometry` (not in the runs) | C5 prepared with the far-range repair, held (amplitude calibration) |
| Radar Fields | C2; A320, X-59, fire truck, race car done, loader and B787 held; tree adds source locks (same values) | C2 plan, held, not run |
| RadarSplat | C2; all six done; runs record the pre-gate parity status, tree drops that override | C2 plan, held; GOTCHA fit path never run on a card |
| Sugavanam–Ertin | `se_spgl1_20260922` (SPGL1 Stage 1); tree adds only the operator check | on hold |

## 10. Points to watch when writing about these recipes

These are facts that a methods or experiments section can easily misstate. Each one points to
where it is recorded.

### Across methods

- **Budgets are per method, not compute-matched.** Each method runs its selected budget,
  not equal updates, passes or GPU time:

  | Method | Budget (collection) |
  | --- | --- |
  | Adaptive RIFT | 150 passes, 360 000 updates |
  | SpINR | 150 passes, 90 000 updates of 4 views |
  | GeRaF | 50 000 steps, ≈ 20.8 passes |
  | Radar Fields | 4 epochs, 960 updates |
  | RadarSplat | 2 000 updates, 0.83 passes |
  | Sugavanam–Ertin | one SPGL1 solve per sub-aperture + 5 000 SDF steps |

  On Camry the counts differ again (sections 3–8). See [SCENE_BUDGET.md](SCENE_BUDGET.md).
- **The collection data is a subset of the archive.** Runs use one physical pair (Tx 0/Rx 0
  of the 16 × 16 array) and 2 400 of the 3 200 parent training views. State this whenever the
  dataset is described; the per-method pages often describe the full-data defaults.
- **The Camry training acquisition is changed, not just subsampled.**
  - The 16-pulse cap and frequency stride 2 apply in every role.
  - The frequency selection records that the objective is a changed training acquisition, not
    an unbiased full-frequency loss (`frequency_selection.objective`).
  - See [GOTCHA_PULSE_SELECTION.md](GOTCHA_PULSE_SELECTION.md) and
    [GOTCHA_FREQUENCY_SELECTION.md](GOTCHA_FREQUENCY_SELECTION.md).
- **Provenance is a snapshot, not a commit.** Every campaign root ran "commit + local edits"
  (section 1). A commit hash alone does not reproduce a run. The run's `source/` copy and
  `source_provenance.json` (with its `worktree_status`) are the provenance.
- **PVC re-implementations are separate backends.**
  - Radar Fields uses a tinycudann torch shim (fp32 instead of fp16, a different initialization
    RNG stream, fixed-order weight gradients).
  - RadarSplat uses torch mirrors of its CUDA ops and a torch SSIM (autograd backward, IEEE
    arithmetic instead of fast-math).
  - Describe these as re-implementations, not as the original code. Strict numerical parity of
    the Radar Fields shim is not certified.
- **Runs on PVC are not bitwise reproducible.** Several XPU float32 reductions (large-K GEMMs,
  `index_add_`, `scatter_add_`) are order-nondeterministic. RIFT on the collection did match an
  H100 run to printed precision over epochs 1–3. For SE Stage 1, two runs of the same
  sub-aperture with different BLAS summation orders ended at different SPGL1 stopping points
  (see Sugavanam–Ertin below).
- **Seeds differ by method.** The split seed is 42 everywhere. Training seeds: RIFT, SpINR,
  GeRaF, RadarSplat and SE use 42; Radar Fields uses the release's seed 0.
- **Results from the collection runs are not all in yet.** As of section 1's timestamp, RIFT
  and SpINR on loader and B787 were still running, Radar Fields on loader and B787 was held,
  SE Stage 1 was still solving A320 and X-59, and no SE Stage 2 had run. No Camry method has a
  completed production run.

### Adaptive RIFT

- **The Camry recipe is not final.**
  - The only run (C4: sum2, random start, no priors) was cancelled after epoch 1.
  - The current tree's backprojection start plus priors has only a 2-epoch smoke run. It was
    mechanically clean but did not pass on quality (validation projected RelMSE 59.4 then 12.2).
  - Its optimizer and schedule are still not aligned with the collection recipe (Adam ε 1e-20,
    position lr 1e-4, no scheduler), pending a user decision.
- **Camry and collection use different recipes**, beyond the data:
  - projected, power-normalized loss instead of raw complex MSE;
  - per-epoch reshuffle instead of fixed view order;
  - refinement every 100 updates up to level 3, instead of every 10 epochs up to level 1;
  - point chunk 16 384 instead of 65 536.

  Do not describe them as one recipe.
- **The collection learning-rate schedule** is cosine with warm restarts (10/20/40/80 epochs).
  The 150-epoch budget ends exactly at the end of the fourth cycle.

### SpINR

- **Batch size.** Each update uses 4 measurement channels at 1t1r, not the paper's 1 024.
  - The C1 checkpoints' recipe text still says "1024 measurement channels per update".
  - The corrected wording is only in the current tree (committed in 9416aa9), not in any run.
- **Budget.** 150 epochs, against the paper's 1 500 (user-selected, matching RIFT's pass count).
- **Quadrature.** The G48 midpoint rule is not qualified for fitted fields. The phase change
  across one cell is about 3.0 rad on the collection and 86.5 rad on Camry.
- **Open audit items** (reported in
  [GOTCHA_FORWARD_MODEL_ALIGNMENT.md](GOTCHA_FORWARD_MODEL_ALIGNMENT.md) §13, not yet resolved):
  - best validation arrives early (epochs 5–10);
  - gradient clipping is active on 48–100 % of updates.

### GeRaF

- **Step hooks.** The released runner holds the model step at 0, so `anneal_end` and
  `freeze_inv_s` never act.
- **Learning-rate schedule.** The cosine schedule's η_min is an absolute 5e-4, so the SDF
  learning rate *rises* from 1e-4 toward 5e-4. Both of these points are release behaviour
  kept on purpose ([GERAF_V1_HARDENING.md](GERAF_V1_HARDENING.md)), but a reader will not
  expect them.
- **Target resolution.** Targets use a 48³ matched-filter lattice, not the release's 601³.
- **No radiometric calibration.** `trans_power` is 1. On Camry this leaves the model predicting
  ≈ 0 (validation RelMSE 1.0) until the amplitude-calibration decision is made.
- **Receiver geometry.** The collection runs use the release's float32 receiver cast. The
  far-range float64 repair exists only in the Camry C5 root and the current tree.

### Radar Fields

- **Selection matters.** Every completed run chose its step-240 checkpoint by validation, and
  every final step was worse. Report best-validation selection as part of the recipe.
- **Budget.** The whole budget is 960 updates (about 13 min on one card), fixed by the released
  800-update learning-rate clock.
- **Parity status.** The completed runs record the shim parity status as "diagnostic" (replay
  2154358). The closed gate (2155072) came after their snapshot.

### RadarSplat

- **Selection and pass count.** It is evaluated at the final step only (no best-checkpoint
  selection). 2 000 updates is 0.83 passes over the training views.
- **Gaussian count.** 112 000 Gaussians against the release default of 20 000
  ([RADARSPLAT_FIDELITY.md](RADARSPLAT_FIDELITY.md)).
- **Nominal vs executed values.** The `.model_recipe` values `spectral_leakage_width_m` 2.0 and
  `max_size_threshold_m` 1.0 are nominal release values. The leakage width actually used is the
  target cache's 0.19986 m.
- **Parity status.** The completed runs' `backend.json` records parity job 2153858 with a
  "diagnostic" status. That comes from an environment override in their snapshot, not from the
  parity result; gate 1 was closed by jobs 2154965/2155075.
- **Camry.** The GOTCHA fitting path has never run on a card.

### Sugavanam–Ertin

- **Stage-1 solver.** It is SPGL1 with package defaults, a user decision that fills a gap the
  paper leaves open. Do not describe it as the authors' solver or as the repository's own
  solver. The repository solver's runs ended 0/72 converged and are superseded.
- **What convergence means.** SPGL1's default exit guarantees the residual budget (1 % energy),
  not minimal L1. The repository's stricter optimality test is recorded per sub-aperture but
  not met (pilot relative Frank–Wolfe gap 0.35–1.33).
- **Stage-1 fields depend on summation order.**
  - Two runs of X-59 sub-aperture 0 differed by 6.5 % (correlation 0.998) with different BLAS
    paths.
  - Above the 15 %-of-peak cloud threshold, 656 of about 675 voxels were shared.
  - Stage 2's input cloud is stable; the exact field is not.
- **Budget and gate.** The 1 % residual budget is a local choice; the paper gives no noise
  budget. The Stage-1 gate is also local.
- **Initialization.** The SDF initialization std of 0.05 is user-selected; the literal
  standard-Gaussian std 1 gives a degenerate network (see [SUGAVANAM_ERTIN_PAPER.md](SUGAVANAM_ERTIN_PAPER.md)).
- **Known transfer limitation.** The 0.3 m PCA normal radius is larger than the 0.1 m objects,
  so the neighbourhood is not local.
- **The sub-aperture partition's mean directions are node-dependent in their last bits.**
  - numpy/libm pick CPU-specific SIMD kernels, so digests differ across nodes.
  - A320's two Stage-1 halves bound different partition digests, and assemble refused them.
    The half was re-solved.
  - The assignments, which decide what each sub-aperture fits, were identical. The difference
    is at most about 1e-15 in the mean directions.
  - The lane now pins the partition per scene. Stage 2's unchanged resume check still requires
    the Stage-2 node to reproduce the bits exactly.
- **Stage 2 on real data.** It had not yet run on PVC at this timestamp; its runtime and
  behaviour there are unmeasured. SE on GOTCHA is on hold.
