# Radar Fields adaptation: RIFT and GOTCHA

Native GOTCHA supports shared [pulse](GOTCHA_PULSE_SELECTION.md) and
[frequency](GOTCHA_FREQUENCY_SELECTION.md) selection in train/validation/test.
Matched-power targets use selected bins; the released 100-profile draw is unchanged.

The comparison profile is **`source-adapted-v3`**. It uses the released TCNN
`RadarField` and restores documented training details. It remains an acquisition
adaptation, not a claim that coherent MIMO or GOTCHA phase histories are Navtech
FFT images. Neither adapter synthesizes coherent complex radar responses.

`audited-v2` and `legacy-v1` preserve earlier implementations/checkpoint identities.
The former's 8,000 updates, batch size one, deterministic 64-ray integration,
step-based mask and finite loss safeguards are **not** the original recipe.
They are no longer selected by the collection default. Results from these
profiles must not be presented as the new comparison implementation.

## Authority and source conflicts

References:

- [Paper, version 2](https://arxiv.org/html/2405.04662v2).
- [Supplement](https://light.princeton.edu/wp-content/uploads/2024/07/Radar-Fields-Supplement.pdf).
- [Released implementation, pinned commit](https://github.com/princeton-computational-imaging/RadarFields/tree/ee76d76570f58b3d8539eafd7df0c188b58af333),
  `ee76d76570f58b3d8539eafd7df0c188b58af333`.

When the supplement and executable release disagree, this profile consistently
follows the release. This is not a selection among alternatives based on their
performance on our datasets.

| Detail | Paper/supplement | Released code followed here |
| --- | --- | --- |
| Optimizer | AdamW, beta2 0.999 | Adam, betas (0.9, 0.99), eps 1e-15, no weight decay; original model parameter groups |
| Budget and sampling | 500 iterations, 16 frames, 200 azimuths, 900 ranges | LR clock 800; batches of 10 frames; 100 profiles; complete epochs; all selected ROI range cells |
| Beam sampling | Elliptical cone description | Uniform pitch/yaw within rectangular angular bounds, one central ray, pitch sorting, 10 samples |
| Occupancy noise rejection | 2 times paired medians, non-strict comparison | Multiplier 1.5 and strict comparison; radial part retained for these acquisitions |
| Range law | R^-2 and R^-4 appear in different equations | Default `approximate_fft=True`: log10(RCS + 1), no explicit spreading factor |

`configs/radarfields.ini`, `parse.py`, `main.py`, model, sampler, trainer,
occupancy, pose and utility files are checked for availability by
`rift/radar_fields_upstream.py`. GitHub source is trusted without content-hash
verification; line endings do not gate execution.
`rift/radar_fields_released.py` can construct the complete original Trainer for
source-format batches, including its original grounding, height and pose terms.
The RIFT/GOTCHA adapters reuse the network and numerical helpers; they do not
pretend that calling that source-format Trainer alone converts our measurements.

## Restored implementation details

- Original 16-level hash encoding (16 to 512), two features per level,
  log2 hash size 19, 64-wide networks, 32 spatial features, BatchNorm, sigmoid
  occupancy and softplus reflectance. The original TCNN SH convention is kept.
- The TCNN alpha-times-reflectance product is formed in its native output
  dtype, followed by the original LUT integrator's float32 weighting. The new
  profile removes the old wrapper's extra direction renormalization and its
  singleton BatchNorm fallback. FP64 geometric directions are converted to the
  model's input dtype without an alternative SH convention.
- The original learning rate is 0.001, decaying to 0.0001 by update 800.
  The release trains `ceil(800 / batches_per_epoch)` complete epochs. Thus
  3,200 RIFT training frames give 3 epochs / **960 updates**; 2,000 GOTCHA
  pass-sectors give 4 epochs / **800 updates**. Other selected pass counts are
  resolved from their frame inventory, not from the RIFT budget. The production
  campaign's 2,400 collection frames give 240 batches per epoch, so 4 epochs /
  **960 updates**. Its 1,500 Camry pass-sectors give 150 batches per epoch, so
  6 epochs / **900 updates**.
- The original sine mask uses **epochs**, not steps. Its nonmonotonic final
  value is preserved: the RIFT three-epoch mask is approximately 0.757, 1,
  0.757. This is not silently replaced by a smoother schedule. In production the
  four-epoch collection mask is 0.55, 0.916, 1, 0.916 and the six-epoch Camry mask
  is 0.359, 0.638, 0.859, 1, 1, 1.
- Training seed 0 is distinct from both datasets' sealed split seed 42.
  Frames use Torch `SubsetRandomSampler`; profiles use the released sorted
  `randint` sampler with replacement. All native profiles remain eligible.
  Our recovery loop materializes/persists each permutation rather than relying
  on a DataLoader iterator; its RNG stream is not claimed identical to the
  original DataLoader's worker-seeding bookkeeping.
- Intensity L1 / occupancy KL / bimodality weights are 0.60 / 0.36 / 0.03.
  Occupancy distributions normalize over the full physical frame batch; KL
  divides by that frame count. Bimodality groups use target probability >0.01
  and sample standard deviations. The v3 objective retains the original
  `nan_to_num` placement and adds no epsilon floor or empty-group replacement.
  The fixed-exterior KL adaptation is specified below. Other undefined upstream
  edge gradients cause an explicit failure, not an unreported
  numerical repair. Ragged profiles are concatenated for these same reductions;
  no fake padded observations enter the loss.
- BatchNorm sees one combined neural query batch per head and optimizer batch.
  Evaluation uses frozen BatchNorm and may chunk queries. A small query-chunk
  setting does not reduce the training BatchNorm population or training memory.

## Differences required by the data formulation

| Choice | RIFT collection | GOTCHA | Reason |
| --- | --- | --- | --- |
| Measurement target | Average static chirps; abs(IFFT over native frequency samples))² for each Tx/Rx pair | Matched range power at exact native frequencies, separately for every actual pulse | Inputs are coherent frequency responses, not supplied vendor FFT images |
| Range surface | Exact bistatic half-path ellipsoid, FP64 | Monostatic spheres using each pulse's actual position | Preserve the actual acquisition geometry |
| Spatial support | Registered object cube, extent 0.15 m; native bins and existing 0.05 m ROI guard | Registered region cube; Rayleigh-spaced range grid anchored at the ROI-center distance, with two guard cells | A common finite scene is required; nonuniform GOTCHA frequencies have no inherited FFT-bin lattice |
| Profile index | Actual Tx-outer/Rx-inner pair ID | Actual pulse ID within a pass-sector | These indices are not fictitious rotating azimuth beams |
| Profile minibatch | 100 sampled pair IDs, with replacement | 100 sampled native pulse IDs, with replacement | Preserve the released profile sampling rule; no permanent pulse/channel decimation |
| Frame | Physical acquisition viewpoint | Pass-sector, with all native pulses retained in the eligible inventory | Honor the registered dataset selection units |
| Occupancy filter | Per-pair radial median over full profile before cropping | Per-pulse radial median over the guarded ROI profile | A cross-pair/cross-pulse median would compare co-pointed returns rather than different Navtech azimuth beams |
| Occupancy recurrence | Original Bayesian recurrence, offset -0.15, scale 2, decay 10 bins | Same recurrence | Preserve the available algorithm; GOTCHA does not supply a complete unambiguous vendor radial image |
| Ground/height priors | Disabled | Disabled | Their pitch ordering assumes a ground vehicle's horizontal radar and mounting height. Our rays view a 3-D object or a region from an airborne platform; sensor-local pitch is not that road-height coordinate |
| Pose refinement | Disabled; use calibrated element positions | Restored (user decision 2, 2026-09-22): the release's per-frame SE3 PoseOptimizer (Adam 9e-4) per pass-sector, moving the sector's pulses rigidly; held-out sectors interpolated within the pass; `pose_refinement: disabled` is the earlier recipe | Collection poses are exact and spiral-ordered. GOTCHA sectors follow each pass in azimuth order, so the release's trajectory interpolation is defined there (alignment doc section 16) |
| Channels | One scalar head per object | Independent HH/HV/VH/VV heads, each shared across selected passes | Polarizations measure different scattering functions; never borrow autofocus or parameters from another channel |

GOTCHA's matched-range target is

`|sum_f S(f) exp(+i 4 pi f (r-r0_effective)/c) / Nf|²`.

Frequencies remain ragged and unchanged. Factoring out a carrier phase by
centering the frequency vector does not change this power. The native loader
applies each HH/VV channel's published correction exactly once; HV/VH remain
raw. Responses are neither resampled nor coherently averaged across distinct
pulses or passes. All training pulses contribute to normalization; validation
uses every validation pulse. Random profile minibatches are optimizer sampling,
not a reduction of the registered input inventory.

The field is zero outside the registered cube, and those rays remain in the
integration denominator. This makes the support contract explicit instead of
aliasing out-of-box hash queries onto the object. One combined query batch is
formed from in-support samples; unlike the original automotive scene box,
exterior samples do not update BatchNorm. This boundary difference follows the
finite-scene data contract and is recorded, not claimed to be source parity.

A zero fixed exterior is absent from the original strictly positive sigmoid
field. Applying its KL literally would introduce `log(0)` and NaN gradients.
For those fixed exterior cells only, v3 omits the divergent, model-independent
`log(alpha_i=0)` constant while **retaining** `-log(sum(alpha))` for every
cell. Thus in-support gradients equal the original KL's limit as fixed exterior
occupancy approaches zero. It does not clamp learned probabilities, normalize
targets over a selected mask, or drop the normalization gradient of exterior
target mass. The absolute reported KL differs by that constant and is not an
original-domain likelihood score. A test compares gradients to an independent
positive-exterior limit. In-support zero/empty-group numerical failures are
still reported rather than repaired. Validation selection uses intensity error,
not this finite-part KL value.

## Missing information and declared fallbacks

**Antenna calibration.** The supplied simulator files are:

- `data/RIFT_dataset/sim_mesh_interp_fmcw_16t16r_10ghz.py`, SHA-256
  `167084aeb27ace69adf0277d88dfb858de179e49e56b67521123e2d593187958`.
- `data/RIFT_dataset/sim_pec_sphere_fmcw_16t16r_79ghz_2k.py`, SHA-256
  `7303f4fa0b5f718d0db852070d4cddb6d6f9e52ef58e4aceb4af8798fb3c3e71`.

They specify 10° horizontal and vertical **HPBW** for both Tx and Rx,
vertical polarization and a device +X boresight aimed at the origin; Tx runs
along local Y and Rx along local Z. RIFT obtains that orientation from the
original full-array element positions, retained as pose metadata before
physical-channel selection. The selected Tx/Rx coordinates still define each
bistatic ray origin and range surface. A 1t1r selection has no measurable array
span, and reversed index lists must not rotate the sensor frame. Missing,
degenerate or nonorthogonal source axes fail explicitly before per-view response
access; the strict unit-ray check remains in place. The five newer objects
record the beam width in their metadata. B787's older archive lacks that operating-point record; the
mesh script's statement of a shared operating point is weaker evidence than
an archive-bound calibration.

The parametric gain function is inside proprietary `RssPy`, which the user
confirmed is inaccessible. HPBW does not specify a complete gain curve or a
hard integration cutoff. We do **not** identify a Gaussian, the Navtech LUT, or
a ±5° truncation as the acquisition model. The fallback is explicitly unit gain
over an aperture covering the registered ROI. The original random sampler and
LUT averaging are still used, with a unit LUT. This is an uncalibrated fallback,
not proof that antenna effects are negligible. GOTCHA's native shards likewise
lack antenna attitude/gain LUTs; its sampling frame looks toward the registered
ROI with a declared world-up convention. The aperture is a scene integration
domain, not a claimed sensor field of view.

**Image preprocessing.** The release loads precomputed thresholded FFT and
occupancy `.npy` files but does not provide their complete generating pipeline.
Vendor FFT-image scaling is also unavailable for our inputs. Both adapters use
a fixed training-only peak and the existing declared 60 dB normalization to
[0,1], independently per object/polarization. The source's known global noise
floor 0.1525 is applied to the intensity target using a strict comparison.
The exact preprocessing threshold inequality was not supplied; this is a
declared reconstruction. Occupancy uses its separate dynamic threshold and
the published recurrence, not this global FFT floor. Peak scaling and the
60 dB range are fallbacks, not original-author hyperparameters.

**Runtime version.** The release lists Python 3.9 / Torch 2.0.1 / CUDA 11.7,
without pinning its tiny-cuda-nn revision. The compatible local environment is
Python 3.10 / Torch 2.6.0+cu124 and tinycudann 2.0 pinned at
`749dd70c5afc5a9dadb85e5652ed65d55e0ba187`, compiled for SM75/80/90.
No existing RIFT packages were upgraded. This is dependency compatibility,
not a reconstruction of an unavailable historical TCNN binary. Installation
evidence is in `external/RadarFields_dependencies/installation.json`.

**Geometry and validation.** The supplement uses a 360×360×16 occupancy grid,
threshold 0.5 and a reflectance pruning threshold whose numerical value is not
given. RIFT's common geometry interface samples occupancy on its registered
48³ cube; it does not silently choose a reflectance cutoff. GOTCHA checkpoints
store the continuous per-polarization field, with region and recipe provenance;
the old B787 geometry/readout CLI does not accept them. Held-out validation and
recovery checkpoints are added comparison infrastructure. Validation uses mask
1 and a reproducible ray seed, isolated from training RNG. Reserved test/unused
responses remain sealed. No settings were selected using reserved-test scores.

## Root integration and boundaries

**Both dataset training entrypoints are implemented.** The GOTCHA entrypoint
is an executable native backend, not a pending hook or a simulated-data wrapper.

| Dataset | Public entrypoint | Implementation reached | Default RF schedule |
| --- | --- | --- | --- |
| RIFT collection | `train_rift_dataset.py --object <object> --method radar_fields` | `train_radar_fields.py`, `source-adapted-v3` | 3 complete epochs / 960 updates |
| GOTCHA | `train_gotcha_dataset.py --method radar_fields` | `train_radar_fields.run_gotcha` → `rift/radar_fields_gotcha.py` | 4 complete epochs / 800 updates for all eight passes |

```bash
python train_rift_dataset.py --object a320 --method radar_fields --dry-run
python train_gotcha_dataset.py --method radar_fields --dry-run
```

GOTCHA defaults to the registered Camry region, passes 1–8 jointly, and HH.
Its sealed split contains 2000/440/440 train/validation/reserved-test
pass-sectors at split seed 42. The production campaign uses the user-pinned
subset instead: 1500/440/440 pass-sectors, a 16-pulse cap per sector and frequency
stride 2 in every role (see [TRAINING_RECIPES.md](TRAINING_RECIPES.md)). `--passes`, `--region`/`--region-config`, and
`--polarizations` select the acquisition; HH/HV/VH/VV are supported by independent
heads shared across the selected passes. All native pulses remain eligible,
and every training pulse contributes to the training-only normalization scan.
The source profile samples 100 profiles per frame during optimization; it does
not permanently discard the remaining native pulses.

Backend options use the root's `--method-config` JSON mapping under the
`radar_fields` key. An empty mapping selects `source-adapted-v3` and the original
TCNN backend. The common root's RIFT-specific `--epochs`/`--lr` settings do not
override this RF recipe. Fixed source settings are checked by
`recipe_from_config`; the explicitly different `audited-v2` profile remains
available for engineering checks. Training seed 0 does not change split seed 42.

GOTCHA writes `run.json`, `history.json`, and `checkpoint_latest.pt`,
`checkpoint_best.pt`, `checkpoint_final.pt` under
`training_checkpoints/GOTCHA_dataset/<region>/<dataset-identity-prefix>/radar_fields/`
by default. Root `--resume /path/to/checkpoint_latest.pt` restores the matching
model heads, optimizer/scheduler, RNG, frame-permutation/exposure state and
pending validation. Source, region, roles, polarization selection, recipe and
normalization provenance are checked before response access. `--output-root`
changes the destination; it does not permit reusing RIFT or historical GOTCHA
transfer checkpoints.

The commands above only plan from metadata. Actual training remains an
experiment-manager action inside a SLURM allocation; the GOTCHA root enforces
that allocation requirement. Entry-point completion does not authorize a launch.

The maintained root trainer also accepts `train_radar_fields.py --object a320`.
Named-object invocation selects v3 and the registered roles. The bare historical
`--npz-path` invocation keeps `legacy-v1` unless a recipe is explicitly selected.
Collection v3 readout uses `protocols/radar_fields_rift_source_adapted_v3.json`.
GOTCHA dispatches the literal `GOTCHA_BACKEND` / `run_gotcha` hook in the same
root module to `rift/radar_fields_gotcha.py`; source/region/split/polarization/
recipe-bound `.pt` checkpoints are distinct from RIFT `.pth.tar` checkpoints.

All recipe changes require new output roots. An explicit Torch backend is for
bounded CPU checks/ablations, never an automatic substitute for TCNN. GOTCHA
backend availability means the implementation is wired, not that real-data
CUDA memory, throughput or convergence has been validated.

The original audit identified genuine normalization, representation, occupancy
and renderer/readout problems, but did not exhaust the full caller's defaults.
The re-audit additionally found the released epoch mask, complete-epoch budget,
sampling, dtype order and source-enabled priors/pose model. Their disposition
is above. Noncoherent RF still cannot reproduce phase-dependent interference in
coherent range power; no wiring or loss-parity test removes that model limitation.

## Validation

Bounded tests check the original loss values/gradients, original Trainer terms
and mask clock, native range-power phase/reference convention, multiple passes
and polarization heads, sealed-role rejection, source versus historical root
dispatch, actual CPU updates and exact interrupted/resumed optimizer/model/RNG
state. The CPU model is an explicitly identified portable probe. CUDA tests skip
when no GPU is allocated. Real metadata-only planning passes for all six RIFT
objects and joint eight-pass GOTCHA HH. No real response fitting, scheduler submission, reserved
test evaluation or convergence claim is part of this implementation task.

Recorded implementation verification: **291 passed, two CUDA-only skipped**; the historical
validator passed **94 checks**. These counts cover the RF/source/GOTCHA tests
and the shared collection/GOTCHA contract suites, not GPU convergence.

Remaining RF work is allocated CUDA forward/backward and resource/convergence
validation, a GOTCHA geometry/readout interface for these checkpoints, and
evaluation of the disclosed antenna/target-model assumptions. The GOTCHA
training entrypoint itself is complete. Reserved-test reporting requires a
separate explicitly authorized evaluation path.

## Selected collection 1t1r acquisition

The canonical collection default now uses source Tx 0/Rx 0. Ordered antenna
selection precedes power normalization, occupancy and fitting; statistics,
resume and readout bind the selected acquisition and source geometry/frequency
identity. The released 100-profile sampler samples with replacement and thus
repeats the sole pair. Its model, sampling rule and optimizer budget remain;
repetition is not additional independent measurements or a measured speedup.
Native GOTCHA remains one pair per pulse, with every selected pulse retained.
See [antenna selection](ANTENNA_SELECTION.md); original TCNN CUDA qualification
and the existing native readout limitations remain separate gates.
