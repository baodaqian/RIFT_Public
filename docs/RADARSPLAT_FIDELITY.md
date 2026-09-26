# RadarSplat: released implementation and dataset adaptations

Native GOTCHA supports shared [pulse](GOTCHA_PULSE_SELECTION.md) and
[frequency](GOTCHA_FREQUENCY_SELECTION.md) selection in train/validation/test.
Conversion uses selected acquisition means; Gaussian fitting retains its budget.

The user-selected `budget48` profile uses **112000 Gaussians** and a G48
occupancy-union support readout. Gaussian count is an explicit departure from
the released 20000; `upstream` preserves the original count and checkpoints.
The source manifest, CUDA renderer, losses, 2000 updates and raster stay unchanged. See the
[scene-budget contract](SCENE_BUDGET.md) for configuration and recovery.


The collection default is now **`budget48`**, using the authors' actual CUDA
renderer, initializer and Adam groups. `audit_v1` and `legacy` are retained only
for their historical checkpoints. Neither historical profile is an exact
implementation of the released baseline. This document supersedes the earlier
claim that correcting the independent Torch implementation was sufficient for
source fidelity.

The intended comparison is **released RadarSplat with explicitly converted
RIFT or GOTCHA acquisitions**, not a reproduction of the Boreas experiment.
Both root dataset trainers now reach the same source model/training engine;
GOTCHA has its own native conversion and checkpoint identities. Original CUDA
execution and real-data convergence have not been validated. Backend discovery
means the adapter is wired; it does not certify installed CUDA dependencies or
benchmark readiness.

## Authority and reproducibility

References are the supplied [paper v1](https://arxiv.org/html/2506.01379v1) and
[official release at `ea9c8f530c708622cc3b1b560436b5557ac6a49b`](https://github.com/umautobots/radarsplat/tree/ea9c8f530c708622cc3b1b560436b5557ac6a49b).
The executable authority is the complete chain
`examples/demo_scripts/run_all_radarsplat.sh` → `run_radarsplat.sh` →
`examples/radar_simple_trainer.py`, with `gsplat/rendering.py` and its CUDA code.
A Python dataclass default alone does not describe the released experiment.

`protocols/radarsplat_official_reference.json` records source SHA-256 hashes,
launch settings, GLM commit `33b4a621a697a305bc3a7610d290677b96beb181` and
fused-SSIM commit `1272e21a282342e89537159e4bad508b19b34157`. The unmodified
source, including license and attribution, is under
`external/radarsplat_reference/<commit>/`. The upstream license is
CC BY-NC-SA 4.0; bundled components retain their own licenses. Source retrieval
is reproducible through `scripts/fetch_radarsplat_reference.py`. It checks the
archive's file inventory, preserves existing files, and does not install
dependencies or start training. Historical source hashes are documentation
only; GitHub source and dependency commits are recorded without hash gates.

`rift/radarsplat_release.py` verifies these source bytes before executing them.
It imports the original rasterizer and CUDA kernels. To avoid the upstream
trainer's GUI/dataset side effects, it executes the unchanged function ASTs for
initialization and preprocessing. It requires the pinned fused-SSIM package and
rejects an already imported different gsplat. There is no automatic CPU renderer
or generic gsplat substitution. The older local denoising helpers retain the
same attribution and license.

## Paper/release conflicts: use the executable release

The paper's supplementary configuration and the subsequently released launcher
do not specify the same experiment. The following choices follow the release,
not a mixture selected for better results. The missing information is an
executable recipe proving which conflicting settings generated the v1 tables.

| Setting | Paper v1 | Pinned release used here |
| --- | --- | --- |
| Updates | 3,000 | 2,000 |
| Gaussian count | 20,000 | 112,000 in user-selected `budget48`; explicit `upstream` retains 20,000 |
| Initial occupancy/noise | 0.1 / 0.1 | 0.5 / 0.5 |
| Initial size | 0.5 m | Argument 0.5 is applied twice; actual initial scale is 0.25 source units |
| SH | Level 10 | Argument 5, advanced every 200 steps; actual CUDA kernel implements through degree 4 |
| Occupancy / probability-penalty weights | 5 / 100 | 10 / 1,000 |
| Occupancy map | Ten frames; threshold 0.15 | Window radius 5 gives eleven interior frames; launcher threshold 0.10 |
| Decay-mask smoothing | Sigma 5 | Sigma 3 |
| FFT detector | Normalized description | Executable uses unnormalized magnitude and a half-spectrum, DC-inclusive ratio |

The source initializer is planar (`z=0`); this is retained even though RIFT
views cover a sphere. The original code's Cartesian-to-spherical covariance
path and its SH direction computation are also retained. In particular, our
previous independent implementation's covariance rotation and Cartesian SH
direction correction are **not** inserted into the author's renderer. These
source behaviors may deserve author clarification, but silently fixing them
would create a different comparator.

The full objective is
`0.8 * mean_power_L1 + 0.2 * (1 - fused_SSIM) + 10 * mean_occupancy_L1 +
100 * max_size_penalty + 1000 * probability_penalty`.
The size penalty uses the source log-scale clamp at 10 and size threshold 1.
The probability penalty remains `mean(relu(sigmoid(alpha) + sigmoid(eta) - 1))`;
it does not independently drive noise to zero.

The source keeps separate power/occupancy products, the per-product 1/255
raster cutoff, raw clipping to [1e-6, 1], original spectral/antenna filters,
0.6 multipath addition and final [0, 1] power clipping. The launcher sets
`refine_stop_iter=0`, so no pruning or splitting occurs. We do not add the old
independent opacity-pruning schedule, 3-D initialization or ablations.

Adam uses the source parameter groups and epsilon 1e-15. Means start at
`1.6e-4 * scene_scale`, scales at .005, quaternions at .001, occupancy/noise
at .05, and both SH groups at .0025. Only the means LR decays exponentially
to 1% over 2,000 updates. Betas remain .9/.999. The source's 1.1 scene-scale
margin is retained after the coordinate conversion below.

## Necessary dataset conversion: complete decision ledger

These choices address missing scanning-radar inputs, different physical units
and the common sealed evaluation protocol. They were not selected from fitting
results. They change the acquisition adapter and must be disclosed with any
reported baseline result.

| Boundary | Implemented choice | Why it differs and practical limit |
| --- | --- | --- |
| Observable | Convert native complex responses to `sum_elevation(abs(MF_complex)**2)` on calibrated polar samples. | The collection supplies bistatic frequency responses, not scanning-radar PNGs. The source model predicts real intensity, not complex phase. |
| Intensity units | New runs (user decision 3, 2026-09-22; `intensity_mapping = log_train_peak_60db_v1`): TRAIN-peak-relative power over 60 dB to [0,1], the release's dB-quantised Navtech image domain and our Radar Fields mapping; every release threshold unchanged. Earlier runs (`linear_train_peak_v1`): divide by a single training-only peak and clip to [0,1]. | Scanner byte calibration is unavailable. Validation uses the same peak, never a fitted validation normalizer. Metrics explicitly say **clipped normalized power**. |
| Coordinate units and support | Map the registered cube `[-e,e]^3` to `[-50,50]^3` source units: `units_per_m=50/e`. Set parser-equivalent scene scale 100, then retain the source 1.1 margin, yielding scale 110 and initial xy support ±55. | The source combines a driving trajectory and 50 m sensing radius to establish road-scene support; that convention is unavailable for object-centred spherical views. The deterministic conversion keeps road-scale size constants meaningful for the 0.30 m scene. It is an explicit geometry-only normalization, not a claim of identical physical Gaussian sizes. Export divides positions/scales by this factor. |
| Pixel lattice | Use scene-support sampling, 33 range/azimuth/elevation samples and Q=10, from registered extent and calibrated standoff. | Native scanning axes are absent. This samples the scene; it does not improve physical sensor resolution. Stored bin centres and poses are validated against acquisition metadata. |
| Original filter execution | Rasterize a complete circular azimuth lattice, filter with the original kernels, then sample the requested calibrated crop. Add the original range-filter support as a halo. | The released azimuth helper assumes 360°; applying it directly to a narrow image would give the wrong stride and boundary behavior. Full-circle length is rounded to a multiple of the integer stride. Local output uses periodic bilinear sampling. |
| Sensor filter inputs | Supply the target cache's range spacing, leakage width and azimuth beamwidth; retain the original filter formula, sigma-four-pixel antenna kernel and cutoff behavior. | Boreas sensor dimensions do not describe this acquisition. These cache values are an acquisition approximation, not measured PSF equivalence. They are recorded, not tuned. |
| Near-range mask | Apply the source 2.5-unit exclusion at physical range after conversion, rather than masking the first bins of a distant crop. | Local bin zero is not the sensor origin. |
| Occupancy donors | Eleven nearest **training** sensor directions, deterministic view-ID tie breaking; apply the released decay mask, spatially reproject to 3-D polar samples, average visible powers, threshold at .10, and project occupied elevation samples. | There is no ordered driving sequence or ground plane. Out-of-coverage samples are unknown; labels are restricted to the registered cube. A 3-D lift replaces temporal ground-plane mapping. Small synthetic subsets use only their available donors. |
| FFT sample-count units | Compare detector magnitude to `30 * N / floor(50/.0596)` for a crop with N range samples; DC ratios and skipped frequencies stay unchanged. | The release's FFT is unnormalized. Keeping threshold 30 on 33 samples bounded by one would make non-DC detection impossible. This algebraic sample-count conversion preserves the source amplitude criterion; it is not a fitted noise threshold. It applies to both denoising and multipath detection. |
| Multipath input | Execute the original FFT and periodic/decay fitting functions on training images in the converted distance units. Reproject the nearest training direction's fitted image to the query. Keep source weight .6. | Precomputed multipath images and a driving source map are absent. A held-out query never fits its own background. The fitting origin is the local crop origin; this and the nearest-direction transfer are disclosed approximations. An image with no detected source contributes zero background. Fit failures fail explicitly. |
| Roles and updates | Use the registered 3200/1000 train/validation roles (production campaign: 2400/1000 at Tx 0/Rx 0). Sample shuffled training cycles with seed 42 and run exactly 2,000 updates. | Source every-fifth-frame holdout does not describe this collection. The sampler has an explicit resumable state. Its precise order differs from the source PyTorch DataLoader, but its sampling policy and fixed update budget are retained. This is not a 150-epoch/equal-compute comparison. |
| Selection and geometry | Evaluate the final source step on sealed validation. Export the same checkpoint's metric Gaussian means, scales, quaternion, occupancy and noise, with checkpoint hash. | Our paper requires signal and geometry from the same field. No LiDAR, mesh, surface prior or fabricated coherent phase enters training. The export is occupancy support, not a recovered surface. |
| Recovery | Save original model/Adam/LR/sampler state and strict source/object/role/cache/calibration identity. | This is repository infrastructure. Raw tensor dtypes, shapes, finite state and source hyperparameters are validated before Torch can coerce them. Historical checkpoints cannot resume under this schema. |

The launcher also retries NaNs and eventually changes the probability weight
from 1,000 to 10. That is a different objective selected after failure. This
integration preserves the successful-path recipe and stops on nonfinite state;
it does not silently execute the fallback or choose a successful retry for a
reported score. A needed fallback would require its own disclosed recipe and
failure report, not relabeling the resulting model as the default release.

## Audit closure and remaining limits

The initial discrepancy audit identified replacing occupancy supervision,
changing its reduction, conflating noise and occupancy behavior, and assuming
MF images matched the Gaussian operator. The source profile now uses separate
multiview occupancy, the released mean loss/SSIM/regularizers and original
renderer products/cutoffs. Recovery and signal/geometry provenance are explicit.
The old independent renderer is not used for the new comparator.

The **measurement-model discrepancy remains structural**. A coherent matched
filter contains interference and an acquisition-specific PSF; an additive
nonnegative Gaussian renderer does not. A two-scatterer diagnostic gives power
4 or 0 for coherent relative phase 0 or pi, versus additive power 2. Prior tiny
operator probes establish MF/gradient arithmetic, but their local-Torch Gaussian
comparison is not CUDA parity evidence for the source profile. Changing the
source model to synthesize complex phase would defeat the requested comparison.

These adaptations still need real-data inspection of occupancy masks, fitted
multipath, signal scaling, GPU allocation and convergence. No measured result
shows whether the unmodified planar initialization and fixed source budget
work well on these acquisitions. No real fitting, reserved-test evaluation,
scheduler action, manager launch request or publication was performed here.
The native GOTCHA implementation below preserves native ragged frequencies,
per-pulse reference ranges, source-owned autofocus and pass-sector roles. It
does not reinterpret a simulated-data cache or historical transfer checkpoint.

## GOTCHA root integration and additional conversion choices

`train_gotcha_dataset.py --method radarsplat` discovers the literal
`GOTCHA_BACKEND` / `run_gotcha` hook in `train_radarsplat.py` and calls the owned
`rift/radarsplat_gotcha.py` backend. At the user's subsequent request,
the separate `train_gotcha_dataset_radarsplat.py` was merged and removed.
Its former detailed planning report is preserved by
`rift/gotcha_baseline_planning.py`. HH remains the intended default channel.
RIFT continues through `rift/radarsplat_collection.py` into the root trainer.
Both paths call `rift/radarsplat_release_training.py`: 112,000 planar Gaussians in `budget48`,
2,000 updates, and the original objective, optimizer groups, schedule and
kernel behavior remain fixed.

GOTCHA supplies monostatic complex phase histories, not a scanner's power
images. For pass-sector v, the adapter first computes

```
MF_v(x) = mean_pulse(mean_native_frequency(
    S_p(f) * exp(+i * 4*pi*f/c * (norm(x - antenna_p) - r0_effective_p))))
P_v(azimuth, range) = sum_elevation(abs(MF_v(x))**2)
```

Every selected pulse contributes before taking power. Native frequencies may be
nonuniform and have different counts in different passes; no FFT, uniform-grid
substitute, discarded pulses or frequency padding enters this calculation. The
production campaign first applies the user-pinned selection (16 pulses per
sector, frequency stride 2, every role), so "every pulse" means every selected
pulse there.
FP64 geometry/phase and complex128 accumulation are used. The public
`GOTCHADataset` supplies the corrected reference range and response: published
HH/VV autofocus is applied once, and cross-polarizations remain raw. Unit native
phase-history weighting is retained; no synthetic RIFT range-spreading factor
is added.

| GOTCHA adaptation | Choice and reason |
| --- | --- |
| Training unit | One image per registered pass-sector, made from all its native pulses. All selected passes share one scene per polarization. Different polarizations have independent original models, optimizers, normalizers, targets and checkpoints; channels are not extra viewpoints. |
| Target grid | Use the registered region-local cube and its circumscribed sphere. Default azimuth/elevation counts are 33/33. Range covers ±sqrt(3) times the half extent about the sector's mean sensor distance, with an odd count of at least 33 and spacing at most half the finest selected native Rayleigh interval `c/(2*bandwidth)`. Current default Camry HH metadata yields 145 range bins. This is sample placement, not super-resolution. |
| Virtual image pose | The sector's mean native sensor position defines an image coordinate frame. Its outward x-axis places the region at azimuth pi, away from the circular seam. This pose only labels the image and supplies RadarSplat's scanning-sensor input; MF formation still uses each actual pulse position and reference range. |
| Native sensor filter approximation | The unchanged range-kernel formula receives width `6*c/(2*bandwidth)` for that pass; source integer discretization determines its actual sigma. The azimuth support uses `wavelength/(2*aperture)`, with aperture measured relative to the sector's first sensor position; at least two intermediate pixels are retained. A single-position synthetic sector uses the ROI angular span. The source sigma-four-pixel kernel is unchanged. These are explicit approximations to a SAR PSF; no author scanning-radar calibration exists for GOTCHA. They were not chosen from fitted results. |
| Narrow angular domain | Rasterize a local crop of the same full-circle intermediate pixel lattice with the complete source filter halo. Offset the projection's pixel origin; rescale only the original helper's degree arguments to retain the identical pixel kernel and stride despite its hard-coded `360/H`. Interior samples do not see crop boundary padding. This avoids allocating an enormous full-circle tensor for an airborne sensor viewing a small region. Synthetic filter values/gradients match full-circle evaluation; CUDA raster parity remains unverified. RIFT's existing full-circle adapter is unchanged. |
| Occupancy and multipath | Reuse the existing eleven-nearest-training-direction occupancy and training-only multipath provider separately for each channel. No validation/test signal creates labels, background fits or normalization. Geometric query poses may use validation metadata. |
| Source model units | The existing registered-cube-to-100-units conversion and source 1.1 initialization margin remain unchanged. Exported geometry is in metres in the registered **region-local frame**. The checkpoint identity records the native translation and rotation. |
| Regional clutter | Restricting MF query locations does not isolate a vehicle response. Clutter at overlapping delays and coherent sidelobes remain. No mesh, survey truth, SDF or surface prior is used in conversion or training. |

The GOTCHA adapter JSON accepts only `azimuth_samples`, `elevation_samples`,
`point_chunk` and `frequency_chunk`. Defaults are 33, 33, 128 and 256.
Azimuth must be odd and at least 11 for the source SSIM window; elevation must
be odd and at least 3. These settings are serialized and immutable on resume.
Model budgets/loss settings are not accepted as adapter overrides.

Preparation is resumable. A run-level `radarsplat_gotcha.pt` binds source,
region, selected passes/channels, split, conversion and released model recipe
before response access. Each head caches identified target files, their hashes,
and a peak bound exclusively to training targets. A stop between target images
leaves a resumable manifest; a training stop saves the original model/Adam/LR
and sampler. Every existing head is validated before another channel reads
native responses. Completed heads require consistent final/recovery checkpoints
and summary; they are not fitted again. Historical RIFT or GOTCHA transfer
checkpoints fail these gates.

Use metadata-only planning to inspect both integrations:

```bash
python train_rift_dataset.py --object a320 --method radarsplat --dry-run
python train_gotcha_dataset.py --method radarsplat --dry-run
python train_gotcha_dataset.py --method radarsplat --passes 1 2 3 4 5 6 7 8 --polarizations hh --dry-run
```

The GOTCHA default root is `training_checkpoints/GOTCHA_dataset`. Use
`--output-root training_checkpoints/GOTCHA_dataset_RadarSplat` to select a
previous dedicated-frontend run. Each root contains
`REGION/DATASET_ID_PREFIX/radarsplat/`, then `hh/` by default. Additional selected
channels retain separate directories. Resume with the explicit run-level
`--resume .../radarsplat_gotcha.pt`.
Actual conversion/fitting requires an experiment-manager allocation. None was
performed in this integration task.

The same-checkpoint GOTCHA readout revalidates current native source metadata,
then reads only the identified target cache:

```bash
python scripts/readout_radarsplat_gotcha.py --run-root RUN_ROOT --polarization hh \
  --checkpoint RUN_ROOT/hh/checkpoints/checkpoint_final.pt --role validation \
  --device cuda --output native_validation.json --geometry native_occupancy.npz
```

Optional `--dataset-root` / `--shard-root` relocate source files without changing
their identity. Reserved test is not exposed. Native target preparation cost,
original CUDA runtime/convergence, real occupancy/multipath quality and sensor
equivalence remain unvalidated.

## Files, usage and validation

RadarSplat collection command construction and recipe choices are owned by
`rift/radarsplat_collection.py`, with `tests/test_radarsplat_collection.py`.
The canonical frontends are `train_rift_dataset.py --method radarsplat` and
`train_gotcha_dataset.py --method radarsplat`. The former dedicated names,
`train_rift_dataset_radarsplat.py` and `train_gotcha_dataset_radarsplat.py`, were
removed after consolidation. Existing `RIFT_dataset_RadarSplat` and
`GOTCHA_dataset_RadarSplat` outputs remain selectable with `--output-root`.
RIFT's dependency gate still
runs before any target conversion. Owner model/recipe modules remain separate;
the merged frontends introduce no new model or data adaptation. Shared interface
details are in `docs/DATASET_FRONTEND_CONSOLIDATION.md`; ownership remains in
`docs/BASELINE_OWNERSHIP.md`.

Metadata planning performs no conversion or fitting:

```bash
python train_rift_dataset.py --object a320 --method radarsplat --dry-run
python train_rift_dataset.py --object a320 --method radarsplat sugavanam_ertin --dry-run
python train_rift_dataset.py --object a320 --method radarsplat --radarsplat-recipe legacy --dry-run
```

For a prepared, compatible target cache, the comparison command is:

```bash
python train_radarsplat.py --fidelity-profile upstream \
  --cache-root NEW_ROOT/targets --checkpoint-dir NEW_ROOT/checkpoints --no-resume
```

Execution belongs to the experiment manager. Use a new output root; direct
unflagged historical root invocations retain `legacy`. The source CLI exposes
no Gaussian-count, loss-weight, refinement or update-budget tuning flags.
The original CUDA extension and the exact fused-SSIM dependency must be built
in an appropriate allocation/environment before this command is runnable.
The local environment has no usable CUDA device; this task did not install or
build those extensions. A missing dependency fails instead of selecting another
model. Install commands and dependency pins are in the original release's
README and `examples/requirements.txt`.

Evaluate/export one final source checkpoint with:

```bash
python scripts/readout_radarsplat_checkpoint.py --object a320 \
  --checkpoint NEW_ROOT/checkpoints/checkpoint_final.pt \
  --cache-root NEW_ROOT/targets --role validation --device cuda \
  --output native_validation.json --geometry native_occupancy.npz
```

The reader also retains explicit historical recipes; reserved test is not
exposed. Checkpoint/cache/object mismatches are rejected before target reads.

`tests/test_radarsplat_release.py` verifies pinned source inventories, executes
the original initializer and loss statements, checks original filter arithmetic
on CPU, rejects a silent CPU model substitution, and tests interrupted/resumed
state, corruption rejection and matching metric geometry export. Renderer and
CUDA transfers are explicitly mocked in the bounded CPU lifecycle test; this
is not a CUDA reproduction test. Historical renderer/contract tests, static
and fullscale-configuration validators remain separate compatibility evidence.
The fetch utility has also verified both local pinned archives end to end.
All six registered objects passed metadata-only planning through the dedicated
entrypoint with response reads forbidden. The latest combined integration
selection passed 170 tests; the historical native validator passed 69 checks,
and static/fullscale-configuration validators passed. These are bounded
software checks, not trained baseline results.
The preceding 46-test RadarSplat-only suite also passed with shared dispatchers and
the SE trainer blocked from import, verifying owner independence.
`tests/test_radarsplat_gotcha.py` adds independent native-frequency/adjoint,
multi-pulse/autofocus, channel routing, train-only normalization, cache identity,
preparation interruption, two-update exact continuation, terminal recovery,
same-checkpoint geometry and crop-filter value/gradient checks. CUDA transfers
and the rasterizer are explicitly mocked in its CPU lifecycle tests. The
combined RadarSplat/collection/GOTCHA selection passes 172 tests; static and
frozen fullscale-configuration validators also pass.
Before the subsequent user-requested frontend merge, all 54 owner tests passed
with shared dispatchers and SE imports blocked; the two shared-root routing
checks pass separately. Real metadata-only planning passes for all six RIFT
objects and all eight GOTCHA passes with HH/HV/VH/VV, with response reads blocked.
The later merge intentionally replaces dispatcher-isolation assertions with
canonical routing, config isolation and failure-propagation checks. Its validation is recorded in
`docs/DATASET_FRONTEND_CONSOLIDATION.md`; model math and recipe gates are unchanged.

## Selected collection 1t1r acquisition

The canonical collection interface now selects Tx 0/Rx 0 before MF-power
conversion, train-only peak estimation and occupancy. Run, target-cache,
normalization and calibration identities include ordered source indices and
source geometry/frequency identity; full-MIMO artifacts are incompatible.
The released CUDA model, loss/optimizer and 2000 updates remain. The selected
`budget48` profile increases the Gaussian count to 112000.

For one pair the previous array-aperture beamwidth formula is undefined. The
explicit `collection_element_hpbw_v1` adaptation uses the simulator's documented
**10° element power HPBW** as the original filter's beamwidth input. This is a
declared sensor-filter approximation, not a verified two-way or MF PSF match.
The existing deterministic tangent fallback defines the polar target frame
when the selected Tx/Rx spans are zero. Asymmetric/reordered antenna selections
measure aperture over all selected positions rather than array endpoints.
The real selected bistatic coordinates still enter MF conversion exactly.
Native GOTCHA preserves its existing pulse-sector conversion and sensor rules.

A bounded synthetic one-pair conversion produces train/validation targets,
loads the resulting cache and resumes without rereading observations. Original
CUDA fitting and physical PSF equivalence remain unqualified. See
[shared acquisition contract](ANTENNA_SELECTION.md).
