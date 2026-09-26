# GeRaF v1: source fidelity and dataset adaptations

Native GOTCHA supports shared [pulse](GOTCHA_PULSE_SELECTION.md) and
[frequency](GOTCHA_FREQUENCY_SELECTION.md) selection in train/validation/test.
Selected MF targets retain released frequency-sum normalization without stride compensation.

Current selection is MF48 with G48 zero-SDF extraction; prior MF101 remains
an explicit recovery configuration. This changes target discretization, while
the source networks/loss/banks stay unchanged. See the
[scene-budget contract](SCENE_BUDGET.md) for configuration and recovery.


Current implementation contract, 2026-09-19. This document supersedes the earlier
hardening assessment. **`source_v1` is the comparison implementation.**
`hardened_v1` and `legacy` are explicitly selected historical adaptations; neither
is an authentic reproduction. Earlier tests established their numerical behavior,
not fidelity to the authors' model. Several earlier changes were discretionary
and were not justified by missing information.

The GeRaF implementation and root-trainer wiring are complete for both the RIFT
collection and native GOTCHA, including guarded preparation, resume and
validation-selected readout. This is an integration status, not experimental
qualification. The configuration table and missing-information sections below
record the unresolved differences that must carry into the paper. Destination
setup and launch instructions are in the tracked
[NCSA Delta handoff](../NCSA_Delta_Production_Handoff.md); the local project-memory
ledger mirrors these disclosures but is not required by a fresh clone.

## References and the limit of the reproduction claim

- User-provided [GeRaF v1 paper](https://arxiv.org/html/2605.29097v2).
  The arXiv revision suffix is not GeRaF 2.0.
- [Official release, pinned commit 38266cb6e194e2f3dcbead614069a7281ffd21a5](https://github.com/VictorLlu/GeRaF-SENS/tree/38266cb6e194e2f3dcbead614069a7281ffd21a5).
- [Pinned data-preparation contract](https://github.com/VictorLlu/GeRaF-SENS/blob/38266cb6e194e2f3dcbead614069a7281ffd21a5/docs/PrepareData.md): per-frame float32 MF cubes and their accumulated sum.
- `geraf/models/rendering/rf_rendering.py::GeRaFStage1`, the class mentioned by
  the README; `geraf/models/networks/sdf_network.py`; `model_utils/field.py`;
  `datasets/transforms/{sample,loading,transform}.py`; the signal/MF CUDA kernels;
  and `tools/train.py`.
- The example launcher is `configs/geraf/bunnyboxv1/geraf2_bunnyboxv1_stage1.py`.
  The inspected initial release has no parent history recovering a historical v1
  launcher/checkpoint. A stage named "1" is not proof of a v1 experiment recipe.

The defensible description is **the released GeRaF stage-1 implementation
configured to the explicit v1 paper settings, with native-acquisition bindings
and disclosed source fallbacks**. Exact reproduction of the historical v1
experiment has not been established. Public implementation details are reused;
missing historical settings are not grounds for redesigning available code.
No choice below was selected using comparative RIFT/GeRaF performance.

## Implementation map

| Responsibility | Maintained code |
| --- | --- |
| Root trainer and native GOTCHA hook | `train_geraf.py` |
| RIFT collection selection | `train_rift_dataset.py`, default `--geraf-implementation source_v1` |
| GOTCHA selection | `train_gotcha_dataset.py --method geraf`; discovers the literal hook without model imports |
| Same source model on both datasets | `rift/geraf_source.py`, `rift/vendor/geraf_sens/rf_rendering.py` |
| Native acquisition operators | `rift/geraf_source_ops.py` |
| Guarded dataset ingress | `rift/geraf_source_data.py`, `rift/geraf_gotcha.py` |
| Targets, training, source banks, resume and selection | `rift/geraf_source_training.py` |
| Native validation readout, both datasets | `scripts/eval_geraf_source.py` |
| Collection zero-SDF geometry and metric mesh scoring | `scripts/eval_geraf_geometry.py`, source checkpoint branch |

The complete `GeRaFStage1` and shared base definitions are copied, including
`loss()`, its detached initial signal bank, rotating strided antenna groups,
current-versus-stale MF mixture, and checkpoint cache handling. SDF, reflectivity,
variance, primary/target sampling, measured mask, target-volume interpolation,
and LR scheduler definitions are also copied. Unmodified example configuration
and training-loop references accompany them. Registry/import plumbing is
replaced with a small local shim. `source_manifest.json` records the pinned
source definition AST hashes and original CUDA file hashes. No independent
replacement loss is substituted inside the copied stage. License and attribution
are included in `rift/vendor/geraf_sens/`.

## Paper, released implementation, and the chosen configuration

| Item | Evidence and current choice | Reason/status |
| --- | --- | --- |
| SDF architecture | Source `SDFNetwork`, `n_layers=8`, width 256, skip 4, scalar output, 10 PE levels | Explicit v1 dimensions/PE; source implementation of layers, raw XYZ, skip sizes, geometric initialization and weight normalization |
| Reflectivity | Source `ReflectivePowerNetwork(feats_dim=0)`, normalized position only; four **linear layers**, 256-wide hidden layers, ReLU, weight normalization, sigmoid | V1 learned position-dependent reflectivity. The implementation is available; the earlier separate softplus network was unjustified |
| Global transmit amplitude | Source `exp(light_power)`, trainable scalar; source class default `light_power=0` | Source default; historical v1 initialization is unavailable. No data-fitted initialization or extra complex gain |
| Sharpness | Source `SingleVarianceNetwork`, variance=.3, `exp(10*variance)`, source clips | Release example fallback; preserves both parameterization and optimizer scaling, not the old independent log-sharpness parameter |
| Loss | Source `loss_mode='l2'`, mean squared MF **magnitude** residual | Explicit v1 objective overrides the v2 example's Charbonnier choice; no complex/MF-power auxiliary objective |
| Gradient penalty | Source gradient-norm term, weight .1 | Historical v1 coefficient unestablished; released stage-1 example fallback, **not** claimed as specified by the v1 paper |
| Sampling | Exact source uniform sphere sampler: aperture grid 32, jittered disk cells, random coarse depth 32; target depth 64 and original interpolation/inbounds rules | Retain release implementation and dimensions, including unusual cell offsets, repeated final spacing and endpoint behavior. No midpoint/AABB replacement |
| Opacity | Original `EPS=1e-5` numerator/denominator, annealed cosine, alpha clipping; `TRANS_EPS=1e-7` cumulative survival | Keep source numerical behavior, including nonzero empty-cell alpha; no log-CDF repair or exact-cell quadrature substitution |
| Lensless correction | Source selected-receiver mean, unit-sphere intersection, first-sampled-point CDF, detached additive correction, **no added clamp** | Exact stage-1 routine; no separate Tx/Rx correction or near-edge replacement |
| Two-way weighting | Source signal network receives transmission, then multiplies `alpha*transmission`; source color floor 1e-6 remains | Keep the authors' finite-cell approximation and floor, even where a different integral could be derived |
| Dynamic mask | Source measured current/accumulated whole-ray mask, applied **before both rendering and MF query selection** | Exact transform and stage loss; no prediction history and no loss-only reinterpretation |
| Mask thresholds | .04 current, .15 accumulated | Released example fallback. Source class defaults .05/.1 are not the example's effective values; historical v1 thresholds are unknown |
| Antenna bank | Exact strided groups, bank size 2, detached initial signals, round-robin updates and stale other groups | Release example fallback for bank size; paper and code establish the signal-bank mechanism. No full refresh during training |
| Optimizer | AdamW, SDF LR 1e-4, others 1e-3; source example weight decay 0 and gradient clip 35 | Rates/optimizer from v1; coefficient/clip are disclosed release fallbacks, not old local defaults |
| LR schedule | Copied source scheduler, 50,000 steps, **absolute** eta_min=5e-4 for each group | Released example fallback. This increases SDF LR from 1e-4 toward 5e-4. It is not silently reinterpreted as a per-group 0.5 ratio |
| Step-dependent schedules | Default `model_step_policy='released_runner_zero'` | Supplied `tools/train.py` never calls `update_step`; the config names an absent/unexecuted `StepHook`. Thus cosine ratio stays zero and sharpness stays frozen under the example's freeze threshold. This is observed release behavior, **not proof of historical v1 behavior** |
| Optional step policy | Explicit `model_step_policy='advance'` calls `update_step` | A different checkpoint-bound recipe, not the default or an automatic correction. Author clarification is needed before calling it historical v1 |
| Coordinate PE | Source `sin(2**l*x)` on normalized coordinates, with raw XYZ | Paper writes a pi-scaled PE. This is a documented paper/code conflict; retain the available implementation rather than invent a third encoding |

The source class default learned power network is used instead of the example's
`FixedPowerNetwork`, and the SDF does not emit 256 reflectivity features. These
are **v1-versus-v2 configuration distinctions supported by the paper**, not
performance improvements. The remaining example fallbacks above cannot be
certified as v1 hyperparameters without further author evidence.

## Necessary native-data bindings

The original kernels are specialized to the authors' radar configuration. The
following changes are explicit acquisition/protocol changes, not model changes.

| Difference | RIFT | GOTCHA | Why required / limit |
| --- | --- | --- | --- |
| Measured samples | All 256 calibrated bistatic pairs and all 600 native frequencies; average redundant static chirps | Every native pulse in each pass-sector; frequencies may differ by pass/polarization | Neither dataset is the original decimated TI capture. No invented channels, padding, frequency resampling, or TI ADC settings |
| Phase kernel | Actual `exp(-i*2*pi*f/c*(Rt+Rr))` | Actual `exp(-i*4*pi*f/c*(distance-r0_effective))` | Native measurement conventions. The release's carrier/slope, sign convention, scaled speed of light and +0.15 m hardware path bias cannot be imposed on these measurements |
| Autofocus/reference | No copied hardware calibration | Published channel-owned HH/VV corrections applied once by the unchanged loader; cross-pols raw | Required by existing GOTCHA metadata; never refitted or borrowed across channels |
| Numerical evaluation | Existing paired range NUFFT with dense checks; float64 phase geometry | Chunked exact sums at the stored nonuniform frequencies, float64 phase geometry | Native near-/far-range coherent phases require the registered precision. Source CUDA interpolation/specular algebra is ported to Torch; CUDA bitwise/kernel performance parity is not claimed |
| Amplitude law | Both retain source `sigma*specular/(Rt+Rr)^2`, back-face/specular gates and division by the total target sample count | Same; monostatic Rt=Rr | Prior product spreading, extra `(4*pi)^-2`, missing point-count normalization and changed gates are removed. Model amplitudes are not retuned to compensate |
| MF normalization | Sum frequencies coherently, average over paired channels, then magnitude | Sum frequencies coherently, average over pulses, then magnitude | Match source normalization; no extra division by frequency count or target peak |
| Scene placement | Registered centered scene, radius/half-extent .15 m | Registered region frame and half-extent (Camry 5 m) | Objects/scenes differ from the original robot scene. No mesh-derived initialization, alignment or target geometry used in fitting |
| Primary-ray orientation | Calibrated array axes projected perpendicular to view direction | Local sector aperture direction, projected perpendicular to scene-center direction | These datasets do not provide the authors' robot `rotation.npy`; deterministic orthogonal fallback if the aperture is degenerate |
| Multiple channels/passes | Separate scenes/checkpoints per registered object | Independent model per selected polarization, shared jointly across selected passes | No published polarimetric GeRaF extension; do not introduce one or pool incompatible channel phases |
| Roles and training coverage | All 3200 registered train IDs, 1000 validation; other roles inaccessible (production campaign: 2400 train at Tx 0/Rx 0) | 2000/440 train/validation pass-sectors by default; reserved 440 inaccessible (production campaign: 1500/440, 16-pulse cap, frequency stride 2) | Required common benchmark splits. Iteration count stays 50k; ordered cyclic minibatches record actual successful exposures. Production values: [TRAINING_RECIPES.md](TRAINING_RECIPES.md) |

Native operators retain the source specular normal Jacobian, including its
epsilon convention, rather than replacing it with a different physical model.
The original CUDA files remain alongside the port as references. Training still
uses the source CUDA autocast sites; CPU engineering tests disable CUDA autocast
through PyTorch. Real CUDA equivalence, memory and throughput remain unmeasured.

## Missing preparation and acquisition information: explicit fallbacks

1. **Accumulated MF construction:** the pinned `docs/PrepareData.md` specifies
   the sum of all per-frame float32 `sarimage.bin` volumes. Summation is therefore
   source-documented, correcting the earlier claim that accumulation itself was
   unknown. The adapter sums **training-only measured MF magnitudes** on the same
   world grid. Sealed role restriction and native MF construction are acquisition
   adaptations; exact upstream preprocessing/calibration and historical-v1
   settings remain unverified. No validation contribution, prediction history,
   coverage weighting, thresholding or fitted per-view weight is inserted.
2. **Selected MF target protocol: 48³, accumulated-only storage.** At the user's
   explicit instruction, both production paths use a 48³ MF grid with
   `target_storage=lazy_trilinear_accumulated_only_v1`: the collection runs use
   `protocols/geraf_mf48_1t1r.json` (bank size 1 at Tx 0/Rx 0), GOTCHA uses
   `protocols/geraf_mf48.json` (bank size 2).
   The released example uses 601³ (0.6 m bounds, 1 mm spacing, inclusive endpoints).
   Choosing 48³ is a disclosed resource-driven change to target interpolation
   and potentially masks/losses; it is not claimed equivalent to the finer grid.
   The registered scene stays unchanged. With inclusive endpoints the 48³ lattice
   spacing is 0.3/47 ≈ 6.38 mm for the 0.3 m collection cube and 10/47 ≈ 0.213 m for
   the 10 m Camry cube. All views, native samples,
   source model/samplers/masks and 50,000-update budget remain unchanged.
3. **Response units:** the original loader divides targets by its radar's known
   `trans_power`. That calibration does not describe these datasets. Default
   `trans_power=1` retains native response units and is recorded explicitly.
   The scalar power remains trainable with the source class initialization.
   No target-peak normalization or first-view amplitude fitting is added.
   Missing radiometric calibration can affect conditioning and must be disclosed.
4. **Native NVS evaluation:** the release's `predict()` extracts fields; it does
   not provide these datasets' NVS evaluator. Validation renders every channel
   freshly without updating the training bank, with fixed seeded source rays,
   no measured dynamic mask, and the source interpolation-validity restriction.
   Select by pooled native MF-magnitude MSE and separately report coherent native
   response error. Training remains the copied stale-bank source loss. These are
   benchmark readout/selection choices, not established historical v1 evaluation.
5. **Geometry protocol:** report the native **zero-SDF** surface, metric units,
   using the same validation-selected checkpoint. The paper defines a zero SDF
   surface; the v2 example's 0.04 offset is not adopted as a v1 setting. The
   common RIFT protocol does not apply the paper's post-hoc mesh alignment or
   its exact point-count/tolerance; use the registered metric mesh and disclose
   benchmark sampling/tolerances. No isolevel tuning or component filtering.

**Final storage implementation:** `SourceTargets` persists only
`accumulated_train_<head>.npy`: **442,496 bytes/head** at MF48, including the NPY header.
It streams each training view's lattice chunks directly into the FP32 sum;
no training or validation view cube is constructed or written. Atomic preparation
recovery stores the last complete-view sum/cursor, bound to source, role order,
head, recipe and checksum. A partial next view is replayed, never double-counted.
The progress file is removed after final checksum publication. Allow about
0.89 MB plus metadata during atomic preparation. Multiple polarization heads
remain independent; the intended HH comparison has one grid.

`LazyMFTarget` evaluates only the unique in-bounds lattice corners needed by
the source's sampled target positions (at most 8 × 807 × 64 before deduplication).
`geraf_target_sampling.sample_lattice` reproduces FP32 coordinate rounding,
zero padding, axis order and trilinear accumulation. Magnitudes are quantized
to FP32 **before** interpolation, as in the original dense writer. The accumulated
mask uses the same training-only sum. Direct MF evaluation at query positions
is not substituted for interpolation, and jitter/source mask definitions are
unchanged. Dense-vs-lazy parity holds at the same grid; it does not establish
48-vs-601 fidelity or convergence.

Preparation still computes every training view over the 48³ lattice, and each
optimizer/validation step now incurs lazy MF work. Previous dense601 timing
estimates are stale. Source response banks remain in checkpoints; the 0.442 MB
figure is the target cache, not total run storage. Bank sizes depend on selected
training counts and antennas. Original CUDA and real-data
runtime/convergence still require manager-run qualification.

## Wiring and compatibility

Metadata-only planning:

```bash
python train_rift_dataset.py --object a320 --method geraf --dry-run
python train_geraf.py --object a320 --checkpoint-dir /tmp/geraf-plan --dry-run
python train_gotcha_dataset.py --method geraf --dry-run
python train_gotcha_dataset.py --list
```

RIFT source runs use `<output-root>/<object>/geraf/source_v1/`; preparation is
inside the new source trainer. The old ray-cache preparer is not invoked for
`source_v1`. `--geraf-source-config FILE` forwards a JSON mapping of explicit
settings. Direct root flags are shown by `train_geraf.py --help`. GOTCHA uses its
existing `--method-config FILE` mechanism, with `{"geraf": {...}}` overrides;
for this single method, `--config protocols/geraf_mf48.json` accepts the same flat
config as collection `--geraf-source-config protocols/geraf_mf48.json`.

New checkpoints bind source/object or region, roles, acquisition metadata,
recipe, target preprocessing, model states, source antenna-bank contents and
pointers, optimizer, LR position, all RNG states, exposures, pending validation,
and validation selection. Identity gates precede response access. Historical
`hardened_v1` or `legacy` checkpoints cannot load into this schema. Those old
recipes remain selectable explicitly for recovery; their existence does not
make them comparison reproductions. The new storage field/cache schema also
rejects older dense `source_v1` caches and checkpoints, even with an explicitly
matching grid size. Use fresh output/cache roots; no automatic migration or
deletion of historical artifacts occurs.

The source trainer writes `source_run.json` for the bound data contract and
recipe, `checkpoint_latest.pth.tar` for recovery, `checkpoint_best.pth.tar` for
MF-validation selection, and `validation.json` for selection history. Its target
cache has a separate `source_targets.json` identity manifest and one checksummed
accumulated volume per head. `--prepare-only` builds only that train accumulator;
it does not read validation responses. Completion is recorded in the latest checkpoint after terminal
validation; this runtime does not write a separate final-checkpoint file.

Readouts when a matching validation-selected checkpoint exists:

```bash
python scripts/eval_geraf_source.py --dataset rift --object a320   --checkpoint PATH/checkpoint_best.pth.tar --cache-root PATH/targets --output PATH/validation.json
python scripts/eval_geraf_source.py --dataset gotcha --region camry   --checkpoint PATH/checkpoint_best.pth.tar --cache-root PATH/source_targets --output PATH/validation.json
python scripts/eval_geraf_geometry.py --object a320   --checkpoint PATH/checkpoint_best.pth.tar --output-dir PATH/geometry
```

The old `eval_geraf_complex_response.py` remains a compatibility evaluator for
old schemas; it is not the new source checkpoint readout. Reserved-test source
readout is not implemented or enabled by this development integration.

## Validation and remaining evidence

The final 101³/lazy integration passes **174 focused synthetic tests**: source
AST/hash checks, interpolation/mask parity, native operators/gradients, the
full-size 4,121,332-byte accumulator, interrupted preparation and integrity,
checkpoint stop/resume/readout, and both dataset frontends. Real RIFT/GOTCHA
metadata plans accept the selected config with response readers blocked and
no output writes. These establish implementation behavior, not CUDA feasibility.

At the time of that verification, no real-data fitting, large target preparation,
CUDA execution, scheduler action or manager-run experiment had been performed. PVC
production runs have since fitted the collection scenes; see
[TRAINING_RECIPES.md](TRAINING_RECIPES.md). Synthetic correctness does
not establish baseline convergence, runtime feasibility, historical v1 parity,
or fair experimental ranking. The remaining work has distinct evidence needs:

- **Author evidence:** recover the historical v1 recipe and resolve the step-hook,
  LR/encoding, per-frame MF preprocessing and radiometry uncertainties above.
  Successful local training cannot establish what the authors originally ran.
- **Experimental qualification:** assess the default target-cache cost, CUDA
  execution, fitted-field sampling/mask consequences and full-data convergence
  through the experiment manager when requested. Any resource-driven recipe
  change must be explicit and added to this adaptation record.
- **Benchmark evaluation:** implement a separately authorized reserved-test
  readout before final test reporting. Use the same validation-selected field
  for native signal and available metric geometry readouts. Native acquisition
  and evaluation adaptations remain paper disclosures even after validation.

## Selected collection 1t1r adaptation

The canonical collection dispatcher now selects physical Tx 0/Rx 0 by default.
A single measured channel cannot populate the released two-bank schedule.
`recipe_for_data` therefore selects `bank_size=1` for this acquisition, with
recipe marker `antenna_bank_adaptation=single_nonempty_bank_v1` and optional
`protocols/geraf_mf48_1t1r.json`. Explicit incompatible bank counts are rejected.
The attributed vendor implementation is unchanged: `SingleBankGeRaFStage1`
overrides only the empty-unselected-bank gather, stores the selected detached
bank, and returns zero-row tensors of the correct shapes. Selected-channel
responses remain differentiable; source loss and antenna normalization remain.
The pointer stays zero modulo one. Tests exercise nonzero finite gradients,
checkpoint bank shapes, exact interrupted recovery and selected-checkpoint loading.

The SDF/reflectivity networks, 48³ accumulated-only target policy and 50000
updates remain. Native GOTCHA still uses two banks over all sector pulses.
Other physical antenna selections retain their ordered source geometry; full
MIMO and different-pair cache/checkpoint identities cannot be relabeled. See
[shared selection contract](ANTENNA_SELECTION.md). CPU checks do not establish
CUDA behavior, fitted quality or runtime savings.
