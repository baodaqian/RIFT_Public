# SpINR reconstruction and dataset adaptations

Native GOTCHA supports shared [pulse](GOTCHA_PULSE_SELECTION.md) and
[frequency](GOTCHA_FREQUENCY_SELECTION.md) selection in train/validation/test.
Opt-in selections bind distinct acquisition identities; full density remains the default.

## Scope and fidelity rule

Reference: Takawale and Roy, **SpINR: Neural Volumetric Reconstruction for FMCW
Radars**, [arXiv:2503.23313v2](https://arxiv.org/html/2503.23313v2), 25 April 2025.
This is the revision of the original SpINR paper supplied by the user, not the
separate SpINRv2 method. No official implementation was recovered for this work.

Follow the paper wherever it specifies a choice. Departures below are either
necessary to consume the benchmark's measurement formulation or explicit choices
where implementation information is absent. Do not change the baseline to obtain
a favorable comparison with RIFT. New source information supersedes a local
choice and requires a new scientific recipe if it changes execution. Neither
synthetic agreement nor matching epoch count establishes original-author parity,
equal compute, numerical convergence, or reproduction of the published scores.

## Root trainer wiring

| Dataset | Root entrypoint | Maintained implementation |
| --- | --- | --- |
| RIFT six-object collection | `train_rift_dataset.py --object OBJECT --method spinr` | `train_spinr_style.py --recipe budget48-direct`; `rift/spinr_style.py`, `rift/spinr_direct.py`, `rift/spinr_fidelity.py` |
| Combined GOTCHA | `train_gotcha_dataset.py --method spinr` | Literal `GOTCHA_BACKEND` and `run_gotcha` in `train_spinr_style.py`; `rift/spinr_gotcha_training.py` and `rift/spinr_native.py` |

The GOTCHA entrypoint discovers the hook without importing or fitting the model
during planning. Its `--method all` includes SpINR for supported polarizations.
Actual execution still requires an experiment-manager allocation. This wiring
task did not request or submit an experiment.

```bash
python train_rift_dataset.py --object all --method spinr --dry-run
python train_gotcha_dataset.py --method spinr --dry-run
python train_gotcha_dataset.py --method spinr --polarizations hh vv --dry-run
python train_gotcha_dataset.py --list
```

For GOTCHA, backend options belong in `--method-config options.json`, under a
`spinr` mapping. The top-level `--epochs`, `--lr`, `--granularity`, etc. configure
the shared RIFT/MFBP backends, not this baseline. Unknown SpINR configuration
keys are rejected by its runtime. The effective defaults are exposed in the
literal hook and saved in `recipe.json`:

```json
{
  "spinr": {
    "epochs": 150,
    "cosine_epochs": 150,
    "grid_size": 48,
    "nodes_per_cell": 1,
    "pulse_batch_size": 1024,
    "seed": 42,
    "neural_point_tile": 4096,
    "renderer_point_tile": 512,
    "checkpoint_every": 10,
    "validation_every": 5
  }
}
```

These are implementation defaults, not a numerical qualification for the Camry
scene. `rift.spinr_gotcha_training.preflight(dataset, config)` resolves them and
reports pulse/update counts, kernel modes, grid spacing and cell phase scale
without opening responses. The root `--dry-run` prints the dataset, supplied
config and declared defaults; it does not execute a training preflight or fit.

The user selected **G48 midpoint on both datasets**: exactly 110592 integration
points. Collection selects the distinct `budget48-direct` recipe; native GOTCHA
uses `grid_size=48, nodes_per_cell=1`, recorded in
`protocols/gotcha_spinr_g48_midpoint.json`. Historical `paper-v1-direct` remains
G96/GL2; native GL2 checkpoints require explicit order 2 plus their saved grid.
The current user-selected comparison budget (2026-09-21) is **150 complete
training passes**, with cosine LR 1e-4 → 1e-5 over those 150 epochs and no
plateau stopping. Collection epochs cover all 2400 views once (600 four-view
updates). The production Camry selection (1,500 sectors × 16 pulses, frequency
stride 2) has 24,000 TRAIN pulses, so a native epoch is 24 batches of 1,024.
Earlier uncapped selections gave 177,605 pulses (174 batches).
This matches RIFT's pass count, not GPU time, FLOPs or optimizer-update count.
It explicitly departs from the paper's 1500 epochs. The network, physics and
objective remain unchanged, with no fitted-field quadrature qualification. See the
[scene-budget contract](SCENE_BUDGET.md); this choice does not authorize runs.

### Recipe identity and checkpoint compatibility

| Path | Scientific identity | Compatibility rule |
| --- | --- | --- |
| Current RIFT collection | `budget48-direct` → `rift_dataset_spinr_v1_passband_direct_bins_g48_midpoint_150_v1` | 150 epochs and 150-epoch cosine; frontend output ends in `spinr/budget48-direct-150/`. Object, roles, acquisition and recipe must match. |
| Historical G48 collection | `budget48-direct-1500` → `rift_dataset_spinr_v1_passband_direct_bins_g48_midpoint_1500_v1` | Preserves the original 1500-epoch recipe/schedule and `spinr/budget48-direct/` output. `paper-v1-direct` separately retains G96/GL2. |
| Earlier RIFT FFT adapter | `paper-v1` → `rift_dataset_spinr_v1_passband_g96_gl2_scene_bins_v1` | Historical recipe only; do not resume it as the direct-bin recipe. |
| Native GOTCHA | Common schema `rift_gotcha_checkpoint_v1`, method format `spinr_gotcha_native_v1` | Source, region, split, channels and effective recipe must match before response access. |

RIFT's retained container name `rift_spinr_style_b787_v2` is a compatibility
identifier, not permission to reuse a B787 checkpoint for another object.
The native GOTCHA format is separate from every RIFT collection and smoke
checkpoint. Use a new output namespace for a different scientific recipe;
relabeling an existing checkpoint does not make it compatible. Native historical
G48 midpoint runs use `protocols/gotcha_spinr_g48_midpoint_1500.json`; explicit
`epochs=1500` also restores the old schedule. Historical shortened native runs
require their saved `epochs` plus `cosine_epochs=1500`. The schedule override
is nested in the generated recipe's optimizer identity, preserving old recipe
dictionaries exactly. New native runs need a fresh output root; the existing
output/recipe gates reject cross-budget recovery before response reads.
An explicit run shorter than the selected schedule remains development-only.
The reviewed September-20 checkpoints retain their original 1500-epoch recipe;
they are not results of the new 150-epoch schedule.

## What follows the paper

The field is position-only and signed real. The operator coherently integrates
that field with product-distance spreading. Supervision uses scene range bins,
with squared magnitude error plus half the complex squared error. The corrected
RIFT path computes selected bins directly. The paper specifies 1,500 epochs
and 1,024 measurement channels per optimizer update (§§3.3–4 and 5.4).
The current 150-epoch comparison budget is the explicit user override above;
historical paper-budget recipes remain available. No angular input, complex output head,
learned complex gain, surface prior, extra regularizer, or SpINRv2 warm-up is added.

The real-field adjoint followed by tiled neural replay is a memory implementation
of the same chain rule. Independent references test values and derivatives; the
tiling does not define a different objective or perform extra optimizer steps.

## Choices absent from the author description

| Missing information | Current explicit choice | Reason and limit |
| --- | --- | --- |
| Exact INR architecture, encoding and activation | Raw normalized XYZ plus six Fourier bands; six width-840 ReLU hidden layers; signed linear scalar head; 3,566,641 parameters | Retain the established local INR without new outcome-based architecture tuning. This is not an author architecture or parameter-count match claim. |
| Weight initialization | Kaiming-normal hidden weights, zero biases, small nonzero normal output weights | The existing local initialization supplies gradients to preceding layers. It is not recovered author initialization. |
| Optimizer, learning-rate schedule and clipping | Adam, LR 1e-4, betas 0.9/0.999, epsilon 1e-8, no weight decay; cosine to 1e-5 over 150 comparison epochs (1500 for historical paper-budget recipes); global gradient-norm clip 1 | Explicit settings. The user-selected pass budget changes the schedule identity. Clipping statistics are recorded. |
| Signal normalization and amplitude units | TRAIN-only mean raw power; fixed output scale 0.1 times the square root of observed/random-field energy ratio on the first 32 TRAIN measurements | No validation/test normalization and no learned gain. RIFT uses 32 whole views; GOTCHA uses up to 32 pulses per channel. This conditions unspecified amplitudes and initialization, not the acquisition phase. |
| Volume quadrature | G48 midpoint on both datasets; physical volume applied once | User-selected resource adaptation. Historical G96/G48 GL2 rules remain explicit. No rule is certified for fitted-field integration; changes bind checkpoint identity. |
| Exact scene-bin boundaries | Geometry-only conservative path/range bounds, floor/ceil bracketing and modular FFT indices | No response amplitudes or target meshes choose bins. Finite-window sidelobes can remain outside the selected bins. |
| Batch ordering and incomplete batches | Deterministic TRAIN permutations; final partial GOTCHA batch is retained | Covers every selected measurement. Matching 1,024 channels does not recover the authors' unknown batch distribution. |
| Selection and geometry extraction | Common benchmark validation selection; explicit signed-field/magnitude export with grid/threshold provenance | These are evaluation conventions, not recovered author extraction code. No test-geometry tuning. The existing export CLI supports RIFT collection checkpoints; a native GOTCHA geometry CLI remains separate work. |

Changing these choices in response to a desired comparison outcome would violate
the fidelity rule. Numerical refinement checks are for integration accuracy,
not a method-performance tuning objective.

## Necessary RIFT dataset adaptations

- The archive supplies exact bistatic frequency-pulse responses, not the authors'
  dechirped acquisition. Retain all 600 frequencies, all 16×16 pairs and the
  native negative phase convention. The direct finite-DFT kernel is the analytic
  transform of these exact frequency responses; no new leakage/RVP correction
  is multiplied into their physical acquisition operator.
- Retain the physical carrier. The paper's general kernel includes its phase;
  reproducing the paper's zero-start-frequency experimental simplification
  would change these benchmark measurements. The position-dependent phase
  cannot generally be absorbed into one gain.
- Use each object's recorded 0.30 m support cube, spherical poses and sealed
  3,200/1,000 train/validation roles. These are known benchmark differences from
  the authors' scan geometry, scene and measurement count. They are not missing
  author information and are not presented as replication of that experiment.
- Production campaign (2026-09-22): 2,400 of the 3,200 parent TRAIN views and only
  physical Tx 0/Rx 0 (see [ANTENNA_SELECTION.md](ANTENNA_SELECTION.md)), so four
  viewpoints give 4 measurement channels per update, not 1,024. See
  [TRAINING_RECIPES.md](TRAINING_RECIPES.md).
- Four complete viewpoints give 1,024 Tx/Rx channels per update. Full native
  coherent validation uses every frequency and pair, including bins outside
  the training selection. Full-role validation selects the earliest minimum.

The old G48/all-bin `legacy-midpoint` recipe and the earlier G96/FFT `paper-v1`
recipe remain unchanged. Their shorter schedule and plateau stop do not define
the new `paper-v1-direct` comparison. A bare root trainer retains its historical
default for existing launchers; the shared collection selector explicitly picks
the corrected recipe. All recipe changes require a fresh checkpoint namespace.

## Necessary GOTCHA adaptations

### Native acquisition and spectral synthesis

GOTCHA is paired monostatic phase history with a per-pulse effective reference
range. In the registered local frame, the SpINR point kernel is

`exp(-i * 4*pi*f/c * (distance - r0_effective)) / distance^2`.

The reference-range phase is necessary to compare in native observation units.
The denominator retains SpINR's product spreading for Tx=Rx; it is not borrowed
from RIFT's unit-spreading native point model. The absolute amplitude calibration
of the measured data versus this physical reflectivity convention is not proven
by this port. Fixed TRAIN-only scaling does not establish that equivalence.

Each pass/channel retains its exact frequency vector and every selected native
pulse. There is no 16×16 synthetic array, common-frequency resampling or padding.
HH/VV source autofocus is applied once by `GOTCHADataset`; the model adds none.
HV/VH remain raw and never borrow co-polarized corrections.

For an **exactly affine** native frequency vector, selected-bin synthesis uses
the closed-form finite geometric series with the carrier/reference phase. For
a nonuniform vector, that formula is not exact. The adapter instead takes the
finite DFT of each exact native point kernel, retains the selected bins, then
sums their weighted contributions. Its adjoint differentiates that same kernel.
It preserves every stored frequency; it does not claim the paper's closed-form
speed advantage. All eight inspected real HH vectors require this exact fallback.

With nonuniform frequency spacing an index-DFT bin is not a unique physical
range. The selection conservatively bounds phase increments using native
minimum/maximum frequency differences and the region's reference-relative
range interval. This is a disclosed acquisition adaptation. The full native
response and common ROI-projected complex metrics expose the fit outside the
selected training observable as well.

### Roles, channels, training and evaluation

The default Camry path uses all eight passes, HH, and 2,000/440/440 pass-sector
train/validation/reserved-test viewpoints. Sector selection retains every pulse.
Each polarization has its own instance of the same real INR; all passes share
that channel's field. An epoch permutes all authorized native TRAIN pulses into
1,024-pulse batches and retains the final partial batch. No sector or frequency
subset is silently used to make a smoke run appear full scale. The production
campaign instead uses the user-pinned GOTCHA recipe: 1,500/440/440 sectors, a fixed
16-pulse cap per sector and frequency stride 2 in every role
([GOTCHA_PULSE_SELECTION.md](GOTCHA_PULSE_SELECTION.md),
[GOTCHA_FREQUENCY_SELECTION.md](GOTCHA_FREQUENCY_SELECTION.md)).

Statistics and fixed initialization scales are per polarization and TRAIN only.
The training loss remains magnitude-plus-half-complex error in selected spectral
bins. Validation processes the complete validation role, reports full-native
complex error, and selects the earliest minimum of the shared geometry-defined
ROI-projected complex error. Magnitude loss is never applied to arbitrary SVD
basis coordinates; the common ROI projector is used for the complex metric.
Range-compatible clutter may remain. The target is not an isolated, surveyed
vehicle, and this work has not exposed a reserved-test reader.

### Recovery and numerical qualification

Native checkpoints bind source, region, split, polarizations, effective recipe,
TRAIN normalization, initialization identities, model/Adam/cosine state, RNG,
cursor, selection and per-view/channel pulse exposure. Synthetic-data and
historical all-training GOTCHA checkpoints are rejected before response access.
Recovery commits completed optimizer batches and can resume the final batch
before scheduler/validation without duplicating fitting exposure. Epoch 150
is the comparison budget; historical epoch-1500 recipes retain their schedule.
Explicit budgets shorter than the selected schedule are tagged development only.

**The Camry integration rule is not qualified for benchmark scoring.** The
registered cube is 10 m wide, so G48 parent cells are about 0.20833 m wide,
compared with 0.00625 m for the G48 collection support. Metadata preflight reports
about 86.54 radians of monostatic phase variation across an axis-aligned parent
cell at the highest native frequency. Exact point-kernel agreement does not
establish accuracy of the midpoint (production) or GL2 integration over such cells. Fitted-field signal and
gradient refinement, and a feasible allocation/runtime assessment, are required
before using a GOTCHA result as a paper baseline. Callable backend availability
means the trainer is wired; it does not certify quadrature or convergence.

## Validation evidence and remaining work

Bounded synthetic tests cover native uniform/nonuniform forward values,
coordinate derivatives and real-field adjoints against independent dense
references; carrier/reference range, physical volumes and product spreading;
ragged passes and separate channel heads; all-pulse sector coverage and
TRAIN-only statistics; actual root dispatch; pre-response checkpoint rejection;
and uninterrupted versus interrupted recovery including pending finalization.
The numerical lifecycle fixtures are deliberately tiny and are not fitted radar
results. The production network is covered by the retained full SpINR validator.
At integration closeout, the combined suite passed 209 tests: the four focused
SpINR files, shared GOTCHA ingress/dispatch tests, collection tests and
smoke-consolidation checks.
This includes 20 native SpINR tests. Recovery compares model, optimizer,
scheduler, RNG, history and exposure state, and repairs terminal best/final
artifacts without reopening measurements.

The checked files were `tests/test_spinr_direct.py`,
`tests/test_spinr_fidelity.py`, `tests/test_spinr_quadrature_audit.py`,
`tests/test_spinr_gotcha.py`, `tests/test_gotcha_dataset.py`,
`tests/test_rift_dataset.py` and `tests/test_smoke_consolidation.py`.
The retained `scripts/validate_spinr_style.py` had separately passed 91 checks
during the fidelity work. These are bounded implementation checks, not measurements
of reconstruction quality. Subsequent documentation edits do not constitute a
new test run.

Real HH metadata preflight found 236,826 TRAIN pulses, 424–434 native frequency
samples per pass, and 232 default optimizer batches per epoch. This read no
response payload and performed no optimization. These counts apply to the
inspected source identity, not every possible custom region/source selection.

The remaining work is broader than smoke tests:

1. **Numerical integration:** use `scripts/check_spinr_quadrature.py` on frozen
   RIFT collection fields to compare saved quadrature against higher-order and
   refined rules, including signal and full-parameter gradients. Implement the
   corresponding acquisition-bound checker for native GOTCHA. The Camry phase
   scale makes this a prerequisite for benchmark interpretation; point-kernel
   parity cannot certify volume integration.
2. **Allocation and convergence evidence:** measure real GPU memory/runtime,
   initialization and clipping behavior, then full-data convergence through
   authorized experiment-manager runs. Preserve the disclosed architecture,
   loss and budget; numerical refinement must address integration error rather
   than optimize the comparison outcome.
3. **Geometry evaluation:** connect the existing RIFT signed-field export to
   the common metric protocol, and add a native GOTCHA checkpoint-bound readout
   in registered local metres. Geometry export does not establish surveyed
   GOTCHA ground truth. Reserved-test evaluation remains a separate explicitly
   authorized path.
4. **Unspecified author details:** architecture, initialization, optimizer
   and other unspecified choices remain disclosed reconstruction assumptions
   unless the authors supply more information. Smoke tests cannot resolve them.

No original-author code parity, final baseline quality, timing advantage or
reserved-test score is claimed here. Root integration is implemented; numerical
qualification and benchmark evidence remain pending.

Detailed earlier numerical/source mapping: `external/SPINR_REFERENCE.md` and
`discrepancy_audit/SpINR_discrepancy_check.md`.

## Selected collection 1t1r acquisition

Canonical collection runs now select physical source Tx 0/Rx 0 through the
shared sealed ingress. Raw complex values, train-only normalization, initial
output scaling, direct-bin rendering and validation use exactly those channels.
Recipe/checkpoint identities bind ordered source indices and geometry;
readout/quadrature CLIs accept matching antenna selectors. Pair tiles are
bounded by available pairs. The signed-real network remains; current quadrature is G48 midpoint and the comparison budget is 150 epochs.
Native GOTCHA retains every pulse/frequency at its existing native 1t1r.
See [antenna selection](ANTENNA_SELECTION.md). This change does not resolve
fitted-field quadrature, convergence or CUDA resource qualification.
