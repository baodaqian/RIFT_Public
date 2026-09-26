# SpINR v1 reference and independent adaptation

Reviewed 2026-09-19 against [SpINR, arXiv:2503.23313v2](https://arxiv.org/html/2503.23313v2)
(25 April 2025), the version returned by the user-supplied unversioned PDF.
The local discrepancy audit describes earlier uploaded helpers; its missing
production-caller evidence is now checked against `train_spinr_style.py`.
No official implementation parity or published-number reproduction is claimed.

## What the paper establishes

Section 3.3 specifies a real spatial field, coherent integration with
`1/(R_T R_R)` spreading, and squared magnitude error plus half the squared
complex error. Figure 1 and §5.4 restrict supervision to scene-relevant DFT
bins; sixteen is their acquisition's count, not a universal setting.
Section 3.1.1 describes zero start frequency in their training setup. Section 4
reports a cylindrical aperture, monostatic conversion, 1,024 measurements per
batch and 1,500 epochs. The paper does not establish our exact MLP, encoding,
initialization, Adam schedule or spatial quadrature settings.

This task targets that paper. SpINRv2 warm-up and regularization are different
method choices and are not silently added to the v1 recipe.

## Fidelity rule and reconstruction decisions

The baseline follows disclosed author choices. Missing details permit a
documented reconstruction choice, not an opportunity to improve or weaken the
method relative to RIFT. Do not change capacity, regularizers, initialization,
objectives or budgets based on the desired comparison outcome. Preserve every
recipe identity; new author information takes precedence over our choices and
requires a new recipe if it changes execution. No official code has been
recovered or executable parity established.

| Decision | Evidence and treatment |
| --- | --- |
| Real spatial field, coherent product-distance integral, selected-bin magnitude-plus-half-complex loss | Specified in §3.3; implemented. No angular inputs, complex head, learned gain, SDF or additional physical priors are added. |
| Direct synthesis of scene DFT bins | Specified in §§3.3, 5.4; implemented in `rift/spinr_direct.py`. The earlier FFT route remains a historical recipe, although mathematically equivalent. |
| 1,500 epochs and 1,024 measurements per batch | Specified in §4; new default uses 1,500 full epochs without plateau stopping. Four complete 256-channel views supply 1,024 channels. Author ordering/grouping is not disclosed and this grouping is a local choice. |
| MLP depth/width/activation/encoding and initialization | Undisclosed. Retain the existing six-layer width-840 ReLU/Fourier field and its initialization without further outcome-based search; parameter-count matching does not establish author architecture parity. |
| Optimizer, learning-rate schedule, clipping and amplitude conditioning | Undisclosed. Retain the declared Adam/cosine, clipping and train-only fixed scale; cosine now spans the specified training duration. |
| Numerical quadrature and exact scene-bin boundary rule | Undisclosed. G96/GL2 and conservative scene bounds are numerical integration choices. Validate against independent references and fitted-field refinement, never choose them to favor the benchmark result. |
| Geometry extraction and thresholding | Undisclosed. Export signed field/magnitude with explicit grid/threshold provenance. No mesh-driven threshold selection is part of training. |
| Carrier, acquisition poses, scene extent and evaluation split | Known benchmark-setting differences, not missing author information. The paper's general §3.3 kernel includes the carrier and both path lengths. Retain native data/geometry for the shared benchmark; never claim replication of their zero-start-frequency cylindrical/monostatic experiment. |

Checkpoint identity records specified choices, undisclosed settings and benchmark
settings separately. Changing the benchmark data to imitate the original
experiment would prevent comparison on the same measurements. This implementation
is a reconstruction of their method on our acquisition, not their original run.

## Current full-training path

The collection dispatcher selects `--recipe paper-v1-direct` in
`train_spinr_style.py`. Its identity is
`rift_dataset_spinr_v1_passband_direct_bins_1500_v1`.

- Signed-real, position-only MLP: raw normalized XYZ plus six Fourier bands,
  six width-840 ReLU hidden layers, scalar linear head, 3,566,641 parameters.
  Kaiming hidden initialization and a small nonzero head are local choices.
- G96 parent cells, tensor two-node Gauss–Legendre integration: 7,077,888
  quadrature queries, with positive physical volumes summing to 0.027 m³.
  These are integration samples, not additional trainable parameters.
  This adopts the existing G96 smoke's integration rule without inheriting
  its subset, budget, checkpoint format or completed-run status.
- Actual passband frequency vector, exact bistatic Tx/Rx geometry, phase -1,
  FP64/complex128 propagation and FP32 network. The carrier is retained.
  Training synthesizes selected bins with the closed-form finite-DFT kernel
  and uses its exact signed-real field adjoint, followed by tiled neural VJP.
  The direct renderer includes physical volume/product spreading once, with
  no RIFT operator prefactor. The native-response evaluation path compensates
  its existing `(4*pi)^-2` factor as before; both match independent direct sums.
- All 3,200 training views, all 256 pairs and all 600 measured frequencies.
  An epoch visits each training view once in 800 deterministic four-view
  updates. The latter is 1,024 Tx/Rx channels; it does not establish equivalence
  to the paper's measurement distribution. Initialization/normalization reads
  do not count as fitting exposure.
- Per-pair scene DFT bins are chosen from the full support cube and acquisition
  geometry only. Minimum point-to-box distances give a conservative lower
  path bound; corner distances give the upper bound. Floor/ceil bracketing
  and modular FFT indexing handle negative phase and alias-boundary crossings.
  No target mesh, response amplitudes or held-out information chooses bins.
- Selected-bin loss is a sum over bins divided by pair count and training-only
  mean raw power. Thus scene plus remainder objectives recover the old all-bin
  objective without pair-dependent rescaling by the number of selected bins.
- Validation reports full-domain pooled coherent RelMSE/RelL2 and selects the
  earliest minimum over the complete 1,000-view validation role. Additional
  scene/remainder objectives, residual/target energies, bin counts and squared
  raw-response cotangent norms expose discarded spectral energy. These gradient
  diagnostics are response gradients, not neural-parameter gradients.
- Checkpoints record per-training-ID exposure counts derived from committed
  updates and deterministic permutations. Compact epoch summaries, wall time,
  update counts, optimizer/scheduler, RNG and partial-epoch cursors are retained.
  Recipe, object, roles, acquisition, normalization and recovery checks precede
  response access; tampered coverage is rejected before metadata loading.

Training defaults to the paper's 1,500 epochs without plateau stopping. Shorter
explicit budgets are development runs, including 150/300 epochs, and cannot be
reported as completing this paper budget. Adam 1e-4 to 1e-5 cosine over 1,500
epochs, fixed training-derived output scaling and norm clipping remain local
choices because the paper omits them. The epoch-150 diagnostic checkpoint is
retained for lifecycle compatibility; it is not a completion criterion. No
production convergence or equal-compute claim follows from matching epoch count.

## Why the acquisition operator stays unchanged

Our archives contain frequency-pulse responses. They are not deramped ADC
signals requiring a new video-phase or leakage correction. On a uniform grid,
the DFT of the exact finite exponential sum equals its finite-geometric-series
form, including off-bin spillover. Independent tests compare both values and
position gradients, retaining the carrier. The current training path synthesizes
selected bins directly, using a stable sinc-ratio form at exact/near bin centers;
only the observed response passes through FFT. Full native-response validation
still uses the existing verified operator. The older FFT training recipe is
algebraically equivalent but does not reproduce the authors' computation path.
No GPU speed advantage is claimed without benchmarking the actual implementation.

## Historical compatibility and invocation

The bare root trainer keeps `legacy-midpoint` as its default for existing
launchers. That recipe retains its original identity, G48 midpoint quadrature,
all-bin objective and historical checkpoint format. `paper-v1` preserves the
earlier G96/GL2 FFT/scene-bin recipe, 300-epoch cosine and plateau stop. The
collection default is explicitly `paper-v1-direct`; select the exact saved
`--spinr-recipe paper-v1` or `legacy-midpoint` for older checkpoints. A recipe
mismatch fails before response reads. Existing G96
16-view smoke and its frozen evaluators are unchanged.

```bash
# Read-only metadata planning:
python train_rift_dataset.py --object a320 --method spinr --dry-run
python train_rift_dataset.py --object a320 --method spinr \
  --spinr-recipe legacy-midpoint --dry-run
```

Actual fitting remains experiment-manager-owned and requires a fresh output
root or an exact same-recipe continuation. No experiment is requested here.

## Geometry readout

`scripts/readout_spinr.py` validates the selected collection object, full-trainer
recipe, saved roles, acquisition and scale before querying the checkpoint.
It reads metadata only, never radar responses. A declared midpoint readout grid
samples signed `sigma * initial_output_scale`; output contains signed values,
magnitude, max-normalized magnitude and support points at an explicit fixed
threshold. Quadrature volumes are not multiplied into the sampled field.
The grid/extent, threshold, object identity, recipe, training coverage and SHA-256
of the loaded checkpoint are recorded inside the NPZ.

```bash
python scripts/readout_spinr.py --object a320 \
  --checkpoint /path/to/checkpoint_best.pth.tar --grid-size 96 \
  --fixed-threshold 0.2 --output figures/a320_spinr_support.npz
```

Use the same selected checkpoint for signal and geometry. This export is a
scattering-support sample, not an occupancy probability, SDF zero surface or
mesh-metric evaluation. Its magnitude threshold differs from the squared-energy
convention in the frozen historical smoke postflight; keep that provenance.

## Frozen-checkpoint quadrature diagnostic

`scripts/check_spinr_quadrature.py` validates a full collection checkpoint and
compares its saved training rule, GL3 at the same parent grid, and GL3 on a
finer grid (default G128 for the corrected G96 recipe). It evaluates the same
first one to four saved training IDs under every rule, retaining all 600
frequencies and 256 pairs. It never optimizes or reads validation/test responses.
The frozen historical smoke has its own diagnostics and is not accepted here.

```bash
# Metadata/checkpoint validation and declared cost only; no response reads:
python scripts/check_spinr_quadrature.py --object a320 \
  --checkpoint /path/to/checkpoint_best.pth.tar --dry-run

# Existing-checkpoint evaluation on an allocated compute node, when requested:
python scripts/check_spinr_quadrature.py --object a320 \
  --checkpoint /path/to/checkpoint_best.pth.tar --device cuda \
  --output /path/to/new_quadrature_report.json
```

The JSON records the exact loaded checkpoint hash, object/recipe, source IDs,
rules, execution settings, timings and peak allocated CUDA memory. Comparisons
include pooled and per-view signal RMS differences normalized by observed signal
RMS, plus differences in the full mean-objective parameter gradient, globally
and for every parameter tensor. Gradient computation follows the saved recipe's
direct-bin or historical FFT path. Existing model mode/gradients are restored.
Default relative tolerance is 1%; near-zero references instead use explicitly
recorded absolute tolerances (signal 1e-12, gradient 1e-10). These are diagnostic
choices, not paper settings. Any failed comparison writes the report and exits
with status 2. No existing output is overwritten.

This closes the missing-checker implementation gap, not the numerical evidence
gap. Even agreement on four views is a finite-view diagnostic, not a proof of
continuous-field convergence. The three default rules query 7,077,888,
23,887,872 and 56,623,104 points respectively, each with neural-gradient replay;
full-grid execution is expensive. No fitted production checkpoint has been
evaluated by this new tool.

## Verification and unresolved evidence

`tests/test_spinr_fidelity.py` exercises geometry-only bin selection for both
phase signs and alias crossings; scene/remainder objective and gradient
partitions; independent dense forward/VJP and finite differences at baseband
and passband; finite-geometric-series/FFT value and position-gradient equality;
masked four-view tiled neural gradients; analytic encoding null and quadrature
refinement; frozen smooth-field signal/parameter-gradient refinement using the
production renderer; full-view exposure accounting; pre-read recipe/coverage
rejection; interrupted-resume equality; object rejection and tiled field readout.
The focused suite plus collection/smoke-consolidation regressions passes 144
tests (21 focused SpINR tests). The existing full SpINR validator passes 91
checks; retained G96 source/numerical, CPU runtime and lifecycle validators pass
24, 22 and 48 checks respectively. Metadata-only planning succeeds for all six
objects. The runtime validator is invoked as
`python -B -m scripts.validate_spinr_style_g96_gauss2_runtime_preflight`;
the source and common lifecycle validators use their ordinary script paths.

The additional `tests/test_spinr_quadrature_audit.py` checks the actual tiled
mean-objective gradient against independent dense-sum autograd, mode/gradient/RNG
preservation, per-view and per-tensor failures hidden by pooled norms, near-zero
handling, pre-response identity rejection, metadata-only cost planning and CLI
report/exit behavior. The initial two-file diagnostic suite passed 31 tests;
the direct-renderer follow-up adds the tests described below.
The CLI report tests mock the expensive full-grid numerical calls; the core
numerical comparisons use bounded synthetic tensors and the production operator.

`tests/test_spinr_direct.py` checks production direct-bin values, coordinate
derivatives and real-field adjoints against independent dense-sum FFT/autograd
at baseband/passband, both phase signs, 24/600 samples and exact/near bin centers.
It checks the real four-view neural update with the frequency renderer disabled,
default paper budget, recipe/scheduler isolation and readout identity. The real
trainer interruption/resume harness covers both scene-bin recipes, including
the direct recipe's disabled plateau stop. The quadrature diagnostic also checks
direct-recipe mean-objective gradients against independent autograd.
The current three focused SpINR files plus collection/smoke-consolidation tests
pass together: 169 tests. The historical full validator still passes 91 checks,
and metadata-only planning selects the new recipe for all six objects.

These are bounded synthetic checks. The G48 midpoint encoding null is absent
on the GL nodes, and analytic cell integration improves under refinement;
neither establishes convergence for a fitted high-frequency ReLU field.
Full-checkpoint G96/GL2 versus independent GL3/refined-grid signal and gradient
checks, GPU memory/throughput, clipping behavior and real-data convergence remain
unmeasured. G96/GL2 uses 64 times as many field queries as G48 midpoint, so its
production budget needs a resource assessment. No real response fitting, test
evaluation, scheduler action or manager mutation was performed.

GOTCHA now has a separate native SpINR hook in the same root trainer, implemented
by `rift/spinr_native.py` and `rift/spinr_gotcha_training.py`. The synthetic-data
FFT selector remains uniform-only; native nonuniform vectors use the exact
finite DFT of each native point kernel with the pulse's reference range.
No frozen loader or synthetic checkpoint contract was relaxed. The complete
two-dataset adaptation ledger, source distinctions, runtime configuration and
GOTCHA quadrature limitations are in `docs/SPINR_ADAPTATION.md`.
