# GOTCHA/RIFT forward-model alignment and baseline audit

Date: 2026-09-22. Original comparison: [RIFT c37adf54935bdfb39fd504004b00da1280608a74](https://github.com/baodaqian/RIFT/tree/c37adf54935bdfb39fd504004b00da1280608a74), `train_rift_dataset.py`/`train.py` versus `train_gotcha_dataset.py`/`rift/gotcha_training.py`. Later release audit: local HEAD `96afb8dbcf587dcc3f63194e648c0f21628cb16f` plus working-tree inspection.

## 1. Verdict and scope

| Subject | Finding |
|---|---|
| RIFT forward model | Shared adaptive point-SH representation; exact known GOTCHA phase-reference adapter; different amplitude law. Direct sum/NUFFT equivalence requires identical kernels/frequencies and numerical qualification. |
| RIFT experiment | Native objective, scaling, initialization, priors and refinement differ from synthetic recipe; manuscript must distinguish them. |
| [SpINR](#spinr-fidelity) | Core field/kernel/loss retained; G48 integration unqualified; four-channel collection batches mislabeled as 1024 channels. |
| [Sugavanam-Ertin](#se-fidelity) | Independent sparse-to-SDF construction with disclosed variants; Stage 1 alone incomplete; acquisition/scale transfer unqualified. |
| [GeRaF](#geraf-fidelity) | Released source agreement; confirmed far-range float32 sphere-intersection defect; historical-v1 settings unresolved. |
| [Radar Fields](#radar-fields-fidelity) | Released default network/core loss; changed priors/acquisition; incomplete source-profile architecture guard. |
| [RadarSplat](#radarsplat-fidelity) | Released CUDA components/successful-path loss; changed acquisition/supervision; selected Gaussian count increased 5.6 times. |

- No renderer/training recipe changed by these audits. No fitted checkpoints inspected, reserved-test responses accessed, or production training launched. Defaults/source labels do not certify saved runs.
- Unconditional unchanged-original-experiment reproduction unsupported for current configurations. Adaptations are not thereby proven incorrect or worse-performing; performance/convergence require run artifacts.
- SpINR/SE author-code unavailability at model freeze: accepted user constraint, not independently established publication history. Disclosed independent choices allowed.
- Supported descriptions: SpINR/SE, "independent implementations with disclosed adaptations"; other three, "released baseline components with explicit native-acquisition adaptations and stated comparison budgets."
- Baselines do not all call RIFT's `native_forward`: [SpINR native kernel](../rift/spinr_native.py#L46) retains `exp(-i*4*pi*f/c*(distance-r0))/distance^2`. Preserve method-specific coherent, magnitude or power objectives.
- This file is the common audit; linked adaptation ledgers retain implementation details. Distinct audit environments/evidence: [section 9](#9-validation-evidence).

## 2. RIFT operator and training contract

### Phase reference: exact adapter

For antenna $\mathbf a_q$, point $\mathbf x_p$, and colocated Tx/Rx:

$$
r_{qp}=\|\mathbf x_p-\mathbf a_q\|_2,\qquad
e^{-\mathrm i4\pi f_{q,i}(r_{qp}-r_q^{\mathrm{ref}})/c}
=D_{q,i}e^{-\mathrm i4\pi f_{q,i}r_{qp}/c},\qquad
D_{q,i}=e^{+\mathrm i4\pi f_{q,i}r_q^{\mathrm{ref}}/c}.
$$

- Synthetic bistatic path becomes $2r_{qp}$. $D$: known, scatterer-independent, unit magnitude, pulse/frequency-dependent; no learned parameters.
- Apply $D$ to predictions or $D^*$ to targets. Consistent conversion preserves unprojected complex residual norm; projected objectives also require consistent projection transformation.
- Do not discard $D$ or absorb it into one scene-wide gain. Apply channel-owned autofocus/effective-reference corrections once.
- Evidence: [renderer](../rift/gotcha_training.py#L44), [ingress/corrections](../rift/gotcha_dataset.py#L294), [geometry](../rift/coherent_radar_geometry.py#L83).

### Amplitude: different physical models

Reviewed synthetic attenuation and its literal native adaptation:

$$
A_{qp}=\frac{1}{(4\pi)^2[(2r_{qp})^2+10^{-9}]},\qquad
\widehat y_{q,i}=D_{q,i}\,g\sum_p A_{qp}\rho_p(\mathbf u_q)e^{-\mathrm i4\pi f_{q,i}r_{qp}/c}.
$$

- Native RIFT currently uses unit amplitude. Adding attenuation costs little arithmetic; processed-GOTCHA calibration/preprocessing must establish whether sum-path-squared, unit weighting or another factor is warranted. Manuscript consistency alone is insufficient.
- Gain absorbs a constant scale, generally not point/pulse-dependent attenuation. Changing attenuation requires new model identity/refit; one gain rescale cannot exactly convert old checkpoints.
- Attenuation requires physical distances; phase uses reference-adjusted paths. `monostatic_near_field_reference` returns virtual reference-range legs: enabling `sum2` on those legs implements the wrong denominator. Native effective reference range and older positive virtual-reference parameter have different meanings.
- Evidence: [amplitude laws](../rift/range_operator.py#L246), [virtual geometry](../rift/coherent_radar_geometry.py#L97), [native contract](../rift/gotcha_dataset.py#L369).

### Direct sum / NUFFT

- Synthetic operator reconstructs uniform float64 frequency grid from endpoints, applies Gaussian gridding/FFT, optionally gathers output bins. Native renderer evaluates supplied frequencies directly.
- `frequency_stride=2` plus appended final endpoint can produce a nonuniform subset from a uniform source vector. Do not treat that subset as a complete uniform NUFFT grid.
- If each complete source grid satisfies a strict phase-error tolerance: render complete grid, gather saved selected-bin indices; preserve per-pass frequencies/counts. Otherwise retain exact-frequency direct sums or qualify a nonuniform-frequency evaluator.
- Existing ingress tolerance, 1% of bin spacing, does not establish phase accuracy. Compare against original frequencies; separate approximation error from input rounding/storage precision. Range referencing can improve conditioning without changing physical attenuation geometry.
- No speedup measured. Benchmark complete forward/backward updates: gridding, FFT, unused full-grid bins, gradient recomputation, pulse batching.
- Evidence: [grid reconstruction](../rift/range_operator.py#L133), [operator/bin selection](../rift/range_operator.py#L399), [native selection](../rift/gotcha_frequency_selection.py#L29), [aligned-view infrastructure](../rift/serialized_range_operator.py#L186).

### Training differences

| Item | Reviewed synthetic RIFT | Native GOTCHA backend/defaults |
|---|---|---|
| Data fit | Full selected complex-response MSE | Geometry-defined range-subspace projected complex MSE |
| Scaling | Native measurement scale | Divide by TRAIN-only mean projected power |
| Initialization | Scaled coherent adjoint, first 100 training-loader views | Random coefficients at 1e-3; projected-observation gain warm start |
| Priors | Gain-scaled point-group sparsity; SH angular penalty | Neither in inspected loop |
| Position LR | 0.003 | 0.0001 |
| Adam epsilon | 1e-8 | 1e-20 |
| Scheduler | Cosine warm restarts | None in inspected backend |
| Update | One training viewpoint | One pass-sector; selected pulses/polarizations accumulated |
| Refinement | Every 10 epochs | Every 100 updates |
| Spatial/angular fractions | 1/512, 1/16 | 0.05, 0.05 |
| Minimum spatial/angular exposures | 3200, 200 | 2, 2 |
| Maximum spatial subdivision level | 1 | 3 |

- Shared: point-SH representation, complete-band unlocks, in-place heir plus zero siblings, prediction-preserving refinement. Recipe differences alone imply no performance ranking.
- Small Camry support cannot be assumed to explain whole-parking-lot returns. Full-response fitting may require larger support, explicit background model or justified measurement treatment. Phase conversion does not remove clutter.
- Report projected-complex/full-native-complex RelMSE separately. Both recorded; native checkpoint selection uses projected pooled RelMSE.
- Evidence: [synthetic recipe](../rift/b7873200_adaptive_fullscale.py#L72), [native defaults](../train_gotcha_dataset.py#L115), [objective](../rift/gotcha_training.py#L69), [initialization](../rift/gotcha_training.py#L132), [training/selection](../rift/gotcha_training.py#L315).

<a id="baseline-fidelity-reaudit"></a>
<a id="spinr-fidelity"></a>

## 3. SpINR

Reference: original SpINR [arXiv:2503.23313v2, §§3.3–5.4](https://arxiv.org/html/2503.23313v2), not SpINRv2. Ledger: [SPINR_ADAPTATION.md](SPINR_ADAPTATION.md).

| Component | Inspection result |
|---|---|
| Paper contract | Signed-real position-only field; coherent volume integration; product-distance spreading; selected DFT bins; magnitude squared error + 0.5 complex squared error; 1024 measurements/batch; 1500 epochs. Network/optimization details insufficient for author-identical recovery. |
| Kernel | Collection physical `R_tx * R_rx`; GOTCHA physical `distance**2`; volume/fixed initialization scale applied once. Reference range changes phase, not attenuation. |
| Native DFT | Nonaffine frequencies: exact finite DFT of point kernel. Exactly affine grid: geometric-series form. Runtime parity with original closed form unestablished. |
| Loss/readout | Selected-bin magnitude-plus-complex spectral training; common projected-complex GOTCHA validation. |
| Accepted local choices | Six layers, width 840, Fourier encoding, initialization, Adam, clipping, fixed TRAIN-only amplitude conditioning; carrier retention/native phase reference. Absolute processed-GOTCHA calibration against spreading unestablished. |

Code: [field](../rift/spinr_style.py#L107), [collection kernel](../rift/spinr_direct.py#L23), [native kernel](../rift/spinr_native.py#L47), [collection loss](../rift/spinr_direct.py#L117), [native loss](../rift/spinr_native.py#L98).

### S1. Incorrect batch metadata

- [Canonical collection](../train_rift_dataset.py#L253): 1 Tx / 1 Rx. [Batch constant](../train_spinr_style.py#L120)/[update](../train_spinr_style.py#L1653): exactly four whole views/optimizer update.
- Actual: four measurement channels/update, each with frequency vector. Historical 16×16 acquisition: 4 * 256 = 1024; frequency bins are not additional independent channels. At 2400 views: 600 updates/pass.
- `batching.all_pairs` updates correctly; `fidelity.specified` still claims `1024 measurement channels per update`; `benchmark_settings` claims `four whole views group 1024 channels`. Both [CUDA recipe](../train_spinr_style.py#L365)/[PVC recipe](../train_spinr_style_pvc.py#L366) inconsistent.
- Four views may remain a disclosed budget. Correct selected-channel metadata with checkpoint-identity compatibility; changing batch size changes optimization and must not silently alter saved runs.

### S2. G48 quadrature unqualified on both datasets

- G48 midpoint: 110592 evaluations of a continuous field. Point-kernel correctness does not qualify volume integration.
- Collection: 0.30 m / 48 = 0.00625 m cell; maximum directional monostatic carrier phase change ≈26.20 rad/cell at 100 GHz. GOTCHA: 10 m / 48 = 0.20833 m; ≈83.83 rad/cell at illustrative 9.6 GHz. Phase scales, not measured selected-observation errors. Reference subtraction leaves spatial phase derivative unchanged.
- [Response-free diagnostic](../scripts/audit_spinr_se_fidelity_20260922.py): exact collinear point/antenna geometry, reference range R; independent oscillatory SciPy integration versus midpoint evaluation of `integral[-L/2,L/2] exp(i*4*pi*f*x/c)/(R-x)^2 dx`.

| Constructed case | G48 magnitude | Reference magnitude | Absolute error / nonoscillatory integral |
|---|---:|---:|---:|
| L=0.30 m, R=10 m, f=95.93358656 GHz | 3.0006749e-3 | 1.4927492e-7 | 0.99999990 |
| L=10 m, R=7000 m, f=9.3535246896 GHz | 2.0408174e-7 | 1.4872085e-13 | 0.9999999998 |

- Frequencies deliberately repeat midpoint phases: severe aliasing of a smooth constant field, not production-3D, actual-native-frequency or fitted-checkpoint error.
- Required: frozen fitted-field signal/full-parameter-gradient refinement checks on both datasets. Equal spatial sample counts across methods do not qualify continuous integration.

### S3. Budget/documentation

- G48, 150 passes, matching 150-pass cosine schedule: explicit user-selected variants; neither hidden defects nor convergence/original-experiment evidence.
- [External reference](../external/SPINR_REFERENCE.md) still describes `paper-v1-direct`, G96/GL2, 1500 epochs, 3200 views, all 256 pairs. Adaptation ledger retains historical counts/1024-channel explanation. Use actual run recipes/acquisitions for manuscript claims.

<a id="se-fidelity"></a>

## 4. Sugavanam-Ertin

Reference: [arXiv:2602.17556v1, §§3–5](https://arxiv.org/html/2602.17556v1). Ledger: [SUGAVANAM_ERTIN_PAPER.md](SUGAVANAM_ERTIN_PAPER.md), including disclosed interpretations of ambiguous equations. No verified author implementation available locally.

- Paper: residual-constrained sparse Fourier inversion/subaperture; magnitude aggregation; PCA normals; eight-layer width-512 Softplus SDF, fourth-layer input skip, tanh output; iso-point resampling; six SDF/normal/Eikonal losses; 5° GOTCHA subapertures. Surface output distinct from complex novel-view-synthesis representation.
- Active [constrained_step](../rift/sugavanam_ertin_sparse.py#L43), not legacy LASSO `proximal_step`: residual tolerance/least-squares subproblem duality-gap checks. [Workflow](../rift/sugavanam_ertin_paper_workflow.py#L465) rejects Stage-2 handoff if any subaperture unconverged.
- [PaperSDF](../rift/sugavanam_ertin_paper.py#L170), [six losses](../rift/sugavanam_ertin_paper.py#L396), aggregation, radius PCA, Newton projection, signed edge weights, priority insertion match documented independent interpretation on inspection. No point-SH substitute, learned complex gain bypassing sparsity, or geometry-truth supervision. No fresh Torch execution/author-code comparison.

### E1. Stage 1 incomplete

- [stage1_only](../rift/sugavanam_ertin_paper_workflow.py#L471) exits before cloud extraction/SDF initialization/training. [September-21 launch record](SUGAVANAM_ERTIN_PAPER.md#stage-1-only-execution): selected for all six collection scenes; current checkpoint completion uninspected.
- Label as sparse subaperture reconstruction, not full two-stage surface result. [Complex readout](../rift/sugavanam_ertin_acquisition.py#L265) tags `stage1_subaperture_diagnostic_not_SDF_NVS`; do not attribute that NVS diagnostic to final SDF.

### E2. Accepted initialization variant

- [Recipe](../rift/sugavanam_ertin_paper_workflow.py#L28): Gaussian std 0.05, disclosed in fidelity identity; bare model retains literal std 1.
- Ledger: saturation at std 1, nonzero spatial/parameter gradients at 0.05; probes not rerun. Acceptance addresses disclosure only; learned zero surface, completed training, convergence/author fidelity remain unestablished.

### E3. Acquisition mismatch

- [Native adapter](../rift/sugavanam_ertin_acquisition.py#L102): first-order Fourier model over small ROI, fitting full complex response; no Camry-return isolation/exact near-field curvature.
- Default residual-energy budget: 1% of observed energy/subaperture, locally chosen rather than measured noise/model error. Feasibility in parking-lot clutter unresolved; constraint failure does not establish poor SDF reconstruction.
- Collection: first-order paths, origin-distance amplitude weighting; disclosed bistatic Fourier transfer, not exact synthetic-RIFT near-field equivalence.
- Illustrative nominal R=10 m, f=100 GHz: omitted monostatic transverse-curvature phase 0.524 rad at 0.05 m, 4.715 rad at 0.15 m. Analytic examples, not selected-array measurements. Quantify data-specific approximation error/achievable residual without silently replacing baseline Fourier model.

### E4. Scale/grid metadata

- Fixed PCA radius 0.3 m versus collection vehicles ≈0.1 m can mix much of object rather than local surface neighborhoods. G40 sparse/G48 surface-readout budgets disclosed; cloud quality, normal locality, geometry resolution unqualified.
- Recipe `grid_pitch_rule="one_native_range_resolution"` with fixed `granularity=40`; [plan](../rift/sugavanam_ertin_paper_workflow.py#L103) uses native resolution only at granularity zero. `voxel_pitch_m` correct; rule label misleading.
- At 0.30 m width: G40 pitch 0.0075 m; 10 GHz bandwidth: native range resolution ≈0.01499 m. Distinguish fixed-grid/native-resolution rules while preserving resume identities.

<a id="geraf-fidelity"></a>

## 5. GeRaF

Reference: [GeRaF-SENS 38266cb6e194e2f3dcbead614069a7281ffd21a5](https://github.com/VictorLlu/GeRaF-SENS/tree/38266cb6e194e2f3dcbead614069a7281ffd21a5): [renderer](https://github.com/VictorLlu/GeRaF-SENS/blob/38266cb6e194e2f3dcbead614069a7281ffd21a5/geraf/models/rendering/rf_rendering.py), [example configuration](https://github.com/VictorLlu/GeRaF-SENS/blob/38266cb6e194e2f3dcbead614069a7281ffd21a5/configs/geraf/bunnyboxv1/geraf2_bunnyboxv1_stage1.py), [runner](https://github.com/VictorLlu/GeRaF-SENS/blob/38266cb6e194e2f3dcbead614069a7281ffd21a5/tools/train.py). Ledger: [GERAF_V1_HARDENING.md](GERAF_V1_HARDENING.md).

- Fresh snapshot comparison: 28/28 recorded items match; 24 AST definitions with only registry decorators removed, four complete text references (two CUDA files, example config, runner). Covers full stage-1 loss, SDF/reflectivity/variance networks, primary/target samplers, dynamic mask, interpolation, scheduler.
- Import shims/native operator replacements inspected separately; entire upstream application not unchanged. No source-content hashes computed/checked.
- [V1 paper](https://arxiv.org/html/2605.29097v2): position-only reflectivity, scalar transmitted amplitude, L2 MF loss, stated network dimensions, 50000 updates. Released example: fixed-power network, extra SDF features, different encoding depth, Charbonnier loss. Local v1 configuration reconstructed on released stage-1 code; historical-v1 launcher not established.

### G1. Confirmed far-range visibility defect

- [sample_frame](../rift/geraf_source.py#L184) casts normalized receiver positions to float32. Copied [intersect_sphere](../rift/vendor/geraf_sens/rf_rendering.py#L446) computes:

```text
b = 2 * dot(center, direction)
c = dot(center, center) - 1
discriminant = b*b - 4*c
```

- [GOTCHA range](../GOTCHA.md#L270) ≈10 km / Camry radius 5 m gives norm(center)≈2000: subtraction of quantities ≈16 million to recover order-one values. Float64 propagation does not protect this separate float32 geometry operation.
- [Reproducer](../scripts/audit_geraf_sphere_precision_20260922.py): actual method, 10000 seeded sphere-interior points, synthetic antenna direction (0.6, 0, 0.8), identical already-rounded inputs for float32/float64; isolates arithmetic from storage precision. No data/fitted field loaded.

| Normalized antenna radius | Float32 false misses / 10000 | Median hit-position difference | Maximum hit-position difference |
|---:|---:|---:|---:|
| 2 | 0 | 1.25e-7 | 1.04e-6 |
| 2000 | 12 | 0.08861 | 0.99955 |
| 2029 (10145 m / 5 m) | 24 | 0.10684 | 0.99966 |

- Float64: zero misses. Differences scene-normalized; final row ≈0.534 m median/4.998 m maximum among mutually valid hits, 5115 valid differences >0.1 normalized units. Synthetic intersection diagnostics, not reconstruction errors/actual pulse statistics; counts may vary by hardware/Torch.
- Intersections feed detached SDF-CDF visibility correction in [_calibrate_transmission](../rift/vendor/geraf_sens/rf_rendering.py#L602); false misses use fallback CDF. Same float32 cast/quadratic in PVC; XPU untested.
- Repair: retain physical receiver geometry in float64; stable sphere intersection before casting intersection coordinates to network dtype; preserve CDF semantics. Check near-range agreement, far-range masks/positions, rendered signals, training impact. Source agreement does not qualify old GOTCHA fits.

### G2. Historical-v1 training unresolved

- [source_step](../rift/geraf_source.py#L122), default `released_runner_zero`: zero throughout fit; cosine annealing remains zero; inverse sharpness detached despite configured 10000-step freeze.
- Fetched runner never calls `update_step`; configured `StepHook` not executed. Matches executable source behavior, not established historical-paper behavior. `advance` is a distinct recipe.
- Absolute cosine `eta_min=5e-4`: SDF LR rises 1e-4 → 5e-4; other groups fall 1e-3 → 5e-4. Matches example; do not silently correct. Historical role, mask thresholds, regularizer weight, variance initialization/related fallbacks require disclosure.

### G3. Target/acquisition variants

- Released example: 601³ MF lattice over 0.6 m. Selected MF48: 48³; spacing ≈6.38 mm across collection 0.3 m cube, ≈212.77 mm across Camry 10 m cube.
- Lazy targets preserve trilinear interpolation on selected lattice, not cross-resolution equivalence; loss targets/measured-mask decisions can change.
- Collection 1t1r: [SingleBankGeRaFStage1](../rift/geraf_source.py#L94), no stale unselected-bank contribution, but differs from released two-bank optimization. GOTCHA: two banks over selected pulses.
- Current selection: 2400 collection views / 1500 GOTCHA sectors; older ledger counts 3200/2000 stale. Record MF48, channel selection, step policy, fallbacks, validation selection, geometry readout in experiment identity.

### G4. Native amplitude retained

- [source_amplitudes](../rift/geraf_source_ops.py#L53): released specular alignment, back-face gate, physical sum-path-squared attenuation, division by total target-point count. MF averages antennas/sums frequencies.
- Native geometry/reference conventions replace original hardware phase/bias; no extra RIFT geometric factor. Algebra/dense forward-gradient checks support adapter, not actual-acquisition Torch/original-CUDA equivalence.
- Verdict: source-copy claim supported; exact historical-v1 reproduction unsupported; GOTCHA defect requires repair/impact assessment before qualification.

<a id="radar-fields-fidelity"></a>

## 6. Radar Fields

Reference: [RadarFields ee76d76570f58b3d8539eafd7df0c188b58af333](https://github.com/princeton-computational-imaging/RadarFields/tree/ee76d76570f58b3d8539eafd7df0c188b58af333): [configuration](https://github.com/princeton-computational-imaging/RadarFields/blob/ee76d76570f58b3d8539eafd7df0c188b58af333/configs/radarfields.ini), [launcher](https://github.com/princeton-computational-imaging/RadarFields/blob/ee76d76570f58b3d8539eafd7df0c188b58af333/main.py), [Trainer](https://github.com/princeton-computational-imaging/RadarFields/blob/ee76d76570f58b3d8539eafd7df0c188b58af333/radarfields/train.py). Ledger: [RADAR_FIELDS_ADAPTATION.md](RADAR_FIELDS_ADAPTATION.md).

| Retained default | Details |
|---|---|
| Released model | Width 64, 32 spatial features, 16 hash levels, final resolution 512, BatchNorm, sigmoid occupancy, softplus reflectance; original parameter groups. |
| Sampling | Ten frames, 100 profiles, ten angular samples; released sampling/integration helpers. Neural queries combined across physical frame batch, avoiding separate training-BatchNorm updates per tile. |
| Optimization | Adam betas 0.9/0.99, epsilon 1e-15; released optimizer/scheduler expressions; LR clock 800 updates, complete epochs, epoch sine mask. |
| Current exposure | 2400 collection frames → 240 batches/epoch × 4 epochs = 960 updates; 1500 GOTCHA sectors → 150 × 6 = 900. Neither exactly 800 updates nor 150 passes; old 3200/2000-frame descriptions obsolete. |

Code: [model wrapper](../rift/radar_fields_upstream.py#L73), [optimizer loader](../rift/radar_fields_released.py#L28). Production requires TCNN; portable Torch/PVC is a separate, uncertified-equivalent backend.

### RF1. Changed objective/acquisition

- Release enables pose refinement, ground-occupancy prior, above-sensor-height penalty; [local recipe](../rift/radar_fields_recipe.py#L46) disables all three. Road-frame assumptions need valid coordinates, but removing them changes optimization.
- [released_scene_directions](../rift/radar_fields_native.py#L159), [renderer](../rift/radar_fields_native.py#L198), [unit_lut](../rift/radar_fields_native.py#L226): ROI-derived aperture/unit response replace measured antenna response. GOTCHA antenna patterns/attitude unavailable; measured point-spread equivalence unestablished.
- Exterior support fixed to zero/omitted from neural queries, changing BatchNorm population. [Finite-part KL](../rift/radar_fields_native.py#L244) removes model-independent exterior divergence, retains normalization gradient. Positive-exterior-limit test supports gradient; absolute KL value differs from release.
- Coherent responses → matched-range power, not original scanner images; GOTCHA uses exact frequencies. TRAIN-only peak, 60 dB scaling, occupancy threshold 0.1525; radial occupancy rule retained, automotive cross-azimuth median removed.
- Unit-gain aperture/normalization/filtering: sensor assumptions, not unit-magnitude phase-reference conversion. Noncoherent renderer lacks coherent interference terms even with coherently generated targets.
- Fresh CPU execution of actual upstream `Trainer.compute_loss`, removed terms disabled: zero loss/prediction-gradient/occupancy-gradient differences versus local `released_batch_loss` on positive interior fixture. Does not validate removed terms, CUDA or fitted results.

### RF2. Confirmed architecture-guard gap

- [Native recipe](../rift/radar_fields_gotcha.py#L45) locks batch, schedule, sampling, LR, three loss weights; accepts `hidden_dim=128` (default 64), `no_batch_norm=True` (False), `hash_levels=8` (16), retaining `profile="source-adapted-v3"` and `fidelity="source_model_and_losses_with_declared_acquisition_adaptations"`.
- [Pure configuration probe](../scripts/audit_released_baseline_profiles_20260922.py) executed all three overrides; separate CPU audit reconfirmed. Wrapper consumes them; backend check accepts them. Collection CLI also locks only part of released settings.
- Defaults match release; full controls saved. Claim-enforcement gap, not proof existing runs used wrong architecture. Lock release-defining architecture or assign variant status; inspect complete checkpoint arguments.

### RF3. Single-pair sampling

- Collection 1 Tx / 1 Rx: 100 profile draws sample one pair with replacement, not 100 independent antenna measurements. Angular samples may differ. Disclosed release-policy choice; count does not establish original information content/proportional compute savings.
- Verdict: `source-adapted-v3` supported, unchanged full-method reproduction unsupported. Separate CPU audit found no additional numerical defect; RF2/acquisition qualification remain open.

<a id="radarsplat-fidelity"></a>

## 7. RadarSplat

Reference: [radarsplat ea9c8f530c708622cc3b1b560436b5557ac6a49b](https://github.com/umautobots/radarsplat/tree/ea9c8f530c708622cc3b1b560436b5557ac6a49b): [launcher](https://github.com/umautobots/radarsplat/blob/ea9c8f530c708622cc3b1b560436b5557ac6a49b/examples/demo_scripts/run_radarsplat.sh), [trainer](https://github.com/umautobots/radarsplat/blob/ea9c8f530c708622cc3b1b560436b5557ac6a49b/examples/radar_simple_trainer.py). Ledger: [RADARSPLAT_FIDELITY.md](RADARSPLAT_FIDELITY.md).

- [create_scene](../rift/radarsplat_release.py#L86) executes released planar random initializer; [ReleasedRenderer](../rift/radarsplat_release.py#L168) calls original fork `_radar_rasterization` on CUDA, without automatic generic-gsplat/independent-CPU fallback. Source initializer/preprocessing functions extracted; historical independent renderer not used.
- [Training](../rift/radarsplat_release_training.py#L115): separate optimizer groups, 2000 updates, SH progression every 200 steps, retained effective SH limit, no refinement/densification; multipath weight 0.6.
- [Successful-path loss](../rift/radarsplat_release.py#L155): released L1/SSIM plus occupancy weight 10, size weight 100, probability weight 1000. Actual upstream loss statements matched CPU values/gradients; original initializer/means-LR schedule passed. CPU SSIM stand-in does not validate fused CUDA SSIM/rasterizer.
- Original NaN retry reduces probability regularization; adapter fails on nonfinite loss/gradients. Disclosed execution difference; failed-run launcher behavior not identical.

### RS1. Sensor/supervision transfer unqualified

- GOTCHA: coherent sector matched-filter power, elevation summed. Collection: coherent measurements → polar power targets. Both: TRAIN-peak normalization/clipping; metrics are clipped normalized power, not coherent complex-response errors.
- Scene cube → 100-unit diameter; Gaussian sizes/regularization thresholds change physical-meter meanings. Image lattice/effective beam/leakage parameters adapted.
- Occupancy: eleven nearest training directions, 3D reprojection, visible-donor averaging; replaces driving-trajectory neighborhoods. Multipath fit to training images, transferred from nearest training direction. FFT thresholds converted for crop length.
- Released spectral/antenna filter formulas reuse adapter-derived inputs; synthetic/native matched-filter point-spread equivalence unestablished.
- Two-scatterer diagnostic: coherent power 4 or 0 versus additive power 2. Establishes non-equivalence, not measured ranking or categorical invalidity of power-domain adaptation.
- Code: [recipe](../rift/radarsplat_release.py#L132), [ReleasedPreprocessing](../rift/radarsplat_release.py#L242), [native calibration](../rift/radarsplat_gotcha.py#L103).

### RS2. Capacity/exposure variants

- [PROFILES](../rift/radarsplat_release.py#L26): canonical user-selected `budget48` = 112000 Gaussians; `upstream`/released launcher = 20000; 5.6× capacity. Discretionary budget, not required acquisition conversion. Equal-looking spatial budgets imply neither equal capacity/compute nor convergence.
- At 2400 collection views, 2000 one-view updates cannot directly optimize every view in one shuffled cycle; all selected TRAIN views can affect preprocessing/occupancy. Same budget over 1500 GOTCHA sectors gives different exposure.
- RadarSplat selects final source step; GeRaF selects by validation. State comparison protocol.
- Verdict: released components with changed capacity/acquisition/supervision; unchanged original hyperparameters/reproduction unsupported. CUDA execution/sensor-model qualification remain open; PVC XPU is a separate renderer.

## 8. PVC boundaries

| Method | Inspection/evidence boundary |
|---|---|
| SpINR | Same four-view/1024-channel metadata defect. Batched native physical distance², reference-relative phase, exact-frequency DFT; extra affine-prefix-plus-final-endpoint branch inspected, not executed. |
| SE | Batched sparse objective; subsequent stages delegated to original workflow via resume. Shared equations/gates do not prove accelerator numerical/trajectory equivalence. |
| Radar Fields | [PVC ledger](RADAR_FIELDS_PVC_ADAPTATION.md): TCNN Torch shim, differences from production fp16; historical nonzero forward/gradient tolerances, not source-CUDA execution/strict equivalence. |
| RadarSplat | [PVC ledger](RADARSPLAT_PVC_ADAPTATION.md): Torch mirror/separate SSIM; reported three-fixture checks pass revised fp32-CUDA-reference gates; production TF32 convolution differences; H100 trajectory comparison open. |
| GeRaF | Same float32 sphere cast/quadratic; no XPU execution. |

PVC ledgers read, not rerun/independently checked against saved dumps. No fresh CUDA/XPU equivalence claimed. Record backend/precision identity; ports are not original CUDA kernels.

## 9. Validation evidence

### Initial SpINR/SE audit

- Fresh: 13 implementation files AST-parsed; assertions locate four-view constants/stale 1024-channel claims in both trainers; independent quadrature/curvature calculations, NumPy 2.3.5/SciPy 1.16.3.
- Base/`spec_bias` lacked Torch. Existing Torch forward/adjoint/gradient/recovery tests inspected, not rerun; historical passes not fresh evidence. No response/geometry-truth data used.
- Reproduce: `& 'C:\ProgramData\Anaconda3\python.exe' scripts/audit_spinr_se_fidelity_20260922.py`.

### Initial Radar Fields/RadarSplat audit

- Fresh: nine implementation files AST-parsed; actual pure native-RF recipe accepted three architecture overrides; revision-specific online configuration/network/launcher/training inspection.
- Local upstream checkouts absent; CUDA extensions, Torch tests, parity dumps, trained results unavailable/not executed. No model behavior/saved recipe changed.
- Reproduce: `python scripts/audit_released_baseline_profiles_20260922.py`.

### Separate GeRaF/Radar Fields/RadarSplat CPU audit

- Independently fetched pinned releases; 28/28 GeRaF source items match structural/text comparison; actual sphere method reproduces far-range failure.
- GeRaF/interpolation checks: 17 passed, one failed, 42 deselected. Failure before operator evaluation: NumPy 2.3.5 removed `np.lib.format._read_array_header`; production pins NumPy 1.26.0. Audit-environment incompatibility, not demonstrated production-kernel failure.
- Nine additional CPU checks passed: source initializer/schedule, executed loss statements, filter/crop sampling, integration dtype order, exterior-KL gradient limit, ROI independence, exact bistatic surfaces, coherent-interference diagnostic.
- Separate actual RF loss execution: zero value/gradient differences for retained core on interior fixture.
- Environment: Windows, Python 3.13, Torch 2.14.0+cpu, NumPy 2.3.5; isolated environment using host packages. MKL sequential for RF/RadarSplat to avoid duplicate host OpenMP runtimes. Production: Python 3.10/Torch 2.6/CUDA; CPU evidence does not certify it.
- No source hashes/historical hash tests, real responses, fitted checkpoints, production GPU or external experiment manager accessed. No fresh CUDA/XPU/convergence evidence. Initial audits' missing Torch does not negate separate CPU evidence.
- Reproduce: `python scripts/audit_geraf_sphere_precision_20260922.py`. Snapshots, standalone comparisons, CPU runners: sibling `../baseline-audit-20260922/`; audit working files, not production dependencies/vendor replacements.

## 10. Actions, cost and existing results

### RIFT sequence

1. Declare common mathematics: physical geometry, amplitude, known measurement-reference adapter, numerical evaluation; identify dataset-specific contracts in manuscript.
2. Share existing native operator first: unit amplitude, exact frequencies, effective reference, autofocus, selected acquisition; compare direct renderer plus independent small dense sum.
3. Verify values, coefficient/position gradients, complex adjoint, reference conversion, selected bins, batching across pulses/passes/ROI points. Separate numerical approximation from input/storage precision.
4. Establish native amplitude provenance; assess attenuation separately. New model identity/refit if changed.
5. Evaluate backprojection initialization, priors, refinement, objective separately; renderer migration does not align training.
6. Report actual loss domain/recipe and linked run artifacts. Common formulation permits known dataset-specific adapters, not literal reuse of synthetic acquisition conventions.

### Baseline follow-through

| Method/scope | Required work |
|---|---|
| SpINR | Correct batch metadata with checkpoint compatibility; frozen-field signal/full-gradient integration refinement on both acquisitions; selected-budget convergence. |
| SE | Correct grid-rule metadata with resume compatibility; inspect constraint satisfaction, Stage-2 completion, iso-point/normal supervision, learned zero surface; separately label Stage 1; quantify acquisition/neighborhood mismatch; retain accepted init/budget labels. |
| GeRaF | Repair/qualify far-range intersections with source visibility semantics; assess prior-fit impact; resolve/label historical step/scheduler/fallbacks; target-grid sensitivity. |
| Radar Fields | Enforce released architecture or variant status; qualify power targets, antenna weighting, occupancy, removed priors. |
| RadarSplat | Qualify power targets, filters, occupancy/multipath; Gaussian-count sensitivity; disclose capacity/exposure/final-step selection. |
| All | Freeze contracts separating released components, mandatory acquisition conventions, budgets, discretionary modeling. Inspect saved recipe/backend/checkpoint identities; source-versus-adapted forward/loss/gradient checks at actual backend/precision; qualify CUDA/PVC separately; establish convergence. Source reuse/equal budgets alone do not establish preserved quality. |

### Cost/results

| Change | Work | Existing-result treatment |
|---|---|---|
| Shared same-kernel implementation | Modest adapter/refactor; forward/gradient/adjoint checks | Retain only with numerical equivalence/unchanged contracts |
| Direct sum → NUFFT, same kernel | Frequency/error qualification; realistic GPU profiling | Numerical equivalence required; trajectories may differ |
| Add synthetic sum2 attenuation | Small kernel edit; calibration/scaling/fitting qualification | New model identity/refit; no relabeling old checkpoints |
| Align initialization/loss/priors/controller | Training/convergence experiments | New recipe/runs |
| Projected → full-response loss | Potentially larger inverse problem/support | Justified setup/fresh evaluation |
| GeRaF visibility repair | Stable physical geometry, near/far checks, signal/training impact | Assess actual old recipes/repair effect before retaining/rerunning |

- RIFT shared-operator adapter/equivalence-test estimate: 2–5 engineer-days, assuming data access/working GPU. Excludes larger scenes, full recipe alignment, convergence, production reruns, GeRaF repair; not measured completion time.
- No reliable GPU-hour/cost estimate. Measure initial/grown-point-count forward/backward updates plus validation/checkpoint overhead; extrapolate explicit budget. [GOTCHA timing notes](../GOTCHA.md) measure their recorded kernels/devices, not proposal.
- RIFT-only changes do not automatically require every baseline rerun if acquisition, roles, calibration, scoring, checkpoint-selection contracts remain intact. Independent baseline defects require remediation/impact assessment. Target resolution, capacity, step-policy, architecture changes require explicit identities.

## 11. Implemented: the amplitude law (2026-09-22, 17:00 CDT)

The user directed that the GOTCHA learned renderer adhere to the RIFT model as implemented for the RIFT dataset. Sections 1–10 above audit the code before this change; for example, section 2 still describes unit amplitude. This section and section 12 record what was implemented afterwards in the working tree. Only the amplitude law changed. Phase handling, native frequencies, the projected objective and the training recipe (section 4) did not.

**Contract.** The learned GOTCHA backends (`rift`, `rift_grid`, `isotropic`) now render

$$
\widehat y_{q,i}=g\sum_p A_{qp}\,\rho_p(\mathbf u_q)\exp\left[-\mathrm i\frac{4\pi f_{q,i}}{c}(r_{qp}-r_q^{\mathrm{ref}})\right],\qquad
A_{qp}=\frac{1}{(4\pi)^2\left[(2r_{qp})^2+10^{-9}\right]}.
$$

This is section 2's equation with $D_{q,i}$ kept inside the native kernel. $A_{qp}$ is computed by calling the RIFT-dataset operator's own `_geom_gain(r, r, r + r, 'sum2', 1/(4π)², 1e-9)` from `rift/range_operator.py`. The same function, constant and ε are used by `train.py`/B787. The inputs are physical monostatic legs, $r_{qp}=\lVert\mathbf x_p-\mathbf a_q\rVert$. The reference range enters only the phase, so the virtual reference-range legs warned about in section 2 are not used. Evaluation stays exact direct summation at the native frequencies in FP64/complex128. The NUFFT is not used (section 2, Direct sum / NUFFT; see section 12).

**Declaration.** The recipe key and CLI flag `--range-model {sum2,unit}` (default `sum2`) are in both `train_gotcha_dataset.py` and `train_gotcha_dataset_pvc.py`.
- `unit` reproduces the earlier kernel bit for bit.
- A recipe or checkpoint without the key means `unit`. A checkpoint written before this change therefore resumes only with `--range-model unit`, and a `sum2` resume of it is refused as a recipe change.
- GOTCHA `unit` is not `train.py`'s `none`, which keeps $1/(4\pi)^2$.
- MFBP is a matched-filter backprojection support, not the RIFT model. It is unchanged and does not declare the key.
- `sum2` is a new model identity. No existing GOTCHA RIFT fit is relabeled.

**Lanes.** The user explicitly authorized changing both the CUDA lane (`rift/gotcha_training.py`, `train_gotcha_dataset.py`) and the PVC lane (`rift_pvc/gotcha_training.py`, `rift_pvc/gotcha_batched.py`, `train_gotcha_dataset_pvc.py`). This is an exception to the AGENTS.md rule that PVC work leaves `rift/` and the root trainers unchanged. The PVC modules import the amplitude definition from `rift/gotcha_training.py`. The batched kernel's hand-written backward includes the amplitude slope $\partial A/\partial r$, taken from autograd of the shared definition, in the per-pulse distance gradient that drives spatial refinement.

**Camry magnitude.** These numbers come from the production acquisition (HH, passes 1–8, 1500 train sectors, 16-pulse cap, stride 2), using TRAIN antenna metadata only, with no responses read.
- Mean $A$ at the ROI centre: $1.53\times10^{-11}$. This constant is absorbed by the log-magnitude complex gain warm start.
- Across all 24 000 TRAIN pulses (1500 sectors × 16) (slant range 9976–10458 m), $A$ varies by a factor of **1.099**. It varies by up to about 6% within a single pass as the orbit's slant range changes. One global gain cannot absorb this pulse-dependent part.
- Across the 10 m cube, $A$ varies by **0.34%** within a pulse.
- The script is at the session scratchpad `camry_sum2_geometry.py`, with results recorded here.

**Verification (CPU, PVC env, 2026-09-22).**
- `tests/test_gotcha_range_model.py` covers the following:
  - `sum2` against an independent dense sum, plus gradcheck.
  - `unit` equals the earlier kernel bit for bit.
  - The GOTCHA `sum2` kernel times $\overline{D}$ equals `range_forward_operator(..., phase_sign=-1, range_model='sum2')` on a monostatic uniform grid to 1e-8 relative.
  - Recipe declaration, and the legacy-checkpoint resume gate.
- `rift_pvc/tests/test_gotcha_range_model_pvc.py` checks the batched forward/backward and per-pulse distance gradients against autograd through the per-pulse renderer to 1e-10.
- The existing `rift_pvc/tests/test_gotcha_batched_pvc.py` now compares the CUDA-file loop with the PVC batched trainer under the `sum2` default and passes.
- GOTCHA/GeRaF suites: 232 passed. Six failures in `test_gotcha_source_af.py`/`test_gotcha_step2_raw_bp_v1.py` also fail on a clean export of 96afb8d; they are not caused by this change.
- PVC card smoke 2156139 (Max 1100, 17:34 CDT): `SMOKE_GATE=PASS` with no warnings. On the device the batched kernel matches per-pulse autograd (forward 5.8e-16, weight gradient 0, position gradient 8.6e-8 with float32 parameters). The production-shape renderer forward+backward is 1.148× the `unit` time (0.0958 s vs 0.0835 s per 16-pulse sector).

**Runs.** On user direction the Camry RIFT fit was relaunched with `sum2` from a campaign prepared from the current tree: C4 `production_20260922_camry_rift_sum2_1job`, job 2156153, started 17:34 CDT, with `range_model: "sum2"` in its plan. Campaign jobs execute their root's saved `source/` snapshot. A `production_relaunch.py` relaunch of `camry-rift-full` from the C3 root would therefore run the pre-change `unit` kernel. A `sum2` Camry fit needs a campaign prepared from the current tree. The Camry RIFT task stays cancelled until the user directs otherwise.

## 12. Implemented: initialization, priors and loss aligned with the RIFT-dataset recipe (2026-09-22, 17:50–18:05 CDT)

User direction: make the GOTCHA adaptive-RIFT initialization, priors and loss the same as the RIFT-dataset recipe, unless the data formulation gives a reason not to, and keep exact direct summation rather than the NUFFT. On PVC the RIFT NUFFT costs about 40 ms per view forward+backward at G48, against 6 ms per pulse for the batched direct kernel. Evaluated along the absolute ~10 km path, the uniform-grid approximation of Camry's native grids would also give about 0.4 rad of phase error. The reference is the production RIFT-dataset command: `train_rift_dataset.py` → `train.py` full-scale adaptive argv (`--bp-init 100 --init-scale 0 --l1-weight 3e-7 --sh-degree-weight 1e-9 --regularizer-normalization fixed_initial --loss complex`, 2400 TRAIN views, 1 Tx × 1 Rx). The optimizer and schedule are unchanged in this round (the section 2 training-differences rows for learning rates, Adam ε, scheduler, update order and refinement). The change applies to `rift` only; `rift_grid` and `isotropic` have their own RIFT-dataset recipes and are unchanged. Both lanes carry it (`rift/gotcha_training.py`, `rift_pvc/gotcha_training.py`, both `train_gotcha_dataset*.py`), under the same user authorization as section 11.

**Initialization (`--initialization backprojection`, recipe `rift_dataset_backprojection_v1`, `bp_views` 100).**
- **Start:** a zero scene, then $b=\sum A^H s$ over the first 100 training views, and $w=\alpha b$ with $\alpha=\langle Ab,s\rangle/\lVert Ab\rVert^2$ in the degree-0 coefficient of the active points (`train.py` `backprojection_init`). This is followed by the loop's first-view gain warm start.
- **The native `sum2` renderer and its exact adjoint (`native_adjoint`) serve as $A$.**
- **Adaptations forced by the data:**
  - A view is a pass-sector (the GOTCHA split and update unit), so the start uses 1600 pulses.
  - The 100 views are the first ones of the trainer's own seed-42 epoch-1 order. B787's loader order is its seed-42 permutation, so its first 100 views are random over the sphere, whereas GOTCHA's sector list is ordered pass by pass. The order is drawn exactly where the loop would draw it, so the stream is unchanged.
  - $s$ is the ROI-projected measurement $\Pi y$, so that $b$ remains the descent direction of GOTCHA's own objective at the empty scene (see Loss).
- **Gauge:** a prediction-preserving gauge $w\to cw$, $g\to g/c$ then sets the mean per-point coefficient norm to B787's starting value, 3.445e-3. Adam steps each parameter by about the learning rate in absolute units, so this is what makes lr 0.003 mean the same thing. Without it, GOTCHA's data units would leave the scene coefficients orders of magnitude larger and effectively frozen.

**Priors (`--priors rift_dataset`, recipe `rift_dataset_dimensionless_priors_v1`).**
- **Form:** `train.py`'s own `regularization_loss` is called unchanged: group L1 × |g| plus SH degree l(l+1) × |g|², normalized by the initial point count (`fixed_initial`), once per update and channel. The prior is excluded from the refinement statistics, as in B787.
- **Why the weights cannot be copied:** they are not dimensionless. The `sum2` amplitude is about 1.6e-5 at B787's range and about 1.5e-11 at Camry's, and the data units differ.
- **Reference measurement:** `scripts_pvc/rift_dataset_prior_reference.py` runs the unchanged production B787 command through its backprojection start and warm start. It reproduces the production log's α = 327.5 and g = 0.7443−0.0566j, then measures:
  - $\sigma^2$ = 6.707e-9 (mean TRAIN power)
  - $m_1=|g|\,\mathrm{mean}\lVert w\rVert$ = 2.571e-3
  - $m_2=|g|^2\mathrm{mean}\lVert w\rVert^2$ = 9.767e-6
- **Dimensionless strengths kept:**
  - $\mu_1=\lambda_1 m_1/\sigma^2$ = **0.115**. At the start, the L1 penalty is 11.5% of the mean power, which agrees with the ≈11% of the epoch-1 logs.
  - $\mu_2=\lambda_2 m_2/\sigma^2$ = **1.46e-6**. This is the SH weight in units of initial scene energy.
- **GOTCHA weights:** $\lambda_1=\mu_1/m_1^{G}$ and $\lambda_2=\mu_2/m_2^{G}$, from its own backprojection start. They are recorded per channel in the checkpoint key `initialization` and in `initialization.json`. Per-epoch prior values are recorded in `history`.

**Loss (unchanged form).**
- **Same as B787:** the complex MSE averaged over each view's samples, one view (pass-sector) per update.
- **Kept, with justification:**
  - The ROI range-subspace projection stays: the measured Camry phase history contains the whole parking lot, which the 10 m cube cannot explain, whereas B787's simulated data contain only the aircraft.
  - The division by TRAIN mean projected power stays as the unit convention. The priors are transferred relative to the same quantity, and Adam ε 1e-20 makes an overall objective scale immaterial.
- **Checkpoint selection:** it already ranks epochs as B787's validation loss does.

**Compatibility.** Recipes and checkpoints without these keys mean `random_1e-3` and `none`, as with `range_model` → `unit`. An earlier checkpoint therefore resumes only with `--range-model unit --initialization random --priors none`.

**Verification.**
- `tests/test_gotcha_rift_dataset_recipe.py` (5) covers:
  - `native_adjoint` is the adjoint to 1e-12.
  - The start equals an independent dense backprojection in the projected domain.
  - The gauge preserves predictions and gives the B787 coefficient size.
  - The L1 prior at the start equals μ1, and the SH prior is 0.
  - The priors leave every refinement buffer unchanged.
  - The epoch-1 order is identical to the legacy run.
  - Legacy checkpoints resume only as legacy.
- The existing batched-vs-loop and exact-resume tests pass under the new defaults in both lanes. Overall: 237 passed, with the same six pre-existing failures.
- Real-data PVC smoke **2156186** (`scripts_pvc/smoke_gotcha_rift_dataset_recipe_pvc.sbatch`, the production Camry command bounded to 2 epochs; ran 18:20–18:44 CDT on ac100):
  - Mechanically clean: exit 0, no XPU→CPU fallback.
  - Backprojection start: 1600 pulses, warm-start gain 3.45−0.62j, gauge factor 3.7e-8 (the backprojected coefficients were ~1e5, so the gauge was needed). Transferred weights: λ1 = 3.53e-7, λ2 = 1.02e-17.
  - Validation projected RelMSE **59.4** after epoch 1 and **12.2** after epoch 2 (full-native 24.2 and 5.4). For comparison, epoch 1 was 16.9 for sum2 with the random start and 20.8/21.2 for unit.
  - The L1 term rose from μ1 = 0.115 at the start to 1.20 after epoch 1, so the scene energy grew about 10×.
  - The start and priors alone therefore do not remove the ≫1 validation. The leading suspect is the unaligned optimizer (section 13, RIFT).

## 13. Final implementation decisions (2026-09-22, 19:00 CDT)

This section records the decided implementation for RIFT and four baselines. Sugavanam–Ertin is handled separately by the SE session and is not covered here.

**Standing rule.** Each method keeps its original (released or paper) behavior. A departure is allowed only where the data formulation forces it or where the user has pinned a budget.

**Lanes.** The user authorized both the CUDA and PVC lanes for these repairs, as an exception to the AGENTS.md rule that `rift/` and the root trainers stay unchanged during the PVC port. Vendored upstream files are never edited; every repair lives in our adapters.

**Evidence base.** Four read-only audits of HEAD 0f53192, one per baseline, verified sections 3–8 against the code and the releases:
- the local release copies `external/RadarFields_reference` and `external/radarsplat_reference`;
- a fresh download of GeRaF-SENS 38266cb, compared with the vendored copy;
- the SpINR paper.

The repairs below were then made and tested on CPU in both lanes.

**Status labels used below.** **retained** = identical to the original. **forced** = adaptation required by the data formulation. **budget** = user-pinned choice. **closed** = repaired now. **open** = needs a user decision or qualification experiments.

### RIFT (GOTCHA adaptive RIFT)

| Item | Decision |
|---|---|
| Forward model | **closed.** RIFT-dataset `sum2` amplitude on physical legs; exact reference-phase adapter (section 11). |
| Numerical evaluation | **forced/decided:** exact direct summation, not the NUFFT. On PVC the NUFFT costs about 40 ms/view against 6 ms/pulse for the batched direct kernel. The uniform-grid error along the absolute path would also be about 0.4 rad. |
| Initialization | **closed.** RIFT-dataset backprojection start with a prediction-preserving coefficient gauge (section 12). |
| Priors | **closed.** `train.py` group-L1 and SH-degree terms at the same dimensionless strength as on B787 (μ1 = 0.115, μ2 = 1.46e-6). |
| Loss | **forced.** ROI range-subspace projected complex MSE, because Camry clutter lies outside the cube. Division by TRAIN mean projected power is the unit convention. |
| Optimizer and schedule | **closed (section 14).** The B787 optimizer and schedule adopted in full; σ² = 1. Previously open: position LR 1e-4 vs 3e-3, Adam ε 1e-20 vs 1e-8, no cosine restarts, epoch reshuffling, refinement cadence and thresholds. See the note below. |
| Physical warrant for `sum2` on processed GOTCHA amplitudes | **open (qualification).** A `--range-model unit` run is the control. |

*Why the optimizer is next.* Smoke 2156186 showed validation RelMSE 59.4 (epoch 1) and 12.2 (epoch 2), and the L1 mass grew about 10× in epoch 1. B787's Adam ε = 1e-8 acts on scene gradients of about 1e-11 to 1e-10. Per `train.py`'s own measurement, that throttles its coefficient steps to 0.05–1.1% of the learning rate. GOTCHA's ε = 1e-20 gives full learning-rate steps, comparable to the gauged coefficient size. Aligning ε only makes sense if the gradient scale is aligned too, for example by expressing the objective in B787 units (σ² = 6.707e-9). This is a hypothesis, not yet a measurement.

### SpINR

| Item | Decision |
|---|---|
| Field, kernel, loss, optimizer | **retained.** Signed-real position-only field. Product spreading on the collection and distance² on GOTCHA, with the reference range in the phase only. Closed-form selected DFT bins. Magnitude² + 0.5·complex² loss. Adam 1e-4 → 1e-5 cosine, clip 1. CUDA and PVC agree numerically. |
| Acquisition | **forced.** Native frequencies and exact geometry. |
| G48 midpoint quadrature; 150 epochs (paper 1500) | **budget.** |
| Batch grouping | **budget.** Collection: 4 views/update = 4 measurement channels at 1 Tx × 1 Rx (paper 1024). GOTCHA: 1024 pulses/update. |
| Batch metadata (S1) | **closed.** The recipe identity now states the actual channel count ("four whole views per update = 4 measurement channels") and drops the paper's 1024 from `specified` when it is not honored. The 16 × 16 identity is unchanged. Checkpoints with the old text still resume and read out through a legacy alias (`tests/test_spinr_batch_metadata.py`). |
| G48 quadrature adequacy (S2) | **kept at G48 (user decision 4, section 16): finer quadrature is not affordable in training time; disclosed.** Corrected numbers: the collection is 10 GHz ± 1.5 GHz, not 100 GHz. Maximum phase change per cell is **3.0 rad** there (pitch 0.96× a quarter wavelength), against **86.5 rad on Camry** (27.5× too coarse). GOTCHA SpINR is therefore the most at-risk result; a frozen-field refinement check is required before relying on it. |
| Collection behavior (A1) | **open (qualification).** Best validation at epoch 5–10 of 150; final coherent RelMSE 0.83–2.01; clipping active on 48–100% of updates. |
| Tooling | **open.** No GOTCHA SpINR readout or test evaluator exists (A2). `scripts/check_spinr_quadrature.py` lacks the production recipe, 2400 views and XPU (A3). |

### GeRaF

| Item | Decision |
|---|---|
| Released stage-1 code | **retained.** All 24 vendored definitions are AST-identical to 38266cb; PVC differs only by the documented autocast swap. Loss, networks, samplers, scheduler, source amplitudes, matched-filter normalization and bilinear port match the reference kernels. |
| Acquisition | **forced.** Native geometry and phase; one bank at 1 Tx × 1 Rx. |
| MF48 (release 601³); 50000 steps | **budget.** |
| Checkpoint selection | **budget.** By validation; the release keeps the final iterate. Disclosed. |
| G1 far-range receiver geometry | **closed, both lanes.** `receiver_geometry = float64_intersection` (default) runs the released `intersect_sphere` unchanged on float64 receiver geometry and casts only its points to the network dtype. `float32_release_cast` is the earlier behavior; it is omitted from recipes, so old checkpoints, manifests and caches replay exactly. Tests (`tests/test_geraf_far_range.py`, PVC twin) run the audit reproducer at R = 2029, 66.7 and 2. The fix equals a float64 reference; legacy equals the release bit for bit. |
| G1 on the collection | The collection itself sits at **R ≈ 66.7**, not 2. There the float32 path has no misses and hits within about 4 mm, so earlier collection runs are essentially unaffected. |
| G1 card validation | **passed.** Smoke **2156223** (18:43–19:08, stopped at the 25-min bound, exit 124) printed `float64_intersection`, had no XPU→CPU fallback, and ran 2550 stable steps with 0.996–1.0 of rays retained. Step-1000 training loss and step-1000/2000 validation equal the legacy-path rate run 2155255 to about 10 digits. rift-cc released Camry GeRaF **2156225** (root C5) on this. |
| Amplitude reachability on GOTCHA (N7) | **decided (user, 2026-09-22): train in native units with a configured `light_power` start.** Under our uncalibrated v1 reconstruction (`trans_power` = 1, class-default `light_power` = 0, trainable), Camry predicted ≈0: validation RelMSE 1.0000, `light_power` 5.5e-7 after 2550 steps, AdamW-ε-throttled. This is **not** the original recipe. The release calibrates amplitude per dataset: its loader divides MF targets by the radar's `trans_power` = 0.0158489319 (and `sub_adc` = 8), and its example starts `light_power` at 7.5644 (upstream `FixedPowerNetwork`, still trainable). GOTCHA supplies no transmit power, so its own calibration array was measured instead (next subsection). Calibrated units would put GeRaF's gradients at ~5e-13, far below ε = 1e-8, so they serve presentation only. Training keeps `trans_power` = 1. The new recipe value `light_power_start = train_warm_start_16_views` (GOTCHA config `protocols/geraf_mf48_gotcha.json`) sets the start to the closed-form least-squares scale between the initial render and the TRAIN MF targets of the first 16 training views: validation readout path, no bank updates, fixed sampling seeds. Recipes without the key keep the class default. Card smoke **2156334**: the warm start set `light_power` = 18.91 (scale 1.63e8). Step-1000 validation `mf_relative_mse` = **0.945** (was 0.99999999983) and native complex RelMSE 1.002; no XPU→CPU fallback. GeRaF now fits, in the same range as its collection runs (best 0.71–0.995). |
| Planner and tests | **closed.** The PVC GOTCHA planner now builds the GeRaF recipe from the PVC twin (N5). A test that hashed vendored source (forbidden by AGENTS.md) now checks definition presence at the pinned commit (N3). |
| Historical-v1 schedule (G2) | **decided: keep executable-release behavior, disclosed.** Under `released_runner_zero`, the inverse sharpness stays at 0.3 in every C1 checkpoint. |
| Other disclosures | **open.** The release divides targets by `sub_adc` = 8 and we do not (N2). The dynamic loss mask never removes a ray at 1 Tx × 1 Rx (N1). N7: GOTCHA's 1/path² scale, (10.1 km / 10 m)², needs about 13.8 nats **beyond** the 3.1–7.2 the collection's `light_power` reaches, i.e. ≈17–21 absolute; the GOTCHA warm start gives 18.9. Collection best checkpoints have native complex RelMSE 1.0–1.23 (qualification). |

### GOTCHA calibration array: a presentation constant

**Sources.** GOTCHA's MAT files carry no transmit power or radiometric constant. The dataset's reference (Casteel et al., Proc. SPIE 6568, 65680D, 2007, Table 1) tabulates a calibration array in the scene: 15-inch and 27-inch trihedrals and dihedrals, with positions and headings. `scripts/gotcha_calibration_array.py` (module `rift/gotcha_calibration.py`) forms coherent sub-aperture images of the trihedrals from the production split's TRAIN sectors only: HH, autofocus applied once, native kernel, ±6° around each boresight, 10 cm then 1 cm peak search.

**Result** (CPU job 2156302):
- All six 15-inch trihedrals are detected in all eight passes, at 18–190× background.
- **K = 7988**, where the data divided by K read as √RCS/(Rt+Rr)². This is the median of 48 measurements, with relative MAD 11%; per-pass medians range 6950–8855. The 27-inch cross-check gives 9501 (+1.5 dB).
- Table 1's positions sit a consistent **(+0.06, −0.49) m** off in the delivered data frame. This is a discrepancy between the documentation and the data, recorded as found.

**Provenance.** The reflector types, sizes, positions and headings are the dataset's. The radar cross-sections (triangular trihedral 4πa⁴/(3λ²); a square trihedral would be 9.5 dB larger) and K are our computation.

**Use.** K is **not** used by any fit, and no existing result changes. Every GOTCHA pipeline already normalizes by a TRAIN statistic, which cancels a constant, and GeRaF trains at `trans_power` = 1. K converts reported reconstructions to physical units (√RCS per resolution cell), after undoing each method's own normalization. It applies to any other object in the scene.

### Radar Fields

| Item | Decision |
|---|---|
| Model and training | **retained.** Released RadarField/TCNN model (hidden 64, feature 32, BN, 16 levels to 512, log2 19, SH bands 0–3); losses 0.6/0.36/0.03; Adam groups 0.9/0.99, ε 1e-15; 800-step clock; 10 frames × 100 profiles × 10 rays. PVC uses the separately identified fp32 torch shim. |
| Supervision and geometry | **forced.** Power targets from coherent data. ROI aperture in place of the unavailable antenna pattern (collection ROI half-angle 1.49°, well inside the 10° beam). |
| Ground-occupancy and above-sensor priors removed | **forced.** No level road frame: the collection spans elevations −90° to +90° with no ground, and GOTCHA looks steeply down. |
| Pose refinement removed on the collection | **forced.** Poses are exact, and the release's frame-index interpolation is undefined on spiral-ordered views. |
| Architecture guard (RF2) | **closed, both lanes and every entry point.** `source-adapted-v3` now rejects any non-release backend, architecture, output mapping or occupancy constant, and any change to its declared adaptation constants (`SOURCE_LOCKS` in `rift/radar_fields_recipe.py`). Every campaign run used these values, so no saved run changes meaning (`tests/test_radar_fields_source_locks.py`). |
| Pose refinement on GOTCHA | **closed (user decision 2, section 16).** Restored as the release's PoseOptimizer per pass-sector (Adam 9e-4, SE3, held-out sectors interpolated). New GOTCHA identity; Camry Radar Fields reruns. |
| Fixed-zero exterior | **kept (user decision 2, section 16).** The common registered cube stays the support for every method. Only 12–28% of collection samples fall inside the cube, so the exterior dominates BatchNorm and the bimodal term; disclosed. |
| Disclosures | The `--object` CLI is fixed at 3200 views (production uses the planner frontend). Best-by-validation checkpoint saved alongside the final one. Abort on nonfinite gradients. Choice of occupancy helper undocumented. |

### RadarSplat

| Item | Decision |
|---|---|
| Released code and recipe | **retained.** Initializer, optimizer groups/LRs/ε 1e-15 and means decay, loss weights 0.8/0.2/10/100/1000, multipath 0.6, SH schedule, no refinement (`refine_stop_iter` 0), rasterizer arguments. |
| PVC renderer | Separately identified torch mirror. It matches the fork's kernels line by line, but autograd replaces the CUDA backward, with IEEE instead of fast-math arithmetic and fp32 instead of TF32. Gate 4 (H100 trajectory) is open, so every RadarSplat result is a PVC-mirror result. |
| `budget48` (112000 Gaussians vs 20000); 2000 updates | **budget.** 0.83 passes over 2400 views; the release makes about 62 passes. |
| Occupancy supervision | **closed (user decision 3, section 16).** Images now use the release's log-power domain (TRAIN peak, 60 dB to [0, 1]), with every release threshold unchanged. Under the earlier linear mapping, 264/300 B787 training views had an all-empty occupancy label (median view maximum 0.0072 against threshold 0.10). New identity; RadarSplat reruns on all seven scenes. |
| Gaussian size relative to the image grid | **kept scene-relative (user decision 3, section 16).** The initial Gaussian is 0.08 range bins (release 4.2), and the size penalty never fires. The scene-relative and grid-relative release ratios cannot both hold with about 16 resolution cells across the scene; disclosed. |
| Production outcome | Collection validation RelMSE 0.81–0.98 (near trivial), consistent with the two items above. |
| Documentation | The 10° collection beamwidth only sets the window length (the kernel fixes σ at 4 px). The recipe records a few unexecuted nominal values (`spectral_leakage_width_m` 2.0, `max_size_threshold_m` 1.0, `effective_initial_scale_m` 0.25). The ledger's 3200/1000 should read 2400/1000. |
| NaN retry | The release restarts up to five times and then lowers the probability weight; the adapter stops. Disclosed; no run was affected. |
| Campaign provenance | `scripts_pvc/production_job.sbatch` exports a stale parity job ID into `backend.json`. Handed to campaign management; no identity change. |

### Corrections to sections 1–10

- **§3 S2:** the collection frequencies are 10 GHz ± 1.5 GHz, so the 26.2 rad/cell figure (quoted at 100 GHz) should be 3.0 rad/cell. The Camry figure is 86.5 rad/cell at 9.91 GHz.
- **§5 G1:** add the collection radius, R ≈ 66.7: 0 float32 misses, median 2.4e-4, maximum ≈ 0.01–0.03 normalized.
- **§8:** GeRaF "no XPU execution" is stale. The collection GeRaF trained on PVC, and the GOTCHA GeRaF card smoke is 2156223.
- **§5, §6, §9:** the cited reproducer scripts (`scripts/audit_*_20260922.py`) are not in the repository. They exist only in the original audit environment.

## 14. Implemented: the B787 optimizer and schedule (2026-09-22, 22:00–23:00 CDT)

User decision (verbatim): "set sigma squared 1 or 0.01, then let the training figure it self out. For the hyperparameter, fully adopt the RIFT B787 setup."

This closes the section 13 RIFT row "Optimizer and schedule". It applies to adaptive `rift` only, in both lanes (`rift/gotcha_training.py`, `rift_pvc/gotcha_training.py`, both `train_gotcha_dataset*.py`), under the same user authorization as sections 11–12. The start, gauge, priors and loss of section 12 are unchanged.

**σ².** The objective keeps its TRAIN mean projected power normalization, σ² = 1. Choosing 0.01 instead would scale every data gradient and, through the transferred weights, every prior gradient by the same factor. Adam is invariant to that while the gradients stay far above ε, so 1 and 0.01 give the same trajectory; 1 keeps the existing convention and recipe identity. At σ² = 1 and ε = 1e-8, Adam is effectively unthrottled on GOTCHA. B787's measured scene gradients (5.9e-12 to 1.1e-10 at σ² = 6.7e-9, `train.py`) correspond to about 9e-4 to 1.6e-2 per unit power, so comparable GOTCHA gradients sit five to six decades above ε. This is an estimate, not a GOTCHA measurement. B787's own ε throttle (scene steps at 0.05–1.1% of the lr) is therefore not reproduced; per the user's direction, training finds its own scale.

**Implemented values.** The reference is `train.py` parsing `fullscale_train_argv` (`rift/b7873200_adaptive_fullscale.py`), which `train_rift_dataset.py` runs for `rift`. The earlier GOTCHA value is given in brackets.

| Item | B787 (`train.py`) | GOTCHA `rift` default | Transfer |
|---|---|---|---|
| Scene and gain lr | 3e-3 | 3e-3 [3e-3] | Literal. |
| Position lr | 3e-3 | 3e-3 [1e-4] | Literal. |
| AdamW | ε 1e-8 on the scene, position and gain groups; betas default; weight decay 0; no clipping (`--clip-grad-norm` 0) | ε 1e-8 on both groups [1e-20]; betas default; weight decay 0; no clipping | Literal. Our groups split the same parameters differently (gain with the scene), which leaves per-parameter AdamW identical. |
| Learning-rate schedule | `CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)`, one `step()` per epoch after its last update | The same, per GOTCHA epoch (1500 pass-sectors) [constant] | Per epoch, as `train.py` counts it. With 150 epochs the restarts follow epochs 10, 30 and 70, and the fourth cycle ends at 150, as on B787. |
| Update unit | One view (`--step-every 1`) | One pass-sector | Forced (unchanged). |
| Training order | Loaders never shuffled: one seed-42 permutation every epoch | The epoch-1 seed-42 permutation, repeated every epoch [fresh permutation per epoch] | Literal. The backprojection start still uses that order's first 100 sectors. |
| Refinement cadence | One joint adaptive-capacity-v2 event after every 10th epoch, after the scheduler step and before validation | The same [every 100 updates, about 15 per epoch] | Per epoch. `train.py` counts epochs; a literal update count (24000) would be 16 GOTCHA epochs. |
| Probe stride | Every 16th view, rotated by the epoch: `(view_index + epoch) % 16 == 0` | Every 16th pass-sector, same rotation [every 10th update, no rotation] | Literal. |
| Spatial / angular fractions | 1/512 / 1/16 | 1/512 / 1/16 [0.05 / 0.05] | Literal (dimensionless). |
| Minimum spatial / angular exposures | 3200 / 200 | 3200 / 200 [2 / 2] | Literal. An exposure is one observation passed to `accumulate_refinement_data_stats`: a view on B787, a pulse on GOTCHA. At the parent 3200 views both equal one epoch of evidence. At a 10-epoch cadence neither binds on either dataset: B787's 2400-view production accrues 24000/1500 per interval and Camry 240000/15000 (1500 sectors × 16 pulses), so a per-epoch translation would select the same points. |
| Floors, cooldown, child maturity | 0, 0, 1 event, 1 event | The same | Literal (cooldown and maturity were already 1). |
| Maximum split level | 1 | 1 [3] | Literal. |
| Maximum active points, epochs, selection | 262144, 150, best validation | The same | Unchanged. |

**Recipe and compatibility.**
- New recipe keys: `optimizer = rift_dataset_b787_optimizer_v1` and `optimizer_schedule`, which holds every value above.
- CLI: `--optimizer b787` (default for adaptive RIFT) or `legacy`.
  - Unset `--lr/--pos-lr/--adam-eps/--refine-every/--probe-every/--max-level` take the selected recipe's values, and explicit values are recorded in the recipe.
  - Under `b787`, `--refine-every` counts epochs; under `legacy` it counts updates.
  - `--refine-fraction` is legacy-only and is rejected under `b787`.
- The earlier optimizer is `gotcha_adamw_constant_lr_v1`. Recipes and checkpoints without the key mean it (`with_legacy_keys`). So smoke 2156186 and the C4 root resume only with `--optimizer legacy`, as well as the section 11–12 legacy flags where those apply.
- `rift_grid`, `isotropic` and `mfbp` recipes are unchanged, key for key and value for value (checked against HEAD fe57027).
- Checkpoints add `scheduler_state_dict`, which a `b787` resume requires. A saved mid-epoch order must equal the fixed order.
- Each epoch's history adds `optimizer.learning_rates` (the rates that epoch used) and, after an event, each channel's split/grown/active counts.

**Verification.**
- `tests/test_gotcha_b787_optimizer.py` (6):
  - The defaults equal `train.parse_args(fullscale_train_argv(...))` field by field. Legacy, `rift_grid`, `isotropic` and `mfbp` values are as before.
  - The probe rotation matches `train.py`.
  - The learning rates of a 12-epoch fit equal a `train.py`-built scheduler, including the restart at epoch 11. The only event follows epoch 10.
  - An interrupt in epoch 2 resumes bit-exactly: model, optimizer, scheduler and history.
  - A `b787` checkpoint without scheduler state is refused.
  - A checkpoint written before the key resumes only as legacy.
- PVC lane:
  - A new batched-vs-loop test under `b787` (with fixture thresholds so an event acts) shows identical refinement decisions and schedule records. States agree within 1e-3/1e-5: ε 1e-8 passes the float32 summation-order noise of the tiny gradients into the steps, where ε 1e-20 did not.
  - The two earlier batched tests and the lifecycle test in `tests/test_gotcha_dataset.py` exercise the per-update cadence and now pin `--optimizer legacy`.
- Counts (CPU):
  - GOTCHA suites in the CUDA env: 160 passed, 6 skipped (HEAD: 154/6).
  - The same suites plus the PVC tests in the PVC env: 191 passed, 6 skipped (HEAD: 184/6).
  - Planner and baseline suites: 126 passed on both HEAD and the tree.
- A metadata-only dry run of the Camry command plans the recipe above.
- The real-data smoke is `scripts_pvc/smoke_gotcha_rift_b787_optimizer_pvc.sbatch`, not yet run.

## 15. Opt-in control: the RIFT-dataset NUFFT and full-native loss (2026-09-22, 22:00–23:15 CDT)

User direction: keep the current GOTCHA recipe (direct summation, ROI range-subspace loss) as the
default, and add, as a control whose result can be compared with it, the RIFT-dataset recipe's two
corresponding choices: the NUFFT forward operator, and the loss computed over the whole response. The
user notes that the NUFFT is correct on the RIFT dataset, so it is taken as the reference
implementation. PVC lane only (`rift_pvc/gotcha_nufft.py`, with small hooks in
`rift_pvc/gotcha_training.py`, `rift_pvc/gotcha_batched.py` and `train_gotcha_dataset_pvc.py`). The CUDA
trainer is unchanged.

**Flags and recipe.** The flags are `--forward-evaluation {direct,nufft}` and
`--loss-domain {roi_projected,full_native}`; the control sets `nufft` and `full_native`. The keys are
written only when not default, so the default recipe is unchanged key for key. The control adds:
- `forward_evaluation = rift_dataset_nufft_v1` and `nufft` (operator parameters);
- `loss_domain = full_native_complex`;
- `loss_normalization = train_only_mean_full_native_power`;
- `checkpoint_selection = validation_full_native_complex_rel_mse`.

Its output directory is `rift_nufft_full_native`, beside `rift`. Everything else is the default recipe:
acquisition, `sum2`, backprojection start, dimensionless priors and the section 14 optimizer.

**Forward evaluation.**
- **Operator:** the kernel of section 11, evaluated by `rift/range_operator.py`'s type-1
  Gaussian-gridding NUFFT at the B787 production settings (oversample 2, kernel width 20, float64). Its
  primitives are imported unchanged.
- **Grid:** as in `range_forward_operator`, the complete uniform grid is rendered and the selected bins
  are gathered afterwards. The grid is each pass's complete native source grid (424–434 bins),
  reconstructed as the linspace through its endpoints and checked by the operator's own 1e-2-bin gate.
  Camry passes deviate from it by ≤ 940 Hz (float32 storage, 6.4e-4 bin).
- **Pairs and path:** the pulses of a sector are the operator's pairs. The path is GOTCHA's two-way
  reference-relative path 2(|x−a|−r0), with phase sign −1, so the kernel is exactly
  exp(−i4πf(|x−a|−r0)/c). The `sum2` amplitude stays on the physical range.
- **Why the reference-relative path:** section 12's "about 0.4 rad" figure was for the absolute ~10 km
  path. On the reference-relative path (|Δr| ≤ 30 m over the cube) the same frequency rounding costs
  ~1e-3 rad.
- **Adjoint and gradients:** the backprojection start uses the operator's exact adjoint. Per-pulse
  position gradients for refinement come from the same backward pass (a hook on the per-pulse range
  tensor). As in `range_operator`, the Gaussian support-jump derivative is omitted.

**Loss domain.**
- **Loss:** `train.py`'s complex MSE over every selected native bin, per pulse, with no projection.
- **Normalization:** the TRAIN mean full-native power. The TRAIN statistics carry
  `loss_domain = full_native_complex`, with the projected values kept as `projected_*`.
- **Start:** the backprojection uses the unprojected measurement (the descent direction of this
  objective at the empty scene), and the gain warm start fits the full response.
- **Selection:** checkpoints are ranked by validation full-native RelMSE. The ROI-projected RelMSE is
  still recorded every epoch.

**Measured accuracy (CPU, 2026-09-22).**

| Check | Result |
| --- | --- |
| NUFFT vs exact direct sum, exact linspace, Camry-scale geometry | 5e-11 relative |
| Same, float32-stored grid | 2e-4 |
| Real Camry pulses, all 8 passes, G48 start grid, random weights (metadata only) | ≤ 3.4e-4 |
| GOTCHA NUFFT × conj(D) vs `range_forward_operator(phase_sign=-1, range_model='sum2')` | 1e-8 |
| Adjoint inner product | 1e-12 |
| Per-pulse range gradients vs per-pulse autograd | 1e-9 |

**What the control can show.**
- **Measurement:** on all 8 Camry passes, the response of any point in the 10 m cube lies in the ROI
  range subspace (rank 86 of ~214 selected bins) to 3e-11 in amplitude. Worst single point
  3.1e-11; random 400-point scene 1.8e-11 (metadata only).
- **Consequence:** the model's prediction is therefore always in that subspace, and for every pulse
  ‖pred − y‖² = ‖P(pred − y)‖² + ‖(I − P)y‖². The full-native loss is the projected loss plus a
  model-independent constant: the out-of-ROI clutter energy.
- **Identical to the default:** the backprojection start (A^H(I−P) = 0), the gain warm start and
  checkpoint ranking (the full-native error is the projected error plus a constant).
- **What changes:** the data gradient is the default's multiplied by E_proj/E_full, the fraction of
  TRAIN energy inside the ROI subspace. The priors are transferred relative to the same unit, so the
  control is the default recipe with the priors effectively E_full/E_proj times stronger relative to
  the fittable data, evaluated with the NUFFT.
- **Size of the change:** smoke 2156186's epoch-1/2 figures imply E_proj/E_full ≈ 0.4 on validation,
  about 2.5× stronger priors. The control smoke records the TRAIN ratio exactly.
- **Reading the result:** the control is the literal RIFT-dataset loss path on GOTCHA. It does not test
  a different fittable signal; clutter outside the ROI subspace adds a floor to its reported RelMSE and
  nothing else.

**Tests.** `rift_pvc/tests/test_gotcha_nufft_pvc.py` has 14 tests:
- the operator checks above;
- grid gates, and the recipe keys being opt-in and adaptive-RIFT-only;
- the control's batched lane matching its per-pulse loop, including a B787 epoch-end refinement event;
- the control start reducing to the default start bit for bit, and the NUFFT start matching the direct
  start to 5e-8;
- the full-native statistics and objective;
- exact interrupt/resume, with the default and single-flag recipes refused.

With the neighbouring GOTCHA suites (batched, range model, recipe, B787 optimizer, dataset): 62 passed.

**Runs.**
- **Smoke:** card smoke **2156462** (`scripts_pvc/smoke_gotcha_rift_nufft_full_pvc.sbatch`, frozen snapshot
  `RIFT_pvc_runs/rift_nufft_control_smoke/source_20260922_2212`). It timed the control against the default
  over 40 TRAIN sectors, then ran 2 epochs of the Camry production command with the control flags.
  It **passed**: ac013, 22:18–22:29 CDT, exit 0, 0 XPU→CPU fallback lines. The recipe carries the control
  keys, the section 14 optimizer and `sum2`.
  - **Cost:** 0.139 s per update against the default's 0.138 (ratio 1.005). TRAIN statistics plus the
    start took 33 s. The two epochs took 9 min 24 s, 4.7 min per epoch at 110,592 points. That projects
    to about 12–22 h for 150 epochs as refinement grows the scene, 25 h at worst.
  - **Validation:** full-native / ROI-projected RelMSE 1.2119 / 1.5334 after epoch 1 and 1.2093 / 1.5267
    after epoch 2. The L1 prior term was 0.125.
  - **Prediction confirmed:**
    - TRAIN E_proj/E_full = 0.4197, so the priors act ≈ 2.38× stronger than in the default.
    - Validation obeys full = 1 + (projected − 1)·0.397 at both epochs, as the subspace argument requires.
    - The start equals the default smoke 2156186's: gain 3.4485 − 0.6235j, gauge 3.709e-8, λ1 3.534e-7,
      λ2 1.023e-17.
- **Production:** user decision (2026-09-22, ~23:20 CDT): "A full production run is definitely worth
  the GPU time because that is the actual control that is worthy of being the standard result in the
  paper."
  - It is launched by rift-cc once smoke 2156462 is mechanically clean.
  - Prepare a campaign from the current tree:
    `python scripts_pvc/production_campaign.py prepare <root> --scenes camry --methods rift --gotcha-rift-variant nufft_full_native`,
    then `submit`.
  - The Camry RIFT task keeps the key `camry-rift-full`, gets job name `pvcprod-camry-rift-nufft-full-native`
    and writes to `.../rift_nufft_full_native`. `campaign.json` records `gotcha_rift_variant`.
  - The default campaign and plans are unchanged (checked byte for byte against HEAD 1d12314).

## 16. Implemented: RadarSplat image domain and Radar Fields GOTCHA pose refinement; decisions on the exterior, Gaussian size and SpINR quadrature (2026-09-22, 22:00–23:30 CDT)

User decisions (verbatim):
- 2: "Restore it on GOTCHA. I agree including the rerun. Zero field outside of support also sounds good."
- 3: "I agree with your option on scale. Keep the original setup so we are not responsible for the result. I agree with the scene-relative size as well."
- 4: "Scale the voxels is not acceptable for training in time. Keep 48."

The same both-lane authorization as sections 11–14 applies. Vendored release files are unchanged.

### RadarSplat: the release's log-power image domain (decision 3; pushed in 1d12314)

- **Why.** The release trains on Navtech polar images. Navtech converts received power to dB and quantises it to 0–255 ([RADIATE SDK](https://github.com/marcelsheeny/radiate_sdk)), and the trainer divides by 255. Its absolute thresholds (occupancy .10, raster cutoff 1/255) therefore live in a log-power domain. Our earlier conversion, linear power over one TRAIN peak, left 264/300 B787 training views with empty occupancy labels.
- **What.** `intensity_mapping = log_train_peak_60db_v1` maps TRAIN-peak-relative power over 60 dB to [0, 1], the Radar Fields adapter's `normalize_power_db`. The vendor's dB per count is not published, so the 60 dB span is our declared choice; it matches our other power-domain baseline.
- **Where.** The mapping is applied at every place the adapter forms an image: training/validation targets (`read_view`), the released multipath fit (`ReleasedPreprocessing.background`) and the occupancy donors (`TrainingOccupancy`).
- **Unchanged.** Every release constant and threshold, the renderer, the loss, scene-relative Gaussian size, and the target cache. The mapping acts after the peak division at training time, so the cache recipe and digest are identical and the Camry cache from 2156165 is reusable.
- **Compatibility.** A fresh run defaults to the log mapping and records `intensity_mapping` in its identity. A saved run resumes in its own mapping; an identity without the key means `linear_train_peak_v1` and stays byte-identical. Metrics keep their keys and add `intensity_mapping`, so a log run's validation errors are in the mapped intensity.
- **Tests.** `tests/test_radarsplat_intensity_mapping.py` (4):
  - the mapping equals `normalize_power_db`;
  - the linear identity is the earlier one and the cache recipe is shared;
  - a view at 0.0072 of the TRAIN peak keeps occupancy labels only in the log domain;
  - fresh runs take the log mapping, legacy checkpoints resume linear, and a mismatched resume is refused.
- **Existing suites.** The RadarSplat suites fail the same two pre-existing tests as HEAD fe57027 (a readout CLI loading a `dummy` checkpoint, and one order-dependent test), and nothing else.

- **Span, decided (user, 2026-09-22 ~23:30).** "I want to use the original implementation of RadarSplat. I don't care about the result from the RadarSplat, I only care about a fair comparison." The log mapping stays at 60 dB, the same declared span as the Radar Fields adapter, and is not tuned. Linear power is not the release's input domain either; its thresholds were set on dB images.
  - B787 check on the first 300 TRAIN views: pixels span −64 to −18 dB re the TRAIN peak (median −42 dB). The release threshold 0.10 sits at −0.9 × span dB.
  - Empty-label views / occupied label cells by span: linear 300/0%; 30 dB 223/2%; 40 dB 4/20%; 50 dB 0/39%; 60 dB 0/56%.
  - The dense 60 dB labels are disclosed, not tuned away.

### Radar Fields: the release's pose refinement on GOTCHA (decision 2)

- **Release.** `configs/radarfields.ini` sets `refine_poses = True`. `PoseOptimizer` keeps one se(3) vector per frame, starting at zero, applied as `pose @ exp_map_SE3(adj)` in the sensor frame. It uses Adam lr 9e-4, betas (0.9, 0.99), eps 1e-15, with no pose regularization, no coplanarity term and no pose schedule. `interp_test_poses` interpolates held-out frames linearly over the frame index, extrapolating at the ends.
- **GOTCHA implementation** (`rift/radar_fields_pose.py`, hooks in both `radar_fields_gotcha.py` lanes):
  - **Frame.** A frame is a pass-sector. Its sensor pose is our RadarSplat GOTCHA sector frame (`frame_for_positions`: mean pulse position, x outward), built from shard metadata.
  - **Correction.** The sector's pulse positions move rigidly, a' = T exp(adj) T⁻¹ a, using the release's own `exp_map_SE3`. The optimizer is the release's, stepped after the field step and before the LR clock, as in the release.
  - **Pointing.** GOTCHA ships no attitude, so ray pointing stays the adapter's look-at-ROI convention, recomputed from the corrected position without a gradient; its off-axis `acos` is singular at zero. Monostatic samples are antenna + range × direction, so they move rigidly with the corrected sensor, and the gradient reaches the pose through them.
  - **Lever arm.** The rotation acts through the rigid motion of the sector's ~100 m track about its centre. That lever arm is comparable to the release's ~50 m scenes, so the release lr applies unchanged. This supersedes the lever-arm rescaling suggested in the relay: with look-at pointing, the rotation cannot swing the beam across the 10 km path.
  - **Range axis.** The measured range bins stay those of the nominal pulse, as the release keeps its FFT bins and moves the sensor.
  - **Held-out sectors.** Validation sectors, and test sectors in any later sealed evaluation, get the release interpolation over sector id within their pass (each pass is one circle in 1° sectors; the 360/1 seam is treated as the sequence ends). The values are detached.
  - **Polarization heads.** Each head keeps its own corrections, as it keeps its own field.
- **Identity.** The GOTCHA recipe adds `pose_refinement` (`release_se3_v1`, all values above), and its `model_recipe.pose_refinement` changes accordingly. `pose_refinement: disabled` reproduces the earlier recipe byte for byte. Checkpoints add `pose_model_state_dict` and `pose_optimizer`, and a resume must match the saved mode. The collection keeps refinement off (exact poses; the spiral order makes the interpolation meaningless).
- **Tests.**
  - `tests/test_radar_fields_gotcha_pose.py` (4):
    - the disabled recipe is the earlier one, and the default is the release optimizer;
    - the correction equals the release `PoseOptimizer.apply_to_poses` result;
    - held-out sectors equal `interp1d` within their pass;
    - training moves and checkpoints the poses, and the mode gates resume.
  - `rift_pvc/tests/test_radar_fields_gotcha_pose_pvc.py` (2) covers the PVC twin.
  - All Radar Fields suites in both lanes: 91 passed, 15 skipped, 0 failed. This includes the existing bit-exact interrupt/resume tests, which now run with refinement on.
- **Campaign.** Camry Radar Fields 2156164 (old recipe) was stopped by rift-cc and is not resumable. The rerun needs a fresh root under this identity.

### Kept as they are, disclosed (decisions 2–4)

- **Radar Fields exterior.** The field stays zero outside the registered cube. The common cube is the support contract for all six methods (section 13 row).
- **RadarSplat Gaussian size.** Stays scene-relative (section 13 row).
- **SpINR quadrature.** Stays at G48. Finer quadrature is not affordable in training time, and no qualification check is run. The Camry figure of 86.5 rad per cell stands as a disclosure: at G48, SpINR on Camry behaves as a 48³ point model with network amplitudes, not a converged volume integral.

