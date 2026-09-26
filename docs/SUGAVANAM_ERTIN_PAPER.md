# Sugavanam–Ertin implementation and fidelity ledger

Native GOTCHA supports shared [pulse](GOTCHA_PULSE_SELECTION.md) and
[frequency](GOTCHA_FREQUENCY_SELECTION.md) selection in train/validation/test.
Stage-1 operators/residual counts bind selected samples; scientific failure gates remain.

The user-selected spatial budget is **G40 Stage 1 and G48 SDF extraction**
on both datasets. Training now defaults to the user-requested Gaussian
initialization standard deviation **0.05** for all linear weights and biases.
These disclosed adaptations preserve the failure gate; they do not make SE benchmark-ready. See the
[scene-budget contract](SCENE_BUDGET.md) for configuration and recovery.

**Production Stage-1 solver (2026-09-22).** The repository's own sparse solver never
converged on the collection scenes (0/72 sub-apertures, residual ≈ 24× the Eq. 4 budget),
because it takes one projected step per sub-aperture per iteration and so never moves the
L1 radius past its first guess. By the user's decision, the collection production runs
solve Eq. 4 with the reference SPGL1 package instead (`spg_bpdn`, package defaults,
https://github.com/drrelyea/spgl1 at `405ca805`), on the same operator and budget. See
[SPGL1 Stage 1](#spgl1-stage-1-production-collection-lane) below and
[TRAINING_RECIPES.md](TRAINING_RECIPES.md). SE on GOTCHA is on hold for the current campaign.


Reference: [Sugavanam and Ertin, arXiv:2602.17556v1](https://arxiv.org/html/2602.17556v1).
There is no verified author implementation available in this project. This is an
independent implementation, not a verified reproduction of their reported results.
The comparison rule is to implement specified behavior and permit a difference
only where the data formulation requires an adaptation, implementation
information is missing, or the user explicitly selects a disclosed variant.
Each such choice is documented below.

## Initialization: selected variant and remaining limits

The literal reading of the stated standard-Gaussian initialization is `N(0,1)`
for every linear weight and bias. With the implemented eight width-512 Softplus
layers and tanh output, a bounded CPU probe at seed 42 produced saturated outputs
and zero spatial gradients at all 256 sampled points, for both the RIFT 0.15 m
and GOTCHA 5 m scene extents. This was initialization evaluation, not training.
It does not establish that the authors' implementation fails.

The maintained training recipe now uses the explicitly requested standard
deviation 0.05 (variance 0.0025), with zero mean, unchanged seed, Fourier features,
architecture and output activation. Both dataset frontends and the direct
`paper-v1` trainer inherit it. Explicit configs retain priority, including 0.05
manager variants. The recipe records `initialization_std` and
`fidelity=user_requested_gaussian_initialization_std_override`; recovery rejects
different recipes. [se_g40_readout48.json](../protocols/se_g40_readout48.json)
records the selected default. [se_paper_std1.json](../protocols/se_paper_std1.json)
restores the literal standard-deviation-1 recipe and its original identity.
The bare `PaperSDF` model still defaults to 1 for historical readouts/fixtures.

The same seed-42 probe at standard deviation 0.1 still yields 256/256 saturated
outputs and zero spatial gradients at collection extent 0.15 m. At GOTCHA
extent 5 m, 251/256 outputs saturate and only 5/256 points have nonzero spatial
gradients. The latter passes the existing nonzero-gradient gate but does not
establish usable initialization or convergence. These are response-free CPU
probes, not fitting results. The local evidence is
`experiment_state/se_init_std01_config_20260920/initialization_probe.json`.

The user's subsequent std-0.05 selection passes the same 256-point seed-42
probe on both scene scales: all points have nonzero spatial gradients and no
outputs saturate. Mean spatial gradient norms are 0.115865 (RIFT) and 0.139885
(GOTCHA). A synthetic SDF/eikonal backward check produces finite, nonzero
gradients for all 18 linear parameter tensors on both scales; no optimizer
step or radar-response access occurs. Initial sampled SDF values remain all
positive, so this does not demonstrate a recovered zero surface or convergence.
Evidence: `experiment_state/se_init_std005_config_20260920/initialization_probe.json`.

Before response reads or fitting, a fresh run performs this probe. A fully degenerate probe
returns `initialization_degenerate`, writes `initialization.json` and `status.json`,
and produces no benchmark checkpoint. This is an unresolved implementation
convention, not a low baseline score. Passing the probe would not establish
convergence either. Author clarification or additional implementation evidence
is needed to establish author fidelity; the selected variant also needs numerical qualification.

To reproduce the bounded check without an allocation or radar responses:

```bash
python train_sugavanam_ertin.py --recipe paper-v1 --object a320 --check-initialization
python train_sugavanam_ertin.py --recipe paper-v1 --object a320 --config protocols/se_paper_std1.json --check-initialization
```

Exit status 2 denotes the unresolved degeneracy. The same probe is available as
`rift.sugavanam_ertin_paper.initialization_audit` for a GOTCHA extent.
The GOTCHA hook's `fidelity_status` exposes this limitation alongside its callable
availability. Availability alone does not mean benchmark readiness.

## Interfaces and compatibility

`train_sugavanam_ertin.py --recipe paper-v1` selects the preprint-based workflow.
Its implementation schema is now `rift_sugavanam_ertin_subaperture_sdf_v2`;
`paper-v1` refers to the paper version, not to the checkpoint schema.
Earlier independent-v1 checkpoints and historical isotropic/stabilized runs
cannot resume this version. No real independent-v1 fitting was performed here.

Implementation responsibilities:

- `train_rift_dataset.py --method se`, `train_gotcha_dataset.py --method se`: canonical dataset entrypoints.
- `rift/sugavanam_ertin_collection.py`: SE-owned collection command/recipe construction.
- `rift/sugavanam_ertin_paper.py`: network, aggregation, normals, sampling, losses.
- `rift/sugavanam_ertin_sparse.py`: streaming residual-constrained sparse solver.
- `rift/sugavanam_ertin_acquisition.py`: sealed inputs and Fourier measurement maps.
- `rift/sugavanam_ertin_paper_workflow.py`: recipe, recovery, two-stage handoff/readout.

The collection dispatcher selects this recipe for SE. `--se-recipe legacy-full` retains
the older workflow. Unflagged direct full calls, standalone stage trainers and
the unified smoke preserve their historical behavior and checkpoints. The
existing 30-total-epoch Stage-1 benchmark remains Stage 1 only.

```bash
python train_sugavanam_ertin.py --recipe paper-v1 --object a320 --dry-run
python train_rift_dataset.py --object loader --method se --dry-run
python train_gotcha_dataset.py --method se --dry-run
```

The user subsequently requested merging and removing the dedicated frontends.
Both datasets now use the canonical dispatchers. RIFT retains the SE-owned command
builder; GOTCHA retains the SE workflow hook and detailed planning through
`rift/gotcha_baseline_planning.py`. Both retain the stable, sealed data ingress.
The former output roots were `training_checkpoints/RIFT_dataset_SE` and
`training_checkpoints/GOTCHA_dataset_SE`. Explicit `--output-root` can select a compatible
existing directory; resume still requires the exact saved identity and recipe.
No model, objective, initialization, data conversion or checkpoint format was
changed by consolidation. Details are in `docs/DATASET_FRONTEND_CONSOLIDATION.md`.

| Canonical frontend | Default run directory below `training_checkpoints/` |
| --- | --- |
| `train_rift_dataset.py --method se` | `RIFT_dataset/<object_id>/sugavanam_ertin/` |
| `train_gotcha_dataset.py --method se` | `GOTCHA_dataset/<region>/<dataset_identity_prefix>/sugavanam_ertin/` |

The RIFT trainer accepts one or more objects, defaulting to all six. SE resume requires
one object and an explicit checkpoint. The GOTCHA trainer defaults to Camry, eight
passes and HH, and accepts an explicit checkpoint. Both expose `--dry-run`,
`--list`, `--config` and `--check-initialization`; the last option is a bounded
CPU diagnostic, not permission to fit real data. Use `--output-root` to select
the original compatible run directory when resuming; consolidation does not
authorize relabeling or migrating a checkpoint.

SE tests and synthetic shard fixtures are also separate:
`tests/test_sugavanam_ertin_paper.py`, `tests/test_sugavanam_ertin_collection.py`,
`tests/test_se_dataset_entrypoints.py`, and `tests/se_dataset_fixtures.py`.
Model tests remain independent; entrypoint tests now verify the shared
dispatchers. The canonical frontends can plan SE and
RadarSplat in one invocation, with separate recipes, outputs and checkpoints.

These commands plan metadata only. `--config` accepts a JSON object of missing
numerical settings for the direct trainer or canonical dataset frontends.
The canonical RIFT flag is `--se-config` (`--config` is an alias); canonical
GOTCHA uses a `sugavanam_ertin` entry in `--method-config`, or `--config` with
SE selected alone. For example, an SE configuration object is:

```json
{"stage1_iterations": 500, "residual_relative_energy": 0.01, "stage2_steps": 5000}
```

These are declared settings, not author values. Specified model settings cannot
be replaced through this interface. Small architectures used by unit tests have
an explicit `synthetic_fixture_not_benchmark` recipe and are not reachable via
production configuration. Real execution still belongs to the experiment
manager; these interfaces do not submit jobs.

## Fixed behavior and equation mapping

| Paper location | Implemented behavior |
| --- | --- |
| Eqs. 2–4 | Far-field Fourier model; independent complex sparse fields; residual-energy constraint. |
| Section 4, GOTCHA | 5° azimuth groups across elevation passes; nine Fourier frequencies. |
| Eqs. 5–7 and following paragraph | Sum of magnitudes; threshold; strongest-aperture fallback; 0.3 m PCA. |
| Section 3.1 | Eight width-512 Softplus hidden layers, Fourier input, fourth-layer input skip, tanh output, standard-Gaussian weights and biases. |
| Eqs. 9–10 | Clipped Newton steps with 1e-4 residual stopping tolerance. |
| Eqs. 11–18 | Radius repulsion, literal signed edge weights, separately clipped updates, priority-based 2:1 insertion. |
| Eqs. 19–25 | Six raw SDF/normal/Eikonal losses; independent iso-point PCA normals. |

There is no extent multiplier on the tanh output or division by extent in the
SDF losses. No signed interior/boundary labels, sphere warm start, strongest-point
cap, topology requirement or added local sign-bracket acceptance rule is used.
Validation does not select a favorable earlier Stage-1 field. The terminal
constraint-satisfying sparse state supplies the SDF cloud.

## Missing information and explicit implementation choices

The following are local choices, not recovered author settings. They have not
been tuned using geometry truth, reserved-test data, or RIFT performance.

| Missing or ambiguous detail | Current choice and consequence |
| --- | --- |
| Eq. 4 numerical noise/model-error energy | Per-aperture `sigma² = residual_relative_energy * sum(abs(Y)²)`; initial ratio 0.01. This is not a measured noise estimate. |
| Sparse optimizer and convergence settings | Repository solver: projected least-squares solves over complex L1 balls, with Pareto radius root-finding and backtracking; 150 maximum outer steps, residual relative tolerance 1e-3, subproblem gap tolerance 1e-5 (never converges on the collection scenes; see below). Production collection lane: SPGL1 `spg_bpdn` with package defaults, one solve per sub-aperture; converged means SPGL1 exits with a root or a BP solution. |
| Precise voxel sampling factor | User-selected G40 on both datasets; voxel centres. Explicit `granularity=0` retains the prior `ceil(box_width / native_range_resolution)` rule; checkpoint identity binds the resolved choice. |
| Threshold value/units | `tau = 0.15 * max(sum_m(abs(S_m)))`, saved numerically. All above-threshold points are kept. |
| Fourier distribution, scale, raw-coordinate inclusion | Fixed Gaussian frequency vectors with standard deviation 2 cycles/m; concatenate raw xyz with sine/cosine features; seed 42. |
| Coordinate preprocessing and Softplus parameter | Metric local coordinates, no normalization; library-standard Softplus beta 1. |
| Optimizer, losses, batches and schedule | Adam, LR 1e-4, eps 1e-8; all six weights 1; off-surface alpha 100; 2048 samples per set; 5000 updates. |
| Sampling schedule and sizes | Attempt 2048 iso-points from step 1, refreshing every 100 steps. Initial proposals jitter cloud points by one voxel pitch. |
| Neighbourhood, bandwidth, clipping and repulsion sizes | Resampling radius 4 pitches, bandwidth 2 pitches, maximum step 2 pitches, repulsion alpha 0.1 pitch. Each value derives from the saved grid. |
| Iso-point PCA radius | Reuse 0.3 m; that paragraph does not restate a separate radius. Sparse/collinear iso-normal estimates are masked rather than substituted with network gradients. |
| Collinear cloud covariance | Use the strongest-aperture direction when the plane normal is underdetermined. No normal-sign orientation is imposed. |
| Projection text uses f <= 1e-4 | Interpret zero-set convergence as abs(f) <= 1e-4. Literal signed stopping would accept arbitrary negative interior values. |
| Undefined arithmetic, ROI and failed projection handling | Reject nonfinite or zero-gradient cases; bound Newton to 24 iterations; retain final samples in the ROI. No intermediate box clamping, metric-distance test or sign bracket. |
| Eq. 17 priority indexing | Choose the globally highest local priority, then its farthest radius neighbour. Stable index ties; at most 2*target_count insertion attempts. |
| Duplicate handling | Do not insert points closer than the projection tolerance. A shortfall remains visible; do not manufacture duplicates. |
| Eq. 21 indexes p while summing background q | Evaluate the off-surface term on the stated background set. |
| Empty validation apertures | Nearest training-aperture direction for the optional Stage-1 diagnostic only. |
| Mesh readout resolution | Native SDF zero on a 48³ evaluation grid; preserve disconnected/open surfaces. |

The signed, unsquared exponent in Eq. 13 is retained even though its dimensions
and normal-sign dependence are questionable. There is no selectable squared
replacement under the paper recipe. Sparse recovery solves the actual constraint;
a fixed LASSO penalty is no longer substituted.

The independent sparse solver follows the general L1-ball/Pareto approach
illustrated by [SPGL1's BPDN documentation](https://spgl1.readthedocs.io/en/latest/tutorials/spgl1s.html).
It does not vendor SPGL1 or claim to be the authors' solver. A duality gap for
each constrained least-squares subproblem and the achieved residual are saved.
Budget exhaustion without satisfying both checks yields `stage1_unconverged`:
no SDF fit or geometry comparison follows. Numerical tolerances are approximate
certificates, not an assertion of exact optimization.

The L1-ball projection clamps its computed multiplier to zero from below.
At a feasible boundary, `sum(abs(x))` and sorted `cumsum(abs(x))` can round to
opposite sides of the radius and otherwise yield a tiny negative threshold.
This is a numerical boundary correction in the independent solver; it changes
neither the residual budget nor convergence tolerances. Invalid user radii and
soft-threshold parameters remain errors. Synthetic boundary/analytic tests and
a saved-checkpoint projection replay cover this case; GPU recovery is unverified.

## Dataset adaptation boundary

The method can be ported to our data, but the original acquisition protocol
cannot be reproduced by relabeling our datasets. These differences remain part
of any comparison disclosure, independently of algorithm fidelity.

For GOTCHA, every native pulse is assigned by its actual angle to a 5° group.
A sector that crosses a boundary is split internally while retaining its sealed
role; all pulses are included exactly once. Mean aperture angles count pulses,
not equally weighted sector means. The published channel-owned HH autofocus
is applied once by the ingress. Native frequency vectors are preserved. The
Fourier operator uses a first-order range expansion about the working ROI and
converts to the existing reference phase. No exact near-field curvature is added.
The Camry ROI and train/validation split differ from the paper's full parking-lot
experiment; clutter remains in the full complex observations. Other polarizations
are not enabled by this hook.

The paper supplies no RIFT-style bistatic MIMO transfer recipe. Our declared
extension uses spatial frequency `k*(u_tx + u_rx)`, reducing to the monostatic
Fourier kernel when Tx equals Rx. All pairs and frequencies remain in native
layout. Fixed origin-reference phase and spreading convert to native units;
there is no learnable gain or exact position-dependent near-field correction.
This extension must not be described as a reproduced author acquisition.
Collection normalization uses training responses only, with exact source/object
and role checks. It rescales coefficients and the Eq. 4 constraint consistently.

The collection also scales vehicles to approximately 0.1 m and uses full-sphere
views. We retain the 0.3 m radius and default 5° azimuth grouping across elevation
instead of silently retuning them. At this scale the neighbourhood is no longer
local to the vehicle. How the authors would transfer their geometry settings to
these objects is unknown. The paper's inconsistent bandwidth-to-resolution text
is not used as a numerical conversion; the grid uses `c/(2*bandwidth)`.

## Artifacts and remaining validation

`checkpoint_latest.pt` records angular/source identities, sparse solver radii and
budgets, committed field hashes, training exposure, SDF/optimizer and RNG state.
`stage1_selected.pt` binds the terminal sparse field and derived cloud; the name
does not imply validation-based selection. `checkpoint_final.pt` reports actual
completion state. `surface.npz` binds its model, acquisition, recipe and readout
provenance, with its SHA-256 in the checkpoint. The SDF supplies geometry only;
Stage-1 complex readout remains a separate diagnostic, not SDF-generated NVS.

Checkpoint resume validates identities and progress before responses. Missing
iso supervision and missing zero sets have explicit statuses. These engineering
checks cannot compensate for unresolved author initialization information.

Focused synthetic tests cover analytic constrained solutions, infeasible budgets,
complex projection, Fourier kernels/gradients, exact resumed trajectories,
5° boundary pulses, raw losses and network definitions, initialization failure,
sealed access and disconnected surface readout. Metadata preflight covers all
six objects and eight GOTCHA HH passes without response reads. Real fitting,
CUDA behavior, convergence and decision-grade comparisons remain unvalidated.

## Recorded handoff verification

The following records the completed SE ownership handoff on 2026-09-19
(America/New_York). It is evidence from that verification, not a claim that all
future versions of the concurrently edited repository have been tested.

| Verification scope | Recorded result |
| --- | --- |
| SE paper/equation/acquisition/recovery tests | 45 passed in `tests/test_sugavanam_ertin_paper.py`. |
| SE-owned collection command tests | 14 passed in `tests/test_sugavanam_ertin_collection.py`. |
| SE-owned dataset entrypoint tests | 14 passed in `tests/test_se_dataset_entrypoints.py`, including frontend import/list checks with shared dispatchers and other baseline modules unavailable. |
| Combined integration selection after the RadarSplat routing handoff | 247 passed, including the six previously failing RadarSplat command cases. |
| Historical SE validators | Full-source validator passed; Stage-2 static validator passed 31 contract/lifecycle checks. |
| Dedicated RIFT frontend, metadata only | All six objects passed planning with registered roles and no response reads. |
| Dedicated GOTCHA frontend, metadata only | Eight HH passes, 2000/440/440 train/validation/reserved-test pass-sector viewpoints, 72 angular groups; no response reads. |

The 73 SE-owned tests are a subset of the 247-test integration selection, not
additional tests. That combined selection comprised the three SE-owned files
above plus `tests/test_rift_dataset.py`, `tests/test_radarsplat_collection.py`,
`tests/test_smoke_consolidation.py`, `tests/test_gotcha_dataset.py`, and
`tests/test_sugavanam_ertin_stage2_stabilized_v3.py`. It was not the entire
repository suite, the entire historical shell-control suite, or a CUDA run.
The earlier concurrent-work failures are resolved in this recorded selection.

No real radar fitting, reserved-test evaluation, long conversion, scheduler
submission, manager launch request or Git publication was part of the SE work.

## Remaining work for the SE owner

1. Resolve the initialization convention using author clarification or additional
   implementation evidence. Preserve the failing literal probe as a numerical
   diagnostic; do not score it as a reconstruction failure or silently replace
   the initialization under the paper recipe.
2. Keep missing noise budgets, numerical settings and transfer assumptions
   explicitly bound to recipes. The bistatic/full-sphere acquisition and the
   0.3 m neighbourhood on 0.1 m objects remain disclosed transfer limitations.
3. Once the initialization is resolved, obtain authorized manager-run evidence
   for sparse-solver convergence, SDF behavior, native CUDA cost and recovery.
   Geometry and optional Stage-1 signal diagnostics must retain their distinct
   meanings. This handoff is not an experiment launch request.

The compact local contract and TODO reference is `PROJECT_MEMORY.md` (not
distributed). Destination setup, execution and the unresolved initialization
gate are covered by the tracked [NCSA Delta handoff](../NCSA_Delta_Production_Handoff.md).
The file ownership boundary is [BASELINE_OWNERSHIP.md](BASELINE_OWNERSHIP.md).

## Selected collection 1t1r acquisition

Canonical collection runs now select physical Tx 0/Rx 0 before Stage-1 data,
normalization or sub-aperture preparation. Sample counts derive from the selected
response shape (600 samples/view for 1t1r), while exact selected bistatic geometry
feeds the existing Fourier extension. The acquisition identity binds indices,
source geometry and frequency grid throughout recovery and exported provenance.
Native GOTCHA remains one pair per pulse and retains every selected pulse.
Stage 2 and the literal `initialization_degenerate` / `stage1_unconverged` gates
remain. Implemented antenna support does not make the unresolved initialization
benchmark-ready. See [shared selection contract](ANTENNA_SELECTION.md).

## Stage-1-only execution

The collection CLI accepts `--se-stage1-only` with `--method se`; the direct
paper-v1 trainer accepts `--stage1-only`. This execution boundary preserves the
full recipe and existing checkpoint identity. It completes the declared sparse
Stage-1 budget, retains the residual/optimality convergence checks, and stops
before SDF initialization or optimization. A converged result writes
`checkpoint_stage1_final.pt` and returns `stage1_complete`; the recoverable
`checkpoint_latest.pt` remains in phase `stage1` at the terminal iteration.
It can later continue the original full workflow if separately authorized.
An unconverged result still reports `stage1_unconverged` and exits nonzero.
Stage-1-only execution rejects checkpoints already in Stage 2 or later.
Default execution without the flag remains the full two-stage workflow.

The September 21 production runs used this flag on all six collection scenes with the
repository solver. Every one that finished ended `stage1_unconverged` (0/72 sub-apertures
converged after 150 iterations), so no Stage 2 ran. They are superseded by the SPGL1 lane
below. Job IDs and paths are in `RIFT_PVC_Adaptation.md` sections 19 and 21, not this guide.

## SPGL1 Stage 1 (production collection lane)

The paper names no solver for Eq. 4. On 2026-09-22 the user chose to fill that gap with the
standard tool rather than a repository solver: SPGL1 (van den Berg and Friedlander), Python
port https://github.com/drrelyea/spgl1 at commit `405ca805a2d56d783a0445e834c801c5b7c2263a`,
vendored unchanged in `rift_pvc/vendor/spgl1` (LGPL-2.1).

- **Problem per sub-aperture:** `minimize ‖x‖₁ subject to ‖A x − b‖₂ ≤ σ`, complex voxel field x
  (G40, 64 000 voxels). A is the declared Eq. 2 operator (first-order bistatic Fourier, native
  reference, origin spreading), stored densely per sub-aperture. b is the response over the
  training RMS. σ = sqrt(2 N · target) = 0.1 ‖b‖, the same 1 % energy budget as before.
- **Solver settings:** `spg_bpdn(..., iscomplex=True)` with every other option at the
  package default: `opt_tol` 1e-4, `bp_tol` 1e-6, `ls_tol` 1e-6, `dec_tol` 1e-4,
  `iter_lim` 10 × rows, `n_prev_vals` 3, `step_min` 1e-16, `step_max` 1e5. The line search
  is SPGL1's projected-arc search, and the radius update is its inexact Newton step.
- **Convergence gate:** a sub-aperture is converged when SPGL1 exits `root_found` or
  `bp_solution_found`. The repository's residual (1e-3) and Frank–Wolfe (1e-5) certificates
  are computed and stored per sub-aperture as diagnostics. SPGL1's default exit certifies
  the residual (a root keeps ‖r‖ within 1e-4 of σ, so the energy stays inside the 1e-3 test)
  but not L1 minimality, so the Frank–Wolfe certificate is not met.
- **Identity:** `rift_pvc.sugavanam_ertin_spgl1.make_recipe` is the paper-v1 recipe plus
  `stage1_solver = spgl1_spg_bpdn_defaults`, the SPGL1 repository and commit, the accepted
  exits, and `stage1_iterations = 1` (one SPGL1 solve per sub-aperture). Old checkpoints
  cannot resume it, and it cannot resume them.
- **Execution:** CPU jobs solve sub-apertures independently (`scripts_pvc/se_spgl1_stage1.py solve`).
  `assemble` writes a terminal Stage-1 checkpoint in the original schema. The unchanged
  workflow then applies its own gate, builds the cloud, and runs Stage 2 on PVC. From
  2026-09-22 on, each solve first checks the dense operator against the batched objective.
- **Measured:** about 2 min per sub-aperture on 32 CPU cores. As of 2026-09-22 evening, B787,
  firetruck, race car and loader passed the gate with 72/72 sub-apertures exiting `root_found`;
  A320 and X-59 were still solving, and no Stage 2 had run yet.

Stage 2 and everything after Stage 1 are unchanged. GOTCHA can be driven by the same lane
(`--gotcha <frontend arguments>`), but SE on GOTCHA is on hold for the current campaign.
