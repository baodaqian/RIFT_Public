# Dataset frontend consolidation

## Transfer to NCSA Delta

The tracked [NCSA Delta production handoff](../NCSA_Delta_Production_Handoff.md)
is the receiving agent's setup and execution guide. Both canonical frontends
accept `--dataset-root` and `--output-root`; GOTCHA additionally accepts
`--shard-root`. PACE paths and the existing PACE Slurm launchers are not portable
defaults. Assemble RIFT links/manifests at the destination from the original
archives and meshes; supply the existing native-v2 GOTCHA shards separately.
Neither dataset nor generated targets/checkpoints travel through Git.

Both frontends enforce a real Slurm allocation for preparation/fitting. They do
not submit jobs. Use Delta's account/partition/environment in the job wrapper;
do not fabricate `SLURM_JOB_ID`. Select one object/method per initial production
job for separate resource and recovery accounting. Multi-method commands remain
sequential; `all` is a planning convenience, not a parallel launch facility.

Model/loss defaults are unchanged by this transfer. The source-adapted GeRaF,
Radar Fields, RadarSplat and SpINR backends are wired on both datasets; SE's
literal initialization remains unresolved. The native GOTCHA `sh_sas` and `fsh`
hooks are absent. Read current `--list`, the method adaptation ledger and the
handoff's readiness table before launching. Source/recipe gates must remain
enabled after data relocation; migrated PACE checkpoints are not automatically
compatible with new paths or freshly prepared caches.

## Canonical interfaces

At the user's request, RadarSplat and Sugavanam–Ertin now share the canonical
`train_rift_dataset.py` and `train_gotcha_dataset.py` frontends. The four
method-specific dataset scripts were removed after merge verification. Their
model engines, recipe modules, acquisition conversions and checkpoint gates
remain separate. There is no implementation conflict between their interfaces.

```bash
python train_rift_dataset.py --object all --method radarsplat sugavanam_ertin --dry-run
python train_gotcha_dataset.py --method radarsplat sugavanam_ertin --dry-run
python train_rift_dataset.py --object a320 --method se --check-initialization
python train_gotcha_dataset.py --method se --check-initialization
```

HH remains the default and intended GOTCHA comparison channel; SE supports HH
only. GOTCHA defaults to Camry, passes 1–8 jointly, and sealed 2000/440/440
train/validation/reserved-test pass-sector roles. RIFT retains all six independent
objects and their 3200/1000/1000/4800 roles. Planning reads metadata and headers
only, with no response access, target conversion, fitting or output creation.
Initialization diagnostics are bounded CPU probes, not training authorization;
a degenerate SE initialization yields exit code 2.

## Removed scripts and output compatibility

| Removed frontend | Replacement | Previous default output root under `training_checkpoints/` |
| --- | --- | --- |
| `train_rift_dataset_radarsplat.py` | `train_rift_dataset.py --method radarsplat` | `RIFT_dataset_RadarSplat` |
| `train_rift_dataset_se.py` | `train_rift_dataset.py --method se` | `RIFT_dataset_SE` |
| `train_gotcha_dataset_radarsplat.py` | `train_gotcha_dataset.py --method radarsplat` | `GOTCHA_dataset_RadarSplat` |
| `train_gotcha_dataset_se.py` | `train_gotcha_dataset.py --method se` | `GOTCHA_dataset_SE` |

Canonical defaults are `training_checkpoints/RIFT_dataset/<object>/<method>/`
and `training_checkpoints/GOTCHA_dataset/<region>/<dataset-id-prefix>/<method>/`.
Use an explicit `--output-root` pointing to the previous root when recovering
an existing compatible run. Nothing is moved, relabeled or silently resumed.
Both plans use JSON with a `plans` list; RIFT entries contain object identity and
commands, while GOTCHA contains a dataset summary and per-method backend details.

Resume still requires exactly one object/method for RIFT or one learned method
for GOTCHA. RIFT RadarSplat uses `--resume auto`; RIFT SE takes an explicit paper
checkpoint. GOTCHA RadarSplat takes the run's `radarsplat_gotcha.pt`; GOTCHA SE
takes its own paper-workflow checkpoint. Backends retain their full
object/source/region/split/recipe checks before response access.

## Preserved frontend features

RIFT uses the owner command builders in `rift/radarsplat_collection.py` and
`rift/sugavanam_ertin_collection.py`. SE's `--se-config FILE` (`--config` alias)
forwards declared paper-recipe settings only to SE, including in a combined
selection. Immutable settings remain rejected. `--check-initialization` requires
SE alone; `--device` currently applies to RadarSplat/SE selections. Default model
budgets stay in owner modules. Source RadarSplat's CUDA/dependency preflight
runs before any subprocess can prepare real targets.

GOTCHA's `rift/gotcha_baseline_planning.py` retains the former detailed reports:
RadarSplat conversion grid, target counts, fixed source recipe and resume file;
SE angular partition and full two-stage recipe. It imports only the selected
baseline's planning dependencies. `--list` parses literal capability declarations
without importing training engines. Execution still uses each maintained root
trainer's `run_gotcha` hook.

GOTCHA `--method-config FILE` maps canonical method names to distinct objects,
for example `{"radarsplat": {}, "sugavanam_ertin": {}}`. Alternatively, `--config
FILE` accepts one selected baseline's object. These flags are mutually exclusive.
Owner validators reject invalid/immutable options before opening the dataset.
Generic RIFT optimizer flags do not override either GOTCHA baseline recipe.

Fitting/preparation requires an experiment-manager allocation. Every selected
output is checked before the first method starts; methods run sequentially in
requested order. An exception or interruption stops the sequence. RIFT preserves
the child exit code; the GOTCHA shared CLI exits 143 on interruption (the removed
SE-specific GOTCHA script used 75). SE's existing backend still raises for an
incomplete scientific/numerical status. Failure is not reported as completion.

## Fidelity and remaining limitations

This merge changes orchestration only. It adds no difference from either paper
or source implementation to the model, objective, optimizer, training budget,
initialization, native response conversion or checkpoint schema. Existing
necessary acquisition adaptations and author-information gaps remain recorded
in [RadarSplat fidelity](RADARSPLAT_FIDELITY.md) and
[Sugavanam–Ertin paper mapping](SUGAVANAM_ERTIN_PAPER.md).

SE's literal published initialization remains unresolved and deliberately stops
fitting when degenerate. RadarSplat still requires the pinned original CUDA
extension and fused-SSIM; real CUDA execution and convergence remain unvalidated.
These are pre-existing baseline limitations, not merge conflicts. Combining CLI
entrypoints does not establish benchmark readiness.

## Verification

After removing the four scripts, the focused combined suite passed **246 tests**:
both dataset contracts, both new frontend integration suites, RadarSplat native
conversion/recovery and collection checks, and SE equation/recovery/collection
and entrypoint checks. Joint dispatch tests mock fitting; numerical tests use
bounded synthetic fixtures. Tests cover config isolation, HH defaults, metadata
planning, destination/dependency preflight, recipe-specific resume, initialization
failure and interruption propagation. Active Python and launcher sources have no
imports or invocations of the removed scripts.

Real metadata-only planning also succeeds for both models across all six RIFT
objects (12 object/method plans) and all eight GOTCHA HH passes (2000/440/440
roles), with native response access and NumPy memory mapping explicitly blocked.
No real target conversion, training, scheduler/manager action or publication was
performed. The model/CUDA readiness limits above remain in force.
