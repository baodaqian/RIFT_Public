# RIFT Public — Intel PVC implementation

RIFT (Radon Implicit Field Transform) learns a direction-dependent complex
scattering field from radar measurements for novel-view signal prediction and
3D reconstruction. This repository publishes the Intel Data Center GPU Max
(PVC / PyTorch XPU) implementation, its dataset frontends, and the incorporated
baselines.

The training and backend source is copied from the working RIFT project. Shared
`rift/` modules and original trainers are included because the PVC entrypoints
import them. Dataset measurements, reference meshes, trained checkpoints,
experiment caches, and credentials are not part of this source release.

## Included methods

| Method | PVC entrypoint | Implementation |
| --- | --- | --- |
| RIFT | `train_pvc.py` | Shared RIFT field/operator with XPU execution |
| SpINR | `train_spinr_style_pvc.py` | Independent implementation of the paper, with disclosed acquisition and budget choices |
| GeRaF | `train_geraf_pvc.py` | Vendored GeRaF-SENS components and PVC bindings |
| Radar Fields | `train_radar_fields_pvc.py` | Upstream source with the separate `upstream-tcnn-torchshim` PVC backend |
| RadarSplat | `train_radarsplat_pvc.py` | Upstream source with separate Torch implementations of its rasterizer and SSIM operations |
| Sugavanam–Ertin | `train_sugavanam_ertin_pvc.py` | Independent paper implementation; Stage 1/2 entrypoints and vendored SPGL1 are included |
| RIFT-SAS / SH-SAS | `train_sas_pvc.py`, `train_sh_sas_pvc.py` | Additional sonar code and the radar-adapted SH-SAS implementation |

The two canonical radar frontends are
[`train_rift_dataset_pvc.py`](train_rift_dataset_pvc.py) for the six synthetic
objects and [`train_gotcha_dataset_pvc.py`](train_gotcha_dataset_pvc.py) for GOTCHA.
Backend availability and compatibility are reported by each frontend's `--list`
option. A method's presence in the source tree does not imply that every dataset
or paper table used that method.

## Environment

Use an Intel GPU allocation with a compatible driver. The environment is pinned
in [`requirements-pvc.yaml`](requirements-pvc.yaml), including Python 3.10.16
and PyTorch 2.12.1+XPU. Do not load CUDA or Intel compiler modules over the XPU
wheel's bundled runtime.

```bash
conda env create -f requirements-pvc.yaml
conda install -n RIFT-PVC -c conda-forge --override-channels \
  open3d=0.19.0 numpy=1.26.0 matplotlib-base=3.10.1 pillow=11.1.0 \
  python=3.10.16 pip=25.0 setuptools=75.8.0 wheel=0.45.1

# Set this to persistent scratch on a cluster.
export RIFT_PVC_CACHE_ROOT=/path/to/persistent/scratch/rift-public-cache
source .local-setup/activate-public-pvc.sh
```

Set `RIFT_PVC_ENV` before activation if using an environment prefix or another
environment name. Copy [`.env.datasets.example.sh`](.env.datasets.example.sh)
to `.env.datasets.sh` and edit the data/output locations; the public activation
helper loads that local file if present. It is excluded from Git.

The original `.local-setup/activate-pvc.sh` and Slurm launchers are preserved as
ACES configurations. They contain the original account, source, data, and
scratch paths. Adapt those paths and scheduler settings before submitting them
from another checkout or machine. Direct Python entrypoints accept explicit
dataset and output paths.

An `Aten Op fallback from XPU to CPU` warning indicates a backend defect; it is
not an accepted execution mode for PVC runs. See
[`docs/PVC_ENVIRONMENT_PLAN.md`](docs/PVC_ENVIRONMENT_PLAN.md).

## Dataset frontends

Run these commands from the repository root after configuring the data paths.
Planning examples below use `--dry-run` and do not start training.

```bash
python train_rift_dataset_pvc.py --list
python train_gotcha_dataset_pvc.py --list

python train_rift_dataset_pvc.py \
  --dataset-root "$RIFT_DATA_ROOT" --object b787 --method rift \
  --num-tx 1 --num-rx 1 --num-train 2400 \
  --output-root "$RIFT_RUN_ROOT" --dry-run

python train_gotcha_dataset_pvc.py \
  --dataset-root "$GOTCHA_DATA_ROOT" --method rift --polarizations hh \
  --num-tx 1 --num-rx 1 --output-root "$GOTCHA_RUN_ROOT" --dry-run
```

Select the intended region, shards, protocol, antenna IDs, frequency selection,
and method configuration before fitting. Removing `--dry-run` executes the
selected plan; the examples are not a replacement for a complete experimental
recipe. Keep caches and checkpoints on persistent scratch. Resume with the same
output root, acquisition, roles, recipe, and saved optimizer state. Reserved
test responses remain sealed during training and model selection.

Prepared AirSAS caches can be used with `train_sas_pvc.py` and
`scripts_pvc_sas/run_airsas_comparison_pvc.sh`. Rebuilding caches from raw Reed
system-data pickles additionally requires the authors'
[Reed source checkout](https://github.com/awreed/Neural-Volumetric-Reconstruction-for-Coherent-SAS)
via `--reed-root` / `REED_ROOT`; that optional checkout was absent from the
working source tree and is not included here.

## Source layout

| Path | Contents |
| --- | --- |
| `rift_pvc/` | XPU adapters, baseline backends, GOTCHA training, regions, and vendored PVC dependencies |
| `scripts_pvc/` | Preparation, launch, recovery, evaluation, rendering, and campaign tools |
| `rift_pvc_sas/`, `scripts_pvc_sas/` | Sonar PVC implementation and tools |
| `rift/`, root trainers, `scripts/` | Shared implementations and tools imported by the PVC paths |
| `external/` | Radar Fields, RadarSplat/GLM, and FRTM reference sources |
| `protocols/` | Dataset and model configurations |
| `docs/` | Dataset contracts, model adaptation details, and numerical recipes |
| `tests/`, package test directories | Existing source tests, copied without running them for this release |
| `slurm/` | Original shared launch configurations |

Start with the [frontend contract](docs/DATASET_FRONTEND_CONSOLIDATION.md),
[scene budgets](docs/SCENE_BUDGET.md), and
[training recipes](docs/TRAINING_RECIPES.md). Baseline-specific notes describe
the distinctions between original implementations, independent reproductions,
and PVC replacements. Complex-signal, power-domain, and geometry metrics have
different meanings and should be reported with their acquisition and recipe.

## Provenance and third-party notices

[`SOURCE_PROVENANCE.json`](SOURCE_PROVENANCE.json) records the parent repository,
selected Git commit, working-tree additions, copied file inventory, and upstream
repositories/commits. Source availability is checked using required files and
APIs, without content-hash acceptance gates. Historical hash inventories copied
from the project are documentation only.

Third-party sources retain their own terms and attribution; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the adjacent license files.
This release does not assign a new blanket license to the combined source tree.
