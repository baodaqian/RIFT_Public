#!/bin/bash
# SH-SAS checkpoint scoring DRY RUN. Scores the EXISTING official SH-SAS checkpoint from
# job 13140226 on four pings -- 1 TRAIN + 3 VALIDATION, fixed in the script itself at
# DRY_RUN_NUM_TRAIN_PINGS=1 / DRY_RUN_NUM_VALIDATION_PINGS=3 (lines 229-230), not by a flag.
# No model is fitted, no paper number is produced, and NO SONAR SLOT IS DRAWN. Slots 7-11
# remain untouched. This exists to verify the slot 9 scoring path before a production
# release depends on it.
#
# GPU, NOT cpu-small, DESPITE THE SCRIPT'S OWN --dry-run HELP TEXT.
# scripts/diagnose_sh_sas_checkpoint_score.py:777 imports network_sh_mlp, whose MODULE-level
# lines 6 and 11 are `import tinycudann as tcnn` and `from eval_sh import EvalSH`. Both are
# compiled CUDA extensions. Neither exists anywhere in the RIFT env; both are installed in
# RIFT_SAS_CU124_L40S (tinycudann 2.0, eval_sh 0.0.0, torch 2.6.0). The help text's
# "for a cpu-small Slurm job" describes an environment this codebase does not have.
# Queued for correction with the lines 195-196 comment fix; not patched here, so the bytes
# that run are the bytes Fable validated.
#
# The environment below MIRRORS scripts/run_official_shsas_smoke.sh -- the preserved recipe
# that produced the very checkpoint being scored. Deviating from it would be scoring a
# checkpoint in an environment other than the one that made it.
#
# PYTHONPATH CARRIES THREE ENTRIES DELIBERATELY. No single --reed-root can satisfy the four
# deferred external imports, because add_reed_root_to_path (741-750) inserts exactly one path:
#   525 forward_model / 526,802 sampling  -> SH_SAS_reference/
#   777 network_sh_mlp                    -> SH_SAS_reference/inr_reconstruction/
# THIRD ENTRY IS THE ISOLATED TREE, NOT CORE RIFT. The official recipe ends its PYTHONPATH
# with $PROJECT; using that here would put core rift/ on the path of a run whose entire point
# is the isolated tree's newly deployed 47-line adapter with the Armadillo-or-Bunny
# allow-list. The script self-inserts PROJECT_ROOT at sys.path[0] so the isolated tree wins
# either way, but there is no reason to have core on the path at all.
#
# AUTHORIZATION: Daqian directly, 2026-09-16, in the manager session -- Inferno gpu-l40s for
# this run, plus a standing rule that dry runs flow freely INCLUDING the GPU lane. Not taken
# from a relay: Fable offered a verbatim quote and the recorded rule says a relay is never
# sufficient for paid Inferno, so it was put to him directly.
#
# Staging only; submitted by hand, not by the manager. Inferno is observation-only for it.
# Remote destination: /storage/project/r-jromberg3-0/dbao31/RIFT/slurm/run_shsas_checkpoint_score_dryrun_20260916.sbatch

#SBATCH --job-name=mgr_shsas_checkpoint_score_dryrun_20260916
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --tmp=12G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE

set -euo pipefail
umask 077

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
SHROOT="$PROJECT/external/SH_SAS_reference"
TREE="$PROJECT/experiment_sources/airsas_armadillo5k_fullgrid150x150x120_smoke_v1_20260914"
ENV_PREFIX=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/RIFT_SAS_CU124_L40S
CKR="$PROJECT/training_checkpoints/airsas_armadillo5k_official_shsas_smoke_v1_20260912"
CACHE="$PROJECT/datasets/sh_sas_reed/cache/armadillo_5khz_trainonly_120rings_v1_20260912"
OUT="$PROJECT/experiment_state/shsas_checkpoint_score_dryrun_20260916"

# INTENT IS A DECLARED FLAG, NEVER INFERRED FROM DISK. This run is fresh by design; it has
# no resume path. Declaring it means the guard below cannot silently block a designed resume
# the way an inferred output-dir-absent check would.
FRESH_RUN=1

: "${SLURM_JOB_ID:?must run in a Slurm allocation}"

if [[ "$FRESH_RUN" -eq 1 && -e "$OUT" ]]; then
  echo "ABORT: fresh run declared but $OUT already exists. Not overwriting; no rm -rf here." >&2
  exit 3
fi

if ! type module >/dev/null 2>&1; then
  for module_init in /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
    if [[ -r "$module_init" ]]; then
      # shellcheck disable=SC1090
      source "$module_init"
      break
    fi
  done
fi
type module >/dev/null 2>&1 || { echo "PACE environment modules are not initialized" >&2; exit 2; }
module load anaconda3/2023.03
module load nvhpc-cuda/12.4

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

export CUDA_HOME=/usr/local/pace-apps/manual/packages/nvhpc/24.5/Linux_x86_64/24.5/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
# sampling.py imports matplotlib.pyplot at module level; without Agg this dies on a headless node.
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
export PYQTGRAPH_QT_LIB=PyQt6
export TORCH_CUDA_ARCH_LIST=8.9
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$SHROOT/inr_reconstruction:$SHROOT:$TREE"

mkdir -p "$OUT"

# Fable's request: record the environment BESIDE the report, because the checkpoint's
# renderability depends on the environment as much as on the recorded args.
ENVLOG="$OUT/run_environment.txt"
{
  echo "slurm_job_id      = ${SLURM_JOB_ID}"
  echo "node              = $(hostname)"
  echo "conda_env         = ${ENV_PREFIX}"
  echo "modules           = anaconda3/2023.03, nvhpc-cuda/12.4"
  echo "cuda_home         = ${CUDA_HOME}"
  echo "torch_cuda_arch   = ${TORCH_CUDA_ARCH_LIST}"
  echo "pythonpath        = ${PYTHONPATH}"
  echo "python            = $(command -v python)"
  echo "recipe_mirrored   = ${PROJECT}/scripts/run_official_shsas_smoke.sh"
  echo "checkpoint_source = job 13140226 / airsas_armadillo5k_official_shsas_smoke_v1_20260912"
  python -c 'import torch, numpy; print("torch             =", torch.__version__); print("numpy             =", numpy.__version__); print("cuda_available    =", torch.cuda.is_available()); print("device_name       =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
  python -c 'import tinycudann, eval_sh; print("tinycudann        = importable"); print("eval_sh           = importable")'
} 2>&1 | tee "$ENVLOG"

echo ""
echo "=== nvidia-smi ==="
nvidia-smi || true

echo ""
echo "=== SCORING DRY RUN: 1 TRAIN + 3 VALIDATION pings ==="
python -B -u "$TREE/scripts/diagnose_sh_sas_checkpoint_score.py" \
  --checkpoint "$CKR/models/000005.tar" \
  --negative-control-checkpoint "$CKR/models/000000.tar" \
  --cache "$CACHE" \
  --render-config-json "$CKR/commandline_args.txt" \
  --reed-root "$SHROOT" \
  --output "$OUT" \
  --device cuda \
  --dry-run
