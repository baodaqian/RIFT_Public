#!/bin/bash
# Bunny5k measured-frontend/cache smoke, staged for final launcher review only.
# No file deployment or Slurm submission is performed by this staged artifact.
#
# Eventual reviewed source list (reference only; do not transfer here):
#   rift/airsas_contract.py
#   scripts/prepare_airsas_cache.py
#   scripts/run_airsas_frontend.sh
#   scripts/validate_airsas_5k_cache.py
#
# Release is contingent on deploying the four reviewed source files together; this
# manager-owned staging step does not transfer or modify those research files.
# Future track: airsas_bunny5k_frontend_cache_l40s_smoke_v1_20260913
# Future cell:  airsas_bunny5k_frontend_cache_l40s_smoke_v1_20260913

#SBATCH --job-name=mgr_rift_airsas_bunny5k_frontend_cache_l40s_smoke_v1_20260913
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --tmp=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0,atl1-1-01-002-8-0,atl1-1-02-006-34-0
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_bunny5k_frontend_cache_l40s_smoke_v1_20260913/mgr_slurm_logs/%j.out
#SBATCH --error=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_bunny5k_frontend_cache_l40s_smoke_v1_20260913/mgr_slurm_logs/%j.err

set -euo pipefail
umask 077

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: Bunny5k frontend/cache launcher accepts no arguments' >&2
  exit 2
}

[[ "${SLURM_JOB_QOS:-}" == 'inferno' ]] || {
  echo "ERROR: expected SLURM_JOB_QOS=inferno, got ${SLURM_JOB_QOS:-unset}" >&2
  exit 2
}
[[ "${SLURM_JOB_PARTITION:-}" == 'gpu-l40s' ]] || {
  echo "ERROR: expected SLURM_JOB_PARTITION=gpu-l40s, got ${SLURM_JOB_PARTITION:-unset}" >&2
  exit 2
}
[[ "${SLURM_JOB_NUM_NODES:-}" == '1' && "${SLURM_NTASKS:-}" == '1' ]] || {
  echo 'ERROR: expected exactly one non-array node/task' >&2
  exit 2
}
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || {
  echo 'ERROR: missing numeric Slurm job ID' >&2
  exit 2
}
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: array jobs are not allowed' >&2
  exit 2
}

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
SYSTEM_DATA=$PROJECT/datasets/sh_sas_reed/airsas_processed/system_data_files/system_data_bunny_5k.pik
FRONTEND=/storage/scratch1/1/dbao31/airsas_bunny5k_frontend_trainonly_v1_20260913
WEIGHTS_DIR=$FRONTEND/deconvolved_measurements/numpy
OUTPUT_CACHE=$PROJECT/datasets/sh_sas_reed/cache/bunny_5khz_trainonly_120rings_v1_20260913
ENV_PREFIX=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/RIFT_SAS_CU124_L40S
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
CUDA_HOME=/usr/local/pace-apps/manual/packages/nvhpc/24.5/Linux_x86_64/24.5/cuda
BUILD_ROOT="${SLURM_TMPDIR:-/tmp}/airsas_bunny5k_frontend_cache_v1_20260913-${SLURM_JOB_ID}/tcnn_build"
TMP_ROOT="${SLURM_TMPDIR:-/tmp}/airsas_bunny5k_frontend_cache_v1_20260913-${SLURM_JOB_ID}"

[[ ! -e "$FRONTEND" ]] || {
  echo "ERROR: FRONTEND already exists: $FRONTEND" >&2
  exit 2
}
[[ ! -e "$OUTPUT_CACHE" ]] || {
  echo "ERROR: OUTPUT_CACHE already exists: $OUTPUT_CACHE" >&2
  exit 2
}

require_nonempty() {
  [[ -s "$1" ]] || {
    echo "ERROR: required deployed file is missing or empty: $1" >&2
    exit 2
  }
}

require_nonempty "$SYSTEM_DATA"
require_nonempty "$PROJECT/rift/airsas_contract.py"
require_nonempty "$PROJECT/scripts/prepare_airsas_cache.py"
require_nonempty "$PROJECT/scripts/run_airsas_frontend.sh"
require_nonempty "$PROJECT/scripts/validate_airsas_5k_cache.py"

module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4

# Directly activate the existing environment; do not invoke bootstrap_airsas_stagea_env.sh.
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

cd "$PROJECT"
export PROJECT_ROOT="$PROJECT"
export ANACONDA_ROOT="$CONDA_BASE"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT/external/Reed_SAS_reference:$PROJECT/external/Reed_SAS_reference/inr_reconstruction:$PROJECT"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
export PYQTGRAPH_QT_LIB=PyQt6
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK must be set}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export BLIS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$TMP_ROOT/tmpdir" "$TMP_ROOT/temp" "$TMP_ROOT/tmp"
export TMPDIR="$TMP_ROOT/tmpdir"
export TEMP="$TMP_ROOT/temp"
export TMP="$TMP_ROOT/tmp"
export TCNN_BUILD_ROOT="$BUILD_ROOT"
export TCNN_BUILD_JOBS="$SLURM_CPUS_PER_TASK"
export TCNN_CUDA_ARCHITECTURES=89
export RIFT_SAS_ENV_PREFIX="$ENV_PREFIX"
export RIFT_SAS_RESUME_EXISTING=1

export SCENE=bunny
export SYSTEM_DATA="$SYSTEM_DATA"
export FRONTEND="$FRONTEND"
export WEIGHTS_DIR="$WEIGHTS_DIR"
export OUTPUT_CACHE="$OUTPUT_CACHE"
export RUN_REED_FRONTEND=1
export NORMALIZATION_MODE=train_only
unset NORMALIZATION_SCALE

echo "AIRSAS_BUNNY5K_TMP_ROOT=$TMP_ROOT"
echo "AIRSAS_BUNNY5K_BUILD_ROOT=$BUILD_ROOT"
echo "AIRSAS_BUNNY5K_GPU_IDENTITY=L40S"
echo "AIRSAS_BUNNY5K_TCNN_CUDA_ARCHITECTURES=$TCNN_CUDA_ARCHITECTURES"
echo "AIRSAS_BUNNY5K_ENV_PREFIX=$ENV_PREFIX"
echo "AIRSAS_BUNNY5K_FRONTEND=$FRONTEND"
echo "AIRSAS_BUNNY5K_OUTPUT_CACHE=$OUTPUT_CACHE"
echo 'AIRSAS_BUNNY5K_NORMALIZATION_SCALE=wrapper-computed-train-only'
echo 'AIRSAS_BUNNY5K_FRONTEND_REQUEST=measured-5k-pulse first-43200-pings 120-rings 360-ping-batches 1000-iterations sparsity-1e-2 unchanged-Reed-crop'

bash scripts/run_airsas_frontend.sh
python -B scripts/validate_airsas_5k_cache.py "$OUTPUT_CACHE"

echo 'AIRSAS_BUNNY5K_FRONTEND_CACHE_PASS'
