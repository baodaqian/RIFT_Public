#!/bin/bash
# AirSAS Armadillo5kHz Stage A on L40S: bootstrap the CUDA frontend environment,
# generate frontend measurements, and prepare the train-only cache.
# This launcher is staged locally for manager-owned review; it does not submit itself.
# The research bootstrap owns the post-install tiny-cuda-nn smoke test.
# Build and source artifacts intentionally remain under the job-unique temporary root
# on failure; there is no cleanup or automatic retry.
#SBATCH --job-name=mgr_rift_airsas_arm5k_frontend_cache_l40s_v1
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --tmp=64G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0,atl1-1-01-002-8-0,atl1-1-02-006-34-0
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_adaptive_vs_sh_sas_v1_20260911/armadillo5k/frontend_cache/mgr_slurm_logs_l40s/%j.out
#SBATCH --error=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_adaptive_vs_sh_sas_v1_20260911/armadillo5k/frontend_cache/mgr_slurm_logs_l40s/%j.err

set -euo pipefail
umask 077

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: AirSAS frontend/cache launcher accepts no arguments' >&2
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
SYSTEM_DATA=$PROJECT/datasets/sh_sas_reed/airsas_processed/system_data_files/system_data_arma_5k.pik
FRONTEND=/storage/scratch1/1/dbao31/airsas_armadillo5k_frontend_trainonly_v1_20260912
CACHE=$PROJECT/datasets/sh_sas_reed/cache/armadillo_5khz_trainonly_120rings_v1_20260912
WEIGHTS_DIR=$FRONTEND/deconvolved_measurements/numpy
BUILD_ROOT="${SLURM_TMPDIR:-/tmp}/airsas_armadillo5k_frontend_cache_v1_20260912-${SLURM_JOB_ID}/tcnn_build"
ENV_PREFIX=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/RIFT_SAS_CU124_L40S

[[ ! -e "$FRONTEND" ]] || {
  echo "ERROR: FRONTEND already exists: $FRONTEND" >&2
  exit 2
}
[[ ! -e "$CACHE" ]] || {
  echo "ERROR: CACHE already exists: $CACHE" >&2
  exit 2
}

require_nonempty() {
  [[ -s "$1" ]] || {
    echo "ERROR: required deployed file is missing or empty: $1" >&2
    exit 2
  }
}

require_nonempty "$SYSTEM_DATA"
require_nonempty "$PROJECT/scripts/bootstrap_airsas_stagea_env.sh"
require_nonempty "$PROJECT/rift/airsas_contract.py"
require_nonempty "$PROJECT/scripts/validate_airsas_5k_cache.py"
require_nonempty "$PROJECT/scripts/prepare_airsas_cache.py"
require_nonempty "$PROJECT/scripts/run_airsas_frontend.sh"

module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

cd "$PROJECT"
export PROJECT_ROOT="$PROJECT"
export ANACONDA_ROOT="$CONDA_BASE"
export PYTHONPATH="$PROJECT"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK must be set}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export BLIS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
TMP_ROOT="${SLURM_TMPDIR:-/tmp}/airsas_armadillo5k_frontend_cache_v1_20260912-${SLURM_JOB_ID}"
mkdir -p "$TMP_ROOT/tmpdir" "$TMP_ROOT/temp" "$TMP_ROOT/tmp"
export TMPDIR="$TMP_ROOT/tmpdir"
export TEMP="$TMP_ROOT/temp"
export TMP="$TMP_ROOT/tmp"
export TCNN_BUILD_ROOT="$BUILD_ROOT"
export TCNN_BUILD_JOBS="$SLURM_CPUS_PER_TASK"
export TCNN_CUDA_ARCHITECTURES=89
export RIFT_SAS_ENV_PREFIX="$ENV_PREFIX"

export SCENE=armadillo
export SYSTEM_DATA="$SYSTEM_DATA"
export RUN_REED_FRONTEND=1
export NORMALIZATION_MODE=train_only
export WEIGHTS_DIR="$WEIGHTS_DIR"
export OUTPUT_CACHE="$CACHE"
export RUN_FRONTEND_AFTER_SMOKE=1

echo "AIRSAS_STAGEA_TMP_ROOT=$TMP_ROOT"
echo "AIRSAS_STAGEA_BUILD_ROOT=$BUILD_ROOT"
echo "AIRSAS_STAGEA_GPU_IDENTITY=L40S"
echo "AIRSAS_STAGEA_TCNN_CUDA_ARCHITECTURES=$TCNN_CUDA_ARCHITECTURES"
echo "AIRSAS_STAGEA_ENV_PREFIX=$ENV_PREFIX"
echo "AIRSAS_STAGEA_FRONTEND=$FRONTEND"
echo "AIRSAS_STAGEA_CACHE=$CACHE"

# bash is a child process, so the bootstrap's activated environment persists through
# its exec of the frontend but does not return to this parent; the final cache validator
# intentionally uses the parent's original RIFT environment.
bash scripts/bootstrap_airsas_stagea_env.sh
python3 -B scripts/validate_airsas_5k_cache.py "$CACHE"

echo 'AIRSAS_ARMADILLO5K_FRONTEND_CACHE_PASS'
