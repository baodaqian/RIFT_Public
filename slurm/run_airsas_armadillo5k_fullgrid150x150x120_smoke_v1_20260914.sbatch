#!/bin/bash
# AirSAS Armadillo5kHz full native 150x150x120 grid smoke staging launcher.
# Staging only for Astra review; this launcher does not submit itself.
# Proposed remote launcher:
# /storage/project/r-jromberg3-0/dbao31/RIFT/slurm/run_airsas_armadillo5k_fullgrid150x150x120_smoke_v1_20260914.sbatch
# The diagnostic runs a fresh dense full-grid fit; it is not an adaptive continuation.

#SBATCH --job-name=mgr_rift_airsas_arm5k_fullgrid150x150x120_smoke_v1_20260914
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --tmp=12G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0,atl1-1-01-002-8-0,atl1-1-02-006-34-0
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_armadillo5k_fullgrid150x150x120_smoke_v1_20260914/mgr_slurm_logs/Report-%j.out
#SBATCH --error=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_armadillo5k_fullgrid150x150x120_smoke_v1_20260914/mgr_slurm_logs/Error-%j.err

set -euo pipefail
umask 077

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
SOURCE_ROOT="$PROJECT/experiment_sources/airsas_armadillo5k_fullgrid150x150x120_smoke_v1_20260914"
CACHE="$PROJECT/datasets/sh_sas_reed/cache/armadillo_5khz_trainonly_120rings_v1_20260912"
REFERENCE_CHECKPOINT="$PROJECT/training_checkpoints/airsas_armadillo5k_adaptive_rift_sas_logpolar_smoke_v1_20260912/checkpoint_best.pt"
OUTPUT_ROOT="$PROJECT/training_checkpoints/airsas_armadillo5k_fullgrid150x150x120_smoke_v1_20260914"
DIAGNOSTIC="$SOURCE_ROOT/scripts/diagnose_sas_fullgrid.py"
TRAIN_SCRIPT="$SOURCE_ROOT/train_sas.py"

require_nonempty() {
  [[ -s "$1" ]] || {
    echo "ERROR: required deployed file is missing or empty: $1" >&2
    exit 2
  }
}

require_directory() {
  [[ -d "$1" ]] || {
    echo "ERROR: required cache directory is missing: $1" >&2
    exit 2
  }
}

[[ ! -e "$OUTPUT_ROOT" ]] || {
  echo "ERROR: full-grid output root already exists: $OUTPUT_ROOT" >&2
  exit 2
}
require_nonempty "$DIAGNOSTIC"
require_nonempty "$TRAIN_SCRIPT"
require_nonempty "$REFERENCE_CHECKPOINT"
require_directory "$CACHE"

module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT

cd "$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

exec python -B "$DIAGNOSTIC" \
  --cache "$CACHE" \
  --reference-checkpoint "$REFERENCE_CHECKPOINT" \
  --output "$OUTPUT_ROOT" \
  --device cuda
