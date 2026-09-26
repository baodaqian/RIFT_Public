#!/bin/bash
# AirSAS Armadillo5kHz Adaptive RIFT-SAS refinement replay diagnostic.
# Staging only for Astra review; this launcher does not submit itself.
# Proposed remote launcher:
# /storage/project/r-jromberg3-0/dbao31/RIFT/slurm/run_airsas_armadillo5k_adaptive_refinement_replay_v1_20260913.sbatch
# The diagnostic restores the saved step-700 recipe, RNG, and optimizer state.
# It owns the fixed validation-ping audit and replay output contract.
#SBATCH --job-name=mgr_rift_airsas_arm5k_adaptive_refinement_replay_v1_20260913
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --tmp=12G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0,atl1-1-01-002-8-0,atl1-1-02-006-34-0
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_armadillo5k_adaptive_refinement_replay_v1_20260913/mgr_slurm_logs/Report-%j.out
#SBATCH --error=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_armadillo5k_adaptive_refinement_replay_v1_20260913/mgr_slurm_logs/Error-%j.err

set -euo pipefail
umask 077

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
CACHE="$PROJECT/datasets/sh_sas_reed/cache/armadillo_5khz_trainonly_120rings_v1_20260912"
CHECKPOINT="$PROJECT/training_checkpoints/airsas_armadillo5k_adaptive_rift_sas_logpolar_smoke_v1_20260912/checkpoint_best.pt"
OUTPUT_ROOT="$PROJECT/training_checkpoints/airsas_armadillo5k_adaptive_refinement_replay_v1_20260913"
DIAGNOSTIC="$PROJECT/scripts/diagnose_sas_refinement.py"
TRAIN_SCRIPT="$PROJECT/train_sas.py"

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
  echo "ERROR: refinement replay output root already exists: $OUTPUT_ROOT" >&2
  exit 2
}
require_nonempty "$DIAGNOSTIC"
require_nonempty "$TRAIN_SCRIPT"
require_nonempty "$CHECKPOINT"
require_directory "$CACHE"

module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT

cd "$PROJECT"
export PYTHONPATH="$PROJECT"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

exec python -B "$DIAGNOSTIC" \
  --cache "$CACHE" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT_ROOT" \
  --device cuda \
  --start-step 700 \
  --end-step 1010 \
  --expected-fixed-val-rel-mse 1.369373 \
  --fixed-val-tolerance 1.0e-4
