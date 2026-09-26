#!/bin/bash
# B787 RadarSplat accepted-budget native validation/TEST evaluation.
# Staging only; the manager supplies account, QoS, exclusions, logs and submission.
# Remote destination: /storage/project/r-jromberg3-0/dbao31/RIFT/slurm/run_radarsplat_b787_native_reevaluation_v1_20260914.sbatch

#SBATCH --job-name=mgr_b787_reeval_radarsplat_20260914
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --tmp=4G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" -eq 2 || "$#" -eq 3 ]] || {
  echo 'ERROR: expected ROLE OUTPUT [resume]' >&2
  exit 64
}
ROLE=$1
OUTPUT=$2
[[ "$ROLE" == validation || "$ROLE" == test ]] || {
  echo 'ERROR: ROLE must be validation or test' >&2
  exit 64
}
RESUME_ARGS=()
if [[ "$#" -eq 3 ]]; then
  [[ "$3" == resume ]] || {
    echo 'ERROR: optional third argument must be literal resume' >&2
    exit 64
  }
  RESUME_ARGS=(--resume)
fi

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
CHECKPOINT=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/radarsplat_b7873200_fullscale_150epochs_v1/native_power_fit_480000steps_2048gaussians/checkpoint_best.pt
CACHE_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/radarsplat_b7873200_fullscale_150epochs_v1/native_power_cache_3200x1000
NPZ=/storage/project/r-jromberg3-0/dbao31/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
ROLE_MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json

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
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

exec python -B -u scripts/eval_radarsplat_b787_budget_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --cache-root "$CACHE_ROOT" \
  --npz-path "$NPZ" \
  --role-manifest "$ROLE_MANIFEST" \
  --out-dir "$OUTPUT" \
  --role "$ROLE" \
  --device cuda \
  --threads 4 \
  --save-every 5 \
  "${RESUME_ARGS[@]}"
