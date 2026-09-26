#!/bin/bash
# B787 frozen Original/Adaptive/SE common range-power reevaluation.
# Staging only; the manager supplies account, QoS, exclusions, logs and submission.
# Remote destination: /storage/project/r-jromberg3-0/dbao31/RIFT/slurm/run_b787_range_reevaluation_v1_20260914.sbatch

#SBATCH --job-name=mgr_b787_reeval_range_20260914
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

[[ "$#" -eq 4 || "$#" -eq 5 ]] || {
  echo 'ERROR: expected CHECKPOINT LABEL ROLE OUTPUT [resume]' >&2
  exit 64
}
CHECKPOINT=$1
LABEL=$2
ROLE=$3
OUTPUT=$4
[[ "$ROLE" == validation || "$ROLE" == test ]] || {
  echo 'ERROR: ROLE must be validation or test' >&2
  exit 64
}
RESUME_ARGS=()
if [[ "$#" -eq 5 ]]; then
  [[ "$5" == resume ]] || {
    echo 'ERROR: optional fifth argument must be literal resume' >&2
    exit 64
  }
  RESUME_ARGS=(--resume)
fi

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
NPZ=/storage/project/r-jromberg3-0/dbao31/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
STATS=/storage/scratch1/1/dbao31/rift_b7873200_radar_fields_production_v1/attempt-12941397/rf_b7873200_production_v1/radar_fields_power_stats.json

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

exec python -B -u scripts/eval_b787_range_power.py \
  --checkpoint "$CHECKPOINT" \
  --label "$LABEL" \
  --role "$ROLE" \
  --npz-path "$NPZ" \
  --stats "$STATS" \
  --out-dir "$OUTPUT" \
  --num-train 3200 \
  --num-val 1000 \
  --seed 42 \
  --device cuda \
  --threads 4 \
  --pair-chunk 64 \
  --point-chunk 65536 \
  --save-every 5 \
  --rf-supported-metrics \
  "${RESUME_ARGS[@]}"
