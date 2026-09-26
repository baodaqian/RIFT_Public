#!/bin/bash
# B787 frozen SpINR G96/Gauss2 common reevaluation.
# Staging only; the manager supplies account, QoS, exclusions, logs and submission.
# Remote destination: /storage/project/r-jromberg3-0/dbao31/RIFT/slurm/run_spinr_b787_reevaluation_v1_20260914.sbatch

#SBATCH --job-name=mgr_b787_reeval_spinr_20260914
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
CHECKPOINT=/storage/scratch1/1/dbao31/rift_b7873200_spinr_style_smallfit_g96_gauss2_v1/b78710k_spinr_style_inr_pm_g96_gauss2_smallfit16_v1/checkpoint_final.pth.tar
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

exec python -B -u scripts/eval_spinr_b787_g96_gauss2.py \
  --checkpoint "$CHECKPOINT" \
  --npz "$NPZ" \
  --stats "$STATS" \
  --role "$ROLE" \
  --label spinr_g96_gauss2_update60 \
  --out-dir "$OUTPUT" \
  --device cuda \
  --threads 4 \
  --save-every 5 \
  "${RESUME_ARGS[@]}"
