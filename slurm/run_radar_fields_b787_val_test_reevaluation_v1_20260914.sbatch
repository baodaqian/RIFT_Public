#!/bin/bash
# B787 Radar Fields validation plus TEST native readout.
# Staging only; the manager supplies account, QoS, exclusions, logs and submission.
# Remote destination: /storage/project/r-jromberg3-0/dbao31/RIFT/slurm/run_radar_fields_b787_val_test_reevaluation_v1_20260914.sbatch

#SBATCH --job-name=mgr_b787_reeval_radar_fields_val_test_20260914
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

[[ "$#" -eq 1 ]] || {
  echo 'ERROR: expected OUTPUT_JSON' >&2
  exit 64
}
OUTPUT_JSON=$1

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
CHECKPOINT=/storage/scratch1/1/dbao31/rift_b7873200_radar_fields_production_v1/attempt-12941397/rf_b7873200_production_v1/checkpoint_final.pth.tar
NPZ=/storage/project/r-jromberg3-0/dbao31/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json

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

exec python -B -u scripts/readout_radar_fields_b7873200_native.py \
  --config protocols/radar_fields_b7873200_production_v1.json \
  --checkpoint "$CHECKPOINT" \
  --npz-path "$NPZ" \
  --sealed-split-manifest "$MANIFEST" \
  --output "$OUTPUT_JSON" \
  --device cuda \
  --roles val test
