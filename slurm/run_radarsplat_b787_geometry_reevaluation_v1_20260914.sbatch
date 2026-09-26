#!/bin/bash
# B787 RadarSplat accepted-budget fixed-geometry evaluation.
# Staging only; the manager supplies account, QoS, exclusions, logs and submission.
# Remote destination: /storage/project/r-jromberg3-0/dbao31/RIFT/slurm/run_radarsplat_b787_geometry_reevaluation_v1_20260914.sbatch

#SBATCH --job-name=mgr_b787_reeval_radarsplat_geometry_20260914
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --tmp=4G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" -eq 1 ]] || {
  echo 'ERROR: expected OUTPUT_DIR' >&2
  exit 64
}
OUTPUT_DIR=$1

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
CHECKPOINT=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/radarsplat_b7873200_fullscale_150epochs_v1/native_power_fit_480000steps_2048gaussians/checkpoint_best.pt
NPZ=/storage/project/r-jromberg3-0/dbao31/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
STL=/storage/project/r-jromberg3-0/dbao31/RIFT/data/B787.stl

module purge
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT
cd "$PROJECT"
export PYTHONPATH="$PROJECT"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

exec python -B -u scripts/eval_radarsplat_b787_fixed_geometry.py \
  --checkpoint "$CHECKPOINT" \
  --npz-path "$NPZ" \
  --stl "$STL" \
  --output-dir "$OUTPUT_DIR"
