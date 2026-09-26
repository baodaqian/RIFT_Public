#!/bin/bash
# One-model B787 selected-checkpoint postflight evaluation. Submit once per approved model.
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --tmp=4G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/b7873200_full_model_matrix_v1_20260906/mgr_slurm_logs/b787-plenoxel-eval-%x-%j.out
#SBATCH --error=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/b7873200_full_model_matrix_v1_20260906/mgr_slurm_logs/b787-plenoxel-eval-%x-%j.err
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" -eq 1 ]] || {
  echo 'usage: run_b787_selected_rift_plenoxel_eval_v1.sbatch original|adaptive' >&2
  exit 2
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
PYTHON=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/RIFT/bin/python
RIFT_SITE=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/RIFT/lib/python3.10/site-packages
EXPORT_SITE=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/sensor_fusion/lib/python3.10/site-packages
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
STL="$PROJECT_ROOT/data/B787.stl"
OUTPUT_PARENT="$PROJECT_ROOT/experiment_state/b7873200_full_model_matrix_v1_20260906/postflight_selected_rift_plenoxel_t0p20_v1"

case "$1" in
  original)
    LABEL=original_rift_epoch40
    CHECKPOINT=/storage/scratch1/1/dbao31/rift_b7873200_full_model_matrix_v1_20260906/original_rift_r7_target20k/b787_full3200_original_best_r7_target20k/checkpoint_best.pth.tar
    ;;
  adaptive)
    LABEL=adaptive_rift_epoch111
    CHECKPOINT=/storage/scratch1/1/dbao31/rift_b7873200_adaptive_fullscale_v1/b78710k_adaptive_rift_fullscale_v1/checkpoint_best.pth.tar
    ;;
  *)
    echo "unknown selected model: $1" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="$OUTPUT_PARENT/$LABEL/attempt_002"
mkdir -p "$OUTPUT_PARENT/$LABEL"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$RIFT_SITE:$EXPORT_SITE"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp}/b787-selected-rift-plenoxel-eval-${SLURM_JOB_ID}-${LABEL}"
mkdir -p "$JOB_TMP_BASE/tmpdir" "$JOB_TMP_BASE/temp" "$JOB_TMP_BASE/tmp"
export TMPDIR="$JOB_TMP_BASE/tmpdir"
export TEMP="$JOB_TMP_BASE/temp"
export TMP="$JOB_TMP_BASE/tmp"

"$PYTHON" scripts/run_b787_selected_rift_plenoxel_eval_v1.py \
  --label "$LABEL" \
  --checkpoint "$CHECKPOINT" \
  --npz-path "$NPZ" \
  --stl "$STL" \
  --output-dir "$OUTPUT_DIR" \
  --threshold 0.20 \
  --extent 0.15 \
  --crop 0.075 \
  --base-grid 48 \
  --upsample 4 \
  --gt-grid 240 \
  --n-surface 20000 \
  --n-volume 50000 \
  --tau 0.00625 \
  --iou-unit 0.005 \
  --seed 0
