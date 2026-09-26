#!/bin/bash
# Evaluation-only six-method B787 postflight.  The manager supplies one
# allocated CPU job and the nine existing artifact paths; this wrapper does
# not train, resume, mutate checkpoints, or retry.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --tmp=4G
#SBATCH --time=02:00:00
#SBATCH --export=NONE

set -euo pipefail

[[ "$#" -eq 9 ]] || {
  echo "usage: $0 OUTPUT_DIR NPZ STL ORIGINAL ADAPTIVE SPINR GERAF RADAR_FIELDS SUGAVANAM_ERTIN" >&2
  exit 64
}
[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo 'ERROR: six-method B787 postflight requires a Slurm allocation' >&2
  exit 94
}

OUTPUT_DIR=$1
NPZ_PATH=$2
STL_PATH=$3
ORIGINAL_CHECKPOINT=$4
ADAPTIVE_CHECKPOINT=$5
SPINR_CHECKPOINT=$6
GERAF_CHECKPOINT=$7
RADAR_FIELDS_CHECKPOINT=$8
SUGAVANAM_ERTIN_CHECKPOINT=$9
PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT

for required in \
  "$PROJECT_ROOT/scripts/run_b7873200_six_method_plenoxel_postflight_v1.py" \
  "$NPZ_PATH" "$STL_PATH" "$ORIGINAL_CHECKPOINT" "$ADAPTIVE_CHECKPOINT" \
  "$SPINR_CHECKPOINT" "$GERAF_CHECKPOINT" "$RADAR_FIELDS_CHECKPOINT" \
  "$SUGAVANAM_ERTIN_CHECKPOINT"; do
  [[ -s "$required" ]] || { echo "ERROR: missing or empty input: $required" >&2; exit 94; }
done

[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "ERROR: refusing to overwrite existing output: $OUTPUT_DIR" >&2
  exit 94
}

module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="$PROJECT_ROOT"

cd "$PROJECT_ROOT"
python -B -u scripts/run_b7873200_six_method_plenoxel_postflight_v1.py \
  --output-dir "$OUTPUT_DIR" \
  --npz-path "$NPZ_PATH" \
  --stl "$STL_PATH" \
  --original-checkpoint "$ORIGINAL_CHECKPOINT" \
  --adaptive-checkpoint "$ADAPTIVE_CHECKPOINT" \
  --spinr-checkpoint "$SPINR_CHECKPOINT" \
  --geraf-checkpoint "$GERAF_CHECKPOINT" \
  --radar-fields-checkpoint "$RADAR_FIELDS_CHECKPOINT" \
  --sugavanam-ertin-checkpoint "$SUGAVANAM_ERTIN_CHECKPOINT" \
  --device cpu
