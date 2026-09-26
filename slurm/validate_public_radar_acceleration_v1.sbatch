#!/bin/bash
# Synthetic CPU release gate for PublicRadar cache and aligned view batching.
# It does not read datasets or enter training/checkpoint code.
#SBATCH --job-name=rift_pr_accel_gate_v1
#SBATCH --account=gts-jromberg3
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --tmp=4G
#SBATCH --time=00:10:00
#SBATCH --export=NONE
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/mgr_slurm_logs/Report-%j.out

set -euo pipefail

[[ $# -eq 0 ]] || {
  echo 'ERROR: PublicRadar acceleration validation takes no arguments' >&2
  exit 94
}
[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo 'ERROR: PublicRadar acceleration validation requires a Slurm allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == embers && "${SLURM_JOB_PARTITION:-}" == cpu-small ]] || {
  echo 'ERROR: PublicRadar acceleration validation requires Embers/cpu-small' >&2
  exit 94
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
SOURCE_ROOT=/storage/scratch1/1/dbao31/rift_public_radar_impl_20260815
VALIDATION_ROOT="$PROJECT_ROOT/experiment_state/public_radar_tuning/acceleration_validation_v1"

for path in \
  "$PROJECT_ROOT/scripts/validate_public_radar_acceleration_v1.py" \
  "$PROJECT_ROOT/scripts/public_radar_interpolation_dense_v2.py" \
  "$PROJECT_ROOT/scripts/public_radar_gotcha_full_domain_v1_partial.py" \
  "$PROJECT_ROOT/scripts/public_radar_camry_tuning.py" \
  "$PROJECT_ROOT/rift/public_radar_tuning.py" \
  "$PROJECT_ROOT/rift/fixed_planar_scene.py" \
  "$PROJECT_ROOT/rift/serialized_range_operator.py" \
  "$PROJECT_ROOT/rift/public_radar_sh_cache.py" \
  "$PROJECT_ROOT/rift/gotcha_domain.py" \
  "$SOURCE_ROOT/rift/coherent_radar_geometry.py" \
  "$SOURCE_ROOT/scripts/activate_exact_numpy.sh"; do
  [[ -f "$path" ]] || {
    echo "ERROR: missing acceleration validation source: $path" >&2
    exit 94
  }
done

mkdir -p "$VALIDATION_ROOT/results" "$VALIDATION_ROOT/logs"
RESULT_PATH="$VALIDATION_ROOT/results/result-${SLURM_JOB_ID}.json"
LOG_PATH="$VALIDATION_ROOT/logs/Report-${SLURM_JOB_ID}.out"
[[ ! -e "$RESULT_PATH" && ! -e "$LOG_PATH" ]] || {
  echo 'ERROR: refusing to overwrite an existing acceleration validation artifact' >&2
  exit 94
}

cd "$SOURCE_ROOT"
module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
source "$SOURCE_ROOT/scripts/activate_exact_numpy.sh"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT"
export RIFT_PUBLIC_SOURCE_ROOT="$SOURCE_ROOT"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

set +e
python -B -u "$PROJECT_ROOT/scripts/validate_public_radar_acceleration_v1.py" \
  --result-path "$RESULT_PATH" >"$LOG_PATH" 2>&1
validation_status=$?
set -e
cat "$LOG_PATH"
exit "$validation_status"
