#!/bin/bash
# Campaign launch 8/8: sequential Camry then TopHat range-focused three-arm comparison.
# Both targets run in one allocation. This launcher never retries or follows on.
#SBATCH --job-name=mgr_rift_gotcha_step3_range_focused_three_arm_launch8_v1_20260910
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --tmp=4G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/mgr_slurm_logs/gotcha-range-focused-three-arm-launch8-%j.out
#SBATCH --error=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/mgr_slurm_logs/gotcha-range-focused-three-arm-launch8-%j.err
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: launch-8 range-focused comparison launcher accepts no arguments' >&2
  exit 2
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
OUTPUT_PARENT="$PROJECT_ROOT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/step3_range_focused_three_arm"

module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp}/rift-gotcha-range-focused-three-arm-launch8-${SLURM_JOB_ID}"
mkdir -p "$JOB_TMP_BASE/tmpdir" "$JOB_TMP_BASE/temp" "$JOB_TMP_BASE/tmp"
export TMPDIR="$JOB_TMP_BASE/tmpdir"
export TEMP="$JOB_TMP_BASE/temp"
export TMP="$JOB_TMP_BASE/tmp"

mkdir -p "$OUTPUT_PARENT/toyota_camry" "$OUTPUT_PARENT/tophat"

python3 scripts/run_gotcha_step3_native_complex_roi_fit_v1.py --mode actual_range_comparison --solver cgls --updates 24 --target-id toyota_camry --archive-root /storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v3_joint8_fullpol/shards --saved-native-root /storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/step3_native_complex_roi_fit_cgls24/toyota_camry/attempt_001 --output-dir /storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/step3_range_focused_three_arm/toyota_camry/attempt_001

python3 scripts/run_gotcha_step3_native_complex_roi_fit_v1.py --mode actual_range_comparison --solver cgls --updates 24 --target-id tophat --archive-root /storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v3_joint8_fullpol/shards --saved-native-root /storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/step3_native_complex_roi_fit_cgls24/tophat/attempt_001 --output-dir /storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/step3_range_focused_three_arm/tophat/attempt_001
