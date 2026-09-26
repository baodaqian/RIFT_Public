#!/bin/bash
# Campaign launch 3/5: bounded measured Camry native-complex/source-AF readiness.
# The driver creates the fresh attempt_001 leaf. This launcher never retries or follows on.
#SBATCH --job-name=mgr_rift_gotcha_step3_camry_native_complex_measured_v1_20260909
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --tmp=2G
#SBATCH --time=00:10:00
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/mgr_slurm_logs/gotcha-camry-native-complex-%j.out
#SBATCH --error=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/mgr_slurm_logs/gotcha-camry-native-complex-%j.err
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: Camry measured-readiness launcher accepts no arguments' >&2
  exit 2
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
ARCHIVE_ROOT=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v3_joint8_fullpol
PROTOCOL="$PROJECT_ROOT/protocols/gotcha_step3_camry_native_complex_measured_v1.json"
OUTPUT_DIR="$PROJECT_ROOT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/step3_camry_native_complex/attempt_001"

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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp}/rift-gotcha-camry-native-complex-${SLURM_JOB_ID}"
mkdir -p "$JOB_TMP_BASE/tmpdir" "$JOB_TMP_BASE/temp" "$JOB_TMP_BASE/tmp"
export TMPDIR="$JOB_TMP_BASE/tmpdir"
export TEMP="$JOB_TMP_BASE/temp"
export TMP="$JOB_TMP_BASE/tmp"

# Do not create OUTPUT_DIR here: the reviewed driver checks freshness and creates it.
python -B -u scripts/run_gotcha_step3_camry_native_complex_measured_v1.py \
  --protocol protocols/gotcha_step3_camry_native_complex_measured_v1.json \
  --archive-root "$ARCHIVE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --stage measured_readiness
