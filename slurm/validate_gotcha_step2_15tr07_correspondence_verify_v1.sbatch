#!/bin/bash
# One bounded GOTCHA 15TR-07 conditional correspondence verification.
# The accepted driver freezes its candidate from the completed TRAIN artifact,
# uses only validation sector 271 for the fixed 9x9 comparison, and never opens
# sealed TEST data or fits a transform, gain, phase, or either cube.
#SBATCH --job-name=mgr_rift_gotcha_step2_15tr07_correspondence_verify_v1_20260909
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --tmp=2G
#SBATCH --time=00:10:00
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/mgr_slurm_logs/gotcha-15tr07-correspondence-verify-%j.out
#SBATCH --error=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/mgr_slurm_logs/gotcha-15tr07-correspondence-verify-%j.err
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: 15TR-07 correspondence verifier accepts no arguments' >&2
  exit 2
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
RUN_ROOT="$PROJECT_ROOT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908"
ARCHIVE_ROOT=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v3_joint8_fullpol
PROTOCOL="$PROJECT_ROOT/protocols/gotcha_step2_15tr07_correspondence_verify_p1_hh_v1.json"
PRIOR_OUTPUT_DIR="$RUN_ROOT/step2_15tr07_nominal_psf/attempt_001"
OUTPUT_DIR="$RUN_ROOT/step2_15tr07_correspondence_verify/attempt_001"

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

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp}/rift-gotcha-correspondence-${SLURM_JOB_ID}"
export TMPDIR="$JOB_TMP_BASE/tmpdir"
export TEMP="$JOB_TMP_BASE/temp"
export TMP="$JOB_TMP_BASE/tmp"
mkdir -p "$TMPDIR" "$TEMP" "$TMP"

# Do not create OUTPUT_DIR here: the accepted driver requires a fresh leaf and
# creates it only after its protocol, prior-artifact, and budget checks pass.
python -B -u scripts/run_gotcha_step2_15tr07_correspondence_verify_v1.py \
  --protocol "$PROTOCOL" \
  --archive-root "$ARCHIVE_ROOT" \
  --prior-output-dir "$PRIOR_OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --stage verification
