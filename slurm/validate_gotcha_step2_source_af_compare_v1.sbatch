#!/bin/bash
# Bounded GOTCHA Step-2 raw-versus-documented-source-AF comparison only.
#
# This launcher runs the accepted compare driver on pass 1 / HH / train-sector
# 002 from the permitted converted_v3_joint8_fullpol archive.  It performs no
# model fitting, target-cube construction, localization, convention closure,
# retry, or follow-on action.  Slurm captures launcher stdout/stderr in the
# manager log root below; the driver owns creation of its fresh output leaf.
#SBATCH --job-name=mgr_rift_gotcha_step2_source_af_compare_v1_20260908
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --tmp=2G
#SBATCH --time=00:10:00
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/mgr_slurm_logs/gotcha-source-af-compare-%j.out
#SBATCH --error=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/mgr_slurm_logs/gotcha-source-af-compare-%j.err
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: source-AF comparison launcher accepts no arguments' >&2
  exit 2
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
RUN_ROOT="$PROJECT_ROOT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908"
ARCHIVE_ROOT=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v3_joint8_fullpol
PROTOCOL="$PROJECT_ROOT/protocols/gotcha_step2_source_af_hh_sector002_h0_compare_v1.json"
OUTPUT_DIR="$RUN_ROOT/step2_source_af_compare/attempt_001"

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
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

# Do not mkdir OUTPUT_DIR here: the accepted driver checks that it is fresh and
# creates the leaf with exist_ok=False after its protocol and budget checks.
python -B -u scripts/run_gotcha_step2_source_af_compare_v1.py \
  --protocol "$PROTOCOL" \
  --archive-root "$ARCHIVE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --stage compare
