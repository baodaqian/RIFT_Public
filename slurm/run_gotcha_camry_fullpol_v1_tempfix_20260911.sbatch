#!/bin/bash
# Camry-only full-polarization Inferno fit: one fail-closed runtime gate, then one fit.
# This launcher is staged locally for manager-owned deployment; it does not submit itself.
#SBATCH --job-name=mgr_rift_gotcha_camry_fullpol_v1_tempfix_20260911
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=16G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0,atl1-1-01-002-8-0,atl1-1-02-006-34-0
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/rift_publicradar_gotcha_camry_fullpol_v1_20260911/mgr_slurm_logs/%j.out
#SBATCH --error=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/rift_publicradar_gotcha_camry_fullpol_v1_20260911/mgr_slurm_logs/%j.err

set -euo pipefail
umask 077

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: Camry full-polarization launcher accepts no arguments' >&2
  exit 2
}

[[ "${SLURM_JOB_QOS:-}" == 'inferno' ]] || {
  echo "ERROR: expected SLURM_JOB_QOS=inferno, got ${SLURM_JOB_QOS:-unset}" >&2
  exit 2
}
[[ "${SLURM_JOB_PARTITION:-}" == 'gpu-h200' ]] || {
  echo "ERROR: expected SLURM_JOB_PARTITION=gpu-h200, got ${SLURM_JOB_PARTITION:-unset}" >&2
  exit 2
}
[[ "${SLURM_JOB_NUM_NODES:-}" == '1' && "${SLURM_NTASKS:-}" == '1' ]] || {
  echo 'ERROR: expected exactly one non-array node/task' >&2
  exit 2
}
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || {
  echo 'ERROR: missing numeric Slurm job ID' >&2
  exit 2
}
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: array jobs are not allowed' >&2
  exit 2
}

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
ARCHIVE=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v3_joint8_fullpol
PROTOCOL=$PROJECT/protocols/gotcha_camry_fullpol_v1.json
OUT=$PROJECT/experiment_state/rift_publicradar_gotcha_camry_fullpol_v1_20260911/attempt_002_tempfix_20260911

[[ ! -e "$OUT" ]] || {
  echo "ERROR: output directory already exists: $OUT" >&2
  exit 2
}

required_files=(
  'rift/gotcha_source_af.py'
  'rift/gotcha_camry_fullpol_core.py'
  'scripts/run_gotcha_camry_fullpol_v1.py'
  'scripts/validate_gotcha_camry_fullpol_v1.py'
  'scripts/validate_gotcha_camry_fullpol_runtime.py'
  'protocols/gotcha_camry_fullpol_v1.json'
  'tests/test_gotcha_camry_fullpol_core.py'
  'tests/test_gotcha_camry_fullpol_runner.py'
  'tests/test_gotcha_source_af.py'
)
for relative_path in "${required_files[@]}"; do
  deployed_file="$PROJECT/$relative_path"
  [[ -s "$deployed_file" ]] || {
    echo "ERROR: required deployed file is missing or empty: $deployed_file" >&2
    exit 2
  }
done

module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

cd "$PROJECT"
export PYTHONPATH="$PROJECT"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export BLIS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

TMP_ROOT="${SLURM_TMPDIR:-/tmp}/gotcha_camry_fullpol_v1_20260911-${SLURM_JOB_ID}"
mkdir -p "$TMP_ROOT/tmpdir" "$TMP_ROOT/temp" "$TMP_ROOT/tmp"
export TMPDIR="$TMP_ROOT/tmpdir"
export TEMP="$TMP_ROOT/temp"
export TMP="$TMP_ROOT/tmp"

echo 'CAMRY_FULLPOL_RUNTIME_GATE_BEGIN'
python3 -B scripts/validate_gotcha_camry_fullpol_runtime.py

echo 'CAMRY_FULLPOL_FIT_BEGIN'
python3 -u -B -m scripts.run_gotcha_camry_fullpol_v1 --archive-root "$ARCHIVE" --protocol "$PROTOCOL" --output "$OUT"
