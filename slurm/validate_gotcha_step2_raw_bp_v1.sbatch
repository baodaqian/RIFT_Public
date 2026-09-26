#!/bin/bash
# GOTCHA Step-2 P1 bounded raw-BP profile.  This is one reviewed, allocated-node
# diagnostic only: no calibration, ROI selection, localization, height claim,
# inversion, fitting, training, or automatic follow-on stage.
# The manager supplies account and resource flags.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" -eq 1 ]] || {
  echo 'ERROR: P1 raw-BP profile expects exactly one fresh output-directory argument' >&2
  exit 2
}
[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo 'ERROR: P1 raw-BP profile requires a Slurm allocation' >&2
  exit 94
}
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: P1 raw-BP profile does not support array jobs' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == 'inferno' ]] || {
  echo 'ERROR: P1 raw-BP profile requires QOS inferno' >&2
  exit 94
}
[[ "${SLURM_JOB_PARTITION:-}" == 'cpu-small' ]] || {
  echo 'ERROR: P1 raw-BP profile requires partition cpu-small' >&2
  exit 94
}
[[ "${SLURM_JOB_NUM_NODES:-1}" == '1' && "${SLURM_NTASKS:-1}" == '1' ]] || {
  echo 'ERROR: P1 raw-BP profile requires one node and one task' >&2
  exit 94
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
ARCHIVE_ROOT=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v3_joint8_fullpol
OUTPUT_ROOT="$PROJECT_ROOT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/step2_p1_raw_bp"
PROTOCOL="$PROJECT_ROOT/protocols/gotcha_step2_p1_hh_sector002_h0_raw_bp_v1.json"
REQUESTED_OUTPUT_DIR=$1

[[ "$REQUESTED_OUTPUT_DIR" == /* ]] || {
  echo 'ERROR: output directory must be absolute' >&2
  exit 94
}
command -v realpath >/dev/null 2>&1 || {
  echo 'ERROR: realpath is required for output-directory containment validation' >&2
  exit 94
}
[[ ! -L "$OUTPUT_ROOT" ]] || {
  echo "ERROR: declared output root must not be a symlink: $OUTPUT_ROOT" >&2
  exit 94
}
OUTPUT_DIR=$(realpath -m -- "$REQUESTED_OUTPUT_DIR")
OUTPUT_ROOT=$(realpath -m -- "$OUTPUT_ROOT")
[[ "$OUTPUT_DIR" != "$OUTPUT_ROOT" && "$OUTPUT_DIR" == "$OUTPUT_ROOT"/* ]] || {
  echo "ERROR: output directory must be a child of $OUTPUT_ROOT" >&2
  exit 94
}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || {
  echo "ERROR: refusing a pre-existing output directory: $OUTPUT_DIR" >&2
  exit 94
}
[[ -d "$ARCHIVE_ROOT" ]] || {
  echo "ERROR: declared converted archive root is missing: $ARCHIVE_ROOT" >&2
  exit 94
}

for path in \
  "$PROJECT_ROOT/rift/gotcha_acquisition.py" \
  "$PROJECT_ROOT/rift/gotcha_step2_controls.py" \
  "$PROJECT_ROOT/protocols/gotcha_step2_p1_hh_sector002_h0_raw_bp_v1.json" \
  "$PROJECT_ROOT/scripts/run_gotcha_step2_raw_bp_v1.py" \
  "$PROJECT_ROOT/tests/test_gotcha_acquisition.py" \
  "$PROJECT_ROOT/tests/test_gotcha_step2_raw_bp_v1.py"; do
  [[ -f "$path" ]] || {
    echo "ERROR: missing deployed P1 source: $path" >&2
    exit 94
  }
done

mkdir -p -- "$OUTPUT_ROOT"
TEST_LOG="$OUTPUT_ROOT/p1-data-free-tests-${SLURM_JOB_ID}.log"

cd "$PROJECT_ROOT"
module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || {
  echo "ERROR: missing conda setup: $CONDA_BASE/etc/profile.d/conda.sh" >&2
  exit 94
}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

mkdir -p -- "$PROJECT_ROOT/batch_a_tmp_test"

python -B -m py_compile \
  rift/gotcha_acquisition.py \
  rift/gotcha_step2_controls.py \
  scripts/run_gotcha_step2_raw_bp_v1.py \
  tests/test_gotcha_acquisition.py \
  tests/test_gotcha_step2_raw_bp_v1.py
python -B -m unittest tests.test_gotcha_acquisition tests.test_gotcha_step2_raw_bp_v1 -q 2>&1 | tee "$TEST_LOG"

python -B -u scripts/run_gotcha_step2_raw_bp_v1.py \
  --protocol "$PROTOCOL" \
  --archive-root "$ARCHIVE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --stage profile 2>&1 | tee "$OUTPUT_ROOT/launcher-${SLURM_JOB_ID}.log"

echo 'GOTCHA_STEP2_P1_RAW_BP_PROFILE_PASS'
