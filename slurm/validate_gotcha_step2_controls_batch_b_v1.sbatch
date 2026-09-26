#!/bin/bash
# Data-free GOTCHA Step-2 controls gate only.  It runs py_compile, the six
# focused synthetic unit tests, and the Step-2 synthetic validator.  It must
# not open a GOTCHA archive, payload, measured response, or test data.
# The manager supplies the account and resource flags for this allocation.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" -eq 1 ]] || {
  echo 'ERROR: Step-2 Batch-B expected exactly one fresh output-directory argument' >&2
  exit 2
}
[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo 'ERROR: Step-2 Batch-B requires a Slurm allocation' >&2
  exit 94
}
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: Step-2 Batch-B does not support array jobs' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == 'inferno' ]] || {
  echo 'ERROR: Step-2 Batch-B requires QOS inferno' >&2
  exit 94
}
[[ "${SLURM_JOB_PARTITION:-}" == 'cpu-small' ]] || {
  echo 'ERROR: Step-2 Batch-B requires partition cpu-small' >&2
  exit 94
}
[[ "${SLURM_JOB_NUM_NODES:-1}" == '1' && "${SLURM_NTASKS:-1}" == '1' ]] || {
  echo 'ERROR: Step-2 Batch-B requires one node and one task' >&2
  exit 94
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
OUTPUT_ROOT="$PROJECT_ROOT/experiment_state/public_radar_tuning/gotcha_validation_steps12_20260908/step2_controls_batch_b"
REQUESTED_OUTPUT_DIR=$1

[[ "$REQUESTED_OUTPUT_DIR" == /* ]] || {
  echo 'ERROR: output directory must be an absolute path under the declared PACE root' >&2
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

for path in \
  "$PROJECT_ROOT/rift/gotcha_step2_controls.py" \
  "$PROJECT_ROOT/tests/test_gotcha_step2_controls.py" \
  "$PROJECT_ROOT/scripts/validate_gotcha_step2_controls.py" \
  "$PROJECT_ROOT/rift/gotcha_acquisition.py"; do
  [[ -f "$path" ]] || {
    echo "ERROR: missing Step-2 source/dependency: $path" >&2
    exit 94
  }
done

mkdir -p -- "$OUTPUT_ROOT"
mkdir -- "$OUTPUT_DIR"

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
export TMPDIR="$OUTPUT_DIR"
export TEMP="$OUTPUT_DIR"
export TMP="$OUTPUT_DIR"

python -B -m py_compile \
  rift/gotcha_step2_controls.py \
  tests/test_gotcha_step2_controls.py \
  scripts/validate_gotcha_step2_controls.py
python -B -m unittest discover -s tests -p 'test_gotcha_step2_controls.py' -q \
  | tee "$OUTPUT_DIR/unittest.log"
python -B -u scripts/validate_gotcha_step2_controls.py \
  | tee "$OUTPUT_DIR/validator.json"

echo 'GOTCHA_STEP2_CONTROLS_BATCH_B_PASS'
