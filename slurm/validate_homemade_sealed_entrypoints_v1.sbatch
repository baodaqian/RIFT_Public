#!/bin/bash
# Allocated, synthetic-only regression gate for the reviewed sealed NPZ
# entrypoints.  It runs the real ordinary-RIFT CLI continuation validator and
# the Radar Fields validator in the normal RIFT environment.
# No scientific archive, B787 payload, production checkpoint, run root, or
# experiment result is opened or written.  The validators create only tiny
# synthetic fixtures beneath one job-owned temporary directory, which this
# launcher removes on exit; the manager-owned Slurm log is the durable record.
#
# This is deliberately a fresh sealed-entrypoint-v1 launcher.  It does not
# alter or reinterpret the completed source-validation v4 Inferno smoke or
# terminal source-validation v1--v3 launcher evidence.
# The manager supplies all scheduler resources and this no-argument launcher
# is intended for a CPU-pool Embers allocation.  The Radar Fields CUDA RNG
# recovery branch was already checked by the immutable v4 GPU smoke; the
# sealed-protocol changes below are CPU-contract gates.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: sealed-entrypoint validation takes no arguments' >&2
  exit 64
}
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: validation requires one non-array Slurm allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == "embers" ]] || {
  echo "ERROR: validation is limited to free Embers, got ${SLURM_JOB_QOS:-<unset>}" >&2
  exit 94
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
for path in \
  "$PROJECT_ROOT/train.py" \
  "$PROJECT_ROOT/train_radar_fields.py" \
  "$PROJECT_ROOT/rift/npz_dataset.py" \
  "$PROJECT_ROOT/rift/radar_fields_dataset.py" \
  "$PROJECT_ROOT/scripts/validate_train_npz_sealed_protocol.py" \
  "$PROJECT_ROOT/scripts/validate_radar_fields.py"; do
  [[ -f "$path" ]] || {
    echo "ERROR: missing reviewed source: $path" >&2
    exit 94
  }
done

module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

validation_parent="${SLURM_TMPDIR:-/tmp}"
[[ -d "$validation_parent" && -w "$validation_parent" && -x "$validation_parent" ]] || {
  echo "ERROR: no usable temporary parent: $validation_parent" >&2
  exit 94
}
validation_parent="$(cd -P -- "$validation_parent" && pwd -P)"
[[ "$validation_parent" != "/" ]] || {
  echo 'ERROR: refusing root as the validation temporary parent' >&2
  exit 94
}
validation_tmp="$(mktemp -d -- "$validation_parent/rift-sealed-entrypoints-v1.${SLURM_JOB_ID}.XXXXXX")"
validation_tmp="$(cd -P -- "$validation_tmp" && pwd -P)"
case "$validation_tmp" in
  "$validation_parent"/rift-sealed-entrypoints-v1."${SLURM_JOB_ID}".*) ;;
  *)
    echo "ERROR: unexpected validation temporary path: $validation_tmp" >&2
    exit 94
    ;;
esac
[[ "$(stat -c '%u' -- "$validation_tmp")" == "$(id -u)" ]] || {
  echo "ERROR: validation temporary child is not job-user owned: $validation_tmp" >&2
  exit 94
}
chmod 700 -- "$validation_tmp"
cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "${validation_tmp:-}" && -d "$validation_tmp" \
        && "$(dirname -- "$validation_tmp")" == "$validation_parent" \
        && "$(stat -c '%u' -- "$validation_tmp")" == "$(id -u)" ]]; then
    rm -rf -- "$validation_tmp" || echo "ERROR: failed to clean $validation_tmp" >&2
  fi
  exit "$status"
}
trap cleanup EXIT
export TMPDIR="$validation_tmp"
export TEMP="$validation_tmp"
export TMP="$validation_tmp"
printf 'VALIDATION_TEMP_PARENT=%s\n' "$validation_parent"

run_validator() {
  local label="$1"
  local marker="$2"
  local script="$3"
  local log="$validation_tmp/$label.log"
  printf 'VALIDATOR_START=%s\n' "$label"
  python -B -u "$script" 2>&1 | tee "$log"
  if grep -Fq "$marker" "$log"; then
    printf 'VALIDATOR_PASS=%s\n' "$label"
  else
    echo "ERROR: validator $label omitted its terminal marker" >&2
    exit 95
  fi
}

run_validator ordinary_rift_sealed_entrypoint \
  'train.py sealed-NPZ entrypoint/continuation checks' \
  scripts/validate_train_npz_sealed_protocol.py
run_validator radar_fields_sealed_entrypoint \
  'ALL RADAR FIELDS GATES PASSED (' \
  scripts/validate_radar_fields.py

radar_fields_log="$validation_tmp/radar_fields_sealed_entrypoint.log"
grep -Fq 'SKIP CUDA RNG restore/topology gate (CUDA unavailable)' "$radar_fields_log" || {
  echo 'ERROR: CPU validator did not record its expected CUDA-unavailable branch' >&2
  exit 95
}

echo 'RIFT_HOMEMADE_SEALED_ENTRYPOINT_VALIDATION_V1_PASS'
