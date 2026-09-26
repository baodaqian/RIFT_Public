#!/bin/bash
# Allocated CPU-only engineering gate for the isolated corrected B7873200
# Sugavanam--Ertin lane. It creates only synthetic temporary artifacts: it
# does not open the B787 archive, responses, a baseline checkpoint, or a real
# output root. The experiment manager supplies the CPU partition, Embers QOS,
# account, resources, time limit, job name, and Slurm report destination.
#
# This is deliberately a fresh, stateless validation attempt. A clean
# interruption is evidence to review, not a scientific training resume.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: B7873200 SE CPU validation takes no arguments' >&2
  exit 64
}
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: B7873200 SE CPU validation requires one non-array allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == "embers" ]] || {
  echo "ERROR: B7873200 SE CPU validation requires Embers, got ${SLURM_JOB_QOS:-<unset>}" >&2
  exit 94
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
for path in \
  "$PROJECT_ROOT/train.py" \
  "$PROJECT_ROOT/train_sugavanam_ertin_stage1.py" \
  "$PROJECT_ROOT/train_sugavanam_ertin_stage2.py" \
  "$PROJECT_ROOT/rift/calibration.py" \
  "$PROJECT_ROOT/rift/npz_dataset.py" \
  "$PROJECT_ROOT/rift/range_operator.py" \
  "$PROJECT_ROOT/rift/sparse_scene.py" \
  "$PROJECT_ROOT/rift/sugavanam_ertin.py" \
  "$PROJECT_ROOT/rift/sugavanam_ertin_a320_stabilized.py" \
  "$PROJECT_ROOT/rift/sugavanam_ertin_b7873200_stage1.py" \
  "$PROJECT_ROOT/rift/sugavanam_ertin_b7873200_stage2_v1.py" \
  "$PROJECT_ROOT/rift/sugavanam_ertin_stage2_runtime_v1.py" \
  "$PROJECT_ROOT/rift/sugavanam_ertin_validzero.py" \
  "$PROJECT_ROOT/scripts/validate_sugavanam_ertin_b7873200_stage1.py" \
  "$PROJECT_ROOT/scripts/validate_sugavanam_ertin_b7873200_stage1_operator.py" \
  "$PROJECT_ROOT/scripts/validate_sugavanam_ertin_b7873200_stage2_static.py" \
  "$PROJECT_ROOT/scripts/validate_sugavanam_ertin_b7873200_stage2_v1.py"; do
  [[ -f "$path" ]] || { echo "ERROR: missing reviewed source: $path" >&2; exit 94; }
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
  echo "ERROR: no usable validation temporary parent: $validation_parent" >&2
  exit 94
}
validation_parent="$(cd -P -- "$validation_parent" && pwd -P)"
[[ "$validation_parent" != "/" ]] || {
  echo 'ERROR: refusing root as the validation temporary parent' >&2
  exit 94
}
validation_tmp="$(mktemp -d -- "$validation_parent/rift-se-b7873200-cpu-v1.${SLURM_JOB_ID}.XXXXXX")"
validation_tmp="$(cd -P -- "$validation_tmp" && pwd -P)"
case "$validation_tmp" in
  "$validation_parent"/rift-se-b7873200-cpu-v1."${SLURM_JOB_ID}".*) ;;
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
export RIFT_VALIDATION_TMPDIR="$validation_tmp"
export TMPDIR="$validation_tmp"
export TEMP="$validation_tmp"
export TMP="$validation_tmp"

printf 'SE_B7873200_CPU_CONTRACT_V1_JOB=%s\n' "$SLURM_JOB_ID"
printf 'SE_B7873200_CPU_CONTRACT_V1_TEMP=%s\n' "$validation_tmp"

run_validator() {
  local label="$1"
  local marker="$2"
  local script="$3"
  local log="$validation_tmp/$label.log"
  printf 'VALIDATOR_START=%s\n' "$label"
  python -B -u "$script" 2>&1 | tee "$log"
  grep -Fq "$marker" "$log" || {
    echo "ERROR: validator $label omitted terminal marker: $marker" >&2
    exit 95
  }
  printf 'VALIDATOR_PASS=%s\n' "$label"
}

run_validator stage1_static \
  'SE_B7873200_STAGE1_STATIC_PASS:' \
  scripts/validate_sugavanam_ertin_b7873200_stage1.py
run_validator stage1_operator \
  'SE_B7873200_STAGE1_OPERATOR_PASS:' \
  scripts/validate_sugavanam_ertin_b7873200_stage1_operator.py
run_validator stage2_static \
  'SE_B7873200_STAGE2_STATIC_PASS:' \
  scripts/validate_sugavanam_ertin_b7873200_stage2_static.py
run_validator stage2_lifecycle \
  'SE_B7873200_STAGE2_CPU_CONTRACT_PASS:' \
  scripts/validate_sugavanam_ertin_b7873200_stage2_v1.py

echo 'SE_B7873200_ALLOCATED_CPU_CONTRACT_V1_PASS'
