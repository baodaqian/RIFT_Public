#!/bin/bash
# Allocated CPU-only engineering validation for the isolated GeRaF B7873200
# repair lane.  It opens no B787 archive, response payload, cache, checkpoint,
# or experiment result.  Every validator creates only synthetic data and tiny
# resume checkpoints below one job-owned temporary directory; the manager's
# Slurm report is the sole durable artifact.
#
# The experiment manager supplies the selected CPU-pool partition, Embers QOS,
# account, four CPUs, 32G RAM, 12G temporary storage, a one-hour limit, job
# name, and the report path.  This launcher takes no arguments and is stateless
# at the scheduler level: a clean interruption should be reviewed as a fresh
# engineering attempt, not resumed as a scientific run.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: GeRaF B7873200 CPU validation takes no arguments' >&2
  exit 64
}
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: GeRaF B7873200 CPU validation requires one non-array allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == "embers" ]] || {
  echo "ERROR: GeRaF B7873200 CPU validation requires Embers, got ${SLURM_JOB_QOS:-<unset>}" >&2
  exit 94
}
PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
for path in \
  "$PROJECT_ROOT/train_geraf.py" \
  "$PROJECT_ROOT/rift/config.py" \
  "$PROJECT_ROOT/rift/forward_operator.py" \
  "$PROJECT_ROOT/rift/geraf.py" \
  "$PROJECT_ROOT/rift/geraf_b7873200_protocol.py" \
  "$PROJECT_ROOT/rift/geraf_b7873200_acquisition.py" \
  "$PROJECT_ROOT/rift/geraf_b7873200_adapter.py" \
  "$PROJECT_ROOT/rift/geraf_b7873200_source.py" \
  "$PROJECT_ROOT/rift/geraf_signal_operator.py" \
  "$PROJECT_ROOT/rift/range_operator.py" \
  "$PROJECT_ROOT/rift/power_baseline_dataset.py" \
  "$PROJECT_ROOT/scripts/prepare_geraf_b7873200_targets.py" \
  "$PROJECT_ROOT/scripts/validate_geraf_b7873200_protocol.py" \
  "$PROJECT_ROOT/scripts/validate_geraf_b7873200_adapter.py" \
  "$PROJECT_ROOT/scripts/validate_geraf_b7873200_acquisition.py" \
  "$PROJECT_ROOT/scripts/validate_geraf_signal_operator.py" \
  "$PROJECT_ROOT/scripts/validate_geraf_b7873200_trainer.py" \
  "$PROJECT_ROOT/scripts/validate_geraf_b7873200_entrypoint.py"; do
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
validation_tmp="$(mktemp -d -- "$validation_parent/rift-geraf-b7873200-cpu-v1.${SLURM_JOB_ID}.XXXXXX")"
validation_tmp="$(cd -P -- "$validation_tmp" && pwd -P)"
case "$validation_tmp" in
  "$validation_parent"/rift-geraf-b7873200-cpu-v1."${SLURM_JOB_ID}".*) ;;
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

printf 'GERAF_B7873200_CPU_CONTRACT_V1_JOB=%s\n' "$SLURM_JOB_ID"
printf 'GERAF_B7873200_CPU_CONTRACT_V1_TEMP=%s\n' "$validation_tmp"

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

run_validator protocol \
  'GeRaF B7873200 sealed protocol passed:' \
  scripts/validate_geraf_b7873200_protocol.py
run_validator adapter \
  'GeRaF B7873200 sealed adapter passed:' \
  scripts/validate_geraf_b7873200_adapter.py
run_validator acquisition \
  'GeRaF B7873200 acquisition compatibility passed:' \
  scripts/validate_geraf_b7873200_acquisition.py
run_validator signal_operator \
  'GeRaF signal-operator gates:' \
  scripts/validate_geraf_signal_operator.py
run_validator trainer_helpers \
  'GeRaF B7873200 trainer checks passed:' \
  scripts/validate_geraf_b7873200_trainer.py
run_validator actual_entrypoint \
  'GeRaF B7873200 actual-entrypoint regression passed:' \
  scripts/validate_geraf_b7873200_entrypoint.py

echo 'GERAF_B7873200_CPU_CONTRACT_V1_PASS'
