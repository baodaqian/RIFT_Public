#!/bin/bash
# Data-free CPU engineering gate for the isolated RadarSplat B7873200 repair.
# It never opens the B787 archive, a prepared cache, a scientific checkpoint,
# or an experiment-result directory.  Its only files are job-private synthetic
# fixtures below SLURM_TMPDIR, removed on every exit.  The manager supplies
# Embers CPU resources and retains the scheduler report as the sole durable
# witness.  This is not B787 target preparation, fitting, evaluation, or a
# production baseline launch.
#
# A clean interruption is stateless and must be reviewed before a fresh
# manager-owned attempt. A genuine terminal failure remains failed. Never
# submit a duplicate and never re-run a completed passing gate.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" -eq 0 ]] || { echo 'ERROR: RadarSplat native contract gate takes no arguments' >&2; exit 64; }
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: one non-array Slurm allocation required' >&2; exit 94;
}
[[ "${SLURM_JOB_QOS:-}" == embers ]] || {
  echo 'ERROR: RadarSplat native contract gate requires Embers' >&2; exit 94;
}

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
for source in \
  "$PROJECT/rift/radarsplat_b7873200.py" \
  "$PROJECT/rift/radarsplat_b7873200_adapter.py" \
  "$PROJECT/rift/radarsplat_b7873200_acquisition.py" \
  "$PROJECT/rift/radarsplat_b7873200_protocol.py" \
  "$PROJECT/rift/power_baseline_dataset.py" \
  "$PROJECT/train_radarsplat.py" \
  "$PROJECT/scripts/validate_radarsplat_b7873200_native_contract.py"; do
  [[ -f "$source" ]] || { echo "ERROR: missing reviewed source: $source" >&2; exit 94; }
done

module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

cd "$PROJECT"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

temp_parent="${SLURM_TMPDIR:-/tmp}"
[[ -d "$temp_parent" && -w "$temp_parent" && -x "$temp_parent" ]] || {
  echo "ERROR: no usable validation temporary parent: $temp_parent" >&2; exit 94;
}
temp_parent="$(cd -P -- "$temp_parent" && pwd -P)"
[[ "$temp_parent" != / ]] || { echo 'ERROR: refusing root as validation temporary parent' >&2; exit 94; }
gate_tmp="$(mktemp -d -- "$temp_parent/radarsplat-b7873200-native-v1.${SLURM_JOB_ID}.XXXXXX")"
gate_tmp="$(cd -P -- "$gate_tmp" && pwd -P)"
case "$gate_tmp" in
  "$temp_parent"/radarsplat-b7873200-native-v1."${SLURM_JOB_ID}".*) ;;
  *) echo 'ERROR: unexpected validation temporary path' >&2; exit 94 ;;
esac
[[ "$(stat -c '%u' -- "$gate_tmp")" == "$(id -u)" ]] || {
  echo 'ERROR: validation temporary directory is not job-user owned' >&2; exit 94;
}
chmod 700 -- "$gate_tmp"
cleanup() {
  local status=$?
  trap - EXIT
  if [[ -d "$gate_tmp" && "$(dirname -- "$gate_tmp")" == "$temp_parent" \
        && "$(stat -c '%u' -- "$gate_tmp")" == "$(id -u)" ]]; then
    rm -rf -- "$gate_tmp" || echo 'WARNING: temporary validation cleanup failed' >&2
  fi
  exit "$status"
}
trap cleanup EXIT
export TMPDIR="$gate_tmp" TEMP="$gate_tmp" TMP="$gate_tmp"
log="$gate_tmp/native_contract.log"

printf 'RADARSPLAT_B7873200_NATIVE_CONTRACT_V1_JOB=%s\n' "$SLURM_JOB_ID"
printf 'RADARSPLAT_B7873200_NATIVE_CONTRACT_V1_START\n'
/usr/bin/time -v python -B -u scripts/validate_radarsplat_b7873200_native_contract.py 2>&1 | tee "$log"
grep -Eq '^RadarSplat B7873200 native contract passed: [1-9][0-9]* checks\.$' "$log" || {
  echo 'ERROR: missing complete RadarSplat native-contract marker' >&2; exit 95;
}
printf 'RADARSPLAT_B7873200_NATIVE_CONTRACT_V1_PASS\n'
