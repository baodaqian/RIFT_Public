#!/bin/bash
# Data-free CPU engineering gate, not B787 fitting or production approval.
# Manager supplies CPU/Embers resources; all synthetic fixtures are temporary.
# Stateless clean interruptions restart from the beginning under a new job ID.
# Preserve genuine terminal failures and never repeat a completed gate.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
set -euo pipefail
umask 077

[[ "$#" -eq 0 ]] || { echo 'ERROR: this gate takes no arguments' >&2; exit 64; }
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: one non-array Slurm allocation required' >&2; exit 94;
}
[[ "${SLURM_JOB_QOS:-}" == embers ]] || {
  echo 'ERROR: this gate requires Embers' >&2; exit 94;
}
PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
[[ -f "$PROJECT/scripts/validate_spinr_style.py" && -x /usr/bin/time ]] || {
  echo 'ERROR: missing validator or timing executable' >&2; exit 94;
}

module purge
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT
cd "$PROJECT"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

temp_parent="${SLURM_TMPDIR:-/tmp}"
[[ -d "$temp_parent" && -w "$temp_parent" && -x "$temp_parent" ]] || exit 94
temp_parent="$(cd -P -- "$temp_parent" && pwd -P)"
[[ "$temp_parent" != / ]] || exit 94
gate_tmp="$(mktemp -d -- "$temp_parent/spinr-contract-v1.${SLURM_JOB_ID}.XXXXXX")"
gate_tmp="$(cd -P -- "$gate_tmp" && pwd -P)"
case "$gate_tmp" in "$temp_parent"/spinr-contract-v1."${SLURM_JOB_ID}".*) ;; *) exit 94 ;; esac
[[ "$(stat -c '%u' -- "$gate_tmp")" == "$(id -u)" ]] || exit 94
cleanup() {
  local status=$?
  trap - EXIT
  if [[ -d "$gate_tmp" && "$(dirname -- "$gate_tmp")" == "$temp_parent" \
        && "$(stat -c '%u' -- "$gate_tmp")" == "$(id -u)" ]]; then
    rm -rf -- "$gate_tmp" || echo 'WARNING: temporary fixture cleanup failed' >&2
  fi
  exit "$status"
}
trap cleanup EXIT
export TMPDIR="$gate_tmp" TEMP="$gate_tmp" TMP="$gate_tmp"
log="$gate_tmp/gate.log"

echo 'SPINR_STYLE_CONTRACT_V1_START'
/usr/bin/time -v python -B -u scripts/validate_spinr_style.py --device cpu 2>&1 | tee "$log"
grep -Eq '^SpINR-style validation passed: [1-9][0-9]* checks\.$' "$log" || {
  echo 'ERROR: missing complete SpINR-style validation marker' >&2; exit 95;
}
echo 'SPINR_STYLE_CONTRACT_V1_PASS'
