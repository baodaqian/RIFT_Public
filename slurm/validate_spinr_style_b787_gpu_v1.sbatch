#!/bin/bash
# First real-data gate on the repaired-models B7873200 track, not production.
# One implementation owner; no hashes, integrity pins or coordination locks.
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
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json
[[ -f "$PROJECT/scripts/validate_spinr_style.py" && -f "$NPZ" && -f "$MANIFEST" && -x /usr/bin/time ]] || {
  echo 'ERROR: missing validator, canonical inputs or timing executable' >&2; exit 94;
}
# This launcher and its fixed memory threshold are for the reviewed 32G recipe.
[[ "${SLURM_MEM_PER_NODE:-}" == 32768 ]] || {
  echo 'ERROR: this gate requires --mem=32G (32768 MiB)' >&2; exit 94;
}

module purge
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT
cd "$PROJECT"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

temp_parent="${SLURM_TMPDIR:-/tmp}"
[[ -d "$temp_parent" && -w "$temp_parent" && -x "$temp_parent" ]] || exit 94
temp_parent="$(cd -P -- "$temp_parent" && pwd -P)"
[[ "$temp_parent" != / ]] || exit 94
gate_tmp="$(mktemp -d -- "$temp_parent/spinr-b787-gpu-v1.${SLURM_JOB_ID}.XXXXXX")"
gate_tmp="$(cd -P -- "$gate_tmp" && pwd -P)"
case "$gate_tmp" in "$temp_parent"/spinr-b787-gpu-v1."${SLURM_JOB_ID}".*) ;; *) exit 94 ;; esac
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

echo 'SPINR_STYLE_B787_GPU_V1_START'
/usr/bin/time -v python -B -u scripts/validate_spinr_style.py \
  --allocated-pace --device cuda --pace-npz "$NPZ" --role-manifest "$MANIFEST" \
  --host-rss-limit-gib 32 2>&1 | tee "$log"
for marker in \
  'Stage H: allocated-node real B787 authorized-only smoke' \
  'PASS: real B787 adapter retains sealed test/unused access denial' \
  'PASS: real B787 all-bin/all-pair B=4 update is finite' \
  'PASS: real B787 initial scale uses exactly first 32 canonical train IDs' \
  'full-update memory:'; do
  grep -Fq "$marker" "$log" || { echo "ERROR: missing $marker" >&2; exit 95; }
done
grep -Eq '^SpINR-style validation passed: [1-9][0-9]* checks\.$' "$log" || {
  echo 'ERROR: missing complete validator marker' >&2; exit 95;
}
echo 'SPINR_STYLE_B787_GPU_V1_PASS'
