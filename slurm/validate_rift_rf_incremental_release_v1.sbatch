#!/bin/bash
# Two independently released engineering gates; not production training.
# The manager supplies CPU/Embers resources. All fixture files are temporary.
# RF additionally reads one canonical B787 training response and official source.
# Durable evidence is the manager log; no scientific checkpoint is created.
# Clean interrupted attempts may restart these stateless gates from the beginning.
# Genuine terminal failures are not retries, and completed gates are not repeated.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
set -euo pipefail
umask 077

[[ "$#" -eq 1 ]] || { echo 'ERROR: expected one released gate name' >&2; exit 64; }
gate="$1"
case "$gate" in
  rift_operator_acd|rf_b787_upstream_contract) ;;
  *) echo "ERROR: unknown gate $gate" >&2; exit 64 ;;
esac
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: one non-array Slurm allocation required' >&2; exit 94;
}
[[ "${SLURM_JOB_QOS:-}" == "embers" ]] || {
  echo 'ERROR: this release requires Embers' >&2; exit 94;
}
PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
if [[ "$gate" == rf_b787_upstream_contract ]]; then
  [[ -f "$NPZ" && -f "$PROJECT/external/RadarFields_reference/radarfields/radar.py" ]] || {
    echo 'ERROR: missing B787 archive or official reference source' >&2; exit 94;
  }
fi

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
gate_tmp="$(mktemp -d -- "$temp_parent/rift-incremental-v1.${SLURM_JOB_ID}.XXXXXX")"
gate_tmp="$(cd -P -- "$gate_tmp" && pwd -P)"
case "$gate_tmp" in "$temp_parent"/rift-incremental-v1."${SLURM_JOB_ID}".*) ;; *) exit 94 ;; esac
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
echo "INCREMENTAL_GATE_START=$gate"

case "$gate" in
  rift_operator_acd)
    python -B -u scripts/validate_range_operator.py --stages A,C,D --device cpu --seed 42 2>&1 | tee "$log"
    for marker in 'Stage A: PASS' 'Stage C: PASS' 'Stage D: PASS'; do
      grep -Fxq "$marker" "$log" || { echo "ERROR: missing $marker" >&2; exit 95; }
    done
    ;;
  rf_b787_upstream_contract)
    python -B -u scripts/validate_radar_fields.py \
      --upstream external/RadarFields_reference --npz-path "$NPZ" \
      --real-num-train 3200 --real-num-val 1000 --real-num-test 1000 \
      --real-seed 42 --real-val-from-tail 2>&1 | tee "$log"
    for marker in \
      'PASS: approximate transform matches the pinned official Python function bit-for-bit' \
      'PASS: non-approximate transform matches the pinned official Python function bit-for-bit' \
      'PASS: real-data gate rejects the selected test source response before payload access' \
      'PASS: the authorized real train view converts to finite nonnegative range power' \
      'ALL RADAR FIELDS GATES PASSED ('; do
      grep -Fq "$marker" "$log" || { echo "ERROR: missing $marker" >&2; exit 95; }
    done
    ;;
esac
echo "RIFT_RF_INCREMENTAL_RELEASE_V1_PASS=$gate"
