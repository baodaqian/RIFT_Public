#!/bin/bash
# One stateless CPU/Embers allocation; no training or checkpoint continuation.
# Manager supplies 8 CPUs, 32G RAM, 16G tmp, 90 minutes, and no GPU.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
set -euo pipefail
umask 077
[[ "$#" -eq 0 ]] || { echo 'ERROR: this diagnostic accepts no arguments' >&2; exit 64; }
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 94
[[ "${SLURM_JOB_QOS:-}" == embers ]] || exit 94
PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json
[[ -f "$NPZ" && -f "$MANIFEST" && -x /usr/bin/time ]] || exit 94
attempt_out="$PROJECT/experiment_state/rf_b7873200_padded_roi_v1/attempt-${SLURM_JOB_ID}"
[[ ! -e "$attempt_out" && ! -L "$attempt_out" ]] || { echo 'ERROR: attempt output already exists' >&2; exit 94; }
mkdir -p -- "$(dirname -- "$attempt_out")"
mkdir -- "$attempt_out"

module purge
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT
cd "$PROJECT"
export PYTHONPATH=. PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
temp_parent="${SLURM_TMPDIR:-/tmp}"
[[ -d "$temp_parent" && -w "$temp_parent" && -x "$temp_parent" ]] || exit 94
temp_parent="$(cd -P -- "$temp_parent" && pwd -P)"
[[ "$temp_parent" != / ]] || exit 94
gate_tmp="$(mktemp -d -- "$temp_parent/rf-roi-v1.${SLURM_JOB_ID}.XXXXXX")"
gate_tmp="$(cd -P -- "$gate_tmp" && pwd -P)"
case "$gate_tmp" in "$temp_parent"/rf-roi-v1."${SLURM_JOB_ID}".*) ;; *) exit 94 ;; esac
[[ "$(stat -c '%u' -- "$gate_tmp")" == "$(id -u)" ]] || exit 94
cleanup() {
    local status=$?
    trap - EXIT
    if [[ -d "$gate_tmp" && "$(dirname -- "$gate_tmp")" == "$temp_parent" \
          && "$(stat -c '%u' -- "$gate_tmp")" == "$(id -u)" ]]; then
        rm -rf -- "$gate_tmp" || echo 'WARNING: diagnostic temp cleanup failed' >&2
    fi
    exit "$status"
}
trap cleanup EXIT
export TMPDIR="$gate_tmp" TEMP="$gate_tmp" TMP="$gate_tmp"
common=(
    --npz-path "$NPZ" --sealed-protocol --sealed-split-manifest "$MANIFEST"
    --num-train 3200 --num-val 1000 --num-test 1000 --seed 42 --val-from-tail
    --checkpoint-root "$gate_tmp" --checkpoint-name diagnostic_only
    --device cpu --extent 0.15 --granularity 48 --steps 1 --view-batch 1
    --train-pairs 0 --val-pairs 0 --stats-max-views 0 --lr 0.001
    --weight-fft 0.60 --weight-occ 0.36 --weight-bimodal 0.03
    --occupancy-threshold 0.05 --dynamic-range-db 60 --range-margin 0.05
    --range-law released --intensity-offset 0.05 --intensity-scaler 1
    --hidden-dim 64 --feature-dim 32 --sh-degree 3 --sigmoid-tightness 1
    --hash-levels 16 --hash-features 2 --hash-base-resolution 16
    --hash-final-resolution 512 --hash-log2-size 19 --query-chunk 32768 --pair-chunk 8
    --diagnose-padded-roi --diagnostic-role-index 0 --diagnostic-pairs 0
)
echo 'RF_B7873200_PADDED_ROI_START: random model, zero optimizer updates, support-ROI measurement only'
/usr/bin/time -v python -B -u train_radar_fields.py "${common[@]}" \
    --diagnostic-role train --diagnostic-json "$attempt_out/train_roi.json"
/usr/bin/time -v python -B -u train_radar_fields.py "${common[@]}" \
    --diagnostic-role val --diagnostic-json "$attempt_out/validation_roi.json"
python -B scripts/validate_rf_roi_artifacts.py --manifest "$MANIFEST" \
    --train "$attempt_out/train_roi.json" --val "$attempt_out/validation_roi.json"
if find "$gate_tmp" "$attempt_out" -name '*.pth*' -print -quit | grep -q .; then
    echo 'ERROR: unexpected checkpoint in diagnostic output' >&2
    exit 95
fi
echo 'RF_B7873200_PADDED_ROI_V1_PASS'
