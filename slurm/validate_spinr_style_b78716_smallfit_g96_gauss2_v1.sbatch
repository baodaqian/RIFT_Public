#!/bin/bash
# Manager-owned opt-in h-refined G96/Gauss2-to-Gauss3 SpINR numerical-repair cell.
# One bounded Inferno allocation: source/numeric checks, renderer/VJP fixture,
# 60-update candidate fit, frozen update-0/update-60 diagnostics, and postflight.
# This launcher has a fresh identity and never resumes the completed recipes.
# The G96/Gauss2 rule is a proposed numerical repair, not a convergence claim.
#SBATCH --job-name=mgr_spinr_style_b78716_smallfit_g96_gauss2_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=inferno
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --tmp=16G
#SBATCH --time=12:00:00
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0,atl1-1-01-002-8-0,atl1-1-02-006-34-0
#SBATCH --export=NONE

set -euo pipefail
umask 077
resume_requested=false
case "$#" in
  0) ;;
  1) [[ "$1" == resume ]] || { echo 'ERROR: use only literal resume' >&2; exit 64; }; resume_requested=true ;;
  *) echo 'ERROR: zero arguments or literal resume only' >&2; exit 64 ;;
esac
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 94
[[ "${SLURM_JOB_QOS:-}" == inferno && "${SLURM_JOB_PARTITION:-}" == gpu-rtx6000 ]] || exit 94
[[ "${SLURM_CPUS_PER_TASK:-}" == 6 && "${SLURM_MEM_PER_NODE:-}" == 32768 ]] || exit 94

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
PARENT_MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json
CELL_ID=spinr_style_b78716_smallfit_g96_gauss2_v1
SMOKE_ROOT=/storage/scratch1/1/dbao31/rift_b7873200_spinr_style_smallfit_g96_gauss2_v1
RUN_ROOT="$SMOKE_ROOT/b78710k_spinr_style_inr_pm_g96_gauss2_smallfit16_v1"
latest_checkpoint="$RUN_ROOT/checkpoint_latest.pth.tar"
final_checkpoint="$RUN_ROOT/checkpoint_final.pth.tar"
report="$RUN_ROOT/smallfit_report.json"
postflight="$RUN_ROOT/smallfit_postflight.json"
launcher_log_root="$SMOKE_ROOT/launcher_logs"

for protected_path in "$SMOKE_ROOT" "$RUN_ROOT" "$latest_checkpoint" "$final_checkpoint" "$report" "$postflight" "$launcher_log_root"; do
  [[ ! -L "$protected_path" ]] || { echo "ERROR: symbolic-link identity path: $protected_path" >&2; exit 94; }
done
for source in \
  "$PROJECT/train.py" \
  "$PROJECT/train_spinr_style.py" \
  "$PROJECT/rift/spinr_smoke_runtime.py" \
  "$PROJECT/train_spinr_style_smoke.py" \
  "$PROJECT/rift/spinr_style.py" \
  "$PROJECT/rift/spinr_quadrature.py" \
  "$PROJECT/rift/spinr_style_b78716_smoke.py" \
  "$PROJECT/rift/range_operator.py" \
  "$PROJECT/rift/npz_dataset.py" \
  "$PROJECT/scripts/validate_spinr_style_g96_gauss2_source.py" \
  "$PROJECT/scripts/validate_spinr_style_g96_gauss2_runtime_preflight.py" \
  "$PROJECT/scripts/validate_spinr_style_b78716_smallfit.py" \
  "$PROJECT/scripts/postflight_spinr_style_b78716_smallfit.py" \
  "$NPZ" "$PARENT_MANIFEST" /usr/bin/time; do
  [[ -f "$source" ]] || { echo "ERROR: missing reviewed input: $source" >&2; exit 94; }
done
resume_args=()
if [[ "$resume_requested" == false ]]; then
  # A normal fresh invocation is allowed only for a completely unused identity.
  [[ ! -e "$SMOKE_ROOT" && ! -e "$RUN_ROOT" && ! -e "$latest_checkpoint" && \
     ! -e "$final_checkpoint" && ! -e "$report" && ! -e "$postflight" && \
     ! -e "$launcher_log_root" ]] || exit 94
else
  if [[ -f "$latest_checkpoint" ]]; then
    # A durable checkpoint resumes only while no terminal artifact exists.
    [[ ! -e "$final_checkpoint" && ! -e "$report" && ! -e "$postflight" ]] || exit 94
    resume_args=(--resume "$latest_checkpoint")
  else
    # A pre-checkpoint interruption may leave only launcher attempt logs.
    # It is safe to rerun the driver fresh under the same recipe identity.
    [[ ! -e "$final_checkpoint" && ! -e "$report" && ! -e "$postflight" ]] || exit 94
    [[ ! -e "$RUN_ROOT" ]] || exit 94
    if [[ -e "$SMOKE_ROOT" ]]; then
      [[ -d "$SMOKE_ROOT" ]] || exit 94
      [[ ! -e "$launcher_log_root" || -d "$launcher_log_root" ]] || exit 94
      shopt -s nullglob dotglob
      prior_entries=( "$SMOKE_ROOT"/* )
      for prior_entry in "${prior_entries[@]}"; do
        [[ "$prior_entry" == "$launcher_log_root" ]] || exit 94
      done
      shopt -u nullglob dotglob
    fi
    echo 'NO_DURABLE_CHECKPOINT_FRESH_RESTART'
    resume_args=()
  fi
fi

module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT
cd "$PROJECT"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("ERROR: exactly one CUDA device is required")
print("SPINR_STYLE_B78716_G96_GAUSS2_GPU=" + torch.cuda.get_device_name(0))
print("SPINR_STYLE_B78716_G96_GAUSS2_GPU_TOTAL_MIB=" + str(torch.cuda.get_device_properties(0).total_memory // 2**20))
PY
command -v nvidia-smi >/dev/null 2>&1 || exit 94
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

temp_parent="${SLURM_TMPDIR:-/tmp}"
[[ -d "$temp_parent" && -w "$temp_parent" && -x "$temp_parent" ]] || exit 94
temp_parent="$(cd -P -- "$temp_parent" && pwd -P)"
[[ "$temp_parent" != / ]] || exit 94
available_kib="$(df -Pk "$temp_parent" | awk 'NR == 2 {print $4}')"
[[ "$available_kib" =~ ^[0-9]+$ ]] && (( available_kib >= 16 * 1024 * 1024 )) || exit 94
gate_tmp="$(mktemp -d -- "$temp_parent/spinr-b78716-g96-gauss2.${SLURM_JOB_ID}.XXXXXX")"
gate_tmp="$(cd -P -- "$gate_tmp" && pwd -P)"
case "$gate_tmp" in "$temp_parent"/spinr-b78716-g96-gauss2."${SLURM_JOB_ID}".*) ;; *) exit 94 ;; esac
[[ "$(stat -c '%u' -- "$gate_tmp")" == "$(id -u)" ]] || exit 94
cleanup() {
  local status=$?
  trap - EXIT
  if [[ -d "$gate_tmp" && "$(dirname -- "$gate_tmp")" == "$temp_parent" && "$(stat -c '%u' -- "$gate_tmp")" == "$(id -u)" ]]; then
    case "$gate_tmp" in "$temp_parent"/spinr-b78716-g96-gauss2."${SLURM_JOB_ID}".*) rm -rf -- "$gate_tmp" ;; esac
  fi
  exit "$status"
}
trap cleanup EXIT
export TMPDIR="$gate_tmp" TEMP="$gate_tmp" TMP="$gate_tmp"

python -B -u scripts/validate_spinr_style_g96_gauss2_source.py
echo 'SPINR_STYLE_B78716_G96_GAUSS2_RUNTIME_PREFLIGHT_START'
python -B -u scripts/validate_spinr_style_g96_gauss2_runtime_preflight.py
echo 'SPINR_STYLE_B78716_G96_GAUSS2_RUNTIME_PREFLIGHT_PASS'
python -B -u scripts/validate_spinr_style_b78716_smallfit.py
attempt_log_dir="$launcher_log_root/${SLURM_JOB_ID}"
[[ ! -e "$attempt_log_dir" ]] || exit 94
mkdir -p -- "$attempt_log_dir"
time_log="$attempt_log_dir/smallfit_time_v.txt"
driver_log="$attempt_log_dir/smallfit_driver.log"
set +e
/usr/bin/time -v -o "$time_log" python -B -u train_spinr_style_smoke.py --checkpoint-root "$SMOKE_ROOT" --npz-path "$NPZ" --parent-role-manifest "$PARENT_MANIFEST" --host-rss-limit-gib 32 "${resume_args[@]}" 2>&1 | tee "$driver_log"
pipeline_status=("${PIPESTATUS[@]}")
driver_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]}"
set -e
if (( driver_status != 0 )); then exit "$driver_status"; fi
if (( tee_status != 0 )); then exit "$tee_status"; fi
[[ -f "$report" && -f "$final_checkpoint" ]] || exit 95
grep -Fxq SPINR_STYLE_B78716_SMALLFIT_QUADRATURE_LIFECYCLE_COMPLETE "$driver_log" || exit 95
time_process_rss_kib="$(awk -F ': *' '/Maximum resident set size/ {print $2; exit}' "$time_log")"
[[ "$time_process_rss_kib" =~ ^[0-9]+$ ]] || exit 95
gpu_total_mib="$(python -c 'import torch; print(torch.cuda.get_device_properties(0).total_memory // 2**20)')"
whole_job_rss_raw=''
if command -v sstat >/dev/null 2>&1; then
  whole_job_rss_raw="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS 2>/dev/null | awk 'NF {print $1; exit}' || true)"
fi
postflight_args=(--report "$report" --output "$postflight" --time-process-rss-kib "$time_process_rss_kib" --host-limit-gib 32 --gpu-total-mib "$gpu_total_mib")
if [[ -n "$whole_job_rss_raw" ]]; then postflight_args+=(--whole-job-rss-raw "$whole_job_rss_raw"); fi
set +e
python -B -u scripts/postflight_spinr_style_b78716_smallfit.py "${postflight_args[@]}"
postflight_status=$?
set -e
(( postflight_status == 0 )) || exit "$postflight_status"
echo 'SPINR_STYLE_B78716_G96_GAUSS2_LIFECYCLE_PASS'
