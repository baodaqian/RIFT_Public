#!/bin/bash
# One manager-owned Adaptive RIFT full-scale candidate lane. This launcher is
# separate from the historical wrapper so driver failures propagate through
# the tee pipeline and terminal evidence is retained.
#SBATCH --job-name=mgr_b7873200_adaptive_rift_fullscale_v1_20260906
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=12G
# Inferno provides the requested 12-hour allocation; scientific and checkpoint
# semantics remain unchanged from the reviewed adaptive candidate.
#SBATCH --time=12:00:00
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0
#SBATCH --export=NONE

set -euo pipefail
umask 077

resume_mode=false
report_only_mode=false
case "$#" in
  0) ;;
  1)
    if [[ "$1" == "resume" ]]; then
      resume_mode=true
    elif [[ "$1" == "report" ]]; then
      report_only_mode=true
    else
      echo 'ERROR: adaptive full-scale accepts only resume or report as an argument' >&2
      exit 64
    fi
    ;;
  *)
    echo 'ERROR: adaptive full-scale accepts zero arguments or the literal resume argument' >&2
    exit 64
    ;;
esac

[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: adaptive full-scale requires one non-array Slurm allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == "inferno" ]] || {
  echo "ERROR: adaptive full-scale requires Inferno, got ${SLURM_JOB_QOS:-<unset>}" >&2
  exit 94
}
[[ "${SLURM_JOB_ACCOUNT:-}" == "gts-jromberg3-ece" ]] || {
  echo "ERROR: adaptive full-scale requires account gts-jromberg3-ece, got ${SLURM_JOB_ACCOUNT:-<unset>}" >&2
  exit 94
}
[[ "${SLURM_JOB_PARTITION:-}" == "gpu-h200" ]] || {
  echo "ERROR: adaptive full-scale requires gpu-h200, got ${SLURM_JOB_PARTITION:-<unset>}" >&2
  exit 94
}
[[ "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || {
  echo "ERROR: adaptive full-scale requires 8 CPUs, got ${SLURM_CPUS_PER_TASK:-<unset>}" >&2
  exit 94
}
[[ "${SLURM_MEM_PER_NODE:-}" == "32768" ]] || {
  echo "ERROR: adaptive full-scale requires 32G/32768MiB node memory, got ${SLURM_MEM_PER_NODE:-<unset>}" >&2
  exit 94
}

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
PARENT_MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json
ADAPTIVE_ROOT=/storage/scratch1/1/dbao31/rift_b7873200_adaptive_fullscale_v1
ADAPTIVE_RUN_ROOT="$ADAPTIVE_ROOT/b78710k_adaptive_rift_fullscale_v1"
LATEST_CHECKPOINT="$ADAPTIVE_RUN_ROOT/checkpoint_latest.pth.tar"
FINAL_CHECKPOINT="$ADAPTIVE_RUN_ROOT/checkpoint_final.pth.tar"
REPORT="$ADAPTIVE_RUN_ROOT/adaptive_fullscale_report.json"
POSTFLIGHT="$ADAPTIVE_RUN_ROOT/adaptive_fullscale_postflight.json"
LAUNCHER_LOG_DIR="$ADAPTIVE_ROOT/launcher_logs/$SLURM_JOB_ID"

# Check identity paths lexically before any existence test follows a link.
for protected_path in \
  "$ADAPTIVE_ROOT" "$ADAPTIVE_RUN_ROOT" "$LATEST_CHECKPOINT" \
  "$FINAL_CHECKPOINT" "$REPORT" "$POSTFLIGHT" "$LAUNCHER_LOG_DIR"; do
  [[ ! -L "$protected_path" ]] || {
    echo "ERROR: adaptive full-scale identity path must not be a symbolic link: $protected_path" >&2
    exit 94
  }
done

driver_args=()
measurement_scope=training_attempt
if [[ "$report_only_mode" == true ]]; then
  [[ -f "$FINAL_CHECKPOINT" ]] || {
    echo 'ERROR: report recovery requires this run\x27s checkpoint_final.pth.tar' >&2
    exit 94
  }
  [[ ! -e "$POSTFLIGHT" ]] || {
    echo 'ERROR: report recovery refuses to overwrite existing postflight evidence' >&2
    exit 94
  }
  driver_args=(--report-only)
  measurement_scope=report_recovery
elif [[ "$resume_mode" == true ]]; then
  [[ ! -e "$FINAL_CHECKPOINT" && ! -e "$REPORT" && ! -e "$POSTFLIGHT" ]] || {
    echo 'ERROR: terminal adaptive full-scale evidence exists; do not resume it' >&2
    exit 94
  }
  [[ -f "$LATEST_CHECKPOINT" ]] || {
    echo "ERROR: literal resume requires this run's checkpoint_latest.pth.tar" >&2
    exit 94
  }
  driver_args=(--resume "$LATEST_CHECKPOINT")
else
  [[ ! -e "$LATEST_CHECKPOINT" && ! -e "$FINAL_CHECKPOINT" && ! -e "$REPORT" && ! -e "$POSTFLIGHT" ]] || {
    echo 'ERROR: adaptive full-scale evidence exists; manager must choose the clean resume route' >&2
    exit 94
  }
fi

for source in \
  "$PROJECT/train.py" \
  "$PROJECT/rift/adaptive_training_workflow.py" \
  "$PROJECT/rift/b7873200_adaptive_fullscale.py" \
  "$PROJECT/rift/sparse_scene.py" \
  "$PROJECT/rift/range_operator.py" \
  "$PROJECT/rift/npz_dataset.py" \
  "$PROJECT/scripts/validate_b7873200_adaptive_fullscale_source.py" \
  "$PROJECT/scripts/postflight_b7873200_adaptive_fullscale.py" \
  "$NPZ" \
  "$PARENT_MANIFEST" \
  /usr/bin/time; do
  [[ -f "$source" ]] || { echo "ERROR: missing reviewed input: $source" >&2; exit 94; }
done

module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT
cd "$PROJECT"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

command -v nvidia-smi >/dev/null 2>&1 || {
  echo 'ERROR: nvidia-smi is unavailable on the allocated GPU node' >&2
  exit 94
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("ERROR: adaptive full-scale requires exactly one allocated CUDA device")
print("RIFT_B7873200_ADAPTIVE_FULLSCALE_GPU=" + torch.cuda.get_device_name(0))
print("RIFT_B7873200_ADAPTIVE_FULLSCALE_GPU_TOTAL_MIB=" + str(torch.cuda.get_device_properties(0).total_memory // 2**20))
PY

mkdir -p "$LAUNCHER_LOG_DIR"
time_log="$LAUNCHER_LOG_DIR/driver_time_v.txt"
driver_log="$LAUNCHER_LOG_DIR/driver.log"
echo 'RIFT_B7873200_ADAPTIVE_FULLSCALE_SOURCE_VALIDATION_START'
python -B -u scripts/validate_b7873200_adaptive_fullscale_source.py
echo 'RIFT_B7873200_ADAPTIVE_FULLSCALE_SOURCE_VALIDATION_PASS'

if [[ "$report_only_mode" == true ]]; then
  echo 'RIFT_B7873200_ADAPTIVE_FULLSCALE_REPORT_RECOVERY_START'
else
  echo 'RIFT_B7873200_ADAPTIVE_FULLSCALE_TRAINING_START'
fi
set +e
/usr/bin/time -v -o "$time_log" python -B -u train.py --workflow adaptive-fullscale \
  --checkpoint-root "$ADAPTIVE_ROOT" \
  --npz-path "$NPZ" \
  --parent-role-manifest "$PARENT_MANIFEST" \
  "${driver_args[@]}" 2>&1 | tee -a "$driver_log"
pipeline_status=("${PIPESTATUS[@]}")
driver_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
set -e

if (( driver_status != 0 )); then
  echo "ERROR: adaptive full-scale driver exited $driver_status; evidence is retained" >&2
  exit "$driver_status"
fi
if (( tee_status != 0 )); then
  echo "ERROR: adaptive full-scale log capture exited $tee_status" >&2
  exit "$tee_status"
fi
[[ -f "$FINAL_CHECKPOINT" && -f "$REPORT" ]] || {
  echo 'ERROR: adaptive full-scale driver exited cleanly without terminal artifacts' >&2
  exit 95
}
if [[ "$report_only_mode" == true ]]; then
  grep -Fxq 'RIFT_B7873200_ADAPTIVE_FULLSCALE_REPORT_ONLY_PASS' "$driver_log" || {
    echo 'ERROR: report-only driver exited without its recovery pass marker' >&2
    exit 95
  }
else
  grep -Fxq 'RIFT_B7873200_ADAPTIVE_FULLSCALE_OBSERVER_PASS' "$driver_log" || {
    echo 'ERROR: adaptive full-scale driver exited without its observer pass marker' >&2
    exit 95
  }
fi

time_process_rss_kib="$(awk -F ': *' '/Maximum resident set size/ {print $2; exit}' "$time_log")"
[[ "$time_process_rss_kib" =~ ^[0-9]+$ ]] || {
  echo 'ERROR: unable to read /usr/bin/time process RSS' >&2
  exit 95
}
gpu_total_mib="$(python - <<'PY'
import torch
print(torch.cuda.get_device_properties(0).total_memory // 2**20)
PY
)"
[[ "$gpu_total_mib" =~ ^[0-9]+$ && "$gpu_total_mib" -gt 0 ]] || {
  echo 'ERROR: unable to read allocated GPU total memory' >&2
  exit 95
}
whole_job_rss_raw=''
if command -v sstat >/dev/null 2>&1; then
  whole_job_rss_raw="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS 2>/dev/null | awk 'NF {print $1; exit}' || true)"
fi
postflight_args=(
  --report "$REPORT"
  --output "$POSTFLIGHT"
  --time-process-rss-kib "$time_process_rss_kib"
  --host-limit-gib 32
  --gpu-total-mib "$gpu_total_mib"
  --measurement-scope "$measurement_scope"
)
if [[ -n "$whole_job_rss_raw" ]]; then
  postflight_args+=(--whole-job-rss-raw "$whole_job_rss_raw")
fi
python -B -u scripts/postflight_b7873200_adaptive_fullscale.py "${postflight_args[@]}"
echo 'RIFT_B7873200_ADAPTIVE_FULLSCALE_SINGLE_JOB_TECHNICAL_PASS'
