#!/bin/bash
# One combined B787 GeRaF 16-train/4-validation repair-and-fit cell.
# The manager remains the sole submitter.  This v2 cell is engineering
# validation only, not a 3200-view result, production fit, or comparison claim.
# It preserves the v1 science recipe and uses a fresh output root after v1's
# terminal inference-cache failure.
#SBATCH --job-name=mgr_geraf_b78716_smallfit_v3_geometrydepsfix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=embers
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --tmp=12G
#SBATCH --time=02:00:00
#SBATCH --signal=TERM@300
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0
#SBATCH --export=NONE

set -euo pipefail
umask 077

terminal_signal() {
  trap - TERM INT
  echo 'ERROR: batch-shell signal arrived without validated clean driver-stop evidence; retaining terminal evidence' >&2
  exit 96
}
trap terminal_signal TERM INT

run_mode=fresh
case "$#" in
  0) ;;
  1)
    case "$1" in
      resume|postflight) run_mode="$1" ;;
      *)
        echo 'ERROR: GeRaF v2 cell accepts zero arguments, resume, or postflight' >&2
        exit 64
        ;;
    esac
    ;;
  *)
    echo 'ERROR: GeRaF v2 cell accepts zero arguments, resume, or postflight' >&2
    exit 64
    ;;
esac

[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: one non-array Slurm allocation is required' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == "embers" && "${SLURM_JOB_PARTITION:-}" == "gpu-rtx6000" ]] || {
  echo 'ERROR: requires the Embers RTX6000 allocation' >&2
  exit 94
}
[[ "${SLURM_CPUS_PER_TASK:-}" == "6" && "${SLURM_MEM_PER_NODE:-}" == "32768" ]] || {
  echo 'ERROR: requires 6 CPUs and 32 GiB RAM' >&2
  exit 94
}

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
PARENT_MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json
CELL_ROOT=/storage/scratch1/1/dbao31/rift_b7873200_geraf_subset16x4_smallfit_v3_geometrydepsfix
V3_EXECUTION_ID=geraf_b78716_smallfit_v3_geometrydepsfix
CACHE_ROOT="$CELL_ROOT/prepared_targets"
FIT_ROOT="$CELL_ROOT/fit"
initial_checkpoint="$FIT_ROOT/checkpoint_initial.pth.tar"
latest_checkpoint="$FIT_ROOT/checkpoint_latest.pth.tar"
best_checkpoint="$FIT_ROOT/checkpoint_best.pth.tar"
final_checkpoint="$FIT_ROOT/checkpoint_final.pth.tar"
report="$FIT_ROOT/smallfit_report.json"
postflight="$FIT_ROOT/smallfit_postflight.json"
run_config="$FIT_ROOT/run_config.json"
launcher_log_root="$CELL_ROOT/launcher_logs"

for source in "$PROJECT/scripts/postflight_geraf_b78716_smallfit.py"; do
  [[ -f "$source" ]] || { echo "ERROR: missing reviewed input: $source" >&2; exit 94; }
done

if [[ "$run_mode" != "postflight" ]]; then
  for source in \
    "$PROJECT/train_geraf.py" \
    "$PROJECT/train_geraf_smoke.py" \
    "$PROJECT/rift/geraf.py" \
    "$PROJECT/rift/geraf_b78716_smallfit.py" \
    "$PROJECT/rift/geraf_b7873200_protocol.py" \
    "$PROJECT/rift/geraf_b7873200_source.py" \
    "$PROJECT/rift/geraf_b7873200_acquisition.py" \
    "$PROJECT/rift/geraf_signal_operator.py" \
    "$PROJECT/rift/range_operator.py" \
    "$PROJECT/rift/coherent_radar_geometry.py" \
    "$PROJECT/rift/serialized_range_operator.py" \
    "$PROJECT/scripts/prepare_geraf_b7873200_targets.py" \
    "$PROJECT/scripts/validate_geraf_b78716_smallfit_source.py" \
    "$PROJECT/scripts/validate_geraf_b78716_smallfit_source_v2.py" \
    "$PROJECT/scripts/validate_geraf_b78716_smallfit.py" \
    "$PROJECT/scripts/validate_geraf_b78716_smallfit_signal_path.py" \
    "$PROJECT/scripts/run_geraf_b78716_smallfit_timed.py" \
    "$PROJECT/scripts/validate_range_operator_inference_cache_transition.py" \
    "$NPZ" \
    "$PARENT_MANIFEST" \
    /usr/bin/time \
    /usr/bin/env; do
    [[ -f "$source" ]] || { echo "ERROR: missing reviewed input: $source" >&2; exit 94; }
  done
fi

recovery_time_log=''
if [[ "$run_mode" == "postflight" ]]; then
  [[ -f "$initial_checkpoint" && -f "$best_checkpoint" && -f "$final_checkpoint" && -f "$report" && -f "$run_config" && ! -e "$postflight" && -d "$launcher_log_root" ]] || {
    echo 'ERROR: postflight recovery requires completed v2 report/final/config evidence, no postflight, and durable launcher logs' >&2
    exit 94
  }
  completed_driver_logs=()
  for candidate in "$launcher_log_root"/*/geraf_driver_v3_geometrydepsfix.log; do
    [[ -f "$candidate" ]] || continue
    if grep -Fxq 'GERAF_B78716_SMALLFIT_LIFECYCLE_COMPLETE' "$candidate"; then
      completed_driver_logs+=("$candidate")
    fi
  done
  (( ${#completed_driver_logs[@]} == 1 )) || {
    echo 'ERROR: postflight recovery requires exactly one completed v2 driver log' >&2
    exit 94
  }
  recovery_time_log="$(dirname "${completed_driver_logs[0]}")/geraf_time_v3_geometrydepsfix.txt"
  [[ -f "$recovery_time_log" ]] || {
    echo 'ERROR: postflight recovery requires the completed driver time record' >&2
    exit 94
  }
  resume_args=()
elif [[ "$run_mode" == "resume" ]]; then
  [[ -f "$initial_checkpoint" && -f "$best_checkpoint" && -f "$run_config" && ! -e "$report" && ! -e "$postflight" ]] || {
    echo 'ERROR: resume requires v2 initial/best/config evidence and no published v2 report/postflight' >&2
    exit 94
  }
  [[ -f "$final_checkpoint" || -f "$latest_checkpoint" ]] || {
    echo 'ERROR: resume requires either a clean v2 latest checkpoint or a final checkpoint with unpublished report' >&2
    exit 94
  }
  resume_args=(--resume "$latest_checkpoint")
else
  [[ ! -e "$CACHE_ROOT" && ! -e "$initial_checkpoint" && ! -e "$latest_checkpoint" && ! -e "$best_checkpoint" && ! -e "$final_checkpoint" && ! -e "$report" && ! -e "$postflight" && ! -e "$run_config" && ! -e "$launcher_log_root" ]] || {
    echo 'ERROR: a fresh GeRaF v2 fit requires its unused v2 output identity' >&2
    exit 94
  }
  resume_args=()
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
export SLURM_EXPORT_ENV=ALL

run_unresumable_stage() {
  stage_name=$1
  shift
  set +e
  "$@"
  stage_status=$?
  set -e
  if (( stage_status == 143 )); then
    echo "ERROR: $stage_name stopped without a validated clean fit checkpoint; retaining terminal evidence" >&2
    return 96
  fi
  return "$stage_status"
}

echo 'GERAF_B78716_SMALLFIT_V2_SIGNAL_RESET_PREFLIGHT_START'
run_unresumable_stage signal-reset-preflight /usr/bin/env --default-signal=TERM -- /usr/bin/true
echo 'GERAF_B78716_SMALLFIT_V2_SIGNAL_RESET_PREFLIGHT_PASS'

read_gpu_total_mib() {
  set +e
  gpu_total_mib="$(python - <<'PY'
import torch
print(torch.cuda.get_device_properties(0).total_memory // 2**20)
PY
)"
  gpu_query_status=$?
  set -e
  if (( gpu_query_status == 143 )); then
    echo 'ERROR: CUDA capacity query stopped without a validated clean fit checkpoint; retaining terminal evidence' >&2
    return 96
  fi
  if (( gpu_query_status != 0 )); then
    return "$gpu_query_status"
  fi
  [[ "$gpu_total_mib" =~ ^[0-9]+$ ]] || {
    echo 'ERROR: CUDA capacity query did not produce an integer MiB value' >&2
    return 95
  }
}

run_unresumable_stage cuda-preflight python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("ERROR: GeRaF v2 cell requires exactly one visible CUDA device")
print("GERAF_B78716_SMALLFIT_V2_GPU=" + torch.cuda.get_device_name(0))
print("GERAF_B78716_SMALLFIT_V2_GPU_TOTAL_MIB=" + str(torch.cuda.get_device_properties(0).total_memory // 2**20))
PY

if [[ "$run_mode" == "resume" ]]; then
  run_unresumable_stage resume-preflight python - "$latest_checkpoint" "$final_checkpoint" <<'PY'
from pathlib import Path
import sys
import torch

latest, final = (Path(value) for value in sys.argv[1:])
checkpoint = final if final.is_file() else latest
if not checkpoint.is_file():
    raise SystemExit("ERROR: no checkpoint is available for GeRaF v2 resume")
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
if not isinstance(payload, dict):
    raise SystemExit("ERROR: GeRaF v2 resume checkpoint is not a mapping")
if final.is_file():
    if payload.get("phase") != "complete" or not isinstance(payload.get("terminal_report"), dict):
        raise SystemExit("ERROR: final GeRaF v2 checkpoint cannot recover an unpublished terminal report")
    print("GERAF_B78716_SMALLFIT_V2_TERMINAL_REPORT_RECOVERY_PREFLIGHT")
else:
    if payload.get("phase") != "interrupted_clean":
        raise SystemExit("ERROR: GeRaF v2 latest checkpoint is not a clean interruption")
    print("GERAF_B78716_SMALLFIT_V2_CLEAN_RESUME_PREFLIGHT")
PY
fi

temp_parent="${SLURM_TMPDIR:-/tmp}"
[[ -d "$temp_parent" && -w "$temp_parent" && -x "$temp_parent" ]] || {
  echo "ERROR: no usable node-local temporary parent: $temp_parent" >&2
  exit 94
}
available_kib="$(df -Pk "$temp_parent" | awk 'NR == 2 {print $4}')"
[[ "$available_kib" =~ ^[0-9]+$ ]] && (( available_kib >= 12 * 1024 * 1024 )) || {
  echo "ERROR: required 12G temporary space unavailable at $temp_parent" >&2
  exit 94
}
gate_tmp="$(mktemp -d -- "$temp_parent/geraf-b78716-smallfit-v3-geometrydepsfix.${SLURM_JOB_ID}.XXXXXX")"
cleanup() {
  status=$?
  trap - EXIT
  [[ -d "${gate_tmp:-}" ]] && rm -rf -- "$gate_tmp"
  exit "$status"
}
trap cleanup EXIT
export TMPDIR="$gate_tmp" TEMP="$gate_tmp" TMP="$gate_tmp"

attempt_log_dir="$launcher_log_root/${SLURM_JOB_ID}"
[[ ! -e "$attempt_log_dir" ]] || {
  echo "ERROR: refusing to overwrite durable v2 launcher logs: $attempt_log_dir" >&2
  exit 94
}
mkdir -p -- "$attempt_log_dir"
echo "GERAF_B78716_SMALLFIT_V3_GEOMETRYDEPSFIX_EXECUTION_ID=$V3_EXECUTION_ID"
printf '%s\n' "GERAF_B78716_SMALLFIT_V3_GEOMETRYDEPSFIX_EXECUTION_ID=$V3_EXECUTION_ID" > "$attempt_log_dir/execution_context.log"

if [[ "$run_mode" == "postflight" ]]; then
  time_log="$recovery_time_log"
  time_process_rss_kib="$(awk -F ': *' '/Maximum resident set size/ {print $2; exit}' "$time_log")"
  [[ "$time_process_rss_kib" =~ ^[0-9]+$ ]] || {
    echo 'ERROR: postflight recovery could not read the completed driver RSS record' >&2
    exit 95
  }
  read_gpu_total_mib
  postflight_log="$attempt_log_dir/geraf_postflight_v3_geometrydepsfix_recovery.log"
  echo "GERAF_B78716_SMALLFIT_V2_POSTFLIGHT_RECOVERY_START source_time_log=$time_log"
  set +e
  python -B -u scripts/postflight_geraf_b78716_smallfit.py \
    --report "$report" --output "$postflight" --time-process-rss-kib "$time_process_rss_kib" \
    --host-limit-gib 32 --gpu-total-mib "$gpu_total_mib" 2>&1 | tee "$postflight_log"
  pipeline_status=("${PIPESTATUS[@]}")
  postflight_status=${pipeline_status[0]}
  tee_status=${pipeline_status[1]}
  set -e
  if (( tee_status != 0 )); then
    if (( tee_status == 143 )); then
      echo "ERROR: GeRaF v2 postflight recovery log stream stopped without a clean fit continuation; retaining terminal evidence" >&2
      exit 96
    fi
    echo "ERROR: GeRaF v2 postflight-recovery log stream failed (postflight=$postflight_status tee=$tee_status); retaining $attempt_log_dir" >&2
    exit "$tee_status"
  fi
  if (( postflight_status != 0 )); then
    if (( postflight_status == 143 )); then
      echo "ERROR: GeRaF v2 postflight recovery stopped without a clean fit continuation; retaining terminal evidence" >&2
      exit 96
    fi
    echo "ERROR: GeRaF v2 postflight recovery failed (postflight=$postflight_status); retaining literal evidence" >&2
    exit "$postflight_status"
  fi
  [[ -f "$postflight" ]] || {
    echo 'ERROR: GeRaF v2 postflight recovery did not write postflight evidence' >&2
    exit 95
  }
  echo 'GERAF_B78716_SMALLFIT_V2_POSTFLIGHT_RECOVERY_PASS'
  exit 0
fi

signal_time_log="$attempt_log_dir/geraf_signal_path_time_v3_geometrydepsfix.txt"
signal_log="$attempt_log_dir/geraf_signal_path_v3_geometrydepsfix.log"
signal_ready_file="$attempt_log_dir/geraf_signal_path_v3_geometrydepsfix.ready"
echo 'GERAF_B78716_SMALLFIT_V2_SIGNAL_PATH_START'
set +e
srun --ignore-signals=TERM --nodes=1 --ntasks=1 --cpus-per-task=6 --exact --export=ALL python -B -u scripts/run_geraf_b78716_smallfit_timed.py --time-log "$signal_time_log" --ready-file "$signal_ready_file" -- python -B -u scripts/validate_geraf_b78716_smallfit_signal_path.py 2>&1 | tee "$signal_log"
pipeline_status=("${PIPESTATUS[@]}")
signal_path_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
set -e
if (( tee_status != 0 )); then
  if (( tee_status == 143 )); then
    echo "ERROR: GeRaF v2 signal-path log stream stopped without a clean fit checkpoint; retaining terminal evidence" >&2
    exit 96
  fi
  echo "ERROR: GeRaF v2 signal-path log stream failed (fixture=$signal_path_status tee=$tee_status); retaining $attempt_log_dir" >&2
  exit "$tee_status"
fi
if (( signal_path_status != 0 )); then
  if (( signal_path_status == 143 )); then
    echo "ERROR: GeRaF v2 signal-path fixture stopped without a clean fit checkpoint; retaining terminal evidence" >&2
    exit 96
  fi
  echo "ERROR: GeRaF v2 signal-path fixture failed (fixture=$signal_path_status); retaining literal evidence" >&2
  exit "$signal_path_status"
fi
[[ -f "$signal_time_log" ]] || {
  echo 'ERROR: GeRaF v2 signal-path fixture did not retain a GNU time record' >&2
  exit 95
}
signal_rss_kib="$(awk -F ': *' '/Maximum resident set size/ {print $2; exit}' "$signal_time_log")"
[[ "$signal_rss_kib" =~ ^[0-9]+$ ]] && (( signal_rss_kib > 0 )) || {
  echo 'ERROR: GeRaF v2 signal-path fixture timing record lacks a positive RSS value' >&2
  exit 95
}
grep -Fxq 'GERAF_B78716_SMALLFIT_SIGNAL_PATH_PASS' "$signal_log" || {
  echo 'ERROR: GeRaF v2 signal-path fixture lacked its pass marker' >&2
  exit 95
}
grep -Fxq 'GERAF_TIMED_WRAPPER_DIRECT_TERM_DEFAULT' "$signal_log" || {
  echo 'ERROR: GeRaF v2 signal-path fixture did not prove default TERM at the direct task' >&2
  exit 95
}
grep -Fq 'GERAF_TIMED_WRAPPER_FORWARD_TERM child_pgid=' "$signal_log" || {
  echo 'ERROR: GeRaF v2 signal-path fixture lacked direct-wrapper TERM forwarding evidence' >&2
  exit 95
}
[[ -f "$signal_ready_file" ]] && grep -Fxq 'GERAF_TIMED_WRAPPER_CHILD_READY' "$signal_ready_file" || {
  echo 'ERROR: GeRaF v2 signal-path fixture lacked its child-ready handshake evidence' >&2
  exit 95
}
echo 'GERAF_B78716_SMALLFIT_V2_SIGNAL_PATH_PASS'

echo 'GERAF_B78716_SMALLFIT_SOURCE_PREFLIGHT_START'
run_unresumable_stage source-preflight python -B -u scripts/validate_geraf_b78716_smallfit_source.py
echo 'GERAF_B78716_SMALLFIT_SOURCE_PREFLIGHT_PASS'
echo 'GERAF_B78716_SMALLFIT_V2_SOURCE_PREFLIGHT_START'
run_unresumable_stage v2-source-preflight python -B -u scripts/validate_geraf_b78716_smallfit_source_v2.py
echo 'GERAF_B78716_SMALLFIT_V2_SOURCE_PREFLIGHT_PASS'
echo 'GERAF_B78716_SMALLFIT_TORCH_FIXTURE_START'
run_unresumable_stage torch-fixture python -B -u scripts/validate_geraf_b78716_smallfit.py
echo 'GERAF_B78716_SMALLFIT_TORCH_FIXTURE_PASS'

cache_transition_log="$attempt_log_dir/range_cache_transition_v3_geometrydepsfix.log"
echo 'GERAF_B78716_SMALLFIT_V2_CACHE_TRANSITION_START'
set +e
python -B -u scripts/validate_range_operator_inference_cache_transition.py 2>&1 | tee "$cache_transition_log"
pipeline_status=("${PIPESTATUS[@]}")
cache_transition_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
set -e
if (( tee_status != 0 )); then
  if (( tee_status == 143 )); then
    echo "ERROR: GeRaF v2 cache-transition log stream stopped without a clean fit checkpoint; retaining terminal evidence" >&2
    exit 96
  fi
  echo "ERROR: GeRaF v2 cache-transition log stream failed (fixture=$cache_transition_status tee=$tee_status); retaining $attempt_log_dir" >&2
  exit "$tee_status"
fi
if (( cache_transition_status != 0 )); then
  if (( cache_transition_status == 143 )); then
    echo "ERROR: GeRaF v2 cache-transition fixture stopped without a clean fit checkpoint; retaining terminal evidence" >&2
    exit 96
  fi
  echo "ERROR: GeRaF v2 cache-transition fixture failed (fixture=$cache_transition_status); retaining literal evidence" >&2
  exit "$cache_transition_status"
fi
grep -Fxq 'RANGE_OPERATOR_INFERENCE_CACHE_TRANSITION_PASS' "$cache_transition_log" || {
  echo 'ERROR: GeRaF v2 cache-transition fixture lacked its pass marker' >&2
  exit 95
}
echo 'GERAF_B78716_SMALLFIT_V2_CACHE_TRANSITION_PASS'

time_log="$attempt_log_dir/geraf_time_v3_geometrydepsfix.txt"
driver_log="$attempt_log_dir/geraf_driver_v3_geometrydepsfix.log"
driver_ready_file="$attempt_log_dir/geraf_driver_v3_geometrydepsfix.ready"
echo 'GERAF_B78716_SMALLFIT_V2_REAL_COMBINED_START'
set +e
  srun --ignore-signals=TERM --nodes=1 --ntasks=1 --cpus-per-task=6 --exact --export=ALL python -B -u scripts/run_geraf_b78716_smallfit_timed.py --time-log "$time_log" --ready-file "$driver_ready_file" -- python -B -u train_geraf_smoke.py \
  --cache-root "$CACHE_ROOT" --checkpoint-dir "$FIT_ROOT" \
  --npz-path "$NPZ" --role-manifest "$PARENT_MANIFEST" --device cuda \
  --host-rss-limit-gib 32 "${resume_args[@]}" 2>&1 | tee "$driver_log"
pipeline_status=("${PIPESTATUS[@]}")
driver_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
set -e
if (( tee_status != 0 )); then
  if (( tee_status == 143 )); then
    echo "ERROR: GeRaF v2 driver log stream stopped before validated clean-stop evidence; retaining terminal evidence" >&2
    exit 96
  fi
  echo "ERROR: GeRaF v2 driver log stream failed (driver=$driver_status tee=$tee_status); retaining $attempt_log_dir" >&2
  exit "$tee_status"
fi
if (( driver_status == 0 || driver_status == 143 )); then
  [[ -f "$driver_ready_file" ]] && grep -Fxq 'GERAF_TIMED_WRAPPER_CHILD_READY' "$driver_ready_file" || {
    echo 'ERROR: GeRaF v2 driver reached a completion path without child-ready handshake evidence' >&2
    exit 95
  }
  grep -Fxq 'GERAF_TIMED_WRAPPER_CHILD_READY' "$driver_log" || {
    echo 'ERROR: GeRaF v2 driver log lacked timed-wrapper child-ready evidence' >&2
    exit 95
  }
fi
if (( driver_status == 143 )); then
  [[ ! -e "$final_checkpoint" && ! -e "$report" && ! -e "$postflight" && -f "$latest_checkpoint" ]] || {
    echo 'ERROR: GeRaF v2 clean-stop exit lacked its exact resumable checkpoint state' >&2
    exit 95
  }
  run_unresumable_stage clean-stop-evidence python - "$latest_checkpoint" <<'PY'
from pathlib import Path
import sys
import torch

payload = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
if not isinstance(payload, dict) or payload.get("phase") != "interrupted_clean":
    raise SystemExit("ERROR: GeRaF v2 clean-stop latest checkpoint is not resumable")
PY
  grep -Fxq 'GERAF_B78716_SMALLFIT_CLEAN_STOP_RESUMABLE' "$driver_log" || {
    echo 'ERROR: GeRaF v2 clean-stop exit lacked the driver lifecycle marker' >&2
    exit 95
  }
  echo "GERAF_B78716_SMALLFIT_V2_CLEAN_STOP_RETAINED logs=$attempt_log_dir"
  exit 143
fi
if (( driver_status == 75 )); then
  grep -Fxq 'GERAF_B78716_SMALLFIT_STOPPED_BEFORE_FIT_NO_CHECKPOINT' "$driver_log" || {
    echo 'ERROR: GeRaF v2 pre-fit stop lacked the driver lifecycle marker' >&2
    exit 95
  }
  # There is no fit checkpoint at this point.  Preserve this terminal attempt
  # rather than treating its prepared cache as a hidden fresh/retry path.
  echo "GERAF_B78716_SMALLFIT_V2_PRE_FIT_STOP_TERMINAL logs=$attempt_log_dir"
  exit 75
fi
if (( driver_status != 0 )); then
  echo "ERROR: GeRaF v2 driver failed (driver=$driver_status tee=$tee_status); retaining literal evidence" >&2
  exit "$driver_status"
fi
[[ -f "$report" && -f "$final_checkpoint" ]] || {
  echo 'ERROR: GeRaF v2 driver did not produce report and final checkpoint' >&2
  exit 95
}
grep -Fxq 'GERAF_B78716_SMALLFIT_LIFECYCLE_COMPLETE' "$driver_log" || {
  echo 'ERROR: GeRaF v2 driver lacked its lifecycle-complete marker' >&2
  exit 95
}

time_process_rss_kib="$(awk -F ': *' '/Maximum resident set size/ {print $2; exit}' "$time_log")"
[[ "$time_process_rss_kib" =~ ^[0-9]+$ ]] || {
  echo 'ERROR: unable to read process RSS from /usr/bin/time' >&2
  exit 95
}
read_gpu_total_mib
postflight_log="$attempt_log_dir/geraf_postflight_v3_geometrydepsfix.log"
echo 'GERAF_B78716_SMALLFIT_V2_POSTFLIGHT_START'
set +e
python -B -u scripts/postflight_geraf_b78716_smallfit.py \
  --report "$report" --output "$postflight" --time-process-rss-kib "$time_process_rss_kib" \
  --host-limit-gib 32 --gpu-total-mib "$gpu_total_mib" 2>&1 | tee "$postflight_log"
pipeline_status=("${PIPESTATUS[@]}")
postflight_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
set -e
if (( tee_status != 0 )); then
  if (( tee_status == 143 )); then
    echo "ERROR: GeRaF v2 postflight log stream stopped without a clean fit continuation; retaining terminal evidence" >&2
    exit 96
  fi
  echo "ERROR: GeRaF v2 postflight log stream failed (postflight=$postflight_status tee=$tee_status); retaining $attempt_log_dir" >&2
  exit "$tee_status"
fi
if (( postflight_status != 0 )); then
  if (( postflight_status == 143 )); then
    echo "ERROR: GeRaF v2 postflight stopped without a clean fit continuation; retaining terminal evidence" >&2
    exit 96
  fi
  echo "ERROR: GeRaF v2 postflight failed (postflight=$postflight_status); retaining literal evidence" >&2
  exit "$postflight_status"
fi
[[ -f "$postflight" ]] || {
  echo 'ERROR: GeRaF v2 postflight did not write postflight evidence' >&2
  exit 95
}
echo 'GERAF_B78716_SMALLFIT_V2_LIFECYCLE_PASS'
