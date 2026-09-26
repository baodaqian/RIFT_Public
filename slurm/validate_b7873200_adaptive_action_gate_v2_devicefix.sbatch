#!/bin/bash
# One bounded, engineering-only B787 adaptive-RIFT action gate.  This is not
# a production/fair-comparison/convergence run.  The manager is the sole
# submitter; the fixed identity below refuses a duplicate or overwrite.
#SBATCH --job-name=mgr_b7873200_repaired_models_v1_rift_b7873200_adaptive_action_gate_v2_devicefix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=embers
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --tmp=12G
#SBATCH --time=01:00:00
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0
#SBATCH --export=NONE

set -euo pipefail
umask 077

# The manager chooses recovery explicitly.  A no-argument invocation is a new
# identity only; the sole literal ``resume`` forwards this fixed run's latest
# checkpoint after the manager has determined that the interruption is clean.
resume_mode=false
case "$#" in
  0) ;;
  1)
    [[ "$1" == "resume" ]] || {
      echo 'ERROR: B787 action gate accepts only the literal resume argument' >&2
      exit 64
    }
    resume_mode=true
    ;;
  *)
    echo 'ERROR: B787 action gate accepts zero arguments or the literal resume argument' >&2
    exit 64
    ;;
esac
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: requires one non-array Slurm allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == "embers" ]] || {
  echo "ERROR: requires Embers, got ${SLURM_JOB_QOS:-<unset>}" >&2
  exit 94
}
[[ "${SLURM_JOB_PARTITION:-}" == "gpu-rtx6000" ]] || {
  echo "ERROR: requires gpu-rtx6000, got ${SLURM_JOB_PARTITION:-<unset>}" >&2
  exit 94
}
[[ "${SLURM_CPUS_PER_TASK:-}" == "6" ]] || {
  echo "ERROR: requires 6 CPUs, got ${SLURM_CPUS_PER_TASK:-<unset>}" >&2
  exit 94
}
[[ "${SLURM_MEM_PER_NODE:-}" == "32768" ]] || {
  echo "ERROR: requires 32G/32768MiB node memory, got ${SLURM_MEM_PER_NODE:-<unset>}" >&2
  exit 94
}

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
PARENT_MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json
ACTION_ROOT=/storage/scratch1/1/dbao31/rift_b7873200_action_gate_v2_devicefix
ACTION_RUN_ROOT="$ACTION_ROOT/b78710k_adaptive_rift_action_gate_v1"
latest_checkpoint="$ACTION_RUN_ROOT/checkpoint_latest.pth.tar"
final_checkpoint="$ACTION_RUN_ROOT/checkpoint_final.pth.tar"
# A fresh v2 identity must never be redirected to the terminal v1 evidence.
# Check the paths lexically before any existence test follows a link.
for protected_path in "$ACTION_ROOT" "$ACTION_RUN_ROOT" "$latest_checkpoint" "$final_checkpoint"; do
  [[ ! -L "$protected_path" ]] || {
    echo "ERROR: action-gate identity path must not be a symbolic link: $protected_path" >&2
    exit 94
  }
done
resume_args=()
if [[ "$resume_mode" == true ]]; then
  [[ ! -e "$final_checkpoint" ]] || {
    echo 'ERROR: a final action-gate checkpoint exists; terminal runs must not resume' >&2
    exit 94
  }
  [[ -f "$latest_checkpoint" ]] || {
    echo "ERROR: literal resume requires this fixed run's existing checkpoint_latest.pth.tar" >&2
    exit 94
  }
  resume_args=(--resume "$latest_checkpoint")
else
  [[ ! -e "$final_checkpoint" && ! -e "$latest_checkpoint" ]] || {
    echo 'ERROR: action-gate checkpoints exist; manager must explicitly choose the clean resume route' >&2
    exit 94
  }
fi
for source in \
  "$PROJECT/train.py" \
  "$PROJECT/train_adaptive_rift_smoke.py" \
  "$PROJECT/rift/b7873200_adaptive_action_smoke.py" \
  "$PROJECT/rift/sparse_scene.py" \
  "$PROJECT/rift/range_operator.py" \
  "$PROJECT/rift/npz_dataset.py" \
  "$PROJECT/scripts/validate_b7873200_adaptive_action_gate.py" \
  "$PROJECT/scripts/validate_b7873200_adaptive_action_postflight.py" \
  "$PROJECT/scripts/postflight_b7873200_adaptive_action_gate.py" \
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
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6

python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("ERROR: action gate requires exactly one allocated CUDA device")
print("RIFT_B7873200_ACTION_GATE_GPU=" + torch.cuda.get_device_name(0))
print("RIFT_B7873200_ACTION_GATE_GPU_TOTAL_MIB=" + str(torch.cuda.get_device_properties(0).total_memory // 2**20))
PY
command -v nvidia-smi >/dev/null 2>&1 || {
  echo 'ERROR: nvidia-smi is unavailable on the allocated GPU node' >&2
  exit 94
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# PACE can omit SLURM_TMPDIR under --export=NONE.  Use a verified job-unique
# child under it when available, otherwise under node-local /tmp.  Nothing
# under the project or scratch result root is ever cleaned by this launcher.
temp_parent="${SLURM_TMPDIR:-/tmp}"
[[ -d "$temp_parent" && -w "$temp_parent" && -x "$temp_parent" ]] || {
  echo "ERROR: no usable node-local temporary parent: $temp_parent" >&2
  exit 94
}
temp_parent="$(cd -P -- "$temp_parent" && pwd -P)"
[[ "$temp_parent" != "/" ]] || { echo 'ERROR: refusing root as temporary parent' >&2; exit 94; }
available_kib="$(df -Pk "$temp_parent" | awk 'NR == 2 {print $4}')"
[[ "$available_kib" =~ ^[0-9]+$ ]] && (( available_kib >= 12 * 1024 * 1024 )) || {
  echo "ERROR: required 12G temporary space unavailable at $temp_parent" >&2
  exit 94
}
gate_tmp="$(mktemp -d -- "$temp_parent/rift-b787-action-gate-v2-devicefix.${SLURM_JOB_ID}.XXXXXX")"
gate_tmp="$(cd -P -- "$gate_tmp" && pwd -P)"
case "$gate_tmp" in
  "$temp_parent"/rift-b787-action-gate-v2-devicefix."${SLURM_JOB_ID}".*) ;;
  *) echo "ERROR: unsafe temporary path $gate_tmp" >&2; exit 94 ;;
esac
[[ "$(stat -c '%u' -- "$gate_tmp")" == "$(id -u)" ]] || {
  echo "ERROR: temporary directory ownership mismatch" >&2
  exit 94
}
cleanup() {
  local status=$?
  trap - EXIT
  if [[ -d "${gate_tmp:-}" && "$(dirname -- "$gate_tmp")" == "$temp_parent" \
        && "$(stat -c '%u' -- "$gate_tmp")" == "$(id -u)" ]]; then
    case "$gate_tmp" in
      "$temp_parent"/rift-b787-action-gate-v2-devicefix."${SLURM_JOB_ID}".*) rm -rf -- "$gate_tmp" ;;
      *) echo "ERROR: refusing unexpected temporary cleanup target $gate_tmp" >&2 ;;
    esac
  fi
  exit "$status"
}
trap cleanup EXIT
export TMPDIR="$gate_tmp" TEMP="$gate_tmp" TMP="$gate_tmp"

echo 'RIFT_B7873200_ACTION_GATE_LOCAL_VALIDATION_START'
python -B -u scripts/validate_b7873200_adaptive_action_gate.py
echo 'RIFT_B7873200_ACTION_GATE_LOCAL_VALIDATION_PASS'
echo 'RIFT_B7873200_ACTION_GATE_POSTFLIGHT_BRANCH_VALIDATION_START'
python -B -u scripts/validate_b7873200_adaptive_action_postflight.py
echo 'RIFT_B7873200_ACTION_GATE_POSTFLIGHT_BRANCH_VALIDATION_PASS'

time_log="$gate_tmp/action_driver_time_v.txt"
driver_log="$gate_tmp/action_driver.log"
echo 'RIFT_B7873200_ACTION_GATE_REAL_SMOKE_START'
set +e
/usr/bin/time -v -o "$time_log" python -B -u train_adaptive_rift_smoke.py \
  --checkpoint-root "$ACTION_ROOT" --npz-path "$NPZ" --parent-role-manifest "$PARENT_MANIFEST" \
  "${resume_args[@]}" 2>&1 | tee "$driver_log"
pipeline_status=("${PIPESTATUS[@]}")
action_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
set -e

report="$ACTION_RUN_ROOT/action_gate_report.json"
if [[ ! -f "$report" ]]; then
  echo 'ERROR: action driver produced no action-gate report; retaining terminal evidence' >&2
  if (( action_status != 0 )); then
    exit "$action_status"
  fi
  if (( tee_status != 0 )); then
    exit "$tee_status"
  fi
  exit 95
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
  --report "$report"
  --output "$ACTION_RUN_ROOT/action_gate_postflight.json"
  --time-process-rss-kib "$time_process_rss_kib"
  --host-limit-gib 32
  --gpu-total-mib "$gpu_total_mib"
)
if [[ -n "$whole_job_rss_raw" ]]; then
  postflight_args+=(--whole-job-rss-raw "$whole_job_rss_raw")
fi
set +e
python -B -u scripts/postflight_b7873200_adaptive_action_gate.py "${postflight_args[@]}"
postflight_status=$?
set -e

if (( action_status != 0 )); then
  echo "ERROR: real action driver exited $action_status; action/memory reports are retained" >&2
  exit "$action_status"
fi
if (( tee_status != 0 )); then
  echo "ERROR: action-driver log capture exited $tee_status; retaining action/memory reports" >&2
  exit "$tee_status"
fi
grep -Fxq 'RIFT_B7873200_ADAPTIVE_ACTION_GATE_PASS' "$driver_log" || {
  echo 'ERROR: action driver exited cleanly without its pass marker' >&2
  exit 95
}
if (( postflight_status != 0 )); then
  echo 'ERROR: action gate passed or ran, but the separate resource acceptance did not' >&2
  exit "$postflight_status"
fi
echo 'RIFT_B7873200_ACTION_GATE_SINGLE_JOB_RELEASE_PASS'
