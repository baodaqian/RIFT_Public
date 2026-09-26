#!/bin/bash
# One combined B787 GeRaF 16-train/4-validation preparation-and-fit cell.
# The experiment manager is the sole submitter.  It is bounded engineering
# evidence only, not a 3200-view result, production fit, or comparison claim.
#SBATCH --job-name=mgr_geraf_b78716_smallfit_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=embers
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --tmp=12G
#SBATCH --time=02:00:00
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0
#SBATCH --export=NONE

set -euo pipefail
umask 077

resume_mode=false
case "$#" in
  0) ;;
  1)
    [[ "$1" == "resume" ]] || {
      echo 'ERROR: bounded GeRaF cell accepts only the literal resume argument' >&2
      exit 64
    }
    resume_mode=true
    ;;
  *)
    echo 'ERROR: bounded GeRaF cell accepts zero arguments or the literal resume argument' >&2
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
CELL_ROOT=/storage/scratch1/1/dbao31/rift_b7873200_geraf_subset16x4_smallfit_v1
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

for source in \
  "$PROJECT/train_geraf.py" \
  "$PROJECT/train_geraf_smoke.py" \
  "$PROJECT/rift/geraf.py" \
  "$PROJECT/rift/geraf_b78716_smallfit.py" \
  "$PROJECT/rift/geraf_b7873200_protocol.py" \
  "$PROJECT/rift/geraf_b7873200_source.py" \
  "$PROJECT/rift/geraf_b7873200_acquisition.py" \
  "$PROJECT/rift/geraf_signal_operator.py" \
  "$PROJECT/scripts/prepare_geraf_b7873200_targets.py" \
  "$PROJECT/scripts/validate_geraf_b78716_smallfit_source.py" \
  "$PROJECT/scripts/validate_geraf_b78716_smallfit.py" \
  "$PROJECT/scripts/postflight_geraf_b78716_smallfit.py" \
  "$NPZ" \
  "$PARENT_MANIFEST" \
  /usr/bin/time; do
  [[ -f "$source" ]] || { echo "ERROR: missing reviewed input: $source" >&2; exit 94; }
done

if [[ "$resume_mode" == true ]]; then
  [[ -f "$initial_checkpoint" && -f "$best_checkpoint" && -f "$run_config" && ! -e "$report" && ! -e "$postflight" ]] || {
    echo 'ERROR: resume requires initial/best/config evidence and no published terminal report/postflight' >&2
    exit 94
  }
  [[ -f "$final_checkpoint" || -f "$latest_checkpoint" ]] || {
    echo 'ERROR: resume requires either a clean latest checkpoint or a final checkpoint with unpublished report' >&2
    exit 94
  }
  resume_args=(--resume "$latest_checkpoint")
else
  [[ ! -e "$initial_checkpoint" && ! -e "$latest_checkpoint" && ! -e "$best_checkpoint" && ! -e "$final_checkpoint" && ! -e "$report" && ! -e "$postflight" && ! -e "$run_config" && ! -e "$launcher_log_root" ]] || {
    echo 'ERROR: a fresh bounded fit requires a new output identity without prior fit evidence' >&2
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

python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("ERROR: bounded GeRaF cell requires exactly one visible CUDA device")
print("GERAF_B78716_SMALLFIT_GPU=" + torch.cuda.get_device_name(0))
print("GERAF_B78716_SMALLFIT_GPU_TOTAL_MIB=" + str(torch.cuda.get_device_properties(0).total_memory // 2**20))
PY

if [[ "$resume_mode" == true ]]; then
  python - "$latest_checkpoint" "$final_checkpoint" <<'PY'
from pathlib import Path
import sys
import torch

latest, final = (Path(value) for value in sys.argv[1:])
checkpoint = final if final.is_file() else latest
if not checkpoint.is_file():
    raise SystemExit("ERROR: no checkpoint is available for bounded GeRaF resume")
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
if not isinstance(payload, dict):
    raise SystemExit("ERROR: bounded GeRaF resume checkpoint is not a mapping")
if final.is_file():
    if payload.get("phase") != "complete" or not isinstance(payload.get("terminal_report"), dict):
        raise SystemExit("ERROR: final bounded GeRaF checkpoint cannot recover an unpublished terminal report")
    print("GERAF_B78716_SMALLFIT_TERMINAL_REPORT_RECOVERY_PREFLIGHT")
else:
    if payload.get("phase") != "interrupted_clean":
        raise SystemExit("ERROR: bounded GeRaF latest checkpoint is not a clean interruption")
    print("GERAF_B78716_SMALLFIT_CLEAN_RESUME_PREFLIGHT")
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
gate_tmp="$(mktemp -d -- "$temp_parent/geraf-b78716-smallfit-v1.${SLURM_JOB_ID}.XXXXXX")"
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
  echo "ERROR: refusing to overwrite durable launcher logs: $attempt_log_dir" >&2
  exit 94
}
mkdir -p -- "$attempt_log_dir"

echo 'GERAF_B78716_SMALLFIT_SOURCE_PREFLIGHT_START'
python -B -u scripts/validate_geraf_b78716_smallfit_source.py
echo 'GERAF_B78716_SMALLFIT_SOURCE_PREFLIGHT_PASS'
echo 'GERAF_B78716_SMALLFIT_TORCH_FIXTURE_START'
python -B -u scripts/validate_geraf_b78716_smallfit.py
echo 'GERAF_B78716_SMALLFIT_TORCH_FIXTURE_PASS'

time_log="$attempt_log_dir/geraf_time_v.txt"
driver_log="$attempt_log_dir/geraf_driver.log"
echo 'GERAF_B78716_SMALLFIT_REAL_COMBINED_START'
set +e
/usr/bin/time -v -o "$time_log" python -B -u train_geraf_smoke.py \
  --cache-root "$CACHE_ROOT" --checkpoint-dir "$FIT_ROOT" \
  --npz-path "$NPZ" --role-manifest "$PARENT_MANIFEST" --device cuda \
  --host-rss-limit-gib 32 "${resume_args[@]}" 2>&1 | tee "$driver_log"
pipeline_status=("${PIPESTATUS[@]}")
driver_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
set -e
if (( tee_status != 0 )); then
  echo "ERROR: bounded GeRaF driver log stream failed (driver=$driver_status tee=$tee_status); retaining $attempt_log_dir" >&2
  exit "$tee_status"
fi
if (( driver_status == 143 )); then
  [[ ! -e "$final_checkpoint" && ! -e "$report" && ! -e "$postflight" && -f "$latest_checkpoint" ]] || {
    echo 'ERROR: bounded GeRaF clean-stop exit lacked its exact resumable checkpoint state' >&2
    exit 95
  }
  python - "$latest_checkpoint" <<'PY'
from pathlib import Path
import sys
import torch

payload = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
if not isinstance(payload, dict) or payload.get("phase") != "interrupted_clean":
    raise SystemExit("ERROR: clean-stop latest checkpoint is not resumable")
PY
  grep -Fxq 'GERAF_B78716_SMALLFIT_CLEAN_STOP_RESUMABLE' "$driver_log" || {
    echo 'ERROR: bounded GeRaF clean-stop exit lacked its lifecycle marker' >&2
    exit 95
  }
  echo "GERAF_B78716_SMALLFIT_CLEAN_STOP_RETAINED logs=$attempt_log_dir"
  exit 143
fi
if (( driver_status == 75 )); then
  grep -Fxq 'GERAF_B78716_SMALLFIT_STOPPED_BEFORE_FIT_NO_CHECKPOINT' "$driver_log" || {
    echo 'ERROR: bounded GeRaF pre-fit stop lacked its lifecycle marker' >&2
    exit 95
  }
  echo "GERAF_B78716_SMALLFIT_PRE_FIT_STOP_RETAINED logs=$attempt_log_dir"
  exit 75
fi
if (( driver_status != 0 )); then
  echo "ERROR: bounded GeRaF driver failed (driver=$driver_status tee=$tee_status); retaining literal evidence" >&2
  exit "$driver_status"
fi
[[ -f "$report" && -f "$final_checkpoint" ]] || {
  echo 'ERROR: bounded GeRaF driver did not produce report and final checkpoint' >&2
  exit 95
}
grep -Fxq 'GERAF_B78716_SMALLFIT_LIFECYCLE_COMPLETE' "$driver_log" || {
  echo 'ERROR: bounded GeRaF driver lacked its lifecycle-complete marker' >&2
  exit 95
}

time_process_rss_kib="$(awk -F ': *' '/Maximum resident set size/ {print $2; exit}' "$time_log")"
[[ "$time_process_rss_kib" =~ ^[0-9]+$ ]] || {
  echo 'ERROR: unable to read process RSS from /usr/bin/time' >&2
  exit 95
}
gpu_total_mib="$(python - <<'PY'
import torch
print(torch.cuda.get_device_properties(0).total_memory // 2**20)
PY
)"
python -B -u scripts/postflight_geraf_b78716_smallfit.py \
  --report "$report" --output "$postflight" --time-process-rss-kib "$time_process_rss_kib" \
  --host-limit-gib 32 --gpu-total-mib "$gpu_total_mib"
echo 'GERAF_B78716_SMALLFIT_LIFECYCLE_PASS'
