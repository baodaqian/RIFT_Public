#!/bin/bash
# One bounded real-data RadarSplat B787 engineering cell.  The manager is the
# only submitter.  It preserves failed contract job 12874322 as terminal
# evidence and never requeues it.  This fresh cell runs the repaired native
# fixture, prepares exactly train[:8]/validation[:4], executes 20 native-power
# updates from 64 Gaussians, verifies completed-resume behaviour, and writes a
# compact postflight report.  It is not a 3200-view, coherent-signal, mesh,
# NVS, reconstruction, convergence, or baseline-comparison run.
#SBATCH --job-name=mgr_radarsplat_b7873200_smallfit8x4_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=inferno
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --tmp=12G
#SBATCH --time=02:00:00
#SBATCH --signal=TERM@120
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

resume_mode=false
case "$#" in
  0) ;;
  1)
    [[ "$1" == "resume" ]] || {
      echo 'ERROR: RadarSplat small fit accepts only the literal resume argument' >&2
      exit 64
    }
    resume_mode=true
    ;;
  *)
    echo 'ERROR: RadarSplat small fit accepts zero arguments or the literal resume argument' >&2
    exit 64
    ;;
esac

[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: one non-array Slurm allocation is required' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == "inferno" && "${SLURM_JOB_PARTITION:-}" == "gpu-rtx6000" ]] || {
  echo 'ERROR: RadarSplat small fit requires the Inferno RTX6000 allocation' >&2
  exit 94
}
[[ "${SLURM_CPUS_PER_TASK:-}" == "6" && "${SLURM_MEM_PER_NODE:-}" == "32768" ]] || {
  echo 'ERROR: RadarSplat small fit requires 6 CPUs and 32 GiB RAM' >&2
  exit 94
}

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
PARENT_MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json
SMOKE_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/radarsplat_b7873200_smallfit8x4_v1
CACHE_ROOT="$SMOKE_ROOT/native_power_cache_8x4"
CHECKPOINT_DIR="$SMOKE_ROOT/native_power_fit_20steps_64gaussians"
TRAIN_LOG="$CHECKPOINT_DIR/training_trace.jsonl"
POSTFLIGHT="$SMOKE_ROOT/smallfit_postflight.json"
LAUNCHER_LOG_ROOT="$SMOKE_ROOT/launcher_logs"

has_attempt_marker() {
  local filename="$1"
  local marker="$2"
  local candidate
  for candidate in "$LAUNCHER_LOG_ROOT"/*/"$filename"; do
    [[ -f "$candidate" ]] || continue
    grep -Fxq "$marker" "$candidate" && return 0
  done
  return 1
}

resume_requires_clean_checkpoint=false

for source in \
  "$PROJECT/rift/radarsplat_b7873200.py" \
  "$PROJECT/rift/radarsplat_b7873200_adapter.py" \
  "$PROJECT/rift/radarsplat_b7873200_acquisition.py" \
  "$PROJECT/rift/radarsplat_b7873200_protocol.py" \
  "$PROJECT/rift/power_baseline_dataset.py" \
  "$PROJECT/train_radarsplat.py" \
  "$PROJECT/scripts/prepare_radarsplat_b7873200_targets.py" \
  "$PROJECT/scripts/validate_radarsplat_b7873200_static.py" \
  "$PROJECT/scripts/validate_radarsplat_b7873200_native_contract.py" \
  "$PROJECT/scripts/postflight_radarsplat_b7873200_smallfit.py" \
  "$NPZ" \
  "$PARENT_MANIFEST" \
  /usr/bin/time; do
  [[ -f "$source" ]] || { echo "ERROR: missing reviewed input: $source" >&2; exit 94; }
done
command -v srun >/dev/null 2>&1 || {
  echo 'ERROR: RadarSplat small fit requires srun for direct allocated compute steps' >&2
  exit 94
}

if [[ "$resume_mode" == false ]]; then
  for path in "$CACHE_ROOT" "$CHECKPOINT_DIR" "$POSTFLIGHT" "$LAUNCHER_LOG_ROOT"; do
    [[ ! -e "$path" ]] || {
      echo "ERROR: fresh RadarSplat small fit refuses existing evidence: $path" >&2
      exit 94
    }
  done
else
  [[ ! -e "$POSTFLIGHT" ]] || {
    echo 'ERROR: completed RadarSplat small-fit evidence exists; this identity must not resume' >&2
    exit 94
  }
  [[ -d "$LAUNCHER_LOG_ROOT" ]] || {
    echo 'ERROR: resume requires the durable launcher evidence root' >&2
    exit 94
  }
  if [[ -f "$CHECKPOINT_DIR/checkpoint_final.pt" ]]; then
    has_attempt_marker 'fit_stdout.log' 'RADARSPLAT_B7873200_NATIVE_TRAINING_COMPLETE' || {
      echo 'ERROR: completed fit lacks its durable training-complete marker and cannot resume postflight' >&2
      exit 94
    }
    has_attempt_marker 'postflight.log' 'RADARSPLAT_B7873200_SMALLFIT8X4_POSTFLIGHT_CLEAN_STOP_RETAINED' || {
      echo 'ERROR: completed fit may resume only after a verified postflight clean stop' >&2
      exit 94
    }
  elif [[ -f "$CHECKPOINT_DIR/checkpoint_latest.pt" ]]; then
    has_attempt_marker 'fit_stdout.log' 'RADARSPLAT_B7873200_SMALLFIT8X4_CLEAN_STOP_RETAINED' || {
      echo 'ERROR: latest checkpoint lacks verified clean-stop evidence and cannot resume' >&2
      exit 94
    }
    resume_requires_clean_checkpoint=true
  else
    echo 'ERROR: resume requires a verified clean-stop latest checkpoint or completed-postflight-stop evidence' >&2
    exit 94
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

run_gpu_step() {
  srun --nodes=1 --ntasks=1 --cpus-per-task=6 --exact --export=ALL --unbuffered "$@"
}

validate_clean_stop_checkpoint() {
  python - "$CHECKPOINT_DIR/checkpoint_latest.pt" "$CACHE_ROOT" <<'PY'
from pathlib import Path
import sys

import train_radarsplat as trainer
from rift.radarsplat_b7873200_protocol import load_cache

checkpoint_path, cache_root = (Path(value) for value in sys.argv[1:])
checkpoint = trainer._load_checkpoint(checkpoint_path, "cpu")
cache = load_cache(cache_root)
identity = checkpoint.get("run_identity")
last_train = checkpoint.get("last_train")
step = checkpoint.get("step")
if (
    checkpoint.get("complete") is not False
    or checkpoint.get("finalization_pending") is not False
    or not isinstance(step, int)
    or not 1 <= step <= 20
    or not isinstance(identity, dict)
    or identity.get("sealed_protocol_identity") != cache.identity
    or identity.get("target_recipe") != cache.recipe
    or not isinstance(last_train, dict)
    or last_train.get("step") != step
):
    raise SystemExit("ERROR: latest checkpoint is not an exact clean RadarSplat small-fit continuation state")
print("RADARSPLAT_B7873200_SMALLFIT8X4_CLEAN_CHECKPOINT_PASS")
PY
}

gpu_total_mib="$(python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('ERROR: RadarSplat small fit requires exactly one CUDA device')
properties = torch.cuda.get_device_properties(0)
print(properties.total_memory // 2**20)
PY
)"
[[ "$gpu_total_mib" =~ ^[0-9]+$ && "$gpu_total_mib" -gt 0 ]] || {
  echo 'ERROR: could not determine the allocated GPU memory' >&2
  exit 94
}
command -v nvidia-smi >/dev/null 2>&1 || {
  echo 'ERROR: nvidia-smi is unavailable on the allocated GPU node' >&2
  exit 94
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ "$resume_requires_clean_checkpoint" == true ]]; then
  validate_clean_stop_checkpoint
fi

# Only a verified job-local child is removed.  The durable cache, checkpoints,
# reports, and per-attempt logs remain on PACE for manager audit.
temp_parent="${SLURM_TMPDIR:-/tmp}"
[[ -d "$temp_parent" && -w "$temp_parent" && -x "$temp_parent" ]] || {
  echo "ERROR: no usable node-local temporary parent: $temp_parent" >&2
  exit 94
}
temp_parent="$(cd -P -- "$temp_parent" && pwd -P)"
[[ "$temp_parent" != / ]] || { echo 'ERROR: refusing root as temporary parent' >&2; exit 94; }
available_kib="$(df -Pk "$temp_parent" | awk 'NR == 2 {print $4}')"
[[ "$available_kib" =~ ^[0-9]+$ ]] && (( available_kib >= 12 * 1024 * 1024 )) || {
  echo "ERROR: required 12G temporary space unavailable at $temp_parent" >&2
  exit 94
}
job_tmp="$(mktemp -d -- "$temp_parent/radarsplat-b7873200-smallfit-v1.${SLURM_JOB_ID}.XXXXXX")"
job_tmp="$(cd -P -- "$job_tmp" && pwd -P)"
case "$job_tmp" in
  "$temp_parent"/radarsplat-b7873200-smallfit-v1."${SLURM_JOB_ID}".*) ;;
  *) echo "ERROR: unsafe temporary path $job_tmp" >&2; exit 94 ;;
esac
[[ "$(stat -c '%u' -- "$job_tmp")" == "$(id -u)" ]] || {
  echo 'ERROR: temporary directory ownership mismatch' >&2
  exit 94
}
cleanup() {
  local status=$?
  trap - EXIT
  if [[ -d "${job_tmp:-}" && "$(dirname -- "$job_tmp")" == "$temp_parent" \
        && "$(stat -c '%u' -- "$job_tmp")" == "$(id -u)" ]]; then
    case "$job_tmp" in
      "$temp_parent"/radarsplat-b7873200-smallfit-v1."${SLURM_JOB_ID}".*) rm -rf -- "$job_tmp" ;;
      *) echo "ERROR: refusing unexpected temporary cleanup target $job_tmp" >&2 ;;
    esac
  fi
  exit "$status"
}
trap cleanup EXIT
export TMPDIR="$job_tmp" TEMP="$job_tmp" TMP="$job_tmp"

mkdir -p -- "$SMOKE_ROOT" "$LAUNCHER_LOG_ROOT"
attempt_log_dir="$LAUNCHER_LOG_ROOT/${SLURM_JOB_ID}"
[[ ! -e "$attempt_log_dir" ]] || {
  echo "ERROR: refusing to overwrite this attempt log directory: $attempt_log_dir" >&2
  exit 94
}
mkdir -p -- "$attempt_log_dir"

echo 'RADARSPLAT_B7873200_SMALLFIT8X4_SOURCE_PREFLIGHT_START'
python -B -u scripts/validate_radarsplat_b7873200_static.py
echo 'RADARSPLAT_B7873200_SMALLFIT8X4_SOURCE_PREFLIGHT_PASS'
echo 'RADARSPLAT_B7873200_SMALLFIT8X4_NATIVE_FIXTURE_START'
python -B -u scripts/validate_radarsplat_b7873200_native_contract.py
echo 'RADARSPLAT_B7873200_SMALLFIT8X4_NATIVE_FIXTURE_PASS'

prepare_log="$attempt_log_dir/prepare.log"
prepare_time_log="$attempt_log_dir/prepare_time_v.txt"
echo 'RADARSPLAT_B7873200_SMALLFIT8X4_PREPARE_START'
set +e
run_gpu_step /usr/bin/time -v -o "$prepare_time_log" python -B -u scripts/prepare_radarsplat_b7873200_targets.py \
  --npz-path "$NPZ" --role-manifest "$PARENT_MANIFEST" --cache-root "$CACHE_ROOT" --device cuda \
  --backend range_nufft --compute-dtype float64 --point-chunk 1024 --pair-chunk 64 --freq-chunk 64 \
  --nufft-oversample 2 --nufft-kernel-width 20 --scene-extent-m 0.15 \
  --n-azimuth 32 --n-elevation 32 --n-range 32 \
  --output-azimuth-resolution-deg 0.9 --elevation-sampling-resolution-deg 0.9 \
  --intermediate-azimuth-resolution-deg 0.09 --occupancy-threshold 0.001 \
  --max-train 8 --max-validation 4 --resume 2>&1 | tee "$prepare_log"
prepare_status=("${PIPESTATUS[@]}")
set -e
if (( prepare_status[0] == 143 )); then
  echo 'ERROR: RadarSplat target preparation stopped before a trainer checkpoint; retaining terminal evidence' >&2
  exit 96
fi
if (( prepare_status[0] != 0 )); then
  echo "ERROR: RadarSplat target preparation exited ${prepare_status[0]}; retaining cache and attempt evidence" >&2
  exit "${prepare_status[0]}"
fi
if (( prepare_status[1] != 0 )); then
  echo "ERROR: RadarSplat target-preparation log capture exited ${prepare_status[1]}" >&2
  exit "${prepare_status[1]}"
fi
grep -Fxq 'RADARSPLAT_B7873200_TARGET_CACHE_COMPLETE' "$prepare_log" || {
  echo 'ERROR: RadarSplat target preparation lacks its completion marker' >&2
  exit 95
}
echo 'RADARSPLAT_B7873200_SMALLFIT8X4_PREPARE_PASS'

# The fresh guard above has already proved this directory did not exist.  It
# must exist before the tee below opens the durable trace, while the trainer's
# resume policy still accepts an empty directory with no artifacts.
mkdir -p -- "$CHECKPOINT_DIR"
[[ -d "$CHECKPOINT_DIR" ]] || {
  echo 'ERROR: cannot create the RadarSplat checkpoint/trace directory' >&2
  exit 95
}

train_log="$TRAIN_LOG"
fit_stdout_log="$attempt_log_dir/fit_stdout.log"
train_tee=(tee)
if [[ "$resume_mode" == true && -f "$train_log" ]]; then
  train_tee=(tee -a)
fi
echo 'RADARSPLAT_B7873200_SMALLFIT8X4_TRAIN_START'
set +e
run_gpu_step python -B -u train_radarsplat.py \
  --cache-root "$CACHE_ROOT" --checkpoint-dir "$CHECKPOINT_DIR" --device cuda \
  --seed 42 --steps 20 --validation-every 5 --checkpoint-every 5 --log-every 1 --resume \
  --allow-development-subset --init-num-gaussians 64 --init-extent-m 0.3 --init-scale-m 0.003 \
  --init-opacity 0.1 --init-noise-probability 0.1 --sh-degree 3 --sh-degree-interval 600 \
  --means-lr-base 0.00016 --scales-lr 0.005 --quaternions-lr 0.001 --opacity-lr 0.05 \
  --noise-probability-lr 0.05 --sh0-lr 0.0025 --shn-lr 0.0025 --adam-beta1 0.9 --adam-beta2 0.999 \
  --adam-eps 1e-15 --ssim-weight 0.2 --occupancy-weight 5 --max-size-weight 100 \
  --opacity-noise-weight 100 --max-scale-m 0.006 --occupancy-threshold 0.001 \
  --prune-opacity 0.0005 --prune-every 5 --gaussian-chunk-size 64 \
  --max-raster-candidate-pairs 2000000 2>&1 | "${train_tee[@]}" "$train_log" | tee "$fit_stdout_log"
fit_status=("${PIPESTATUS[@]}")
set -e
if (( fit_status[1] != 0 || fit_status[2] != 0 )); then
  echo 'ERROR: RadarSplat small-fit training log capture failed' >&2
  exit 95
fi
if (( fit_status[0] == 143 )); then
  grep -Fxq 'RadarSplat B7873200 clean interruption saved checkpoint_latest.pt' "$fit_stdout_log" || {
    echo 'ERROR: RadarSplat trainer stop lacked the native clean-checkpoint marker; retaining terminal evidence' >&2
    exit 96
  }
  [[ -f "$CHECKPOINT_DIR/checkpoint_latest.pt" && ! -e "$CHECKPOINT_DIR/checkpoint_final.pt" \
        && ! -e "$CHECKPOINT_DIR/gaussian_occupancy_geometry.npz" && ! -e "$CHECKPOINT_DIR/summary.json" ]] || {
    echo 'ERROR: RadarSplat trainer stop lacks the exact unfinished checkpoint/artifact state' >&2
    exit 95
  }
  validate_clean_stop_checkpoint
  printf '%s\n' 'RADARSPLAT_B7873200_SMALLFIT8X4_CLEAN_STOP_RETAINED' | tee -a "$fit_stdout_log"
  exit 143
fi
if (( fit_status[0] != 0 )); then
  echo "ERROR: RadarSplat small-fit trainer exited ${fit_status[0]}; retaining cache/checkpoint/log evidence" >&2
  exit "${fit_status[0]}"
fi
[[ -f "$CHECKPOINT_DIR/checkpoint_final.pt" && -f "$CHECKPOINT_DIR/checkpoint_latest.pt" \
      && -f "$CHECKPOINT_DIR/gaussian_occupancy_geometry.npz" && -f "$CHECKPOINT_DIR/summary.json" ]] || {
  echo 'ERROR: RadarSplat small fit lacks a complete terminal artifact set' >&2
  exit 95
}
if ! grep -Fxq 'RADARSPLAT_B7873200_NATIVE_TRAINING_COMPLETE' "$fit_stdout_log"; then
  # A scheduler interruption can occur after the trainer has durably
  # finalized, but before this launcher reaches postflight.  Its literal
  # resume must enter the trainer's verified no-update completion path.
  [[ "$resume_mode" == true ]] && grep -Fxq \
    'RadarSplat B7873200 completion artifacts are present; no update executed.' "$fit_stdout_log" || {
      echo 'ERROR: RadarSplat small fit lacks a training-complete or verified terminal-resume marker' >&2
      exit 95
    }
fi

# A completed same-identity invocation must only validate/recreate derivations,
# never execute another optimizer update.  The postflight reads the resulting
# final/latest payloads directly to verify that invariant.
recovery_log="$attempt_log_dir/completion_resume.log"
echo 'RADARSPLAT_B7873200_SMALLFIT8X4_COMPLETION_RESUME_START'
run_gpu_step python -B -u train_radarsplat.py \
  --cache-root "$CACHE_ROOT" --checkpoint-dir "$CHECKPOINT_DIR" --device cuda \
  --seed 42 --steps 20 --validation-every 5 --checkpoint-every 5 --log-every 1 --resume \
  --allow-development-subset --init-num-gaussians 64 --init-extent-m 0.3 --init-scale-m 0.003 \
  --init-opacity 0.1 --init-noise-probability 0.1 --sh-degree 3 --sh-degree-interval 600 \
  --means-lr-base 0.00016 --scales-lr 0.005 --quaternions-lr 0.001 --opacity-lr 0.05 \
  --noise-probability-lr 0.05 --sh0-lr 0.0025 --shn-lr 0.0025 --adam-beta1 0.9 --adam-beta2 0.999 \
  --adam-eps 1e-15 --ssim-weight 0.2 --occupancy-weight 5 --max-size-weight 100 \
  --opacity-noise-weight 100 --max-scale-m 0.006 --occupancy-threshold 0.001 \
  --prune-opacity 0.0005 --prune-every 5 --gaussian-chunk-size 64 \
  --max-raster-candidate-pairs 2000000 2>&1 | tee "$recovery_log"
grep -Fxq 'RadarSplat B7873200 completion artifacts are present; no update executed.' "$recovery_log" || {
  echo 'ERROR: RadarSplat completion resume did not prove its no-update path' >&2
  exit 95
}

whole_job_rss_raw=''
if command -v sstat >/dev/null 2>&1; then
  whole_job_rss_raw="$(sstat -j "${SLURM_JOB_ID}.batch" --noheader --format=MaxRSS 2>/dev/null | awk 'NF {print $1; exit}' || true)"
fi
postflight_args=(
  --cache-root "$CACHE_ROOT"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --train-log "$TRAIN_LOG"
  --launcher-log-root "$LAUNCHER_LOG_ROOT"
  --completion-resume-log "$recovery_log"
  --output "$POSTFLIGHT"
  --device cuda
  --gaussian-chunk-size 64
  --max-raster-candidate-pairs 2000000
  --host-limit-gib 32
  --gpu-total-mib "$gpu_total_mib"
)
if [[ -n "$whole_job_rss_raw" ]]; then
  postflight_args+=(--whole-job-rss-raw "$whole_job_rss_raw")
fi
echo 'RADARSPLAT_B7873200_SMALLFIT8X4_POSTFLIGHT_START'
postflight_log="$attempt_log_dir/postflight.log"
set +e
run_gpu_step python -B -u scripts/postflight_radarsplat_b7873200_smallfit.py "${postflight_args[@]}" 2>&1 | tee "$postflight_log"
postflight_status=("${PIPESTATUS[@]}")
set -e
if (( postflight_status[1] != 0 )); then
  echo 'ERROR: RadarSplat postflight log capture failed; retaining terminal evidence' >&2
  exit 95
fi
if (( postflight_status[0] == 143 )); then
  grep -Fxq 'RADARSPLAT_B7873200_POSTFLIGHT_CLEAN_STOP' "$postflight_log" || {
    echo 'ERROR: RadarSplat postflight stop lacked its explicit clean-stop marker' >&2
    exit 96
  }
  [[ -f "$CHECKPOINT_DIR/checkpoint_final.pt" && ! -e "$POSTFLIGHT" ]] || {
    echo 'ERROR: RadarSplat postflight stop lacks its exact recoverable terminal-artifact state' >&2
    exit 95
  }
  printf '%s\n' 'RADARSPLAT_B7873200_SMALLFIT8X4_POSTFLIGHT_CLEAN_STOP_RETAINED' | tee -a "$postflight_log"
  exit 143
fi
if (( postflight_status[0] != 0 )); then
  echo "ERROR: RadarSplat postflight exited ${postflight_status[0]}; retaining terminal evidence" >&2
  exit "${postflight_status[0]}"
fi
grep -Fxq 'RADARSPLAT_B7873200_SMALLFIT_POSTFLIGHT_PASS' "$postflight_log" || {
  echo 'ERROR: RadarSplat postflight lacks its completion marker' >&2
  exit 95
}
echo 'RADARSPLAT_B7873200_SMALLFIT8X4_PASS'
