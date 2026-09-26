#!/bin/bash
# Review candidate for one full B787 RadarSplat native-power fit.  The cell
# prepares all 3200 train and 1000 validation targets, keeps 1000 test and
# 4800 unused responses sealed, fits the reviewed 2048-row/480000-update,
# 150-pass recipe,
# and performs cache-only native train/validation readout.  It is not a
# production-clearance or coherent-signal evaluation.
#SBATCH --job-name=mgr_radarsplat_b7873200_fullscale_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=inferno
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --tmp=12G
#SBATCH --time=12:00:00
#SBATCH --signal=TERM@300
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0
#SBATCH --export=NONE

set -euo pipefail
umask 077

terminal_signal() {
  trap - TERM INT
  echo 'ERROR: batch shell received a signal without a validated child clean-stop state; retaining terminal evidence' >&2
  exit 96
}
trap terminal_signal TERM INT

resume_mode=false
case "$#" in
  0) ;;
  1)
    [[ "$1" == "resume" ]] || {
      echo 'ERROR: RadarSplat full-scale accepts zero arguments or the literal resume argument' >&2
      exit 64
    }
    resume_mode=true
    ;;
  *)
    echo 'ERROR: RadarSplat full-scale accepts zero arguments or the literal resume argument' >&2
    exit 64
    ;;
esac

[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: one non-array Slurm allocation is required' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == "inferno" && "${SLURM_JOB_PARTITION:-}" == "gpu-rtx6000" ]] || {
  echo 'ERROR: RadarSplat full-scale requires the Inferno RTX6000 allocation' >&2
  exit 94
}
[[ "${SLURM_CPUS_PER_TASK:-}" == "6" && "${SLURM_MEM_PER_NODE:-}" == "32768" ]] || {
  echo 'ERROR: RadarSplat full-scale requires 6 CPUs and 32 GiB RAM' >&2
  exit 94
}

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
PARENT_MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json
FULL_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/radarsplat_b7873200_fullscale_150epochs_v1
CACHE_ROOT="$FULL_ROOT/native_power_cache_3200x1000"
CHECKPOINT_DIR="$FULL_ROOT/native_power_fit_480000steps_2048gaussians"
TRAIN_LOG="$CHECKPOINT_DIR/training_trace.jsonl"
READOUT="$FULL_ROOT/fullscale_readout.json"
LAUNCHER_LOG_ROOT="$FULL_ROOT/launcher_logs"
ATTEMPT_LOG_DIR="$LAUNCHER_LOG_ROOT/${SLURM_JOB_ID}"

for source in \
  "$PROJECT/protocols/radarsplat_b7873200_fullscale_v1.json" \
  "$PROJECT/slurm/train_radarsplat_b7873200_fullscale_v1.sbatch" \
  "$PROJECT/rift/radarsplat_b7873200.py" \
  "$PROJECT/rift/radarsplat_b7873200_adapter.py" \
  "$PROJECT/rift/radarsplat_b7873200_acquisition.py" \
  "$PROJECT/rift/radarsplat_b7873200_protocol.py" \
  "$PROJECT/rift/power_baseline_dataset.py" \
  "$PROJECT/rift/matched_filter_power.py" \
  "$PROJECT/train_radarsplat.py" \
  "$PROJECT/scripts/prepare_radarsplat_b7873200_targets.py" \
  "$PROJECT/scripts/validate_radarsplat_b7873200_static.py" \
  "$PROJECT/scripts/validate_radarsplat_b7873200_native_contract.py" \
  "$PROJECT/scripts/validate_radarsplat_b7873200_fullscale_config.py" \
  "$PROJECT/scripts/readout_radarsplat_b7873200_fullscale.py" \
  "$NPZ" \
  "$PARENT_MANIFEST" \
  /usr/bin/time; do
  [[ -f "$source" ]] || { echo "ERROR: missing reviewed input: $source" >&2; exit 94; }
done
command -v srun >/dev/null 2>&1 || {
  echo 'ERROR: RadarSplat full-scale requires srun for direct allocated compute steps' >&2
  exit 94
}

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

has_line_prefix() {
  local path="$1"
  local prefix="$2"
  local line
  [[ -f "$path" ]] || return 1
  while IFS= read -r line; do
    [[ "$line" == "$prefix"* ]] && return 0
  done < "$path"
  return 1
}

has_attempt_prefix() {
  local filename="$1"
  local prefix="$2"
  local candidate
  for candidate in "$LAUNCHER_LOG_ROOT"/*/"$filename"; do
    has_line_prefix "$candidate" "$prefix" && return 0
  done
  return 1
}

has_attempt_file() {
  local filename="$1"
  local candidate
  for candidate in "$LAUNCHER_LOG_ROOT"/*/"$filename"; do
    [[ -f "$candidate" ]] && return 0
  done
  return 1
}

if [[ "$resume_mode" == false ]]; then
  for path in "$CACHE_ROOT" "$CHECKPOINT_DIR" "$READOUT" "$LAUNCHER_LOG_ROOT"; do
    [[ ! -e "$path" ]] || {
      echo "ERROR: fresh RadarSplat full-scale refuses existing evidence: $path" >&2
      exit 94
    }
  done
else
  [[ ! -e "$READOUT" ]] || {
    echo 'ERROR: completed RadarSplat full-scale evidence exists; this identity must not resume' >&2
    exit 94
  }
  [[ -d "$LAUNCHER_LOG_ROOT" ]] || {
    echo 'ERROR: full-scale resume requires the durable launcher evidence root' >&2
    exit 94
  }
  if [[ -f "$CHECKPOINT_DIR/checkpoint_final.pt" ]]; then
    has_attempt_marker 'fit_stdout.log' 'RADARSPLAT_B7873200_NATIVE_TRAINING_COMPLETE' || {
      echo 'ERROR: completed fit lacks its durable training-complete marker' >&2
      exit 94
    }
    if has_attempt_file 'readout.log' \
      && ! has_attempt_marker 'readout.log' 'RADARSPLAT_B7873200_READOUT_CLEAN_STOP'; then
      echo 'ERROR: terminal readout evidence exists without a validated clean-stop marker' >&2
      exit 94
    fi
  elif [[ -f "$CHECKPOINT_DIR/checkpoint_latest.pt" ]]; then
    has_attempt_marker 'fit_stdout.log' 'RADARSPLAT_B7873200_FULLSCALE_CLEAN_STOP_RETAINED' || {
      echo 'ERROR: latest checkpoint lacks validated full-scale clean-stop evidence' >&2
      exit 94
    }
  elif [[ -d "$CACHE_ROOT" ]]; then
    if ! has_attempt_marker 'prepare.log' 'RADARSPLAT_B7873200_TARGET_CACHE_COMPLETE' \
      && ! has_attempt_prefix 'prepare.log' 'RADARSPLAT_B7873200_PREPARE_CLEAN_STOP_RETAINED'; then
      echo 'ERROR: partial full-scale cache lacks validated preparation completion or clean-stop evidence' >&2
      exit 94
    fi
    echo 'INFO: resume will continue target preparation from its atomic target files' >&2
  else
    echo 'ERROR: full-scale resume requires a partial cache or a validated latest checkpoint' >&2
    exit 94
  fi
fi

mkdir -p -- "$ATTEMPT_LOG_DIR"

module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT
cd "$PROJECT"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6

RUN_STEP=(srun --nodes=1 --ntasks=1 --cpus-per-task=6 --exact --kill-on-bad-exit=1
  --signal=TERM@300 --export=ALL --unbuffered)

run_step() {
  "${RUN_STEP[@]}" "$@"
}

pipeline_failure_code() {
  local child_status="$1"
  shift
  if (( child_status != 0 )); then
    printf '%s\n' "$child_status"
    return 0
  fi
  local pipeline_status
  for pipeline_status in "$@"; do
    if (( pipeline_status != 0 )); then
      printf '%s\n' "$pipeline_status"
      return 0
    fi
  done
  printf '0\n'
}

GPU_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk 'NR == 1 {gsub(/[[:space:]]/, ""); print; exit}')"
[[ "$GPU_TOTAL_MIB" =~ ^[0-9]+$ && "$GPU_TOTAL_MIB" -gt 0 ]] || {
  echo 'ERROR: could not determine allocated GPU memory' >&2
  exit 94
}

echo 'RADARSPLAT_B7873200_SIGNAL_PROBE_START'
set +e
"${RUN_STEP[@]}" python -B -c 'import os, signal; seen = []; signal.signal(signal.SIGTERM, lambda signum, frame: seen.append(signum)); os.kill(os.getpid(), signal.SIGTERM); print("RADARSPLAT_B7873200_SIGNAL_PROBE_PASS" if seen == [signal.SIGTERM] else "RADARSPLAT_B7873200_SIGNAL_PROBE_FAIL", flush=True); raise SystemExit(143 if seen == [signal.SIGTERM] else 95)'
signal_probe_status=$?
set -e
(( signal_probe_status == 143 )) || {
  echo "ERROR: direct Python srun signal probe failed with status $signal_probe_status" >&2
  exit 96
}
echo 'RADARSPLAT_B7873200_SIGNAL_PROBE_COMPLETE'

echo 'RADARSPLAT_B7873200_FULLSCALE_SOURCE_PREFLIGHT_START'
run_step python -B -u scripts/validate_radarsplat_b7873200_fullscale_config.py \
  --config protocols/radarsplat_b7873200_fullscale_v1.json \
  --launcher slurm/train_radarsplat_b7873200_fullscale_v1.sbatch
run_step python -B -u scripts/validate_radarsplat_b7873200_static.py
echo 'RADARSPLAT_B7873200_FULLSCALE_SOURCE_PREFLIGHT_PASS'

echo 'RADARSPLAT_B7873200_FULLSCALE_NATIVE_CONTRACT_START'
run_step python -B -u scripts/validate_radarsplat_b7873200_native_contract.py
echo 'RADARSPLAT_B7873200_FULLSCALE_NATIVE_CONTRACT_PASS'

prepare_log="$ATTEMPT_LOG_DIR/prepare.log"
prepare_time_log="$ATTEMPT_LOG_DIR/prepare.time"
echo 'RADARSPLAT_B7873200_FULLSCALE_PREPARE_START'
set +e
"/usr/bin/time" -v -o "$prepare_time_log" "${RUN_STEP[@]}" python -B -u scripts/prepare_radarsplat_b7873200_targets.py \
  --npz-path "$NPZ" --role-manifest "$PARENT_MANIFEST" --cache-root "$CACHE_ROOT" --device cuda \
  --backend range_nufft --compute-dtype float64 --point-chunk 1024 --pair-chunk 64 --freq-chunk 64 \
  --nufft-oversample 2 --nufft-kernel-width 20 --scene-extent-m 0.15 \
  --n-azimuth 32 --n-elevation 32 --n-range 32 --output-azimuth-resolution-deg 0.9 \
  --elevation-sampling-resolution-deg 0.9 --intermediate-azimuth-resolution-deg 0.09 \
  --max-train 0 --max-validation 0 --resume 2>&1 | tee "$prepare_log"
prepare_status=(${PIPESTATUS[@]})
set -e
if (( prepare_status[0] == 143 )); then
  (( prepare_status[1] == 0 )) || {
    echo 'ERROR: preparation tee failed while the child reported a clean stop' >&2
    exit 96
  }
  has_line_prefix "$prepare_log" 'RADARSPLAT_B7873200_PREPARE_CLEAN_STOP_RETAINED' || {
    echo 'ERROR: full-scale preparation stop lacked its durable clean-stop marker' >&2
    exit 96
  }
  exit 143
fi
if (( prepare_status[0] != 0 || prepare_status[1] != 0 )); then
  failure_status="$(pipeline_failure_code "${prepare_status[@]}")"
  echo "ERROR: full-scale target preparation failed with status $failure_status; retaining evidence" >&2
  exit "$failure_status"
fi
grep -Fxq 'RADARSPLAT_B7873200_TARGET_CACHE_COMPLETE' "$prepare_log" || {
  echo 'ERROR: full-scale target preparation lacks its complete-cache marker' >&2
  exit 94
}
echo 'RADARSPLAT_B7873200_FULLSCALE_PREPARE_PASS'

mkdir -p -- "$CHECKPOINT_DIR"
fit_stdout_log="$ATTEMPT_LOG_DIR/fit_stdout.log"
if [[ "$resume_mode" == true && -f "$TRAIN_LOG" ]]; then
  train_tee=(tee -a)
else
  train_tee=(tee)
fi
echo 'RADARSPLAT_B7873200_FULLSCALE_TRAIN_START'
set +e
run_step python -B -u train_radarsplat.py \
  --cache-root "$CACHE_ROOT" --checkpoint-dir "$CHECKPOINT_DIR" --device cuda \
  --seed 42 --steps 480000 --validation-every 600 --checkpoint-every 100 --log-every 1 --resume \
  --init-num-gaussians 2048 --init-extent-m 0.3 --init-scale-m 0.003 \
  --init-opacity 0.1 --init-noise-probability 0.1 --sh-degree 3 --sh-degree-interval 600 \
  --means-lr-base 1.6e-4 --scales-lr 5.0e-3 --quaternions-lr 1.0e-3 \
  --opacity-lr 5.0e-2 --noise-probability-lr 5.0e-2 --sh0-lr 2.5e-3 --shn-lr 2.5e-3 \
  --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1.0e-15 \
  --ssim-weight 0.2 --occupancy-weight 5.0 --max-size-weight 100.0 --opacity-noise-weight 100.0 \
  --max-scale-m 0.006 --occupancy-threshold 0.001 --prune-opacity 0.0005 --prune-every 100 \
  --gaussian-chunk-size 64 --max-raster-candidate-pairs 2000000 \
  2>&1 | "${train_tee[@]}" "$TRAIN_LOG" | tee "$fit_stdout_log"
fit_status=(${PIPESTATUS[@]})
set -e
if (( fit_status[0] == 143 )); then
  (( fit_status[1] == 0 && fit_status[2] == 0 )) || {
    echo 'ERROR: training log pipeline failed while the child reported a clean stop' >&2
    exit 96
  }
  grep -Fxq 'RadarSplat B7873200 clean interruption saved checkpoint_latest.pt' "$fit_stdout_log" || {
    echo 'ERROR: full-scale clean stop lacked the trainer checkpoint marker' >&2
    exit 96
  }
  [[ -f "$CHECKPOINT_DIR/checkpoint_latest.pt" && ! -e "$CHECKPOINT_DIR/checkpoint_final.pt" \
        && ! -e "$CHECKPOINT_DIR/gaussian_occupancy_geometry.npz" && ! -e "$CHECKPOINT_DIR/summary.json" ]] || {
    echo 'ERROR: full-scale clean stop did not retain the exact unfinished state' >&2
    exit 96
  }
  printf '%s\n' 'RADARSPLAT_B7873200_FULLSCALE_CLEAN_STOP_RETAINED' | tee -a "$fit_stdout_log"
  exit 143
fi
if (( fit_status[0] != 0 || fit_status[1] != 0 || fit_status[2] != 0 )); then
  failure_status="$(pipeline_failure_code "${fit_status[@]}")"
  echo "ERROR: full-scale training failed with status $failure_status; retaining evidence" >&2
  exit "$failure_status"
fi
if ! grep -Fxq 'RADARSPLAT_B7873200_NATIVE_TRAINING_COMPLETE' "$fit_stdout_log"; then
  if [[ "$resume_mode" != true ]] || ! grep -Fxq 'RadarSplat B7873200 completion artifacts are present; no update executed.' "$fit_stdout_log"; then
    echo 'ERROR: full-scale training lacks its completion or verified no-update marker' >&2
    exit 94
  fi
fi
[[ -f "$CHECKPOINT_DIR/checkpoint_final.pt" && -f "$CHECKPOINT_DIR/checkpoint_latest.pt" \
      && -f "$CHECKPOINT_DIR/gaussian_occupancy_geometry.npz" && -f "$CHECKPOINT_DIR/summary.json" ]] || {
  echo 'ERROR: full-scale training lacks required durable artifacts' >&2
  exit 94
}
echo 'RADARSPLAT_B7873200_FULLSCALE_TRAIN_PASS'

completion_resume_log="$ATTEMPT_LOG_DIR/completion_resume.log"
echo 'RADARSPLAT_B7873200_FULLSCALE_COMPLETION_RESUME_START'
run_step python -B -u train_radarsplat.py \
  --cache-root "$CACHE_ROOT" --checkpoint-dir "$CHECKPOINT_DIR" --device cuda \
  --seed 42 --steps 480000 --validation-every 600 --checkpoint-every 100 --log-every 1 --resume \
  --init-num-gaussians 2048 --init-extent-m 0.3 --init-scale-m 0.003 \
  --init-opacity 0.1 --init-noise-probability 0.1 --sh-degree 3 --sh-degree-interval 600 \
  --means-lr-base 1.6e-4 --scales-lr 5.0e-3 --quaternions-lr 1.0e-3 \
  --opacity-lr 5.0e-2 --noise-probability-lr 5.0e-2 --sh0-lr 2.5e-3 --shn-lr 2.5e-3 \
  --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1.0e-15 \
  --ssim-weight 0.2 --occupancy-weight 5.0 --max-size-weight 100.0 --opacity-noise-weight 100.0 \
  --max-scale-m 0.006 --occupancy-threshold 0.001 --prune-opacity 0.0005 --prune-every 100 \
  --gaussian-chunk-size 64 --max-raster-candidate-pairs 2000000 2>&1 | tee "$completion_resume_log"
grep -Fxq 'RadarSplat B7873200 completion artifacts are present; no update executed.' "$completion_resume_log" || {
  echo 'ERROR: full-scale completion resume did not prove its no-update path' >&2
  exit 94
}

readout_log="$ATTEMPT_LOG_DIR/readout.log"
echo 'RADARSPLAT_B7873200_FULLSCALE_READOUT_START'
set +e
run_step python -B -u scripts/readout_radarsplat_b7873200_fullscale.py \
  --cache-root "$CACHE_ROOT" --checkpoint-dir "$CHECKPOINT_DIR" \
  --launcher-log-root "$LAUNCHER_LOG_ROOT" --output "$READOUT" --device cuda \
  --gpu-total-mib "$GPU_TOTAL_MIB" --host-limit-gib 32 \
  --seed 42 --steps 480000 --epochs 150 --validation-every 600 --checkpoint-every 100 --log-every 1 \
  --init-num-gaussians 2048 --init-extent-m 0.3 --init-scale-m 0.003 \
  --init-opacity 0.1 --init-noise-probability 0.1 --sh-degree 3 --sh-degree-interval 600 \
  --means-lr-base 1.6e-4 --scales-lr 5.0e-3 --quaternions-lr 1.0e-3 \
  --opacity-lr 5.0e-2 --noise-probability-lr 5.0e-2 --sh0-lr 2.5e-3 --shn-lr 2.5e-3 \
  --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1.0e-15 \
  --ssim-weight 0.2 --occupancy-weight 5.0 --max-size-weight 100.0 --opacity-noise-weight 100.0 \
  --max-scale-m 0.006 --occupancy-threshold 0.001 --prune-opacity 0.0005 --prune-every 100 \
  --gaussian-chunk-size 64 --max-raster-candidate-pairs 2000000 2>&1 | tee "$readout_log"
readout_status=(${PIPESTATUS[@]})
set -e
if (( readout_status[0] == 143 )); then
  (( readout_status[1] == 0 )) || {
    echo 'ERROR: readout tee failed while the child reported a clean stop' >&2
    exit 96
  }
  grep -Fxq 'RADARSPLAT_B7873200_READOUT_CLEAN_STOP' "$readout_log" || {
    echo 'ERROR: full-scale readout clean stop lacked its marker' >&2
    exit 96
  }
  exit 143
fi
if (( readout_status[0] != 0 || readout_status[1] != 0 )); then
  failure_status="$(pipeline_failure_code "${readout_status[@]}")"
  echo "ERROR: full-scale readout failed with status $failure_status; retaining evidence" >&2
  exit "$failure_status"
fi
grep -Fxq 'RADARSPLAT_B7873200_FULLSCALE_READOUT_PASS' "$readout_log" || {
  echo 'ERROR: full-scale readout lacks its pass marker' >&2
  exit 94
}
[[ -f "$READOUT" ]] || { echo 'ERROR: full-scale readout JSON is missing' >&2; exit 94; }
echo 'RADARSPLAT_B7873200_FULLSCALE_PASS'
