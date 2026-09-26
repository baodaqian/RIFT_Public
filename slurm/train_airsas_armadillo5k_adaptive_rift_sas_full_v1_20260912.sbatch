#!/bin/bash
# Armadillo5k AirSAS full training, validation, and test readout.
# This launcher is staged locally for manager-owned review; it does not submit itself.
#SBATCH --job-name=mgr_rift_airsas_arm5k_adaptive_full_v1
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=embers
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --tmp=12G
#SBATCH --time=08:00:00
#SBATCH --signal=TERM@300
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0,atl1-1-01-002-8-0,atl1-1-02-006-34-0
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_adaptive_vs_sh_sas_v1_20260911/armadillo5k/adaptive_rift_sas_full/mgr_slurm_logs/%j.out
#SBATCH --error=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_adaptive_vs_sh_sas_v1_20260911/armadillo5k/adaptive_rift_sas_full/mgr_slurm_logs/%j.err

set -euo pipefail
umask 077

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: Armadillo5k AirSAS launcher accepts no arguments' >&2
  exit 64
}
[[ "${SLURM_JOB_QOS:-}" == 'embers' ]] || {
  echo "ERROR: expected SLURM_JOB_QOS=embers, got ${SLURM_JOB_QOS:-unset}" >&2
  exit 2
}
[[ "${SLURM_JOB_PARTITION:-}" == 'gpu-l40s' ]] || {
  echo "ERROR: expected SLURM_JOB_PARTITION=gpu-l40s, got ${SLURM_JOB_PARTITION:-unset}" >&2
  exit 2
}
[[ "${SLURM_JOB_NUM_NODES:-}" == '1' && "${SLURM_NTASKS:-}" == '1' ]] || {
  echo 'ERROR: expected exactly one non-array node/task' >&2
  exit 2
}
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || {
  echo 'ERROR: missing numeric Slurm job ID' >&2
  exit 2
}
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: array jobs are not allowed' >&2
  exit 2
}

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
CACHE=$PROJECT/datasets/sh_sas_reed/cache/armadillo_5khz_trainonly_120rings_v1_20260912
CHECKPOINT_ROOT=$PROJECT/training_checkpoints
MODEL=adaptive_rift_sas
NAME=airsas_armadillo5k_adaptive_rift_sas_full_v1_20260912
RUN_ROOT=$CHECKPOINT_ROOT/$NAME
STATUS=$RUN_ROOT/status.json
ATTEMPT_LOG_ROOT=$RUN_ROOT/attempt_logs
ATTEMPT_LOG=$ATTEMPT_LOG_ROOT/$SLURM_JOB_ID.log
TMP_ROOT="${SLURM_TMPDIR:-/tmp}/airsas_armadillo5k-${MODEL}-${SLURM_JOB_ID}"

require_nonempty() {
  [[ -s "$1" ]] || {
    echo "ERROR: required deployed file is missing or empty: $1" >&2
    exit 2
  }
}

require_nonempty "$PROJECT/rift/airsas_contract.py"
require_nonempty "$PROJECT/scripts/validate_airsas_5k_cache.py"
require_nonempty "$PROJECT/scripts/run_airsas_comparison.sh"
require_nonempty "$PROJECT/train_sas.py"
[[ -d "$CACHE" ]] || {
  echo "ERROR: shared AirSAS cache is missing: $CACHE" >&2
  exit 2
}

module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

cd "$PROJECT"
export PYTHONPATH="$PROJECT"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export SLURM_EXPORT_ENV=ALL
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK must be set}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export BLIS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$TMP_ROOT/tmpdir" "$TMP_ROOT/temp" "$TMP_ROOT/tmp"
export TMPDIR="$TMP_ROOT/tmpdir"
export TEMP="$TMP_ROOT/temp"
export TMP="$TMP_ROOT/tmp"

status_shape_ok() {
  [[ -s "$STATUS" ]] || return 1
  [[ "$(grep -Ec '^[[:space:]]*"done":[[:space:]]*(true|false),?[[:space:]]*$' "$STATUS")" == '1' ]] || return 1
  [[ "$(grep -Ec "^[[:space:]]*\"model\":[[:space:]]*\"${MODEL}\",?[[:space:]]*$" "$STATUS")" == '1' ]] || return 1
  [[ "$(grep -Ec '^[[:space:]]*"step":[[:space:]]*[0-9]+,?[[:space:]]*$' "$STATUS")" == '1' ]] || return 1
}

status_step() {
  local value
  value="$(sed -nE 's/^[[:space:]]*"step":[[:space:]]*([0-9]+).*/\1/p' "$STATUS")"
  [[ "$value" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$value"
}

status_is_clean() {
  status_shape_ok || return 1
  grep -Eq '^[[:space:]]*"done":[[:space:]]*false,?[[:space:]]*$' "$STATUS" || return 1
  [[ "$(grep -Ec '^[[:space:]]*"reason":[[:space:]]*"signal",?[[:space:]]*$' "$STATUS")" == '1' ]] || return 1
  local step
  step="$(status_step)" || return 1
  (( step >= 1 && step <= 26000 )) || return 1
  [[ -s "$RUN_ROOT/checkpoint_latest.pt" ]]
}

status_is_done() {
  status_shape_ok || return 1
  grep -Eq '^[[:space:]]*"done":[[:space:]]*true,?[[:space:]]*$' "$STATUS" || return 1
  local step
  step="$(status_step)" || return 1
  [[ "$step" == '26000' ]]
}

training_needed=true
if [[ -e "$RUN_ROOT" ]]; then
  [[ -d "$RUN_ROOT" ]] || {
    echo "ERROR: run root is not a directory: $RUN_ROOT" >&2
    exit 96
  }
  if [[ -n "$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    [[ -s "$STATUS" && -s "$RUN_ROOT/checkpoint_latest.pt" ]] || {
      echo 'ERROR: nonempty run root lacks a validated status/latest pair; refusing to resume or overwrite' >&2
      exit 96
    }
    if status_is_clean; then
      training_needed=true
    elif status_is_done; then
      training_needed=false
    else
      echo 'ERROR: nonempty run root has ambiguous lifecycle evidence; refusing to resume or overwrite' >&2
      exit 96
    fi
  fi
fi

if [[ "$training_needed" == true ]]; then
  mkdir -p "$RUN_ROOT" "$ATTEMPT_LOG_ROOT"
  set +e
  SCENE=armadillo MODEL="$MODEL" PROFILE=full CACHE="$CACHE" CHECKPOINT_ROOT="$CHECKPOINT_ROOT" CHECKPOINT_NAME="$NAME" \
    srun --ignore-signals=TERM --nodes=1 --ntasks=1 --cpus-per-task=4 --exact --export=ALL \
      bash scripts/run_airsas_comparison.sh 2>&1 | tee -a "$ATTEMPT_LOG"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  (( pipeline_status[0] == 0 )) || { echo "ERROR: srun comparison task failed with status ${pipeline_status[0]}" >&2; exit "${pipeline_status[0]}"; }
  (( pipeline_status[1] == 0 )) || { echo "ERROR: attempt log tee failed with status ${pipeline_status[1]}" >&2; exit "${pipeline_status[1]}"; }
  if status_is_clean; then
    echo 'AIRSAS_ARMA5_ADAPTIVE_CLEAN_STOP'
    exit 0
  fi
  status_is_done || {
    echo 'ERROR: successful comparison task returned ambiguous lifecycle evidence' >&2
    exit 96
  }
fi

[[ -s "$RUN_ROOT/checkpoint_best.pt" ]] || {
  echo 'ERROR: checkpoint_best.pt is missing or empty' >&2
  exit 96
}

run_eval() {
  local role="$1"
  local output="$2"
  [[ -s "$output" ]] && return 0
  python3 -B train_sas.py \
    --cache "$CACHE" \
    --model "$MODEL" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --checkpoint-name "$NAME" \
    --resume "$RUN_ROOT/checkpoint_best.pt" \
    --eval-only \
    --evaluation-role "$role" \
    --eval-pings 0 \
    --eval-bins 0 \
    --require-explicit-splits \
    --sh-degree 3 \
    --num-rays 4900 \
    --max-bins 110 \
    --opacity-scale 500 \
    --lambertian-ratio 0 \
    --normal-step 0.0032 \
    --signal-scale 10 \
    --grad-clip 1.0 \
    --beamwidth-deg 30 \
    --sh-direction rx_to_point \
    --seed 42 \
    --device cuda
  [[ -s "$output" ]] || {
    echo "ERROR: $role readout did not produce a nonempty output: $output" >&2
    exit 96
  }
}

run_eval validation "$RUN_ROOT/validation_eval.json"
run_eval test "$RUN_ROOT/test_eval.json"
[[ -s "$RUN_ROOT/selected_readout.json" && -s "$STATUS" ]] || {
  echo 'ERROR: selected_readout.json and status.json must be nonempty before final PASS' >&2
  exit 96
}
[[ -s "$RUN_ROOT/validation_eval.json" && -s "$RUN_ROOT/test_eval.json" ]] || {
  echo 'ERROR: both full readout outputs must be nonempty before final PASS' >&2
  exit 96
}
echo 'AIRSAS_ARMA5_ADAPTIVE_FULL_PASS'
