#!/bin/bash
# Full-scale GeRaF B7873200 manager lane.
#
# This is a single, one-GPU Inferno allocation charged to gts-jromberg3-ece.  It prepares the complete
# train/validation cache, runs the 50,000-update fit, and performs a terminal
# checkpoint/readout gate while retaining one GNU-time record per phase.  The
# manager remains the only submitter. Every continuation is phase-aware and
# reuses only existing durable preparation/checkpoint/readout artifacts.
#SBATCH --job-name=mgr_geraf_b7873200_fullscale_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
# A single gpu-h100 is limited to an 8:1 CPU:GPU ratio on PACE.
# Keep the one-GPU allocation at the validated maximum of eight CPUs.
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --tmp=20G
# Inferno supplies the requested 12-hour allocation; the 50,000-update budget
# and durable cache/checkpoint identity remain unchanged across continuations.
#SBATCH --time=12:00:00
#SBATCH --signal=TERM@600
#SBATCH --export=NONE

set -euo pipefail
umask 077

run_mode=fresh
case "$#" in
  0) ;;
  1)
    [[ "$1" == "resume" ]] || {
      echo 'ERROR: full-scale GeRaF accepts no arguments or the explicit resume mode' >&2
      exit 64
    }
    run_mode=resume
    ;;
  *)
    echo 'ERROR: full-scale GeRaF accepts no arguments or the explicit resume mode' >&2
    exit 64
    ;;
esac

[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: one non-array Slurm allocation is required' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == "inferno" && "${SLURM_JOB_PARTITION:-}" == "gpu-h100" ]] || {
  echo 'ERROR: full-scale GeRaF requires the Inferno gpu-h100 allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_ACCOUNT:-}" == "gts-jromberg3-ece" ]] || {
  echo 'ERROR: full-scale GeRaF requires account gts-jromberg3-ece' >&2
  exit 94
}

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
ROLE_MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json
CELL_ROOT=/storage/scratch1/1/dbao31/rift_b7873200_geraf_fullscale_v1
CACHE_ROOT="$CELL_ROOT/prepared_targets"
FIT_ROOT="$CELL_ROOT/fit"
LAUNCHER_LOG_ROOT="$CELL_ROOT/launcher_logs"
EXECUTION_ID=geraf_b7873200_fullscale_v1
PREP_STATE="$CACHE_ROOT/geraf_b7873200_preparation_state.json"
PREP_TIME_LOG="$LAUNCHER_LOG_ROOT/$SLURM_JOB_ID/preparation_time.txt"
FIT_ALLOCATION_TIME_LOG="$LAUNCHER_LOG_ROOT/$SLURM_JOB_ID/fit_allocation_time.txt"
FIT_PROCESS_TIME_LOG="$LAUNCHER_LOG_ROOT/$SLURM_JOB_ID/fit_process_time.txt"
READOUT_TIME_LOG="$LAUNCHER_LOG_ROOT/$SLURM_JOB_ID/readout_time.txt"
READOUT_JSON="$FIT_ROOT/fullscale_readout.json"

if [[ "$run_mode" == "fresh" ]]; then
  [[ ! -e "$CELL_ROOT" ]] || {
    echo "ERROR: fresh full-scale run requires an unused output root: $CELL_ROOT" >&2
    exit 94
  }
else
  [[ -d "$CELL_ROOT" && -d "$CACHE_ROOT" ]] || {
    echo 'ERROR: resume requires the existing full-scale cell and preparation roots' >&2
    exit 94
  }
  [[ ! -e "$READOUT_JSON" ]] || {
    echo 'ERROR: completed full-scale readout is already published; do not rerun this identity' >&2
    exit 94
  }
fi

mkdir -p "$LAUNCHER_LOG_ROOT/$SLURM_JOB_ID"
ATTEMPT_LOG="$LAUNCHER_LOG_ROOT/$SLURM_JOB_ID/execution_context.log"
printf '%s\n' \
  "EXECUTION_ID=$EXECUTION_ID" \
  "JOB_ID=$SLURM_JOB_ID" \
  "MODE=$run_mode" \
  "CACHE_ROOT=$CACHE_ROOT" \
  "FIT_ROOT=$FIT_ROOT" > "$ATTEMPT_LOG"

module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT
cd "$PROJECT"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export SLURM_EXPORT_ENV=ALL

run_python_validator() {
  local label="$1"
  local marker="$2"
  local script="$3"
  local log="$LAUNCHER_LOG_ROOT/$SLURM_JOB_ID/${label}.log"
  echo "GERAF_B7873200_FULLSCALE_${label^^}_START"
  set +e
  python -B -u "$script" 2>&1 | tee "$log"
  local -a statuses=("${PIPESTATUS[@]}")
  local script_status="${statuses[0]}"
  local tee_status="${statuses[1]}"
  set -e
  if (( tee_status != 0 || script_status != 0 )); then
    echo "ERROR: $label validator failed (script=$script_status tee=$tee_status); retaining evidence" >&2
    return 95
  fi
  grep -Fq "$marker" "$log" || {
    echo "ERROR: $label validator omitted terminal marker: $marker" >&2
    return 95
  }
  echo "GERAF_B7873200_FULLSCALE_${label^^}_PASS"
}

run_timed_stage() {
  local label="$1"
  local time_log="$2"
  local log="$LAUNCHER_LOG_ROOT/$SLURM_JOB_ID/${label}.log"
  shift 2
  echo "GERAF_B7873200_FULLSCALE_${label^^}_START"
  set +e
  /usr/bin/time -v -o "$time_log" "$@" 2>&1 | tee "$log"
  local -a statuses=("${PIPESTATUS[@]}")
  local command_status="${statuses[0]}"
  local tee_status="${statuses[1]}"
  set -e
  if (( tee_status != 0 )); then
    echo "ERROR: $label log stream failed (command=$command_status tee=$tee_status); retaining evidence" >&2
    return 95
  fi
  if (( command_status != 0 )); then
    return "$command_status"
  fi
  [[ -s "$time_log" ]] || {
    echo "ERROR: $label did not produce its GNU-time record" >&2
    return 95
  }
  echo "GERAF_B7873200_FULLSCALE_${label^^}_PASS"
}

preparation_status() {
  [[ -s "$PREP_STATE" ]] || return 1
  python -B - "$PREP_STATE" <<'PY'
import json
import sys
from pathlib import Path

from rift.geraf_b7873200_preparation_state import validate_preparation_state

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
validate_preparation_state(payload, expected_total=4200)
print(payload["status"])
PY
}

fit_checkpoint_phase() {
  python -B - "$FIT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
run_config_path = root / "run_config.json"
latest_path = root / "checkpoint_latest.pth.tar"
final_path = root / "checkpoint_final.pth.tar"
best_path = root / "checkpoint_best.pth.tar"
if not root.is_dir():
    print("none")
    raise SystemExit(0)
if not run_config_path.is_file():
    print("none" if not any(path.is_file() for path in (latest_path, final_path, best_path)) else "invalid")
    raise SystemExit(0)
try:
    config = json.loads(run_config_path.read_text(encoding="utf-8"))
    identity = config.get("run_identity")
    validation = identity.get("validation") if isinstance(identity, dict) else None
    if not isinstance(identity, dict) or identity.get("steps") != 50000:
        print("invalid")
        raise SystemExit(0)
    if not isinstance(validation, dict) or validation.get("every_optimizer_steps") != 1000:
        print("invalid")
        raise SystemExit(0)
    checkpoint_path = next((path for path in (latest_path, final_path) if path.is_file()), None)
    if checkpoint_path is None:
        print("invalid")
        raise SystemExit(0)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        print("invalid")
        raise SystemExit(0)
    step = checkpoint.get("step")
    complete = checkpoint.get("complete") is True
    pending = checkpoint.get("pending_validation_step")
    if complete and step == 50000 and pending is None:
        print("terminal" if all(path.is_file() for path in (latest_path, final_path, best_path)) else "terminal_checkpoint_only")
    elif isinstance(step, int) and step >= 0 and step < 50000 and not complete:
        print("incomplete")
    else:
        print("invalid")
except Exception:
    print("invalid")
PY
}

has_existing_phase_time_logs() {
  local label="$1"
  find "$LAUNCHER_LOG_ROOT" -type f -name "${label}_time.txt" -size +0c -print -quit 2>/dev/null | grep -q .
}

echo 'GERAF_B7873200_FULLSCALE_SOURCE_PREFLIGHT_START'
run_python_validator source_preflight \
  GERAF_B7873200_FULLSCALE_SOURCE_PASS \
  scripts/validate_geraf_b7873200_fullscale_source.py

run_python_validator protocol_preflight \
  'GeRaF B7873200 sealed protocol passed:' \
  scripts/validate_geraf_b7873200_protocol.py
run_python_validator adapter_preflight \
  'GeRaF B7873200 sealed adapter passed:' \
  scripts/validate_geraf_b7873200_adapter.py
run_python_validator acquisition_preflight \
  'GeRaF B7873200 acquisition compatibility passed:' \
  scripts/validate_geraf_b7873200_acquisition.py
run_python_validator signal_operator_preflight \
  'GeRaF signal-operator gates:' \
  scripts/validate_geraf_signal_operator.py
run_python_validator trainer_preflight \
  'GeRaF B7873200 trainer checks passed:' \
  scripts/validate_geraf_b7873200_trainer.py
run_python_validator entrypoint_preflight \
  'GeRaF B7873200 actual-entrypoint regression passed:' \
  scripts/validate_geraf_b7873200_entrypoint.py

echo 'GERAF_B7873200_FULLSCALE_CACHE_TRANSITION_START'
run_python_validator cache_transition \
  RANGE_OPERATOR_INFERENCE_CACHE_TRANSITION_PASS \
  scripts/validate_range_operator_inference_cache_transition.py

mkdir -p "$CACHE_ROOT"
prepare_needed=1
if [[ "$run_mode" == "resume" ]]; then
  existing_preparation_status="$(preparation_status 2>/dev/null || true)"
  case "$existing_preparation_status" in
    complete|complete_clean_stop_before_fit)
      prepare_needed=0
      ;;
    running|partial|partial_clean_stop)
      prepare_needed=1
      ;;
    *)
      echo 'ERROR: resume requires a valid existing GeRaF preparation-state record' >&2
      exit 95
      ;;
  esac
fi

if (( prepare_needed )); then
  echo 'GERAF_B7873200_FULLSCALE_PREPARE_START'
  set +e
  run_timed_stage preparation "$PREP_TIME_LOG" \
    python -B -u scripts/prepare_geraf_b7873200_targets.py \
      --npz-path "$NPZ" \
      --role-manifest "$ROLE_MANIFEST" \
      --cache-root "$CACHE_ROOT" \
      --scene-extent 0.15 \
      --n-azimuth 32 --n-elevation 32 --n-depth 32 \
      --aperture-scale 1.0 --phase-sign -1.0 \
      --backend range --device cuda --compute-dtype float64 \
      --point-chunk 4096 --pair-chunk 32 --freq-chunk 75 \
      --kernel-width 20 --oversample 2 --resume
  preparation_exit=$?
  set -e
  if (( preparation_exit != 0 )); then
    if (( preparation_exit == 143 )); then
      stopped_preparation_status="$(preparation_status 2>/dev/null || true)"
      case "$stopped_preparation_status" in
        partial_clean_stop|complete_clean_stop_before_fit)
          echo "GERAF_B7873200_FULLSCALE_PREPARATION_CLEAN_STOP_RESUME_REQUIRED status=$stopped_preparation_status"
          exit 143
          ;;
      esac
    fi
    echo "ERROR: preparation failed without a resumable preparation-state record (status=$preparation_exit)" >&2
    exit "$preparation_exit"
  fi
fi

final_preparation_status="$(preparation_status 2>/dev/null || true)"
case "$final_preparation_status" in
  complete|complete_clean_stop_before_fit) ;;
  *)
    echo "ERROR: full-scale preparation is not complete (status=$final_preparation_status)" >&2
    exit 95
    ;;
esac

[[ -f "$CACHE_ROOT/b7873200_sealed_protocol.json" && \
   -f "$CACHE_ROOT/geraf_b7873200_recipe.json" && \
   -f "$CACHE_ROOT/geraf_b7873200_manifest.json" && \
   -f "$CACHE_ROOT/geraf_b7873200_stats.json" && \
   -f "$CACHE_ROOT/geraf_b7873200_acquisition.npz" ]] || {
  echo 'ERROR: full-scale preparation did not produce the complete sealed cache metadata' >&2
  exit 95
}

mkdir -p "$FIT_ROOT"
fit_phase="$(fit_checkpoint_phase)"
case "$fit_phase" in
  terminal)
    fit_needed=0
    ;;
  none|incomplete|terminal_checkpoint_only)
    fit_needed=1
    ;;
  *)
    echo "ERROR: existing GeRaF fit artifacts are invalid (phase=$fit_phase)" >&2
    exit 95
    ;;
esac

if (( fit_needed )); then
  if [[ "$fit_phase" == "none" ]]; then
    fit_resume_args=(--no-resume)
  else
    fit_resume_args=(--resume)
  fi

  driver_ready_file="$LAUNCHER_LOG_ROOT/$SLURM_JOB_ID/fit_ready"
  echo 'GERAF_B7873200_FULLSCALE_FIT_START'
  set +e
  run_timed_stage fit "$FIT_ALLOCATION_TIME_LOG" \
    srun --ignore-signals=TERM --nodes=1 --ntasks=1 --cpus-per-task=8 --exact --export=ALL \
      python -B -u scripts/run_geraf_b78716_smallfit_timed.py \
        --time-log "$FIT_PROCESS_TIME_LOG" \
        --ready-file "$driver_ready_file" -- \
        python -B -u train_geraf.py \
          --npz-path "$NPZ" \
          --role-manifest "$ROLE_MANIFEST" \
          --cache-root "$CACHE_ROOT" \
          --checkpoint-dir "$FIT_ROOT" \
          --seed 42 --scene-extent 0.15 \
          --n-azimuth 32 --n-elevation 32 --n-depth 32 \
          --aperture-scale 1.0 --phase-sign -1.0 \
          --steps 50000 --validation-every 1000 \
          --compute-dtype float64 --oversample 2 --kernel-width 20 \
          --pair-chunk 32 --point-chunk 4096 --device cuda \
          "${fit_resume_args[@]}"
  fit_exit=$?
  set -e
  if (( fit_exit != 0 )); then
    if (( fit_exit == 143 )); then
      stopped_fit_phase="$(fit_checkpoint_phase)"
      case "$stopped_fit_phase" in
        incomplete)
          echo 'GERAF_B7873200_FULLSCALE_FIT_CLEAN_STOP_RESUME_REQUIRED phase=incomplete'
          exit 143
          ;;
        terminal_checkpoint_only)
          echo 'GERAF_B7873200_FULLSCALE_FIT_CLEAN_STOP_RESUME_REQUIRED phase=terminal_checkpoint_only'
          exit 143
          ;;
        terminal)
          ;;
        *)
          echo "ERROR: fit stopped without a resumable checkpoint phase (phase=$stopped_fit_phase)" >&2
          exit 95
          ;;
      esac
    else
      echo "ERROR: full-scale fit failed (status=$fit_exit)" >&2
      exit "$fit_exit"
    fi
  else
    [[ -f "$driver_ready_file" ]] || {
      echo 'ERROR: full-scale trainer did not publish the timed-wrapper readiness marker' >&2
      exit 95
    }
    [[ -s "$FIT_PROCESS_TIME_LOG" ]] || {
      echo 'ERROR: full-scale timed wrapper did not produce its child GNU-time record' >&2
      exit 95
    }
  fi
fi

fit_phase="$(fit_checkpoint_phase)"
[[ "$fit_phase" == "terminal" ]] || {
  echo "ERROR: full-scale fit did not reach a terminal checkpoint phase (phase=$fit_phase)" >&2
  exit 95
}

echo 'GERAF_B7873200_FULLSCALE_READOUT_START'
set +e
run_timed_stage readout "$READOUT_TIME_LOG" \
  python -B - "$FIT_ROOT" "$READOUT_JSON" "$LAUNCHER_LOG_ROOT" "$READOUT_TIME_LOG" <<'PY'
import json
import os
import sys
from pathlib import Path

import torch


fit_root, output_path, launcher_log_root, readout_time = map(Path, sys.argv[1:])
final_path = fit_root / "checkpoint_final.pth.tar"
latest_path = fit_root / "checkpoint_latest.pth.tar"
best_path = fit_root / "checkpoint_best.pth.tar"
run_config_path = fit_root / "run_config.json"
history_path = fit_root / "history.json"
for path in (final_path, latest_path, best_path, run_config_path, history_path):
    if not path.is_file():
        raise SystemExit(f"missing terminal full-scale artifact: {path}")

final = torch.load(final_path, map_location="cpu", weights_only=False)
latest = torch.load(latest_path, map_location="cpu", weights_only=False)
if not isinstance(final, dict) or not isinstance(latest, dict):
    raise SystemExit("terminal GeRaF checkpoints must be mappings")
if final.get("complete") is not True or latest.get("complete") is not True:
    raise SystemExit("terminal GeRaF checkpoints must be complete")
if final.get("step") != 50_000 or latest.get("step") != 50_000:
    raise SystemExit("terminal GeRaF checkpoints must end at step 50000")
if final.get("pending_validation_step") is not None or latest.get("pending_validation_step") is not None:
    raise SystemExit("terminal GeRaF checkpoints cannot retain pending validation")
if final.get("history") != latest.get("history"):
    raise SystemExit("final/latest GeRaF validation histories disagree")
history = final.get("history")
if not isinstance(history, list) or [row.get("step") for row in history] != list(range(1000, 50_001, 1000)):
    raise SystemExit("full-scale GeRaF history must contain every validation step 1000..50000")
run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
identity = run_config.get("run_identity")
if not isinstance(identity, dict) or identity.get("steps") != 50_000:
    raise SystemExit("run configuration is not the full 50000-step identity")
validation = identity.get("validation")
if not isinstance(validation, dict) or validation.get("every_optimizer_steps") != 1000:
    raise SystemExit("run configuration has the wrong validation cadence")


def existing_time_logs(label: str) -> list[str]:
    return sorted(
        str(path)
        for path in launcher_log_root.rglob(f"{label}_time.txt")
        if path.is_file() and path.stat().st_size > 0
    )


phase_time_logs = {
    "preparation": existing_time_logs("preparation"),
    "fit_allocation": existing_time_logs("fit_allocation"),
    "fit_process": existing_time_logs("fit_process"),
    "readout": existing_time_logs("readout"),
}
for phase in ("preparation", "fit_allocation", "fit_process"):
    if not phase_time_logs[phase]:
        raise SystemExit(f"missing retained GNU-time record for phase: {phase}")
current_readout = str(readout_time)
if current_readout not in phase_time_logs["readout"]:
    phase_time_logs["readout"].append(current_readout)
phase_time_logs["readout"].sort()
output = {
    "schema": "rift_geraf_b7873200_fullscale_terminal_readout_v1",
    "status": "terminal_training_readout",
    "production_clearance": False,
    "steps": 50_000,
    "validation_steps": [1000, 50_000],
    "validation_views": 1000,
    "reserved_test_accessed": False,
    "unused_views_accessed": False,
    "checkpoint_final": str(final_path),
    "checkpoint_latest": str(latest_path),
    "checkpoint_best": str(best_path),
    "phase_time_logs": phase_time_logs,
}
temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output_path)
print("GERAF_B7873200_FULLSCALE_READOUT_PASS", flush=True)
PY
readout_exit=$?
set -e
if (( readout_exit != 0 )); then
  if (( readout_exit == 143 )) && [[ ! -e "$READOUT_JSON" ]] && \
     has_existing_phase_time_logs preparation && \
     has_existing_phase_time_logs fit_allocation && \
     has_existing_phase_time_logs fit_process && \
     [[ "$(fit_checkpoint_phase)" == "terminal" ]]; then
    echo 'GERAF_B7873200_FULLSCALE_READOUT_CLEAN_STOP_RESUME_REQUIRED phase=terminal_fit'
    exit 143
  fi
  echo "ERROR: terminal readout failed without a phase-safe resume condition (status=$readout_exit)" >&2
  exit "$readout_exit"
fi

echo 'GERAF_B7873200_FULLSCALE_COMPLETE'
