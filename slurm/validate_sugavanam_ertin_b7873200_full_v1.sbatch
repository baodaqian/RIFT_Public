#!/bin/bash
# Proposed full B7873200 Sugavanam--Ertin package.
# The experiment manager reviews and submits this one Inferno allocation on
# account gts-jromberg3-ece.  It
# is not a production-clearance job and does not tune the topology gate.
# Auto-resume is disabled: continuation is literal and operator-selected.
#SBATCH --job-name=mgr_se_b7873200_full_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --tmp=24G
#SBATCH --time=12:00:00
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0,atl1-1-01-002-8-0,atl1-1-02-006-34-0
#SBATCH --signal=B:TERM@600
#SBATCH --export=NONE

set -euo pipefail
umask 077

# Install TERM handling before source/GPU preflights. Signals set a flag and
# are acted on at explicit boundaries; no driver PID is assumed to exist.
driver_pid=''
tee_pid=''
term_requested=0
term_forwarded=0
forward_term() {
  term_requested=1
  if [[ -n "$driver_pid" ]]; then
    if (( term_forwarded == 0 )); then
      term_forwarded=1
      echo 'SE_B7873200_FULL_FORWARDING_TERM_TO_DRIVER' >&2
      kill -TERM -- "-$driver_pid" 2>/dev/null || kill -TERM "$driver_pid" 2>/dev/null || true
    fi
  else
    if [[ -n "$tee_pid" ]]; then
      echo 'SE_B7873200_FULL_TERM_DURING_TEE_DRAIN' >&2
    else
      echo 'SE_B7873200_FULL_TERM_BEFORE_DRIVER_START' >&2
    fi
  fi
}
abort_before_driver_if_requested() {
  if (( term_requested )); then
    trap - TERM INT
    echo 'SE_B7873200_FULL_TERM_BEFORE_DRIVER_EXIT' >&2
    exit 143
  fi
}
trap forward_term TERM INT

compute_cap_mode=0
timeout_recovery_mode=0
case "$#" in
  0) resume_mode=0 ;;
  1)
    if [[ "$1" == "resume" ]]; then
      resume_mode=1
    elif [[ "$1" == "stage1-compute-cap30" ]]; then
      resume_mode=1
      compute_cap_mode=1
    elif [[ "$1" == "stage1-compute-cap30-recover-timeout12995517" ]]; then
      resume_mode=1
      compute_cap_mode=1
      timeout_recovery_mode=1
    else
      echo 'ERROR: full package accepts zero arguments, literal resume, stage1-compute-cap30, or stage1-compute-cap30-recover-timeout12995517' >&2
      exit 64
    fi
    ;;
  *) echo 'ERROR: full package accepts zero arguments, literal resume, stage1-compute-cap30, or stage1-compute-cap30-recover-timeout12995517' >&2; exit 64 ;;
esac

[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: one non-array Slurm allocation is required' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == "inferno" && "${SLURM_JOB_PARTITION:-}" == "gpu-rtx6000" ]] || {
  echo 'ERROR: full package requires Inferno gpu-rtx6000' >&2
  exit 94
}
[[ "${SLURM_JOB_ACCOUNT:-}" == "gts-jromberg3-ece" ]] || {
  echo 'ERROR: full package requires account gts-jromberg3-ece' >&2
  exit 94
}
[[ "${SLURM_CPUS_PER_TASK:-}" == "6" && "${SLURM_MEM_PER_NODE:-}" == "65536" ]] || {
  echo 'ERROR: full package requires 6 CPUs and 64 GiB RAM' >&2
  exit 94
}

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
PARENT_MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json
FULL_ROOT=/storage/scratch1/1/dbao31/rift_b7873200_sugavanam_ertin_full3200_replacement_v1
FULL_RUN_NAME=b78710k_sugavanam_ertin_full3200_stage1_stage2_replacement_v1
RUN_ROOT="$FULL_ROOT/$FULL_RUN_NAME"
STAGE1_CHECKPOINT_NAME=sugavanam_ertin_b7873200_stage1_full3200_replacement_v1
STAGE1_DIR="$RUN_ROOT/stage1/$STAGE1_CHECKPOINT_NAME"
STAGE1_LATEST="$STAGE1_DIR/checkpoint_latest.pth.tar"
STAGE1_FINAL="$STAGE1_DIR/checkpoint_final.pth.tar"
STAGE1_CLEAN="$STAGE1_DIR/full_stage1_clean_interruption.json"
STAGE1_BUNDLE="$RUN_ROOT/stage1/sugavanam_ertin_b7873200_stage1_final_v1.pth.tar"
REPORT="$RUN_ROOT/full3200_report.json"
CAP_RUN_NAME=b78710k_sugavanam_ertin_stage1_compute_capped_epoch30_v1
CAP_ROOT="$FULL_ROOT/$CAP_RUN_NAME"
CAP_STAGE1_CHECKPOINT_NAME=sugavanam_ertin_b7873200_stage1_compute_capped_epoch30_v1
CAP_STAGE1_DIR="$CAP_ROOT/stage1/$CAP_STAGE1_CHECKPOINT_NAME"
CAP_STAGE1_LATEST="$CAP_STAGE1_DIR/checkpoint_latest.pth.tar"
CAP_STAGE1_CLEAN="$CAP_STAGE1_DIR/full_stage1_clean_interruption.json"
CAP_STAGE1_FINAL="$CAP_STAGE1_DIR/checkpoint_final.pth.tar"
CAP_STAGE1_BUNDLE="$CAP_ROOT/stage1/sugavanam_ertin_b7873200_stage1_final_v1.pth.tar"
CAP_REPORT="$CAP_ROOT/stage1_compute_capped_epoch30_report.json"
LAUNCHER_LOG_ROOT="$FULL_ROOT/launcher_logs"
if (( timeout_recovery_mode )); then
  echo 'SE_B7873200_FULL_MODE=stage1_compute_capped_epoch30_recover_timeout12995517'
elif (( compute_cap_mode )); then
  echo 'SE_B7873200_FULL_MODE=stage1_compute_capped_epoch30'
fi

for source in \
  "$PROJECT/train.py" \
  "$PROJECT/rift/npz_dataset.py" \
  "$PROJECT/rift/range_operator.py" \
  "$PROJECT/rift/sugavanam_ertin.py" \
  "$PROJECT/rift/sugavanam_ertin_a320_stabilized.py" \
  "$PROJECT/rift/sugavanam_ertin_b7873200_full.py" \
  "$PROJECT/rift/sugavanam_ertin_b7873200_stage1.py" \
  "$PROJECT/rift/sugavanam_ertin_b7873200_stage2_v1.py" \
  "$PROJECT/rift/sugavanam_ertin_stage2_runtime_v1.py" \
  "$PROJECT/rift/sugavanam_ertin_validzero.py" \
  "$PROJECT/train_sugavanam_ertin.py" \
  "$PROJECT/train_sugavanam_ertin_smoke.py" \
  "$PROJECT/train_sugavanam_ertin_stage2.py" \
  "$PROJECT/scripts/validate_sugavanam_ertin_b7873200_full_source.py" \
  "$PROJECT/scripts/postflight_sugavanam_ertin_b7873200_full.py" \
  "$PROJECT/scripts/validate_sugavanam_ertin_b7873200_stage1.py" \
  "$PROJECT/scripts/validate_sugavanam_ertin_b7873200_stage1_operator.py" \
  "$PROJECT/scripts/validate_sugavanam_ertin_b7873200_stage2_static.py" \
  "$PROJECT/scripts/validate_sugavanam_ertin_b7873200_stage2_v1.py" \
  "$NPZ" \
  "$PARENT_MANIFEST" \
  /usr/bin/time \
  /usr/bin/setsid \
  /usr/bin/mkfifo; do
  [[ -f "$source" ]] || { echo "ERROR: missing reviewed full-package input: $source" >&2; exit 94; }
done
abort_before_driver_if_requested

stage1_latest_resume_ok() {
  local latest="${1:-$STAGE1_LATEST}"
  local clean="${2:-$STAGE1_CLEAN}"
  [[ -f "$latest" && -f "$clean" ]] || return 1
  python - "$clean" "$latest" <<'PY'
import json
import sys
from pathlib import Path

clean = Path(sys.argv[1])
latest = Path(sys.argv[2])
try:
    record = json.loads(clean.read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
checkpoint = Path(record.get("checkpoint_path", ""))
if (record.get("state") != "clean_interrupted"
        or record.get("resume_allowed") is not True
        or checkpoint.resolve() != latest.resolve()
        or not latest.is_file()):
    raise SystemExit(1)
PY
}

timeout_recovery_checkpoint_ok() {
  local latest="$1"
  python - "$latest" <<'PY'
from collections.abc import Mapping
from pathlib import Path
import sys
import torch

latest = Path(sys.argv[1]).resolve()
if not latest.is_file():
    raise SystemExit("ERROR: timeout recovery source checkpoint is absent")

try:
    checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
except Exception as exc:
    raise SystemExit(f"ERROR: timeout recovery checkpoint is unreadable: {exc}") from exc
if not isinstance(checkpoint, Mapping):
    raise SystemExit("ERROR: timeout recovery checkpoint is not a mapping")
if checkpoint.get("epoch") != 16:
    raise SystemExit("ERROR: timeout recovery checkpoint epoch is not exactly 16")

for key in ("optimizer_state_dict", "scheduler_state_dict", "rng_state"):
    value = checkpoint.get(key)
    if not isinstance(value, Mapping) or not value:
        raise SystemExit(f"ERROR: timeout recovery checkpoint lacks required {key}")
rng_state = checkpoint["rng_state"]
if not {"python", "numpy", "torch_cpu", "freq_cpu"}.issubset(rng_state):
    raise SystemExit("ERROR: timeout recovery checkpoint RNG state is incomplete")

expected_contract = {
    "schema": "rift_checkpoint_execution_contract_v1",
    "label": "rift_sugavanam_ertin_b7873200_stage1_v1",
    "observation": {
        "data_format": "npz",
        "sealed_npz_protocol": True,
        "num_train": 3200,
        "num_validation": 1000,
        "num_reserved_test": 1000,
        "num_freq_wanted": 600,
        "validation_cap_axis": None,
        "validation_from_tail": False,
    },
    "scene": {
        "representation": "grid",
        "extent_m": 0.15,
        "granularity": 48,
        "initial_scale": 0.0,
        "backprojection_views": 400,
        "shell_init_radius_m": 0.0,
        "normalize_scene_scale": False,
        "sh_max_degree": 10,
        "sh_init_degree": 0,
    },
    "physics": {
        "forward_operator": "range",
        "range_model": "product",
        "compute_dtype": "float64",
        "phase_sign": -1.0,
        "coordinate_source": "npz_per_view_positions",
        "num_rx": 16,
        "num_tx": 16,
        "point_chunk": 65536,
        "pair_chunk": 64,
    },
    "fit": {
        "epochs": 150,
        "loss": "complex",
        "step_every": 1,
        "clip_grad_norm": 0.0,
        "learn_global_gain": True,
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "l1_weight": 3.0e-7,
        "sh_smooth_weight": 0.0,
        "regularizer_normalization": "active_mean",
        "adam_eps": 1.0e-20,
        "checkpoint_metric": "val",
        "seed": 42,
        "scheduler": {"t0": 10, "t_mult": 2, "eta_min": 1.0e-6},
        "pruning": {
            "every": 0,
            "threshold": 0.0,
            "criterion": "energy",
            "start_epoch": 0,
            "mode": "mass",
            "target_active": 0,
            "end_epoch": 0,
            "min_active": 0,
        },
        "view_weight_alpha": 0.0,
        "view_weight_max_ratio": 0.0,
        "magnitude_weight": 0.0,
        "magnitude_warmup_epochs": 0,
        "resume_requires_full_state": True,
        "optimizer": {
            "name": "AdamW",
            "scene_learning_rate": 0.003,
            "gain_learning_rate": 0.003,
            "betas": [0.9, 0.999],
            "eps": 1.0e-20,
            "weight_decay": 0.0,
        },
    },
}
if checkpoint.get("execution_contract") != expected_contract:
    raise SystemExit("ERROR: timeout recovery checkpoint execution contract differs")
print("SE_B7873200_FULL_TIMEOUT_RECOVERY_CHECKPOINT_PASS")
PY
}

if (( compute_cap_mode )); then
  if (( timeout_recovery_mode )); then
    [[ "$CAP_STAGE1_LATEST" == "$CAP_ROOT/stage1/$CAP_STAGE1_CHECKPOINT_NAME/checkpoint_latest.pth.tar" ]] || {
      echo 'ERROR: timeout recovery source identity is not the exact compute-capped latest' >&2
      exit 94
    }
    [[ -f "$CAP_STAGE1_LATEST" ]] || {
      echo 'ERROR: timeout recovery requires the existing compute-capped Stage-1 latest' >&2
      exit 94
    }
    [[ ! -e "$CAP_STAGE1_FINAL" && ! -e "$CAP_STAGE1_BUNDLE" ]] || {
      echo 'ERROR: timeout recovery refuses terminal compute-capped artifacts' >&2
      exit 94
    }
    if [[ -d "$CAP_ROOT/stage2" ]] && [[ -n "$(find "$CAP_ROOT/stage2" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
      echo 'ERROR: timeout recovery refuses existing Stage-2 artifacts' >&2
      exit 94
    fi
    CAP_SOURCE_LATEST="$CAP_STAGE1_LATEST"
  elif [[ -e "$CAP_ROOT" ]] && [[ -n "$(find "$CAP_ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    # A continuation is allowed only from the cap root's own clean latest;
    # the driver performs the exact checkpoint/epoch validation.
    if ! [[ -f "$CAP_STAGE1_LATEST" && -f "$CAP_STAGE1_CLEAN" ]]; then
      echo 'ERROR: compute-capped output root is nonempty without its clean latest pair' >&2
      exit 94
    fi
    CAP_SOURCE_LATEST="$CAP_STAGE1_LATEST"
  else
    CAP_SOURCE_LATEST="$STAGE1_LATEST"
  fi
  if (( ! timeout_recovery_mode )); then
    stage1_latest_resume_ok "$CAP_SOURCE_LATEST" "${CAP_SOURCE_LATEST%/*}/full_stage1_clean_interruption.json" || {
      echo 'ERROR: compute-capped Stage-1 requires a clean Stage-1 latest pair' >&2
      exit 94
    }
  fi
elif (( resume_mode == 0 )); then
  [[ ! -e "$RUN_ROOT" ]] || { echo 'ERROR: fresh full package requires a new output identity' >&2; exit 94; }
else
  [[ ! -d "$RUN_ROOT/stage2/complete" ]] || {
    echo 'ERROR: completed or scientific-negative full package is terminal' >&2
    exit 94
  }
  if [[ -f "$REPORT" ]]; then
    python - "$REPORT" "$STAGE1_DIR" "$STAGE1_BUNDLE" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "clean_interruption":
    raise SystemExit("ERROR: only a clean-interruption full report may be resumed")
phase = payload.get("phase")
if phase not in {"stage1", "stage1_terminal_recovery", "stage2"}:
    raise SystemExit("ERROR: clean full report has no resumable phase")
run_root = Path(sys.argv[1]).parent
stage1_dir = Path(sys.argv[2])
bundle = Path(sys.argv[3])
if phase == "stage2":
    latest = run_root / "stage2/checkpoint_latest.pth.tar"
    lifecycle = json.loads((run_root / "stage2/lifecycle.json").read_text(encoding="utf-8"))
    if not latest.is_file() or lifecycle.get("state") != "clean_interrupted" or lifecycle.get("resume_allowed") is not True:
        raise SystemExit("ERROR: clean Stage-2 report lacks a validated lifecycle/latest pair")
    if Path(lifecycle.get("checkpoint_path", "")).resolve() != latest.resolve():
        raise SystemExit("ERROR: Stage-2 lifecycle checkpoint path does not match latest")
elif phase in {"stage1", "stage1_terminal_recovery"}:
    clean = stage1_dir / "full_stage1_clean_interruption.json"
    final = stage1_dir / "checkpoint_final.pth.tar"
    latest = stage1_dir / "checkpoint_latest.pth.tar"
    entry = run_root / "stage2_entry_clean_interruption.json"
    if phase == "stage1_terminal_recovery" and entry.is_file():
        entry_record = json.loads(entry.read_text(encoding="utf-8"))
        if (entry_record.get("schema") != "rift_sugavanam_ertin_b7873200_full_stage2_entry_v1"
                or entry_record.get("state") != "clean_interrupted"
                or entry_record.get("resume_allowed") is not True
                or Path(entry_record.get("resume_target", "")).resolve() != bundle.resolve()
                or not bundle.is_file()):
            raise SystemExit("ERROR: Stage-2 entry clean record does not name an existing bundle continuation")
        if (run_root / "stage2").exists() and any((run_root / "stage2").iterdir()):
            raise SystemExit("ERROR: Stage-2 entry continuation has unexpected Stage-2 artifacts")
    if clean.is_file():
        clean_record = json.loads(clean.read_text(encoding="utf-8"))
        checkpoint = Path(clean_record.get("checkpoint_path", ""))
        if clean_record.get("state") != "clean_interrupted" or clean_record.get("resume_allowed") is not True:
            raise SystemExit("ERROR: Stage-1 clean record is not resumable")
        if checkpoint.resolve() not in {latest.resolve(), final.resolve()} or not checkpoint.is_file():
            raise SystemExit("ERROR: Stage-1 clean record does not name an existing latest/final checkpoint")
    if not (clean.is_file() or final.is_file() or bundle.is_file()):
        raise SystemExit("ERROR: clean Stage-1 report lacks a checkpoint/bundle recovery artifact")
PY
  fi
  if ! [[ -f "$RUN_ROOT/stage2/checkpoint_latest.pth.tar" ]] && \
     ! stage1_latest_resume_ok && \
     ! [[ -f "$STAGE1_FINAL" ]] && \
     ! [[ -f "$STAGE1_BUNDLE" ]]; then
    echo 'ERROR: literal resume requires the full identity latest checkpoint' >&2
    exit 94
  fi
fi

module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

cd "$PROJECT"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
RIFT_SITE="$CONDA_PREFIX/lib/python3.10/site-packages"
EXPORT_SITE=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/sensor_fusion/lib/python3.10/site-packages
[[ -d "$RIFT_SITE" && -d "$EXPORT_SITE" ]] || { echo 'ERROR: required dependency sites are unavailable' >&2; exit 94; }
export SE_FULL_RIFT_SITE="$RIFT_SITE" SE_FULL_EXPORT_SITE="$EXPORT_SITE"
export PYTHONPATH="$PROJECT:$RIFT_SITE:$EXPORT_SITE"

if (( timeout_recovery_mode )); then
  timeout_recovery_checkpoint_ok "$CAP_SOURCE_LATEST" || {
    echo 'ERROR: timeout recovery checkpoint validation failed before driver' >&2
    exit 94
  }
fi

command -v nvidia-smi >/dev/null 2>&1 || { echo 'ERROR: nvidia-smi is unavailable' >&2; exit 94; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -B - <<'PY'
import os
from pathlib import Path
import sys
import torch
import skimage
from skimage.measure import marching_cubes

rift_site = Path(os.environ["SE_FULL_RIFT_SITE"]).resolve()
export_site = Path(os.environ["SE_FULL_EXPORT_SITE"]).resolve()
search_path = [str(Path(entry).resolve()) for entry in sys.path if entry]
if search_path.index(str(rift_site)) >= search_path.index(str(export_site)):
    raise SystemExit("ERROR: RIFT site-packages must precede scoped scikit-image site")
if rift_site not in Path(torch.__file__).resolve().parents:
    raise SystemExit("ERROR: Torch resolved outside the RIFT site")
if export_site not in Path(skimage.__file__).resolve().parents:
    raise SystemExit("ERROR: scikit-image resolved outside the scoped export site")
field = torch.zeros((2, 2, 2), dtype=torch.float32)
field[1, :, :] = 1.0
vertices, faces, _normals, _values = marching_cubes(field.numpy(), level=0.5)
if len(vertices) == 0 or len(faces) == 0:
    raise SystemExit("ERROR: marching_cubes dependency preflight returned an empty surface")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("ERROR: full package requires exactly one visible CUDA device")
print("SE_B7873200_FULL_GPU=" + torch.cuda.get_device_name(0))
print("SE_B7873200_FULL_GPU_TOTAL_MIB=" + str(torch.cuda.get_device_properties(0).total_memory // 2**20))
print("SE_B7873200_FULL_SKIMAGE=" + skimage.__version__)
PY
abort_before_driver_if_requested

echo 'SE_B7873200_FULL_SOURCE_PREFLIGHT_START'
python -B -u scripts/validate_sugavanam_ertin_b7873200_full_source.py
echo 'SE_B7873200_FULL_SOURCE_PREFLIGHT_PASS'
abort_before_driver_if_requested

run_validator() {
  local label="$1"
  local marker="$2"
  local script="$3"
  local log="$LAUNCHER_LOG_ROOT/${SLURM_JOB_ID}.${label}.preflight.log"
  mkdir -p -- "$LAUNCHER_LOG_ROOT"
  echo "VALIDATOR_START=$label"
  python -B -u "$script" 2>&1 | tee "$log"
  grep -Fq "$marker" "$log" || { echo "ERROR: validator $label omitted marker $marker" >&2; exit 95; }
  echo "VALIDATOR_PASS=$label"
  abort_before_driver_if_requested
}

run_validator stage1_static \
  'SE_B7873200_STAGE1_STATIC_PASS:' \
  scripts/validate_sugavanam_ertin_b7873200_stage1.py
run_validator stage1_operator \
  'SE_B7873200_STAGE1_OPERATOR_PASS:' \
  scripts/validate_sugavanam_ertin_b7873200_stage1_operator.py
run_validator stage2_static \
  'SE_B7873200_STAGE2_STATIC_PASS:' \
  scripts/validate_sugavanam_ertin_b7873200_stage2_static.py
run_validator stage2_lifecycle \
  'SE_B7873200_STAGE2_CPU_CONTRACT_PASS:' \
  scripts/validate_sugavanam_ertin_b7873200_stage2_v1.py
abort_before_driver_if_requested

if (( compute_cap_mode )); then
  resume_args=(
    --resume "$CAP_SOURCE_LATEST"
    --compute-capped-stage1-epoch30
    --compute-capped-stage1-source-checkpoint "$CAP_SOURCE_LATEST"
  )
  if (( timeout_recovery_mode )); then
    resume_args+=(--recover-timeout12995517)
  fi
elif (( resume_mode == 1 )); then
  if [[ -f "$RUN_ROOT/stage2/checkpoint_latest.pth.tar" ]]; then
    resume_args=(--resume "$RUN_ROOT/stage2/checkpoint_latest.pth.tar")
  elif [[ -f "$STAGE1_BUNDLE" ]]; then
    resume_args=(--resume "$STAGE1_BUNDLE")
  elif [[ -f "$STAGE1_FINAL" ]]; then
    resume_args=(--resume "$STAGE1_FINAL")
  elif stage1_latest_resume_ok; then
    resume_args=(--resume "$STAGE1_LATEST")
  fi
else
  resume_args=()
fi

abort_before_driver_if_requested
mkdir -p -- "$LAUNCHER_LOG_ROOT"
time_log="$LAUNCHER_LOG_ROOT/${SLURM_JOB_ID}.time.txt"
driver_log="$LAUNCHER_LOG_ROOT/${SLURM_JOB_ID}.driver.log"
driver_fifo="$LAUNCHER_LOG_ROOT/${SLURM_JOB_ID}.driver.fifo"
[[ ! -e "$driver_fifo" ]] || { echo 'ERROR: driver FIFO already exists' >&2; exit 94; }
abort_before_driver_if_requested
mkfifo -- "$driver_fifo"
tee "$driver_log" < "$driver_fifo" &
tee_pid=$!
if (( term_requested )); then
  set +e
  trap - TERM INT
  kill "$tee_pid" 2>/dev/null || true
  wait "$tee_pid" 2>/dev/null || true
  rm -f -- "$driver_fifo" "$driver_log" "$time_log"
  set -e
  echo 'SE_B7873200_FULL_TERM_BEFORE_DRIVER_EXIT' >&2
  exit 143
fi
/usr/bin/setsid /usr/bin/time -v -o "$time_log" python -B -u train_sugavanam_ertin.py \
  --checkpoint-root "$FULL_ROOT" \
  --npz-path "$NPZ" \
  --parent-role-manifest "$PARENT_MANIFEST" \
  --device cuda \
  "${resume_args[@]}" > "$driver_fifo" 2>&1 &
driver_pid=$!
if (( term_requested )); then
  forward_term
fi
set +e
wait "$driver_pid"
driver_status=$?
if (( term_requested && (driver_status == 143 || driver_status == 130) )); then
  # Bash wait can return the forwarding signal before the cooperative child
  # exits.  Reap exactly once more, without a polling loop or process-group
  # liveness spin; a 127 means the first wait already reaped the child.
  echo 'SE_B7873200_FULL_REAP_AFTER_INTERRUPTED_WAIT' >&2
  wait "$driver_pid"
  reap_status=$?
  if (( reap_status != 127 )); then
    driver_status="$reap_status"
  fi
fi
driver_pid=''
wait "$tee_pid"
tee_status=$?
if (( term_requested && (tee_status == 143 || tee_status == 130) )); then
  echo 'SE_B7873200_FULL_REAP_AFTER_INTERRUPTED_TEE_WAIT' >&2
  wait "$tee_pid"
  tee_reap_status=$?
  if (( tee_reap_status != 127 )); then
    tee_status="$tee_reap_status"
  fi
fi
set -e
tee_pid=''
trap - TERM INT
rm -f -- "$driver_fifo"
if (( tee_status != 0 )); then
  echo "ERROR: driver log tee exited $tee_status" >&2
  exit "$tee_status"
fi
if (( driver_status != 0 )); then
  report_for_signal="$REPORT"
  if (( compute_cap_mode )); then
    report_for_signal="$CAP_REPORT"
  fi
  if (( driver_status == 143 )) && [[ -f "$report_for_signal" ]] && python - "$report_for_signal" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "clean_interruption" else 1)
PY
  then
    echo 'SE_B7873200_FULL_CLEAN_INTERRUPTION_RECORDED' >&2
    exit 143
  fi
  echo "ERROR: full driver exited $driver_status; retaining partial evidence" >&2
  exit "$driver_status"
fi
if (( compute_cap_mode )); then
  [[ -f "$CAP_REPORT" ]] || { echo 'ERROR: compute-capped driver did not produce its isolated report' >&2; exit 95; }
  grep -Fq 'SE_B7873200_FULL_STAGE1_COMPUTE_CAPPED_EPOCH30' "$driver_log" || {
    echo 'ERROR: compute-capped driver omitted its epoch-30 outcome marker' >&2
    exit 95
  }
  echo "SE_B7873200_FULL_STAGE1_COMPUTE_CAPPED_EPOCH30: $CAP_REPORT"
  exit 0
fi
[[ -f "$REPORT" ]] || { echo 'ERROR: full driver did not produce its report' >&2; exit 95; }
if grep -Fq 'SE_B7873200_FULL_LIFECYCLE_COMPLETE' "$driver_log"; then
  echo 'SE_B7873200_FULL_DRIVER_OUTCOME=complete'
elif grep -Fq 'SE_B7873200_FULL_SCIENTIFIC_NEGATIVE_TOPOLOGY_COMPLETE' "$driver_log"; then
  echo 'SE_B7873200_FULL_DRIVER_OUTCOME=scientific_negative_topology'
else
  echo 'ERROR: full driver omitted a recognized terminal outcome marker' >&2
  exit 95
fi

python -B -u scripts/postflight_sugavanam_ertin_b7873200_full.py \
  --report "$REPORT" \
  --output "$RUN_ROOT/postflight.json" \
  --host-limit-gib 64 \
  --time-limit-hours 12 \
  --time-file "$time_log"
echo 'SE_B7873200_FULL_EXPERIMENT_COMPLETE'
