#!/bin/bash
# Versioned CPU-source plus Radar Fields CUDA-recovery gate for the reviewed
# homemade RIFT repair batch. The manager supplies the account, QOS,
# partition, GPU, CPU, memory, temporary-space, time, job-name, and output-log
# arguments. Use only Embers with gpu-v100, gpu-rtx6000, gpu-a100, gpu-h100,
# or gpu-h200. The adaptive/split/NPZ validators are CPU contracts; CUDA is
# required here specifically to exercise Radar Fields CUDA RNG recovery.
# This launcher never trains, opens an existing scientific dataset, creates a
# run root, or resumes, changes, or writes a production checkpoint. Validators
# may create cleanup-only synthetic fixtures under node temporary storage. Its
# only persistent evidence is the manager-owned Slurm log.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE

set -euo pipefail

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: validation gate takes no arguments' >&2
  exit 64
}
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: validation gate requires one non-array Slurm allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == "embers" ]] || {
  echo "ERROR: validation gate requires Embers, got ${SLURM_JOB_QOS:-<unset>}" >&2
  exit 94
}
case "${SLURM_JOB_PARTITION:-}" in
  gpu-v100|gpu-rtx6000|gpu-a100|gpu-h100|gpu-h200) ;;
  *)
    echo "ERROR: unsupported/non-approved partition: ${SLURM_JOB_PARTITION:-<unset>}" >&2
    exit 94
    ;;
esac

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
for path in \
  "$PROJECT_ROOT/train.py" \
  "$PROJECT_ROOT/train_radar_fields.py" \
  "$PROJECT_ROOT/rift/sparse_scene.py" \
  "$PROJECT_ROOT/rift/dataset.py" \
  "$PROJECT_ROOT/rift/npz_dataset.py" \
  "$PROJECT_ROOT/rift/radar_fields.py" \
  "$PROJECT_ROOT/rift/radar_fields_dataset.py" \
  "$PROJECT_ROOT/scripts/validate_adaptive_split.py" \
  "$PROJECT_ROOT/scripts/validate_adaptive_capacity_v2.py" \
  "$PROJECT_ROOT/scripts/validate_npz_sealed_ingest.py" \
  "$PROJECT_ROOT/scripts/validate_radar_fields.py"; do
  [[ -f "$path" ]] || { echo "ERROR: missing reviewed source: $path" >&2; exit 94; }
done

printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID}"
printf 'SLURM_JOB_QOS=%s\n' "${SLURM_JOB_QOS}"
printf 'SLURM_JOB_PARTITION=%s\n' "${SLURM_JOB_PARTITION}"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"

module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
# Deliberately do not source activate_exact_numpy.sh: it targets the stale
# scratch checkout and would replace this synchronized project on PYTHONPATH.
export PYTHONPATH=.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

validation_parent="${SLURM_TMPDIR:-}"
[[ -n "$validation_parent" && "$validation_parent" != "/" && -d "$validation_parent" ]] || {
  echo 'ERROR: validation gate requires a non-root SLURM_TMPDIR directory' >&2
  exit 94
}
validation_parent="${validation_parent%/}"
validation_tmp="$(mktemp -d "$validation_parent/rift-homemade-source-cuda-v1.XXXXXX")"
case "$validation_tmp" in
  "$validation_parent"/rift-homemade-source-cuda-v1.*) ;;
  *)
    echo "ERROR: unsafe validation temporary path: $validation_tmp" >&2
    exit 94
    ;;
esac
cleanup() {
  if [[ -d "${validation_tmp:-}" ]]; then
    case "$validation_tmp" in
      "$validation_parent"/rift-homemade-source-cuda-v1.*)
        rm -rf -- "$validation_tmp"
        ;;
      *)
        echo "ERROR: refusing to remove unexpected temporary path: $validation_tmp" >&2
        return 1
        ;;
    esac
  fi
}
trap cleanup EXIT
export TMPDIR="$validation_tmp"
export TEMP="$validation_tmp"
export TMP="$validation_tmp"

python -B -u - <<'PY'
import json
import os

import numpy as np
import torch

visible = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    visible.append(
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": int(properties.total_memory),
        }
    )
payload = {
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "numpy_version": np.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()),
    "visible_devices": visible,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
print("CUDA_PREFLIGHT " + json.dumps(payload, sort_keys=True), flush=True)
if torch.__version__ != "2.6.0+cu124" or torch.version.cuda != "12.4":
    raise RuntimeError((torch.__version__, torch.version.cuda))
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise RuntimeError("at least one real CUDA device is required")
probe = torch.arange(8, device="cuda:0", dtype=torch.float32).square().sum()
torch.cuda.synchronize()
if not bool(torch.isfinite(probe)):
    raise RuntimeError("CUDA probe produced a non-finite value")
print("CUDA_PREFLIGHT_PASS", float(probe.detach().cpu()), flush=True)
PY

run_validator() {
  local label="$1"
  local marker="$2"
  local script="$3"
  local log="$validation_tmp/$label.log"
  printf 'VALIDATOR_START=%s\n' "$label"
  python -B -u "$script" 2>&1 | tee "$log"
  if grep -Fq "$marker" "$log"; then
    printf 'VALIDATOR_PASS=%s\n' "$label"
  else
    local grep_status=$?
    if [[ "$grep_status" -eq 1 ]]; then
      echo "ERROR: validator $label omitted its terminal marker" >&2
    else
      echo "ERROR: cannot read validator log for $label" >&2
    fi
    exit 95
  fi
}

run_validator adaptive_split 'PASS validate_adaptive_split (' scripts/validate_adaptive_split.py
run_validator adaptive_capacity 'ALL ADAPTIVE-CAPACITY V2 GATES PASS' scripts/validate_adaptive_capacity_v2.py
run_validator npz_sealed_ingest 'NPZ sealed-ingestion checks' scripts/validate_npz_sealed_ingest.py
run_validator radar_fields 'ALL RADAR FIELDS GATES PASSED (' scripts/validate_radar_fields.py

radar_fields_log="$validation_tmp/radar_fields.log"
if grep -Fq 'SKIP CUDA RNG restore/topology gate (CUDA unavailable)' "$radar_fields_log"; then
  echo 'ERROR: Radar Fields CUDA recovery gate was skipped' >&2
  exit 95
else
  skip_grep_status=$?
  if [[ "$skip_grep_status" -ne 1 ]]; then
    echo 'ERROR: cannot read Radar Fields validation log for CUDA-skip check' >&2
    exit 95
  fi
fi

echo 'RIFT_HOMEMADE_SOURCE_CPU_AND_RADAR_FIELDS_CUDA_VALIDATION_PASS'
