#!/bin/bash
# AirSAS/SH-SAS allocated validation: fail-closed H200 precheck, then three gates.
# This launcher is staged locally for manager-owned deployment; it does not submit itself.
#SBATCH --job-name=mgr_rift_airsas_allocated_validation_v1_20260911
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=12G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0,atl1-1-01-002-8-0,atl1-1-02-006-34-0
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_adaptive_vs_sh_sas_v1_20260911/validation/mgr_slurm_logs/%j.out
#SBATCH --error=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_adaptive_vs_sh_sas_v1_20260911/validation/mgr_slurm_logs/%j.err

set -euo pipefail
umask 077

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: AirSAS validation launcher accepts no arguments' >&2
  exit 2
}

[[ "${SLURM_JOB_QOS:-}" == 'inferno' ]] || {
  echo "ERROR: expected SLURM_JOB_QOS=inferno, got ${SLURM_JOB_QOS:-unset}" >&2
  exit 2
}
[[ "${SLURM_JOB_PARTITION:-}" == 'gpu-h200' ]] || {
  echo "ERROR: expected SLURM_JOB_PARTITION=gpu-h200, got ${SLURM_JOB_PARTITION:-unset}" >&2
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
SYSTEM_DATA=$PROJECT/datasets/sh_sas_reed/airsas_processed/system_data_files/system_data_bunny_20k.pik

require_nonempty() {
  [[ -s "$1" ]] || {
    echo "ERROR: required deployed file is missing or empty: $1" >&2
    exit 2
  }
}

require_nonempty "$PROJECT/scripts/validate_sas.py"
require_nonempty "$PROJECT/scripts/validate_sh_sas.py"
require_nonempty "$PROJECT/scripts/validate_sh_sas_dataset.py"
require_nonempty "$PROJECT/train_sh_sas.py"
require_nonempty "$PROJECT/rift/__init__.py"
require_nonempty "$PROJECT/rift/rift_sas.py"
require_nonempty "$PROJECT/rift/sas_dataset.py"
require_nonempty "$PROJECT/rift/sas_operator.py"
require_nonempty "$PROJECT/rift/sh_sas.py"
require_nonempty "$PROJECT/rift/sparse_scene.py"
require_nonempty "$PROJECT/rift/config.py"
require_nonempty "$PROJECT/rift/range_operator.py"
require_nonempty "$PROJECT/rift/coherent_radar_geometry.py"
require_nonempty "$PROJECT/rift/encoding.py"
require_nonempty "$PROJECT/rift/occlusion.py"
require_nonempty "$PROJECT/rift/radar_fields.py"
require_nonempty "$PROJECT/rift/spherical_harmonics.py"
require_nonempty "$PROJECT/rift/distributed.py"
require_nonempty "$PROJECT/rift/calibration.py"
require_nonempty "$PROJECT/rift/radar_fields_dataset.py"

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
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK must be set}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export BLIS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
: "${SLURM_TMPDIR:?SLURM_TMPDIR must be set}"
TMP_ROOT="$SLURM_TMPDIR/airsas_allocated_validation_v1_20260911"
mkdir -p "$TMP_ROOT/tmpdir" "$TMP_ROOT/temp" "$TMP_ROOT/tmp"
export TMPDIR="$TMP_ROOT/tmpdir"
export TEMP="$TMP_ROOT/temp"
export TMP="$TMP_ROOT/tmp"

python3 - <<'PY'
import torch
import tinycudann

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"expected exactly one CUDA device, found {torch.cuda.device_count()}")
device_name = torch.cuda.get_device_name(0)
if "H200" not in device_name.upper():
    raise RuntimeError(f"expected an H200 device, found {device_name}")

device = torch.device("cuda:0")
x = torch.tensor([1.0, -2.0, 3.0], device=device, requires_grad=True)
w = torch.tensor([0.5, 1.5, -0.25], device=device, requires_grad=True)
loss = (x.square() * w).sum()
if not torch.isfinite(loss).item():
    raise RuntimeError("non-finite CUDA forward result")
loss.backward()
if not torch.isfinite(x.grad).all().item() or not torch.isfinite(w.grad).all().item():
    raise RuntimeError("non-finite CUDA backward result")
torch.cuda.synchronize()
print(f"AIRSAS_H200_IMPORT_GPU_PASS device={device_name}")
PY

python3 -B scripts/validate_sas.py --system-data "$SYSTEM_DATA"
python3 -B scripts/validate_sh_sas.py
python3 -B scripts/validate_sh_sas_dataset.py datasets/sh_sas_reed

echo 'AIRSAS_ALLOCATED_VALIDATION_PASS'
