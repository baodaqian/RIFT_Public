#!/bin/bash
# Versioned accelerated launcher for the GOTCHA pass-2 full-domain cells.
# Legacy identities, roots, and checkpoints are immutable; this launcher accepts only v1 names.
# Resumes may use the FP64-capable V100/RTX6000/A100/H100/H200 pools or the separately
# validated CUDA 12.8/SM120 environment on RTX Pro Blackwell. L40S stays excluded.
#SBATCH --account=gts-jromberg3
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100,gpu-rtx6000,gpu-a100,gpu-h100,gpu-h200,gpu-rtxpro-blackwell
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=20G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --export=NONE

set -euo pipefail

[[ "$#" -eq 2 ]] || {
  echo 'usage: launcher gotcha_p2_full_domain_bp{400|1600}_deg{0|3}_vb8_shcache_p131072_v1 {fresh|resume}' >&2
  exit 64
}
CELL="$1"
MODE="$2"
case "$CELL" in
  gotcha_p2_full_domain_bp400_deg0_vb8_shcache_p131072_v1)
    BASE_CELL=gotcha_p2_full_domain_bp400_deg0
    ;;
  gotcha_p2_full_domain_bp400_deg3_vb8_shcache_p131072_v1)
    BASE_CELL=gotcha_p2_full_domain_bp400_deg3
    ;;
  gotcha_p2_full_domain_bp1600_deg0_vb8_shcache_p131072_v1)
    BASE_CELL=gotcha_p2_full_domain_bp1600_deg0
    ;;
  gotcha_p2_full_domain_bp1600_deg3_vb8_shcache_p131072_v1)
    BASE_CELL=gotcha_p2_full_domain_bp1600_deg3
    ;;
  *) echo "invalid accelerated GOTCHA cell: $CELL" >&2; exit 64 ;;
esac
case "$MODE" in
  fresh|resume) ;;
  *) echo "invalid launch mode: $MODE" >&2; exit 64 ;;
esac

EXPECTED_JOB="mgr_rift_publicradar_gotcha_full_domain_v1_${CELL}"
[[ "${SLURM_JOB_NAME:-}" == "$EXPECTED_JOB" ]] || {
  echo "ERROR: expected Slurm job $EXPECTED_JOB, got ${SLURM_JOB_NAME:-unset}" >&2
  exit 94
}
case "${SLURM_JOB_QOS:-}" in
  embers|inferno) ;;
  *)
    echo 'ERROR: accelerated GOTCHA full-domain training requires embers or inferno' >&2
    exit 94
    ;;
esac
case "${SLURM_JOB_PARTITION:-}" in
  gpu-v100|gpu-rtx6000|gpu-a100|gpu-h100|gpu-h200|gpu-rtxpro-blackwell) ;;
  *)
    echo 'ERROR: accelerated GOTCHA full-domain training requires V100, RTX6000, A100, H100, H200, or RTX Pro Blackwell' >&2
    exit 94
    ;;
esac

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
SOURCE_ROOT=/storage/scratch1/1/dbao31/rift_public_radar_impl_20260815
DATA_ROOT=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v2_interleaved
RUN_ROOT="$PROJECT_ROOT/experiment_state/public_radar_tuning/gotcha_p2_full_domain_v1_vb8_shcache_p131072_v1"
ENTRYPOINT="$PROJECT_ROOT/scripts/public_radar_gotcha_full_domain_v1_partial.py"
SCENE=gotcha_pass2_hh
DATASET="$DATA_ROOT/$SCENE/$SCENE.npz"
SH_BASIS_CACHE="$DATA_ROOT/$SCENE/sh_basis_degree6_v1"

if [[ "$MODE" == fresh && -e "$RUN_ROOT/$BASE_CELL" ]]; then
  echo "ERROR: fresh launch refuses existing accelerated run directory: $RUN_ROOT/$BASE_CELL" >&2
  exit 94
fi

for path in \
  "$SOURCE_ROOT" \
  "$DATASET" \
  "$ENTRYPOINT" \
  "$PROJECT_ROOT/scripts/public_radar_camry_tuning.py" \
  "$PROJECT_ROOT/rift/public_radar_tuning.py" \
  "$PROJECT_ROOT/rift/fixed_planar_scene.py" \
  "$PROJECT_ROOT/rift/serialized_range_operator.py" \
  "$PROJECT_ROOT/rift/gotcha_domain.py" \
  "$PROJECT_ROOT/rift/public_radar_sh_cache.py" \
  "$SH_BASIS_CACHE/manifest.json" \
  "$SH_BASIS_CACHE/real_sh_degree6_f32.npy" \
  "$SH_BASIS_CACHE/theta_rad_f32.npy" \
  "$SH_BASIS_CACHE/phi_rad_f32.npy" \
  "$SH_BASIS_CACHE/viewpoint_positions.npy"; do
  [[ -e "$path" ]] || { echo "ERROR: required path is absent: $path" >&2; exit 94; }
done

cd "$SOURCE_ROOT"
module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
if [[ "${SLURM_JOB_PARTITION:-}" == gpu-rtxpro-blackwell ]]; then
  SM120_ENV="$PROJECT_ROOT/.experiment_manager/envs/rift-baselines-sm120-v1"
  [[ -x "$SM120_ENV/bin/python" ]] || { echo "ERROR: missing SM120 runtime: $SM120_ENV" >&2; exit 94; }
  module load nvhpc-cuda/12.9
  conda activate "$SM120_ENV"
  export CUDA_HOME
  CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
else
  module load nvhpc-cuda/12.4
  conda activate RIFT
  source scripts/activate_exact_numpy.sh
fi

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT"
export RIFT_PUBLIC_SOURCE_ROOT="$SOURCE_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_EXTENSIONS_DIR="$TMPDIR/rift_publicradar_gotcha_accel_v1.$SLURM_JOB_ID"
mkdir -m 700 "$TORCH_EXTENSIONS_DIR"
export NO_FAST_MATH=0

python -B - <<'PY'
import os
import numpy as np
import torch

partition = os.environ["SLURM_JOB_PARTITION"]
if partition == "gpu-rtxpro-blackwell":
    if torch.__version__ != "2.7.1+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError((partition, torch.__version__, torch.version.cuda))
    if np.__version__ != "1.26.0" or "sm_120" not in torch.cuda.get_arch_list():
        raise RuntimeError((np.__version__, torch.cuda.get_arch_list()))
    expected_capability = (12, 0)
else:
    if torch.__version__ != "2.6.0+cu124" or torch.version.cuda != "12.4":
        raise RuntimeError((partition, torch.__version__, torch.version.cuda))
    expected_capability = {
        "gpu-v100": (7, 0),
        "gpu-rtx6000": (7, 5),
        "gpu-a100": (8, 0),
        "gpu-h100": (9, 0),
        "gpu-h200": (9, 0),
    }[partition]
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("exactly one CUDA device is required")
major, minor = torch.cuda.get_device_capability(0)
if (major, minor) != expected_capability:
    raise RuntimeError(f"unexpected CUDA capability for {partition}: {(major, minor)}")
print(torch.cuda.get_device_name(0), flush=True)
PY

runner_pid=''
forward_signal() {
  if [[ -n "$runner_pid" ]] && kill -0 "$runner_pid" 2>/dev/null; then
    kill -TERM "$runner_pid"
  fi
}
trap forward_signal TERM INT
set +e
python -B -u "$ENTRYPOINT" "$BASE_CELL" "$MODE" \
  --npz-path "$DATASET" \
  --run-root "$RUN_ROOT" \
  --seed 42 \
  --updates-per-epoch 1800 \
  --epochs 15 \
  --scene-lr 3e-5 \
  --adam-eps 1e-15 \
  --view-batch-size 8 \
  --point-chunk 131072 \
  --pair-chunk 1 \
  --sh-basis-cache "$SH_BASIS_CACHE" \
  --require-sh-basis-cache &
runner_pid=$!
wait "$runner_pid"
rc=$?
set -e
trap - TERM INT
exit "$rc"
