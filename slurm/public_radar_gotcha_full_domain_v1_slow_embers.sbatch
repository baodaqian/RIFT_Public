#!/bin/bash
# Fresh V100-or-RTX6000/Embers launcher for the complete GOTCHA pass-2 plane.
# The 1776-square representation is about 3.01x the retired 1024-square pilot.
# Training and long metric passes both checkpoint so clean preemptions progress.
#SBATCH --account=gts-jromberg3
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100,gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=20G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --export=NONE

set -euo pipefail

[[ "$#" -eq 2 ]] || {
  echo 'usage: launcher gotcha_p2_full_domain_bp{400|1600}_deg{0|3} {fresh|resume}' >&2
  exit 64
}
CELL="$1"
MODE="$2"
case "$CELL" in
  gotcha_p2_full_domain_bp400_deg0|gotcha_p2_full_domain_bp400_deg3|\
  gotcha_p2_full_domain_bp1600_deg0|gotcha_p2_full_domain_bp1600_deg3) ;;
  *) echo "invalid GOTCHA full-domain v1 cell: $CELL" >&2; exit 64 ;;
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
[[ "${SLURM_JOB_QOS:-}" == embers ]] || {
  echo 'ERROR: GOTCHA full-domain v1 requires embers' >&2
  exit 94
}
case "${SLURM_JOB_PARTITION:-}" in
  gpu-v100|gpu-rtx6000) ;;
  *)
    echo 'ERROR: GOTCHA full-domain v1 requires gpu-v100 or gpu-rtx6000' >&2
    exit 94
    ;;
esac

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
SOURCE_ROOT=/storage/scratch1/1/dbao31/rift_public_radar_impl_20260815
DATA_ROOT=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v2_interleaved
RUN_ROOT="$PROJECT_ROOT/experiment_state/public_radar_tuning/gotcha_p2_full_domain_v1"
ENTRYPOINT="$PROJECT_ROOT/scripts/public_radar_gotcha_full_domain_v1_partial.py"

SCENE=gotcha_pass2_hh
DATASET="$DATA_ROOT/$SCENE/$SCENE.npz"

for path in \
  "$SOURCE_ROOT" \
  "$DATASET" \
  "$ENTRYPOINT" \
  "$PROJECT_ROOT/scripts/public_radar_camry_tuning.py" \
  "$PROJECT_ROOT/rift/public_radar_tuning.py" \
  "$PROJECT_ROOT/rift/fixed_planar_scene.py" \
  "$PROJECT_ROOT/rift/serialized_range_operator.py" \
  "$PROJECT_ROOT/rift/gotcha_domain.py"; do
  [[ -e "$path" ]] || { echo "ERROR: required path is absent: $path" >&2; exit 94; }
done

cd "$SOURCE_ROOT"
module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
source scripts/activate_exact_numpy.sh

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT"
export RIFT_PUBLIC_SOURCE_ROOT="$SOURCE_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_EXTENSIONS_DIR="$TMPDIR/rift_publicradar_gotcha_full_v1.$SLURM_JOB_ID"
mkdir -m 700 "$TORCH_EXTENSIONS_DIR"
export NO_FAST_MATH=0

python -B - <<'PY'
import torch

if torch.__version__ != "2.6.0+cu124" or torch.version.cuda != "12.4":
    raise RuntimeError((torch.__version__, torch.version.cuda))
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("exactly one CUDA device is required")
major, minor = torch.cuda.get_device_capability(0)
if major != 7:
    raise RuntimeError(f"Volta/Turing capability expected, got {(major, minor)}")
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
python -B -u "$ENTRYPOINT" "$CELL" "$MODE" \
  --npz-path "$DATASET" \
  --run-root "$RUN_ROOT" \
  --seed 42 \
  --updates-per-epoch 1800 \
  --epochs 15 \
  --scene-lr 3e-5 \
  --adam-eps 1e-15 \
  --point-chunk 32768 \
  --pair-chunk 1 &
runner_pid=$!
wait "$runner_pid"
rc=$?
set -e
trap - TERM INT
exit "$rc"
