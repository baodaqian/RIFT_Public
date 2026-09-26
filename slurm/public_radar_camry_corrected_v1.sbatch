#!/bin/bash
# Isolated Camry initialization diagnostic and degree-0/3 corrected pilots.
# Resource fields are also pinned in the manager track; the launcher validates
# the paid-QOS/job identity but does not touch any frozen PublicRadar artifact.
#SBATCH --account=gts-jromberg3
#SBATCH --qos=inferno
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --tmp=20G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --export=NONE

set -euo pipefail

[[ "$#" -eq 2 ]] || {
  echo 'usage: launcher {initdiag|deg0|deg3} {fresh|resume}' >&2
  exit 64
}
CELL="$1"
MODE="$2"
case "$CELL" in
  initdiag|deg0|deg3) ;;
  *) echo "invalid cell: $CELL" >&2; exit 64 ;;
esac
case "$MODE" in
  fresh|resume) ;;
  *) echo "invalid launch mode: $MODE" >&2; exit 64 ;;
esac

EXPECTED_JOB="mgr_rift_camry_corrected_v1_v100fix1_${CELL}"
[[ "${SLURM_JOB_NAME:-}" == "$EXPECTED_JOB" ]] || {
  echo "ERROR: expected Slurm job $EXPECTED_JOB, got ${SLURM_JOB_NAME:-unset}" >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == inferno ]] || {
  echo 'ERROR: corrected Camry pilots require the explicitly authorized Inferno QOS' >&2
  exit 94
}
[[ "${SLURM_JOB_PARTITION:-}" == gpu-v100 ]] || {
  echo 'ERROR: corrected Camry pilots are validated for gpu-v100' >&2
  exit 94
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
SOURCE_ROOT=/storage/scratch1/1/dbao31/rift_public_radar_impl_20260815
DATASET=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v1/cvdomes_camry/cvdomes_camry.npz
CHECKPOINT=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/public_radar/cvdomes_gotcha_v1/cvdomes_camry/rift/checkpoints/cvdomes_hh_rift/checkpoint_best.pth.tar
RUN_ROOT="$PROJECT_ROOT/experiment_state/public_radar_tuning/cvdomes_camry_corrected_v1"
ENTRYPOINT="$PROJECT_ROOT/scripts/public_radar_camry_tuning.py"

for path in "$SOURCE_ROOT" "$DATASET" "$CHECKPOINT" "$ENTRYPOINT" "$PROJECT_ROOT/rift/public_radar_tuning.py"; do
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TORCH_EXTENSIONS_DIR="$TMPDIR/rift_camry_corrected_torch_extensions.$SLURM_JOB_ID"
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
    raise RuntimeError(f"gpu-v100 capability expected, got {(major, minor)}")
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
  --checkpoint-best "$CHECKPOINT" \
  --run-root "$RUN_ROOT" \
  --extent 2.886751345948129 \
  --granularity 64 \
  --seed 42 \
  --bp-views 100 \
  --updates-per-epoch 1800 \
  --epochs 15 \
  --scene-lr 3e-5 \
  --adam-eps 1e-15 &
runner_pid=$!
wait "$runner_pid"
rc=$?
set -e
trap - TERM INT
exit "$rc"
