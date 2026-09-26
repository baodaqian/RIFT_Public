#!/bin/bash
# Isolated B=1/2/4/8 full-domain GOTCHA degree-3 memory measurements.
#SBATCH --job-name=rift_viewbatch_mem_v1
#SBATCH --account=gts-jromberg3
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=20G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --export=NONE
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/public_radar_tuning/viewbatch_memory_v1/Report-%A_%a.out

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: view-batch memory validation requires a Slurm array allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == embers && "${SLURM_JOB_PARTITION:-}" == gpu-v100 ]] || {
  echo 'ERROR: view-batch memory validation requires Embers/gpu-v100' >&2
  exit 94
}

BATCH_SIZES=(1 2 4 8)
BATCH_SIZE="${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}"
PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
SOURCE_ROOT=/storage/scratch1/1/dbao31/rift_public_radar_impl_20260815
DATASET=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v2_interleaved/gotcha_pass2_hh/gotcha_pass2_hh.npz
RUN_ROOT="$PROJECT_ROOT/experiment_state/public_radar_tuning/viewbatch_memory_v1"
ENTRYPOINT="$PROJECT_ROOT/scripts/benchmark_public_radar_view_batch_memory.py"

for path in \
  "$SOURCE_ROOT" \
  "$DATASET" \
  "$ENTRYPOINT" \
  "$PROJECT_ROOT/scripts/public_radar_gotcha_full_domain_v1_partial.py" \
  "$PROJECT_ROOT/rift/fixed_planar_scene.py" \
  "$PROJECT_ROOT/rift/serialized_range_operator.py"; do
  [[ -e "$path" ]] || { echo "ERROR: missing benchmark path: $path" >&2; exit 94; }
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
export TORCH_EXTENSIONS_DIR="$TMPDIR/rift_viewbatch_mem_v1.${SLURM_JOB_ID}"
mkdir -m 700 "$TORCH_EXTENSIONS_DIR"
export NO_FAST_MATH=0

python -B - <<'PY'
import torch

if torch.__version__ != "2.6.0+cu124" or torch.version.cuda != "12.4":
    raise RuntimeError((torch.__version__, torch.version.cuda))
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("exactly one CUDA device is required")
if torch.cuda.get_device_capability(0) != (7, 0):
    raise RuntimeError(f"V100 capability expected, got {torch.cuda.get_device_capability(0)}")
print(torch.cuda.get_device_name(0), flush=True)
PY

python -B -u "$ENTRYPOINT" \
  --npz-path "$DATASET" \
  --run-root "$RUN_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --point-chunk 32768
