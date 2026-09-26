#!/bin/bash
# Short paired Inferno/V100 timing gate for former scalar versus aligned
# viewpoint rendering.  Each cell writes one immutable benchmark JSON only;
# it never enters a training run root and never reads or writes checkpoints.
#SBATCH --job-name=rift_vbspd_v5i
#SBATCH --account=gts-jromberg3
#SBATCH --qos=inferno
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-7%4
#SBATCH --export=NONE
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/mgr_slurm_logs/viewbatch-speedup-v5-inferno-%A_%a.out

set -euo pipefail

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: Inferno view-batch speed benchmark takes no arguments' >&2
  exit 64
}
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: view-batch speed benchmark requires a Slurm array allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == inferno && "${SLURM_JOB_PARTITION:-}" == gpu-v100 ]] || {
  echo 'ERROR: view-batch speed benchmark requires Inferno/gpu-v100' >&2
  exit 94
}
[[ "${SLURM_ARRAY_TASK_MIN:-}" == 0 && "${SLURM_ARRAY_TASK_MAX:-}" == 7 && \
   "${SLURM_ARRAY_TASK_COUNT:-}" == 8 ]] || {
  echo 'ERROR: view-batch speed benchmark requires the sealed eight-cell array' >&2
  exit 94
}

DEGREES=(0 0 0 0 3 3 3 3)
BATCH_SIZES=(1 2 4 8 1 2 4 8)
case "$SLURM_ARRAY_TASK_ID" in
  0|1|2|3|4|5|6|7) ;;
  *) echo "ERROR: invalid array index: $SLURM_ARRAY_TASK_ID" >&2; exit 94 ;;
esac
DEGREE="${DEGREES[$SLURM_ARRAY_TASK_ID]}"
BATCH_SIZE="${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}"

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
SOURCE_ROOT=/storage/scratch1/1/dbao31/rift_public_radar_impl_20260815
DATA_ROOT=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v2_interleaved
DATASET="$DATA_ROOT/gotcha_pass2_hh/gotcha_pass2_hh.npz"
RUN_ROOT="$PROJECT_ROOT/experiment_state/public_radar_tuning/viewbatch_speedup_v5_inferno"
ENTRYPOINT="$PROJECT_ROOT/scripts/benchmark_public_radar_viewbatch_speedup_v5.py"
RESULT_PATH="$RUN_ROOT/gotcha_full_deg${DEGREE}_b${BATCH_SIZE}.json"

for path in \
  "$SOURCE_ROOT" \
  "$DATASET" \
  "$ENTRYPOINT" \
  "$PROJECT_ROOT/scripts/public_radar_gotcha_full_domain_v1_partial.py" \
  "$PROJECT_ROOT/scripts/public_radar_camry_tuning.py" \
  "$PROJECT_ROOT/rift/public_radar_tuning.py" \
  "$PROJECT_ROOT/rift/fixed_planar_scene.py" \
  "$PROJECT_ROOT/rift/serialized_range_operator.py" \
  "$PROJECT_ROOT/rift/public_radar_sh_cache.py" \
  "$PROJECT_ROOT/rift/gotcha_domain.py"; do
  [[ -e "$path" ]] || { echo "ERROR: missing benchmark path: $path" >&2; exit 94; }
done
[[ ! -e "$RESULT_PATH" ]] || {
  echo "ERROR: refusing to overwrite existing benchmark: $RESULT_PATH" >&2
  exit 94
}

cd "$SOURCE_ROOT"
module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
source "$SOURCE_ROOT/scripts/activate_exact_numpy.sh"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT"
export RIFT_PUBLIC_SOURCE_ROOT="$SOURCE_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_EXTENSIONS_DIR="$TMPDIR/rift_vbspd_v5i.${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}"
mkdir -m 700 "$TORCH_EXTENSIONS_DIR"
export NO_FAST_MATH=0

python -B - <<'PY'
import torch

if torch.__version__ != "2.6.0+cu124" or torch.version.cuda != "12.4":
    raise RuntimeError((torch.__version__, torch.version.cuda))
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("exactly one CUDA device is required")
if torch.cuda.get_device_capability(0) != (7, 0):
    raise RuntimeError(
        f"V100 capability expected, got {torch.cuda.get_device_capability(0)}"
    )
print(torch.cuda.get_device_name(0), flush=True)
PY

python -B -u "$ENTRYPOINT" \
  --npz-path "$DATASET" \
  --run-root "$RUN_ROOT" \
  --degree "$DEGREE" \
  --batch-size "$BATCH_SIZE"
