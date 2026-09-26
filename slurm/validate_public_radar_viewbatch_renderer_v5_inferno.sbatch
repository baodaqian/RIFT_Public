#!/bin/bash
# One-cell Inferno/V100 renderer compatibility and timing gate.  Degree 3 / B4
# exercises the deployed batched renderer before the full matrix is allowed.
# It never enters a training run root and never reads or writes checkpoints.
#SBATCH --job-name=rift_vbrgate_v5
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
#SBATCH --export=NONE
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/mgr_slurm_logs/viewbatch-renderer-gate-v5-inferno-%j.out

set -euo pipefail

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: Inferno renderer gate takes no arguments' >&2
  exit 64
}
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: renderer gate requires one non-array Slurm allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == inferno && "${SLURM_JOB_PARTITION:-}" == gpu-v100 ]] || {
  echo 'ERROR: view-batch speed benchmark requires Inferno/gpu-v100' >&2
  exit 94
}
DEGREE=3
BATCH_SIZE=4

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
SOURCE_ROOT=/storage/scratch1/1/dbao31/rift_public_radar_impl_20260815
DATA_ROOT=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v2_interleaved
DATASET="$DATA_ROOT/gotcha_pass2_hh/gotcha_pass2_hh.npz"
RUN_ROOT="$PROJECT_ROOT/experiment_state/public_radar_tuning/viewbatch_renderer_gate_v5_inferno"
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
export TORCH_EXTENSIONS_DIR="$TMPDIR/rift_vbrgate_v5.${SLURM_JOB_ID}"
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

python -B - "$RESULT_PATH" <<'PY'
import json
import math
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

require(payload.get("schema") == "rift.public_radar.viewbatch_speedup_v5_inferno", "schema")
require(payload.get("schema_version") == 5, "schema_version")
require(payload.get("degree") == 3, "degree")
require(payload.get("requested_aligned_batch_size") == 4, "batch_size")
require(payload.get("point_chunk") == 32768, "point_chunk")
require(payload.get("status") == "pass", f"status={payload.get('status')!r}")
require(payload.get("stage") == "complete", "stage")
for key in (
    "measurement_completed",
    "runtime_nonzero",
    "timing_valid",
    "numerical_finite",
    "memory_safe",
):
    require(payload.get(key) is True, key)
require(payload.get("unsafe_reasons") == [], "unsafe_reasons")
require(payload.get("noisy_components") == [], "noisy_components")
require(payload["tiny_cuda_equivalence_gate"].get("passed") is True, "tiny gate")
require(payload["tiny_cuda_equivalence_gate"].get("all_finite") is True, "tiny finite")
full = payload["full_window_equivalence_gate"]
require(full.get("passed") is True, "full-window gate")
require(full.get("all_finite") is True, "full-window finite")
require(full.get("same_measurement_exact") is True, "measurement identity")
require(math.isfinite(float(payload["runtime_seconds"])) and float(payload["runtime_seconds"]) > 0.0, "runtime")

cuda_cap = int(payload["cuda_cap_bytes"])
host_cap = int(payload["host_cap_kib"])
for key in (
    "cuda_whole_process_peak_allocated_bytes",
    "cuda_whole_process_peak_reserved_bytes",
):
    require(0 <= int(payload[key]) <= cuda_cap, key)
require(0 < int(payload["host_VmHWM_kib"]) <= host_cap, "host peak")

for phase_name in ("former_scalar", "aligned"):
    phase = payload[phase_name]
    require(phase.get("timing_valid") is True, f"{phase_name}.timing_valid")
    require(phase.get("numerical_finite") is True, f"{phase_name}.numerical_finite")
    for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        require(0 <= int(phase[key]) <= cuda_cap, f"{phase_name}.{key}")
    timing = phase["timing"]
    cv_limit = float(timing["noisy_cv_limit"])
    for component in ("forward", "backward", "optimizer", "total"):
        median = float(timing["median_window_seconds"][component])
        cv = float(timing["coefficient_of_variation"][component])
        require(math.isfinite(median) and median > 0.0, f"{phase_name}.{component}.median")
        require(math.isfinite(cv), f"{phase_name}.{component}.cv")
        if component != "optimizer":
            require(cv <= cv_limit, f"{phase_name}.{component}.cv_limit")

speedup = payload["paired_speedup"]
require(
    speedup.get("definition")
    == "former_scalar_seconds_divided_by_aligned_seconds",
    "paired speedup definition",
)
for component in ("forward", "backward", "optimizer", "total"):
    paired_values = speedup["paired_same_view_window_speedups"][component]
    require(len(paired_values) == 3, f"{component}.paired_count")
    for value in paired_values:
        require(
            math.isfinite(float(value)) and float(value) > 0.0,
            f"{component}.paired_speedup",
        )
    for family_name in (
        "median_paired_same_view_window_speedup",
        "ratio_of_median_window_times",
    ):
        value = float(speedup[family_name][component])
        require(
            math.isfinite(value) and value > 0.0,
            f"{family_name}.{component}",
        )

print("RENDERER_GATE_PASS", path, flush=True)
PY
