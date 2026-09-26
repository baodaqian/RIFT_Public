#!/bin/bash
# One-shot compatibility gate before moving the paired PublicRadar benchmark
# from V100 to an RTX Pro 6000 Blackwell node.  This does not train a model.
#SBATCH --job-name=rift_bwcompat_v1
#SBATCH --account=gts-jromberg3
#SBATCH --qos=inferno
#SBATCH --partition=gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --tmp=8G
#SBATCH --time=00:05:00
#SBATCH --export=NONE
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/mgr_slurm_logs/blackwell-cuda124-v1-%j.out

set -euo pipefail

[[ "$#" -eq 0 ]] || { echo 'ERROR: compatibility gate takes no arguments' >&2; exit 64; }
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'ERROR: Slurm allocation required' >&2; exit 94; }
[[ "${SLURM_JOB_QOS:-}" == inferno && \
   "${SLURM_JOB_PARTITION:-}" == gpu-rtxpro-blackwell ]] || {
  echo 'ERROR: Inferno RTX Pro Blackwell allocation required' >&2
  exit 94
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
SOURCE_ROOT=/storage/scratch1/1/dbao31/rift_public_radar_impl_20260815
RESULT_ROOT="$PROJECT_ROOT/experiment_state/public_radar_tuning/blackwell_cuda124_compat_v1"
RESULT_PATH="$RESULT_ROOT/result.json"
[[ -d "$SOURCE_ROOT" ]] || { echo "ERROR: missing source root: $SOURCE_ROOT" >&2; exit 94; }
[[ -f "$SOURCE_ROOT/scripts/activate_exact_numpy.sh" ]] || {
  echo 'ERROR: missing exact NumPy activation script' >&2
  exit 94
}
[[ ! -e "$RESULT_PATH" ]] || { echo "ERROR: refusing to overwrite $RESULT_PATH" >&2; exit 94; }

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

python -B -u - "$RESULT_PATH" <<'PY'
import json
import os
from pathlib import Path
import sys
import time

import torch

result_path = Path(sys.argv[1])
if torch.__version__ != "2.6.0+cu124" or torch.version.cuda != "12.4":
    raise RuntimeError((torch.__version__, torch.version.cuda))
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("exactly one CUDA device is required")

started = time.perf_counter()
device = torch.device("cuda:0")
generator = torch.Generator(device=device)
generator.manual_seed(20260901)
x = torch.randn((1024, 1024), device=device, generator=generator, requires_grad=True)
y = torch.randn((1024, 1024), device=device, generator=generator)
loss = (x @ y).square().mean()
loss.backward()
torch.cuda.synchronize(device)
if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(x.grad).all()):
    raise RuntimeError("Blackwell CUDA forward/backward produced non-finite values")

payload = {
    "schema": "rift.public_radar.blackwell_cuda124_compat_v1",
    "schema_version": 1,
    "status": "pass",
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "device": torch.cuda.get_device_name(device),
    "device_capability": list(torch.cuda.get_device_capability(device)),
    "device_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
    "loss": float(loss.detach().cpu()),
    "gradient_norm": float(torch.linalg.vector_norm(x.grad).detach().cpu()),
    "elapsed_seconds": time.perf_counter() - started,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
result_path.parent.mkdir(parents=True, exist_ok=True)
temporary = result_path.with_suffix(
    result_path.suffix + f".{os.environ['SLURM_JOB_ID']}.{os.getpid()}.tmp"
)
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.link(temporary, result_path)
temporary.unlink()
print(json.dumps(payload, sort_keys=True), flush=True)
PY
