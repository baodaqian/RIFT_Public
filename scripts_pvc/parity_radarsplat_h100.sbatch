#!/bin/bash -l
# Gate 1 reference for Package F: the fork's CUDA ops on an H100 (gpu_debug),
# recorded by scripts_pvc/parity_radarsplat_dump.py for replay on PVC.
# Environment as the H100 campaign's tools/job.sh + tools/dependencies.sh recipe:
# CUDA env, JIT build of the fork's extension and the pinned fused-SSIM into a
# job-local site under the Package F runtime root (never the campaign root).
#SBATCH --job-name=parity-radarsplat-h100
#SBATCH --partition=gpu_debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:55:00
#SBATCH --account=158648339640
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageF/logs/%x-%j.log
set -eo pipefail
module load GCCcore/12.3.0 CUDA/12.4.1
source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate /scratch/user/u.db364833/.conda/envs/RIFT
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg WANDB_MODE=offline
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MAX_JOBS=${MAX_JOBS:-8} TORCH_CUDA_ARCH_LIST=9.0 CC=gcc CXX=g++ CUDA_HOME=/sw/eb/sw/CUDA/12.4.1
F=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageF
RUNTIME=$F/runtime/h100
export XDG_CACHE_HOME=$RUNTIME/xdg TORCH_EXTENSIONS_DIR=$RUNTIME/torch_extensions CUDA_CACHE_PATH=$RUNTIME/cuda
export PIP_CACHE_DIR=$RUNTIME/pip MPLCONFIGDIR=$RUNTIME/matplotlib TMPDIR=$RUNTIME/tmp/$SLURM_JOB_ID
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR" "$RUNTIME/site"
cd /home/u.db364833/RIFT
source .env.datasets.sh
export PYTHONPATH="$RUNTIME/site:/home/u.db364833/RIFT"
hostname; date; nvidia-smi -L; module -t list
SNAP=/scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645/code/external/fused-ssim
if ! python -c 'import fused_ssim' 2>/dev/null; then
  python -m pip install --no-deps --no-build-isolation --no-index --upgrade --target "$RUNTIME/site" \
    "git+file://$SNAP@1272e21a282342e89537159e4bad508b19b34157"
fi
echo "=== fork extension build + pinned fused-SSIM check: $(date '+%T') ==="
python - <<'PY'
import torch
from rift.radarsplat_release import load_cuda_reference
rendering, fused_ssim = load_cuda_reference(device='cuda')
x = torch.rand(1, 3, 32, 32, device='cuda', requires_grad=True)
fused_ssim(x, torch.rand_like(x), padding='valid').backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
print('PINNED_RADARSPLAT_IMPORT_AND_SSIM_CUDA_PASS', flush=True)
PY
echo "=== parity dump: $(date '+%T') ==="
MANIFEST=${PARITY_MANIFEST:-$F/parity/plan/train2400/1t1r_2010b2bbe725/b787/role_manifest.json}
python -u scripts_pvc/parity_radarsplat_dump.py --cache-root "$F/parity/targets_subset" \
  --npz-path "$RIFT_DATA_ROOT/objects/b787.npz" --role-manifest "$MANIFEST" \
  --max-train "${PARITY_MAX_TRAIN:-24}" --max-validation "${PARITY_MAX_VALIDATION:-4}" \
  --output "$F/parity/radarsplat_parity_h100_${SLURM_JOB_ID}.pt"
date; echo "PARITY_JOB_EXIT=$?"
