#!/bin/bash
# One full-budget production task from the recorded campaign manifest.
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@180
set -eo pipefail
CAMPAIGN=$1
TASK=$2
cd "$CAMPAIGN/source"
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=xpu PYTHONUNBUFFERED=1 WANDB_MODE=offline
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
export RIFT_PVC_TCNN_HALF=0 RIFT_PVC_TCNN_WEIGHT_GRAD=bmm:256
# No RIFT_PVC_RADARSPLAT_PARITY_* overrides: they pinned the pre-gate-1 diagnostic
# status and shadowed the current constants in rift_pvc/radarsplat_xpu_backend.py,
# so backend.json recorded stale parity provenance (audit F6).
unset RIFT_PVC_ALLOW_BACKEND RIFT_PVC_RADARSPLAT_ALLOW_CUDA_RESUME
unset RIFT_PVC_RADARSPLAT_PARITY_JOB RIFT_PVC_RADARSPLAT_PARITY_STATUS
export MPLCONFIGDIR="$CAMPAIGN/runtime/$TASK/mpl"
export XDG_CACHE_HOME="$CAMPAIGN/runtime/$TASK/cache"
export TMPDIR="$CAMPAIGN/runtime/$TASK/tmp"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TMPDIR"
# Without B:, --signal reaches only job steps, and this script runs no srun step, so the
# runner was killed at the wall clock without its graceful stop. B: signals the batch
# shell, and exec makes the runner that process (checked with CPU jobs 2156326/2156327,
# 2026-09-22: only B: + exec delivered SIGTERM to Python before the limit).
exec python -u scripts_pvc/run_production_task.py "$CAMPAIGN" "$TASK"
