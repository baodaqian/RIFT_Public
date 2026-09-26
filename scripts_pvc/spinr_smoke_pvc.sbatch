#!/bin/bash
#SBATCH --job-name=pvc-spinr-A
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageA/logs/%x-%j.log
set -eo pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=xpu PYTHONUNBUFFERED=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
export WANDB_MODE=offline
export MPLCONFIGDIR="/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageA/runtime/mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageA/runtime/cache-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python -m pytest -q -rs -p no:cacheprovider -W 'error:Aten Op fallback from XPU to CPU' rift_pvc/tests/test_spinr_xpu.py
python -u scripts_pvc/smoke_spinr_pvc.py --output "/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageA/b787-${SLURM_JOB_ID}"
