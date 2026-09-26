#!/bin/bash
# TRAIN-mean constant reference (scripts_pvc/eval_train_constant_pvc.py) on the CPU partition (no GPU cap).
# Usage: export ARGS="gotcha --campaign-root ... --task ... --mf-stats ... --heldout-metrics ... --output ..."
#        sbatch --export=ALL --job-name eval-train-constant --output OUT/NAME-%j.log scripts_pvc/eval_train_constant_cpu.sbatch
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -eo pipefail
: "${ARGS:?}"
cd "${SOURCE_ROOT:-/home/u.db364833/RIFT}"
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu RIFT_PVC_ALLOW_BACKEND=cpu RIFT_PVC_TCNN_SHIM=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
hostname; date
python -u scripts_pvc/eval_train_constant_pvc.py $ARGS
date
