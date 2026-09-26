#!/bin/bash
# Production arguments and fixed 1000-step clock, cooperative stop then resume.
#SBATCH --job-name=pvc-smoke-b787-sh-sas
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageH/logs/%x-%j.log
set -eo pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export RIFT_ACCELERATOR=xpu WANDB_MODE=offline
OUT=${PVC_OUTPUT_ROOT:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageH/outputs/b787_sh_sas_${SLURM_JOB_ID}}
python -u scripts_pvc/smoke_sh_sas_pvc.py --dataset-root "$RIFT_DATA_ROOT" --output-root "$OUT" \
  --term-after "${PVC_TERM_AFTER:-1500}" --resume-after "${PVC_RESUME_AFTER:-900}"
