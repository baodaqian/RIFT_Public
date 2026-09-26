#!/bin/bash
# Resume check for the adaptive RIFT trainer on one PVC card: continue the smoke
# run in PVC_OUTPUT_ROOT from PVC_RESUME (checkpoint_latest.pth.tar) to
# PVC_EPOCHS total epochs (default 3), exercising checkpoint load, RNG restore
# (torch_xpu_all) and the adaptive-v2 full-resume contract. Not a production run.
#SBATCH --job-name=pvc-resume-b787-rift
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
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageD/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 WANDB_MODE=offline PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
: "${PVC_OUTPUT_ROOT:?set PVC_OUTPUT_ROOT to the smoke output root}"; : "${PVC_RESUME:?set PVC_RESUME to checkpoint_latest.pth.tar}"
hostname; date; echo "output root: $PVC_OUTPUT_ROOT"; echo "resume: $PVC_RESUME"; ls -la "$PVC_RESUME"
python -u train_rift_dataset_pvc.py --dataset-root "$RIFT_DATA_ROOT" --output-root "$PVC_OUTPUT_ROOT" \
  --object b787 --method rift --num-train 2400 --num-tx 1 --num-rx 1 --pvc-epochs "${PVC_EPOCHS:-3}" --resume "$PVC_RESUME"
rc=$?; date; echo "RESUME_EXIT=$rc"; exit $rc
