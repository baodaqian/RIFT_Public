#!/bin/bash
# Bounded real-data smoke of the adaptive RIFT trainer on one PVC card: B787,
# production argument list (2400 train views, 1t1r, adaptive-capacity v2.2),
# epochs overridden by PVC_EPOCHS (default 2). Not a production run.
#SBATCH --job-name=pvc-smoke-b787-rift
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageD/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 WANDB_MODE=offline PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
OUT=${PVC_OUTPUT_ROOT:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageD/outputs/rift_smoke_${SLURM_JOB_ID}}
hostname; date; echo "output root: $OUT"; python -c "from rift_pvc import accelerator as a; print(a.describe())"
python -u train_rift_dataset_pvc.py --dataset-root "$RIFT_DATA_ROOT" --output-root "$OUT" \
  --object b787 --method rift --num-train 2400 --num-tx 1 --num-rx 1 --pvc-epochs "${PVC_EPOCHS:-2}"
rc=$?; date; echo "SMOKE_EXIT=$rc"; ls -la "$OUT"/train2400/*/b787/* 2>/dev/null | head -20; exit $rc
