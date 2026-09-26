#!/bin/bash
# Bounded real-data smoke of the built-in GOTCHA RIFT backend on one PVC card:
# Camry, HH, eight passes, 1500 training pass-sectors (the production command),
# stopped by SIGTERM after PVC_SECONDS (default 1500 s) so the trainer writes its
# interrupted-status checkpoint exactly as under the campaign manager's
# pre-time-limit signal. One GOTCHA epoch is ~2.8 h on H100, so no epoch completes.
# Set PVC_RESUME=<checkpoint_latest.pt> and PVC_OUTPUT_ROOT=<same root> to resume.
#SBATCH --job-name=pvc-smoke-gotcha-rift
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
OUT=${PVC_OUTPUT_ROOT:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageD/outputs/gotcha_smoke_${SLURM_JOB_ID}}
hostname; date; echo "output root: $OUT"; python -c "from rift_pvc import accelerator as a; print(a.describe())"
timeout -s TERM -k 300 "${PVC_SECONDS:-1500}" python -u train_gotcha_dataset_pvc.py --dataset-root "$GOTCHA_DATA_ROOT" --output-root "$OUT" \
  --region camry --polarizations hh --passes 1 2 3 4 5 6 7 8 --method rift --num-train 1500 --num-tx 1 --num-rx 1 ${PVC_RESUME:+--resume "$PVC_RESUME"}
rc=$?; date; echo "SMOKE_EXIT=$rc (124 = stopped by timeout, 143 = trainer reported interrupted)"; find "$OUT" -maxdepth 5 -type f | head -20; exit $rc
