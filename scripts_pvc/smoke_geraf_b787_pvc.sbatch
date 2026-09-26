#!/bin/bash
# Production recipe and 50000-step cosine clock; bounded by cooperative SIGTERM
# after committed updates. Both legs must advance durable optimizer/RNG state.
# STAGE=prepare|train|all; keep PVC_OUTPUT_ROOT when splitting prepare and train.
#SBATCH --job-name=pvc-smoke-b787-geraf
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
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageB/logs/%x-%j.log
set -eo pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=xpu PYTHONUNBUFFERED=1 WANDB_MODE=offline
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
STAGE=${STAGE:-all}
OUT=${PVC_OUTPUT_ROOT:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageB/outputs/geraf_production_clock_smoke}
CONFIG=${PVC_GERAF_CONFIG:-protocols/geraf_mf48_1t1r.json}
PLAN="$OUT/plan_${SLURM_JOB_ID}.json"
export MPLCONFIGDIR="$OUT/runtime/mpl" XDG_CACHE_HOME="$OUT/runtime/cache"
mkdir -p "$OUT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python -u scripts_pvc/plan_geraf_pvc.py --dataset-root "$RIFT_DATA_ROOT" --output-root "$OUT" \
  --object b787 --num-train 2400 --num-tx 1 --num-rx 1 --geraf-source-config "$CONFIG" > "$PLAN"
python -u scripts_pvc/smoke_geraf_pvc.py --plan "$PLAN" --stage "$STAGE" \
  --first-update "${PVC_FIRST_UPDATE:-100}" --resume-updates "${PVC_RESUME_UPDATES:-100}"
