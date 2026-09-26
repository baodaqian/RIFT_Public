#!/bin/bash
# Bounded real-data smoke of released RadarSplat (budget48) on one PVC card (Package F, gate 3).
#
# Production argument list (B787, 2400 train views, 1t1r, budget48: 112000 planar
# Gaussians, 2000 updates) obtained from the PVC frontend's own planner; the
# release CLI has no step flag. The controller SIGTERMs each process group
# at the existing 100-update logging cadence, validates its checkpoint, then
# requires advancing recovery. PVC_STOP_AFTER/PVC_RESUME_AFTER are watchdogs;
# timeouts and CPU fallback warnings fail. The 2000-update recipe is unchanged.
#
#   STAGE=prepare  target-cache preparation only (2400 train + 1000 validation views)
#   STAGE=train    bounded fit + resume (needs the prepared cache under the same root)
#   STAGE=all      prepare then train (default)
#
# Keep PVC_OUTPUT_ROOT stable across jobs so the train stage finds the cache.
#SBATCH --job-name=pvc-smoke-b787-radarsplat
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageF/logs/%x-%j.log
set -eo pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export RIFT_ACCELERATOR=xpu
unset RIFT_PVC_ALLOW_BACKEND RIFT_PVC_RADARSPLAT_ALLOW_CUDA_RESUME
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 WANDB_MODE=offline PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
export RIFT_PVC_RADARSPLAT_TIMING_EVERY=${RIFT_PVC_RADARSPLAT_TIMING_EVERY:-5}
export RIFT_PVC_RADARSPLAT_PARITY_JOB=${RIFT_PVC_RADARSPLAT_PARITY_JOB:-}

STAGE=${STAGE:-all}
OUT=${PVC_OUTPUT_ROOT:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageF/outputs/radarsplat_b787_smoke}
STOP_AFTER=${PVC_STOP_AFTER:-900}        # watchdog seconds for the fresh leg
RESUME_AFTER=${PVC_RESUME_AFTER:-600}    # watchdog seconds for the resume leg
PLAN="${OUT}/plan_${SLURM_JOB_ID}.json"
LOG=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageF/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log

hostname; date; echo "stage=$STAGE output root: $OUT stop_after=${STOP_AFTER}s resume_after=${RESUME_AFTER}s"
python -c "from rift_pvc import accelerator as a; print(a.describe())"
mkdir -p "$OUT" || exit 1
# Planner + derived manifest + PVC dependency gate (loads the fork on the card).
python -u scripts_pvc/plan_radarsplat_pvc.py --dataset-root "$RIFT_DATA_ROOT" --output-root "$OUT" \
  --object b787 --num-train 2400 --num-tx 1 --num-rx 1 --radarsplat-recipe budget48 > "$PLAN" || exit 1
cat "$PLAN"
python -u scripts_pvc/smoke_radarsplat_pvc.py --plan "$PLAN" --stage "$STAGE" \
  --first-step "${PVC_FIRST_STEP:-100}" --resume-steps "${PVC_RESUME_STEPS:-100}" \
  --deadline "$STOP_AFTER" --resume-deadline "$RESUME_AFTER" \
  --prepare-deadline "${PVC_PREPARE_DEADLINE:-7200}" --per-job-checkpoints
rc=$?
echo "SMOKE_EXIT=$rc"
exit "$rc"
