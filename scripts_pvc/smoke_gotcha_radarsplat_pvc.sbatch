#!/bin/bash
# Bounded real-data smoke of the RadarSplat GOTCHA backend on one PVC card (Package F):
# Camry, HH, eight passes, 1500 training pass-sectors (the production selection),
# through train_gotcha_dataset_pvc.py -> train_radarsplat_pvc.run_gotcha ->
# rift_pvc.radarsplat_gotcha (native MF conversion per pass-sector on the card,
# then the released budget48 engine on the torch-mirror renderer).
#
# Leg 1 runs the fresh command until SIGTERM after PVC_SECONDS (default 1200 s):
# the backend returns status=interrupted and the frontend exits 143. Leg 2 resumes
# from the run's control file (radarsplat_gotcha.pt) for PVC_RESUME_SECONDS
# (default 1500 s) or to completion. Not a production run.
#SBATCH --job-name=pvc-smoke-gotcha-radarsplat
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
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageF/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export RIFT_ACCELERATOR=xpu
unset RIFT_PVC_ALLOW_BACKEND RIFT_PVC_RADARSPLAT_ALLOW_CUDA_RESUME
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 WANDB_MODE=offline PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
export RIFT_PVC_RADARSPLAT_TIMING_EVERY=${RIFT_PVC_RADARSPLAT_TIMING_EVERY:-5}
OUT=${PVC_OUTPUT_ROOT:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageF/outputs/gotcha_radarsplat_smoke}
LOG=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageF/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log
COMMON=(--dataset-root "$GOTCHA_DATA_ROOT" --output-root "$OUT" --region camry --polarizations hh
        --passes 1 2 3 4 5 6 7 8 --method radarsplat --num-train 1500 --num-tx 1 --num-rx 1)
hostname; date; echo "output root: $OUT"; python -c "from rift_pvc import accelerator as a; print(a.describe())"
mkdir -p "$OUT"
echo "=== leg 1 (fresh, SIGTERM after ${PVC_SECONDS:-1200}s): $(date '+%F %T') ==="
start=$SECONDS
timeout -s TERM -k 300 "${PVC_SECONDS:-1200}" python -u train_gotcha_dataset_pvc.py "${COMMON[@]}"
rc1=$?; echo "=== leg 1 exit=${rc1} wall=$((SECONDS-start))s (143 = interrupted with control file, 0 = complete) ==="
if [[ "$rc1" != 0 && "$rc1" != 124 && "$rc1" != 143 ]]; then
  echo "Fresh GOTCHA leg failed; refusing to mask its error with a resume."
  exit "$rc1"
fi
CONTROL=$(find "$OUT" -name radarsplat_gotcha.pt | head -1)
echo "control file: ${CONTROL:-none}"
rc=$rc1
if [[ -n "$CONTROL" && "$rc1" -ne 0 ]]; then
  echo "=== leg 2 (resume, SIGTERM after ${PVC_RESUME_SECONDS:-1500}s): $(date '+%F %T') ==="
  start=$SECONDS
  timeout -s TERM -k 300 "${PVC_RESUME_SECONDS:-1500}" python -u train_gotcha_dataset_pvc.py "${COMMON[@]}" --resume "$CONTROL"
  rc=$?; echo "=== leg 2 exit=${rc} wall=$((SECONDS-start))s ==="
fi
date
echo "--- XPU->CPU operator fallbacks in this job ---"; grep -c 'fallback from XPU' "$LOG" 2>/dev/null || echo 0
if grep -q 'fallback from XPU to CPU' "$LOG"; then
  echo "GOTCHA smoke failed: an XPU operator fell back to CPU."
  rc=1
fi
echo "--- conversion / update progress (last lines) ---"
grep -E 'RADARSPLAT_PVC_UPDATE_TIMING_JSON|^\{"step"|sector|prepared|status' "$LOG" | tail -8
echo "--- outputs ---"; find "$OUT" -maxdepth 6 \( -name '*.pt' -o -name 'backend.json' -o -name 'summary.json' -o -name 'recipe.json' -o -name '*.json' \) -printf '%10s  %p\n' 2>/dev/null | head -20
echo "SMOKE_EXIT=$rc (143/124 = bounded stop, 0 = complete)"
exit $rc
