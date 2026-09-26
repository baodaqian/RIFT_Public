#!/bin/bash
# Matched-filter backprojection (Backprojection row of the manuscript) on one PVC card.
#
# Runs scripts_pvc/eval_rift_dataset_model_free_pvc.py --method mfbp (the unchanged
# scripts/eval_rift_dataset_model_free.py): the train-only coherent matched-filter
# backprojection of each scene's 2,400 TRAIN views (production role manifest, 1 Tx x 1 Rx),
# phase-only matched filter over all 600 frequencies (phase_sign -1, range_nufft, float64),
# accumulated directly on the 48^3 evaluator lattice over +-0.15 m. Reads TRAIN responses only.
#
# Usage: export OUT=/scratch/user/u.db364833/RIFT_runs/final_eval_20260923/backprojection \
#               [OBJECTS="b787 a320 x59 firetruck race_car loader"] [METHOD=mfbp] [RESUME=1]
#        sbatch --export=ALL --job-name pvc-mfbp --output $OUT/pvc-mfbp-%j.log \
#               scripts_pvc/eval_rift_dataset_model_free_pvc.sbatch
# Writes $OUT/<object>/{model_free_identity.json,accumulator_latest.npz,matched_filter.npz}.
# The objects run one after another on the same card. The accumulator is saved every 5 views;
# RESUME=1 continues an interrupted object, and an object whose matched_filter.npz exists is skipped.
# Any "Aten Op fallback" line (XPU -> CPU) fails the job.
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
set -eo pipefail
: "${OUT:?set OUT}"
cd "${SOURCE_ROOT:-/home/u.db364833/RIFT}"
source .local-setup/activate-pvc.sh
set -u
unset RIFT_PVC_ALLOW_BACKEND
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=xpu PYTORCH_DEBUG_XPU_FALLBACK=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
D=/scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645/inputs/RIFT_dataset
P=/scratch/group/p.cis261724.000/RIFT_pvc_runs/production_20260921_35jobs/outputs/rift/train2400/1t1r_2010b2bbe725
declare -A SCENE=([b787]=b787 [a320]=airliner_a320 [x59]=supersonic_x59 [firetruck]=firetruck
                  [race_car]=race_car [loader]=loader)
EXTRA=()
[ "${RESUME:-0}" = 1 ] && EXTRA+=(--resume)
mkdir -p "$OUT"
hostname; date
python -c "from rift_pvc import accelerator as a; print(a.describe())"
FAILED=0
for OBJECT in ${OBJECTS:-b787 a320 x59 firetruck race_car loader}; do
  : "${SCENE[$OBJECT]:?unknown object $OBJECT}"
  if [ -f "$OUT/$OBJECT/matched_filter.npz" ]; then echo "$OBJECT: matched_filter.npz exists, skipped"; continue; fi
  LOG="$OUT/$OBJECT-pvc-$SLURM_JOB_ID.log"
  echo "$OBJECT: start $(date +%T) log $LOG"
  set +e
  python -u scripts_pvc/eval_rift_dataset_model_free_pvc.py --object "$OBJECT" --dataset-root "$D" \
    --role-manifest "$P/${SCENE[$OBJECT]}/role_manifest.json" --method "${METHOD:-mfbp}" \
    --output-dir "$OUT/$OBJECT" --device xpu "${EXTRA[@]}" 2>&1 | tee "$LOG"
  STATUS=${PIPESTATUS[0]}
  set -e
  FALLBACKS=$(grep -c 'Aten Op fallback' "$LOG" || true)
  echo "$OBJECT: exit $STATUS, ATEN_FALLBACK_LINES=$FALLBACKS, end $(date +%T)"
  if [ "$STATUS" -ne 0 ] || [ "$FALLBACKS" -ne 0 ]; then FAILED=1; fi
done
date
echo "MFBP_RESULT=$([ $FAILED = 0 ] && echo pass || echo fail)"
exit $FAILED
