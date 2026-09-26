#!/bin/bash
# GOTCHA held-out signal scoring (scripts_pvc/eval_gotcha_heldout_pvc.py) on the CPU partition (no GPU cap).
# Usage: export METHOD=rift CAMPAIGN_ROOT=... TASK=camry-rift-full CHECKPOINT=... OUT=... LABEL=... ROLE=validation [TEST_OPT_IN=1]
#        (METHOD=zero takes no CHECKPOINT: the zero reference and the constant-predictor floors)
#        sbatch --export=ALL --job-name eval-gotcha-$LABEL --output $OUT/$LABEL-%j.log scripts_pvc/eval_gotcha_heldout_cpu.sbatch
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
set -eo pipefail
: "${METHOD:?}" "${CAMPAIGN_ROOT:?}" "${TASK:?}" "${OUT:?}" "${LABEL:?}" "${ROLE:?}"
[ "$METHOD" = zero ] || : "${CHECKPOINT:?}"      # METHOD=zero: the checkpoint-free reference (floors)
cd "${SOURCE_ROOT:-/home/u.db364833/RIFT}"
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu RIFT_PVC_ALLOW_BACKEND=cpu RIFT_PVC_TCNN_SHIM=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
EXTRA=()
[ "$METHOD" = zero ] || EXTRA+=(--checkpoint "$CHECKPOINT")
if [ "$ROLE" = test ]; then
  [ "${TEST_OPT_IN:-0}" = 1 ] || { echo "ROLE=test needs TEST_OPT_IN=1"; exit 2; }
  EXTRA+=(--allow-reserved-test)
fi
[ -n "${MAX_SECTORS:-}" ] && EXTRA+=(--max-sectors "$MAX_SECTORS")
mkdir -p "$OUT"
hostname; date
python -u scripts_pvc/eval_gotcha_heldout_pvc.py --method "$METHOD" --campaign-root "$CAMPAIGN_ROOT" --task "$TASK" \
  --label "$LABEL" --out-dir "$OUT" --role "$ROLE" --device cpu \
  --mf-stats "${MF_STATS:-$OUT/${LABEL}_mf_train_peak.json}" "${EXTRA[@]}"
date
