#!/bin/bash
# NEW-TARGET COPY of camry_paper_provisional_20260924/jobs/spinr_role_fit_pvc.sbatch (2026-09-25, writing agent, for
# the GOTCHA Tables 2/8 TRAIN cells): RIFT's role_fit TRAIN and VAL numbers for a validation-selected SpINR checkpoint
# of a new target (Sentra, Santa Fe), on one PVC card. Identical reading otherwise:
# scripts_pvc/gotcha_spinr_role_fit_pvc.py, every unit of both roles (--unit-stride 1): full-native RelMSE, e, rho,
# e/rho^2 over the 2,410 TRAIN units (48,200 pulses) and the 30 pass-4 VAL units, rendered by SpINR's own validation
# render; the dataset is rebuilt from the production arguments of scripts_pvc/run_gotcha_target_spinr_pvc.sbatch (full
# split: stride 1, pass 4, fraction 0.1) and the tool refuses a checkpoint whose dataset identity differs. TEST stays
# sealed. Any "Aten Op fallback" line fails the job.
#
# COPY FIRST (campaign seat, 2026-09-25): the new-target SpINR runs are live and validate every 5 epochs, so their
# checkpoint_best.pt can be rewritten mid-read. The job copies SPINR_CKPT to a frozen file next to OUT, checks the
# copy is byte-identical and the source did not change during the copy, and reads only the copy.
#
# Knobs (no spaces; sbatch --export=ALL,VAR=...):
#   TARGET_REGION          sentra_box_v1 | santafe_box_v1
#   TARGET_REGION_CONFIG   (rift_pvc/regions/gotcha_new_targets.json)
#   SPINR_SHARDS           the target's carphase_v2 keep6 shard root (as the run used)
#   SPINR_RUN              the run's output root (holds spinr_config_e150_v5.json)
#   SPINR_CKPT             checkpoint to read (a live checkpoint_best.pt is fine: it is copied first)
#   OUT                    output JSON (must not exist); the frozen copy and the log go beside it
#
#   sbatch --export=ALL,TARGET_REGION=sentra_box_v1,SPINR_SHARDS=...,SPINR_RUN=...,SPINR_CKPT=...,OUT=... \
#     --job-name pvc-sentra-spinr-rolefit --output DIR/%x-%j.log scripts_pvc/run_gotcha_target_spinr_role_fit_pvc.sbatch
#SBATCH --job-name=pvc-target-spinr-rolefit
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=ac023
set -eo pipefail
: "${TARGET_REGION:?}" "${SPINR_SHARDS:?}" "${SPINR_RUN:?}" "${SPINR_CKPT:?}" "${OUT:?}"
REGION_CONFIG=${TARGET_REGION_CONFIG:-rift_pvc/regions/gotcha_new_targets.json}
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
set -u
unset RIFT_PVC_ALLOW_BACKEND
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=xpu PYTORCH_DEBUG_XPU_FALLBACK=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
[ -e "$OUT" ] && { echo "$OUT exists"; exit 3; }
DIR=$(dirname "$OUT"); STEM=$(basename "$OUT" .json)
mkdir -p "$DIR"
FROZEN="$DIR/$STEM.checkpoint.pt"
LOG="$DIR/$STEM-$SLURM_JOB_ID.log"
CONFIG="$SPINR_RUN/spinr_config_e150_v5.json"
[ -f "$CONFIG" ] || { echo "missing $CONFIG"; exit 3; }
for p in 1 2 3 4 5 6 7 8; do [ -f "$SPINR_SHARDS/pass${p}_hh.npz" ] || { echo "missing $SPINR_SHARDS/pass${p}_hh.npz"; exit 3; }; done
hostname; date
BEFORE=$(stat -c %Y:%s "$SPINR_CKPT")
cp -p "$SPINR_CKPT" "$FROZEN"
AFTER=$(stat -c %Y:%s "$SPINR_CKPT")
if [ "$BEFORE" != "$AFTER" ] || ! cmp -s "$SPINR_CKPT" "$FROZEN"; then
  echo "checkpoint changed during the copy ($BEFORE -> $AFTER); resubmit"; rm -f "$FROZEN"; exit 4
fi
echo "frozen copy: $FROZEN (source $SPINR_CKPT, mtime:size $BEFORE)"
python -c "from rift_pvc import accelerator as a; print(a.describe())"
set +e
python -W always -u scripts_pvc/gotcha_spinr_role_fit_pvc.py "$FROZEN" \
  --roles validation train --unit-stride 1 --device xpu --output "$OUT" -- \
  --dataset-root /scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined \
  --shard-root "$SPINR_SHARDS" --region "$TARGET_REGION" --region-config "$REGION_CONFIG" \
  --output-root "$DIR/${STEM}_plan_root" --polarizations hh --passes 1 2 3 4 5 6 7 8 --num-tx 1 --num-rx 1 \
  --pulses-per-sector 0 --frequency-stride 2 \
  --unit-split-stride 1 --unit-split-heldout-pass 4 --unit-split-heldout-fraction 0.1 \
  --method spinr --method-config "$CONFIG" --device xpu 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e
FALLBACKS=$(grep -c 'Aten Op fallback' "$LOG" || true)
echo "exit $STATUS, ATEN_FALLBACK_LINES=$FALLBACKS"; date
if [ "$STATUS" -eq 0 ] && [ "$FALLBACKS" -eq 0 ] && [ -f "$OUT" ]; then echo ROLE_FIT=pass; else echo ROLE_FIT=fail; exit 1; fi
