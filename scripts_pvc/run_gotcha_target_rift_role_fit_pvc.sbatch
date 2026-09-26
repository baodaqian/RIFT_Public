#!/bin/bash
# TRAIN/VAL role-fit for a RIFT target checkpoint. Header copied from the SpINR role-fit wrapper,
# INCLUDING --exclude=ac023: that node hung three role-fits on 2026-09-25 (CPU accruing, no forward
# progress, dying at XPU init/teardown) and 46 of our launchers already exclude it.
# Knobs: RIFT_CKPT, TARGET_REGION, TARGET_REGION_CONFIG, RIFT_SHARDS, OUT, RIFT_SPLIT_STRIDE,
#        RIFT_HELDOUT_FRACTION.
#SBATCH --job-name=pvc-target-rift-rolefit
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
: "${RIFT_CKPT:?}" "${TARGET_REGION:?}" "${RIFT_SHARDS:?}" "${OUT:?}"
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
set -u
unset RIFT_PVC_ALLOW_BACKEND
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=xpu PYTORCH_DEBUG_XPU_FALLBACK=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
[ -e "$OUT" ] && { echo "$OUT exists"; exit 3; }
mkdir -p "$(dirname "$OUT")"
# the checkpoint is a frozen copy; record it either side so a silent change cannot pass unnoticed
SIG_BEFORE=$(stat -c '%Y:%s' "$RIFT_CKPT")
hostname; date
python -c "from rift_pvc import accelerator as a; print(a.describe())"
rc=0
python -u scripts_pvc/gotcha_rift_role_fit_pvc.py "$RIFT_CKPT" \
  --roles validation train --output "$OUT" -- \
  --dataset-root /scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined \
  --shard-root "$RIFT_SHARDS" --region "$TARGET_REGION" \
  --region-config "${TARGET_REGION_CONFIG:-rift_pvc/regions/gotcha_new_targets.json}" \
  --polarizations hh --passes 1 2 3 4 5 6 7 8 --num-tx 1 --num-rx 1 --pulses-per-sector 0 \
  --frequency-stride 2 --unit-split-stride "${RIFT_SPLIT_STRIDE:-1}" --unit-split-heldout-pass 4 \
  --unit-split-heldout-fraction "${RIFT_HELDOUT_FRACTION:-0.1}" || rc=$?
SIG_AFTER=$(stat -c '%Y:%s' "$RIFT_CKPT")
[ "$SIG_BEFORE" = "$SIG_AFTER" ] || { echo "checkpoint CHANGED during the read ($SIG_BEFORE -> $SIG_AFTER)"; rc=4; }
n=$(grep -c 'Aten Op fallback' "/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_newcar_final_20260925/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log" 2>/dev/null || true)
date; echo "fallback lines: ${n:-0}"; echo "ROLE_FIT=$([ $rc -eq 0 ] && echo pass || echo fail:$rc)"; exit $rc
