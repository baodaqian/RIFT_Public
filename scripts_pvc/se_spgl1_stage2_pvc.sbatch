#!/bin/bash
# SE SPGL1 Stage 2 (scripts_pvc/se_spgl1_stage1.py stage2) on one PVC card.
#
# Resumes the unchanged paper-v1 Stage-2 workflow from the Stage-1 checkpoint that
# `assemble` wrote, so the source flags below MUST be byte-identical to the ones the
# solve/assemble jobs used: the acquisition and recipe identity are re-checked on
# resume. The driver is taken from the campaign root's own source snapshot, never
# from the working tree, for the same reason.
#
# Usage: export SCENE=b787 SE_ROOT=/scratch/.../se_spgl1_20260922
#        sbatch --export=ALL --job-name se-spgl1-b787-stage2-smoke \
#               --output $SE_ROOT/logs/<name>-%j.log scripts_pvc/se_spgl1_stage2_pvc.sbatch
#
# Bounded smoke: the default 40 min wall clock plus the cooperative TERM below stops
# it cleanly at exit 75 with a resumable checkpoint. Raise --time on the sbatch command
# line for a full run; resuming is re-running this identical script.
#
# Exit codes (paper-v1 trainer): 0 complete, 2 incomplete (see status.json, e.g.
# surface_unavailable / incomplete_iso_supervision), 75 interrupted -> resume.
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
# B: is required. Without it Slurm signals the job STEPS, and this script runs
# python directly in the batch step, so nothing receives the signal and the job
# is killed at the wall clock with no clean exit 75 (observed on smoke 2156277).
#SBATCH --signal=B:TERM@180
set -eo pipefail
: "${SCENE:?set SCENE}"
: "${SE_ROOT:?set SE_ROOT}"
cd "$SE_ROOT/source"
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=xpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLCONFIGDIR="$SE_ROOT/runtime/$SCENE-stage2/mpl"
export XDG_CACHE_HOME="$SE_ROOT/runtime/$SCENE-stage2/cache"
export TMPDIR="$SE_ROOT/runtime/$SCENE-stage2/tmp"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TMPDIR"

C1=/scratch/group/p.cis261724.000/RIFT_pvc_runs/production_20260921_35jobs
MANIFEST=$C1/outputs/rift/train2400/1t1r_2010b2bbe725/$SCENE/role_manifest.json
OUT=$SE_ROOT/outputs/rift/train2400/1t1r_2010b2bbe725/$SCENE/sugavanam_ertin_spgl1

date
echo "scene=$SCENE job=$SLURM_JOB_ID node=$SLURMD_NODENAME output=$OUT"

# Slurm's --signal=B: delivers TERM to THIS shell only, and bash does not forward
# signals to a foreground child. So run the driver in the background and forward
# TERM explicitly; `exec python` would also work but would discard the epilogue.
set +e
python -u scripts_pvc/se_spgl1_stage1.py stage2 \
  --npz-path /scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645/inputs/RIFT_dataset/objects/$SCENE.npz \
  --parent-role-manifest "$MANIFEST" \
  --config $C1/source/protocols/se_g40_readout48.json \
  --output "$OUT" &
PID=$!
trap 'echo "forwarding TERM to $PID"; kill -TERM "$PID" 2>/dev/null' TERM
wait "$PID"
STATUS=$?
# A trap firing during `wait` makes it return >128; wait again for the real status.
if [ "$STATUS" -gt 128 ]; then wait "$PID"; STATUS=$?; fi
set -e

echo "STAGE2_EXIT=$STATUS"
case $STATUS in
  0)  echo "STAGE2_RESULT=complete" ;;
  75) echo "STAGE2_RESULT=interrupted_resumable (re-run this identical script)" ;;
  2)  echo "STAGE2_RESULT=incomplete (see $OUT/status.json)" ;;
  *)  echo "STAGE2_RESULT=error" ;;
esac
grep -c 'Aten Op fallback' "$SE_ROOT/logs/"*stage2*"$SLURM_JOB_ID"*.log 2>/dev/null | sed 's/^/ATEN_FALLBACK_LINES=/' || true
date
exit $STATUS
