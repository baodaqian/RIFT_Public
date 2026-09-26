#!/bin/bash
# Recovery-contract check for the adaptive RIFT trainer on one PVC card, mirroring
# the campaign manager: launch the production command bounded to PVC_EPOCHS
# (default 3), SIGTERM the whole process group after PVC_TERM_AFTER seconds
# (default 600: after epoch 1 completes), then relaunch the IDENTICAL command with
# --resume checkpoint_latest so the remaining epochs run from the restored state
# (model, optimizer, scheduler, gain, RNG incl. torch_xpu_all). Not a production run.
#SBATCH --job-name=pvc-recovery-b787-rift
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:45:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageD/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 WANDB_MODE=offline PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
OUT=${PVC_OUTPUT_ROOT:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageD/outputs/rift_recovery_${SLURM_JOB_ID}}
EPOCHS=${PVC_EPOCHS:-3}; TERM_AFTER=${PVC_TERM_AFTER:-600}
CMD=(python -u train_rift_dataset_pvc.py --dataset-root "$RIFT_DATA_ROOT" --output-root "$OUT" --object b787 --method rift --num-train 2400 --num-tx 1 --num-rx 1 --pvc-epochs "$EPOCHS")
hostname; date; echo "output root: $OUT"; python -c "from rift_pvc import accelerator as a; print(a.describe())"
echo "=== phase 1: fresh run, SIGTERM to the process group after ${TERM_AFTER}s ==="
setsid "${CMD[@]}" & PID=$!; sleep 2; PGID=$(ps -o pgid= "$PID" | tr -d ' '); echo "child pid=$PID pgid=$PGID"
sleep "$TERM_AFTER"; date; echo "TIME_LIMIT_OR_TERMINATION: sending SIGTERM to process group $PGID"; kill -TERM -- "-$PGID"
wait "$PID"; rc1=$?; date; echo "PHASE1_EXIT=$rc1"
CK=$(ls "$OUT"/train2400/*/b787/rift_pvcsmoke/checkpoint_latest.pth.tar 2>/dev/null | head -1)
[ -n "$CK" ] || { echo "no checkpoint after interruption"; echo "RECOVERY_EXIT=2"; exit 2; }
ls -la "$CK"
echo "=== phase 2: identical command with --resume ==="
"${CMD[@]}" --resume "$CK"; rc2=$?; date; echo "PHASE2_EXIT=$rc2"; echo "RECOVERY_EXIT=$rc2"; exit $rc2
