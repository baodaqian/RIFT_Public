#!/bin/bash
# Read-out-only split of the production renderer's loss into λ gate and opacity (docs/RIFT_SAS_Train.md, A29;
# the S5 split of B20/A25) on one PVC card: scripts_pvc_sas/rift_sas_renderer_split.py reads each checkpoint out on
# TRAIN64/VAL64 (never TEST) under the renderers L (1, 0), G (0, 0), O (1, 500) and P (0, 500). No training; the
# checkpoints are only read, so pass copies (checkpoint_step0.pt, checkpoint_valread_<job>.pt,
# checkpoint_resume_start_<job>.pt), never a live run's checkpoint_latest.pt.
# Knobs travel in --export (no commas in values); SPLIT_CHECKPOINTS is colon-separated:
#   sbatch --export=ALL,SPLIT_CHECKPOINTS=<ckpt a>:<ckpt b> scripts_pvc_sas/rift_sas_renderer_split_pvc.sbatch
# SPLIT_EXIT=0: every read-out written, 0 fallback lines; 1 otherwise.
#SBATCH --job-name=pvc-sas-renderer-split
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
unset RIFT_ACCELERATOR RIFT_PVC_ALLOW_BACKEND   # never inherit a forced backend from the submitting shell
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTORCH_DEBUG_XPU_FALLBACK=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
source .env.datasets.sh
PKG=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG
LOG=$PKG/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log
OUT=$PKG/rift_sas_train/renderer_split_${SLURM_JOB_ID}
IFS=: read -r -a CKPTS <<< "${SPLIT_CHECKPOINTS:?set SPLIT_CHECKPOINTS (colon-separated)}"
for ckpt in "${CKPTS[@]}"; do
  [ -s "$ckpt" ] || { echo "missing checkpoint $ckpt"; echo "SPLIT_EXIT=1"; exit 1; }
  case "$(basename "$ckpt")" in checkpoint_latest.pt) echo "refusing a live checkpoint_latest.pt: $ckpt"; echo "SPLIT_EXIT=1"; exit 1;; esac
done
hostname; date; python -c "from rift_pvc import accelerator as a; print(a.describe())"
rc=0
python -u scripts_pvc_sas/rift_sas_renderer_split.py "${CKPTS[@]}" --out-dir "$OUT" --device xpu || rc=1
date
n=$(grep -c 'fallback from XPU to CPU' "$LOG" || true); echo "XPU->CPU fallback lines in this log: $n"
[ "$n" != "0" ] && rc=1
echo "SPLIT_OUT=$OUT"
echo "SPLIT_EXIT=$rc"; exit $rc
