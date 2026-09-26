#!/bin/bash
# Read-out-only estimate of a declared output band-limit (docs/RIFT_SAS_Train.md, A43; §4 item 2 option (a)) on one PVC
# card: scripts_pvc_sas/rift_sas_band_limit_readout.py renders each checkpoint on TRAIN64 (gain fit) and on all 4,320
# VAL pings (never TEST) and scores the output with bands zeroed and per-band gains (none / neg / neg_high / in_band / pb_neg_high / pb_all). No training.
# Every checkpoint is first copied into this job's output directory and only the copy is read, so a live
# checkpoint_latest.pt of a finished run may be passed.
# Knobs travel in --export (no commas in values); BAND_CHECKPOINTS is colon-separated:
#   sbatch --export=ALL,BAND_CHECKPOINTS=<ckpt a>:<ckpt b> scripts_pvc_sas/rift_sas_band_limit_readout_pvc.sbatch
# BAND_EXIT=0: report written, 0 fallback lines; 1 otherwise.
#SBATCH --job-name=pvc-sas-band-limit
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
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
OUT=$PKG/rift_sas_train/band_limit_${SLURM_JOB_ID}
mkdir -p "$OUT" || exit 1
IFS=: read -r -a SOURCES <<< "${BAND_CHECKPOINTS:?set BAND_CHECKPOINTS (colon-separated)}"
COPIES=()
for i in "${!SOURCES[@]}"; do
  src=${SOURCES[$i]}
  [ -s "$src" ] || { echo "missing checkpoint $src"; echo "BAND_EXIT=1"; exit 1; }
  cp "$src" "$OUT/ckpt${i}.pt" || { echo "BAND_EXIT=1"; exit 1; }
  echo "ckpt${i}.pt <- $src"
  COPIES+=("$OUT/ckpt${i}.pt")
done
hostname; date; python -c "from rift_pvc import accelerator as a; print(a.describe())"
rc=0
python -u scripts_pvc_sas/rift_sas_band_limit_readout.py "${COPIES[@]}" --device xpu --train-pings 64 --val-pings 0 \
  --json "$OUT/band_limit.json" || rc=1
rm -f "${COPIES[@]}"   # the copies are only for reading; the report names them and the mapping is printed above
date
n=$(grep -c 'fallback from XPU to CPU' "$LOG" || true); echo "XPU->CPU fallback lines in this log: $n"
[ "$n" != "0" ] && rc=1
echo "BAND_OUT=$OUT"
echo "BAND_EXIT=$rc"; exit $rc
