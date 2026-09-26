#!/bin/bash
# Linear capacity check of the sonar shell renderer (docs/RIFT_SAS_Train.md, B2/B3) on one PVC card.
# Knobs travel in --export and use ':' as the list separator (sbatch --export splits on commas):
#   sbatch --export=ALL,CAP_TAG=rays4900,CAP_GRIDS=32:64:128,CAP_NUM_RAYS=4900,CAP_TIME_BUDGET=900 scripts_pvc_sas/rift_sas_capacity_check_pvc.sbatch
#   sbatch --export=ALL,CAP_TAG=rays19600,CAP_GRIDS=128,CAP_NUM_RAYS=19600,CAP_TIME_BUDGET=2700 scripts_pvc_sas/rift_sas_capacity_check_pvc.sbatch
#   sbatch --export=ALL,CAP_TAG=adaptive_p64_r64,CAP_FIELD=adaptive,CAP_POINTS=64,CAP_GRIDS=64,CAP_TIME_BUDGET=2700 scripts_pvc_sas/rift_sas_capacity_check_pvc.sbatch
#SBATCH --job-name=pvc-sas-capacity
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
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
unset RIFT_ACCELERATOR RIFT_PVC_ALLOW_BACKEND   # never inherit a forced backend from the submitting shell
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTORCH_DEBUG_XPU_FALLBACK=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
source .env.datasets.sh
LOG=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log
OUT=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG/rift_sas_train
mkdir -p "$OUT"
CACHE="${AIRSAS_CACHE_ROOT:?}/armadillo_5khz_trainonly_120rings_v1_20260912"
TAG="${CAP_TAG:-default}"
GRIDS="${CAP_GRIDS:-32:64:128}"; FIT_RINGS="${CAP_FIT_RINGS:-4:6}"
POINTS_ARGS=(); [ -n "${CAP_POINTS:-}" ] && POINTS_ARGS=(--points ${CAP_POINTS//:/ })
hostname; date; python -c "from rift_pvc import accelerator as a; print(a.describe())"
python -u scripts_pvc_sas/rift_sas_capacity_check.py --cache "$CACHE" --device xpu \
  --grids ${GRIDS//:/ } --num-rays "${CAP_NUM_RAYS:-4900}" --degree "${CAP_DEGREE:-0}" \
  --fit-rings ${FIT_RINGS//:/ } --heldout-ring "${CAP_HELDOUT_RING:-5}" \
  --azimuth-stride "${CAP_AZIMUTH_STRIDE:-4}" --iterations "${CAP_ITERATIONS:-20}" \
  --time-budget "${CAP_TIME_BUDGET:-900}" \
  --field "${CAP_FIELD:-grid}" "${POINTS_ARGS[@]}" \
  --json "$OUT/capacity_${TAG}_${SLURM_JOB_ID}.json"
rc=$?; date
n=$(grep -c 'fallback from XPU to CPU' "$LOG" || true); echo "XPU->CPU fallback lines in this log: $n"
[ "$n" != "0" ] && rc=1
echo "CAPACITY_EXIT=$rc"; exit $rc
