#!/bin/bash
# Authoritative full-VAL read-out of a sonar pilot checkpoint (docs/RIFT_SAS_Train.md §2: --eval-only
# --evaluation-role validation --eval-pings 0 --eval-bins 0, all 4,320 VAL pings x 326 bins; never TEST).
# Runs the unchanged trainer entry point in eval-only mode on <run>/checkpoint_latest.pt with the pilot's recipe
# flags (the trainer refuses a mismatch); writes <run>/validation_eval.json (moved to validation_eval_<job>.json).
# Knobs in --export (no commas): PILOT_RESUME_RUN=<run dir> and the run's PILOT_* knobs.
#   sbatch --export=ALL,PILOT_RESUME_RUN=<run dir>,PILOT_COEFFICIENT_LR=1e-4,PILOT_INIT_SCALE=1e-2 scripts_pvc_sas/rift_sas_val_readout_pvc.sbatch
# VAL_EXIT=0: validation_eval.json written, complete, 0 fallback lines; 1 otherwise. No training.
#SBATCH --job-name=pvc-sas-val-readout
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
unset RIFT_ACCELERATOR RIFT_PVC_ALLOW_BACKEND
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTORCH_DEBUG_XPU_FALLBACK=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
source .env.datasets.sh
PKG=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG
LOG=$PKG/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log
CACHE="${AIRSAS_CACHE_ROOT:?}/armadillo_5khz_trainonly_120rings_v1_20260912"
RUN="${PILOT_RESUME_RUN:?set PILOT_RESUME_RUN}"; RUN="${RUN%/}"
[ -s "$RUN/checkpoint_latest.pt" ] || { echo "no checkpoint_latest.pt in $RUN"; echo "VAL_EXIT=1"; exit 1; }
cp "$RUN/checkpoint_latest.pt" "$RUN/checkpoint_valread_${SLURM_JOB_ID}.pt" || exit 1
hostname; date; python -c "from rift_pvc import accelerator as a; print(a.describe())"
python -u train_sas_pvc.py --cache "$CACHE" --model adaptive_rift_sas \
  --checkpoint-root "$(dirname "$RUN")" --checkpoint-name "$(basename "$RUN")" \
  --resume "$RUN/checkpoint_valread_${SLURM_JOB_ID}.pt" --eval-only --evaluation-role validation --eval-pings 0 --eval-bins 0 \
  --require-explicit-splits --sh-degree 3 --num-rays 4900 --max-bins 110 --grad-clip 1.0 --beamwidth-deg 30 \
  --sh-direction rx_to_point --seed 42 \
  --initial-granularity 64 --granularity 128 --adaptive-capacity 262144 --max-active 262144 \
  --refine-every 1000000000 --probe-every 1000000000 \
  --coefficient-lr "${PILOT_COEFFICIENT_LR:?}" --position-lr "${PILOT_POSITION_LR:-1e-4}" \
  --init-scale "${PILOT_INIT_SCALE:?}" --lambertian-ratio "${PILOT_LAMBERTIAN_RATIO:-0.0}" \
  --opacity-scale "${PILOT_OPACITY_SCALE:-500}"
rc=$?
if [ $rc -eq 0 ] && [ -s "$RUN/validation_eval.json" ]; then
  mv "$RUN/validation_eval.json" "$RUN/validation_eval_${SLURM_JOB_ID}.json"
  python -c "import json,sys; r=json.load(open(sys.argv[1])); print('full VAL rel_mse', r['rel_mse'], 'views', r['views'], 'complete', r['complete'], 'checkpoint_step', r['checkpoint_step']); sys.exit(0 if r['complete'] == 1.0 and int(r['views']) == 4320 else 1)" "$RUN/validation_eval_${SLURM_JOB_ID}.json" || rc=1
else
  rc=1
fi
date
n=$(grep -c 'fallback from XPU to CPU' "$LOG" || true); echo "XPU->CPU fallback lines in this log: $n"
[ "$n" != "0" ] && rc=1
echo "VAL_EXIT=$rc"; exit $rc
