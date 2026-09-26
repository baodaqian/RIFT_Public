#!/bin/bash
# CGLS warm start of adaptive RIFT-SAS (docs/RIFT_SAS_Train.md, A17/B13/A20/A21) on one PVC card.
# Builds the step-0 trainer checkpoint of a pilot run (R1 representation, recipe flags, lambertian_ratio from
# PILOT_LAMBERTIAN_RATIO, default the recipe's 0) with scripts_pvc_sas/rift_sas_warm_start.py, keeps a copy as
# checkpoint_step0.pt, and reads it out on TRAIN64/VAL64 by elevation group and band under the production renderer
# before any Adam step (never TEST). No training happens here; the run continues through
# rift_sas_pilot_pvc.sbatch in resume mode (PILOT_RESUME_RUN=<run dir>, PILOT_TAG=<same tag>,
# PILOT_COEFFICIENT_LR=<the CONTINUE_COEFFICIENT_LR printed here>, the other PILOT_* knobs unchanged).
# Knobs travel in --export (no commas in values):
#   sbatch --export=ALL,PILOT_TAG=W1_cgls_s8 scripts_pvc_sas/rift_sas_warm_start_pvc.sbatch
# Chained CGLS leg (A29): WARM_CONTINUE_STATE=<earlier run>/cgls_state.pt continues that CGLS exactly;
# WARM_RESTART_FROM=<earlier run>/checkpoint_step0.pt restarts it from that field (a warm start without a state
# file, i.e. W1). Either writes a new step-0 run under the new PILOT_TAG; the source run is only read.
#   sbatch --export=ALL,PILOT_TAG=W2_cgls_s8,WARM_RESTART_FROM=<W1 run>/checkpoint_step0.pt,WARM_CGLS_SECONDS=4200 scripts_pvc_sas/rift_sas_warm_start_pvc.sbatch
# WARM_EXIT=0: step-0 checkpoint written and read out, no fallback; 1: anything else. One leg; never a run end.
#SBATCH --job-name=pvc-sas-warm-start
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:20:00
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
CACHE="${AIRSAS_CACHE_ROOT:?}/armadillo_5khz_trainonly_120rings_v1_20260912"
TAG="${PILOT_TAG:?set PILOT_TAG}"
OUT=$PKG/rift_sas_train/pilot/${TAG}_${SLURM_JOB_ID}
NAME=airsas_armadillo5k_adaptive_rift_sas_r1pilot_${TAG}
RUN=$OUT/$NAME
mkdir -p "$OUT" || exit 1
hostname; date; python -c "from rift_pvc import accelerator as a; print(a.describe())"
python scripts/validate_airsas_5k_cache.py "$CACHE" || { echo "WARM_EXIT=1"; exit 1; }

# the exact pilot argv (rift_sas_pilot_pvc.sbatch); --coefficient-lr is replaced by the warm start's value
ARGS=(
  --cache "$CACHE" --model adaptive_rift_sas
  --checkpoint-root "$OUT" --checkpoint-name "$NAME"
  --require-explicit-splits --sh-degree 3 --num-rays 4900 --max-bins 110 --grad-clip 1.0
  --beamwidth-deg 30 --sh-direction rx_to_point --seed 42 --profile full
  --initial-granularity 64 --granularity 128 --adaptive-capacity 262144 --max-active 262144
  --refine-every 1000000000 --probe-every 1000000000
  --checkpoint-every "${PILOT_CHECKPOINT_EVERY:-500}"
  --coefficient-lr "${PILOT_COEFFICIENT_LR:-1e-4}" --position-lr "${PILOT_POSITION_LR:-1e-4}"
  --init-scale "${PILOT_INIT_SCALE:-1e-2}" --lambertian-ratio "${PILOT_LAMBERTIAN_RATIO:-0.0}"
  --opacity-scale "${PILOT_OPACITY_SCALE:-500}"
)
echo "warm start $TAG: ${ARGS[*]}"
echo "CGLS source: continue-state=${WARM_CONTINUE_STATE:-none} restart-from=${WARM_RESTART_FROM:-none} budget=${WARM_CGLS_SECONDS:-2700}s"
rc=0
python -u scripts_pvc_sas/rift_sas_warm_start.py --fit-stride "${WARM_FIT_STRIDE:-8}" \
  --iterations "${WARM_CGLS_ITERATIONS:-20}" --time-budget "${WARM_CGLS_SECONDS:-2700}" \
  --target-t-min "${WARM_TARGET_T_MIN:-0.8}" --lr-fraction "${WARM_LR_FRACTION:-0.01}" \
  ${WARM_CONTINUE_STATE:+--continue-state "$WARM_CONTINUE_STATE"} ${WARM_RESTART_FROM:+--restart-from "$WARM_RESTART_FROM"} \
  -- "${ARGS[@]}" || rc=1
if [ $rc -eq 0 ] && [ -s "$RUN/checkpoint_latest.pt" ]; then
  cp "$RUN/checkpoint_latest.pt" "$RUN/checkpoint_step0.pt" || rc=1
  echo "=== step-0 read-out (production renderer, before any Adam step; TRAIN64/VAL64, never TEST) ==="
  python -u scripts_pvc_sas/rift_sas_checkpoint_readout.py "$RUN/checkpoint_step0.pt" --device xpu \
    --train-pings 64 --val-pings 64 --json "$OUT/readout_checkpoint_step0.json" || rc=1
else
  rc=1
fi
date
n=$(grep -c 'fallback from XPU to CPU' "$LOG" || true); echo "XPU->CPU fallback lines in this log: $n"
[ "$n" != "0" ] && rc=1
echo "RUN_DIR=$RUN"
[ -s "$RUN/cgls_state.pt" ] && echo "CGLS_STATE=$RUN/cgls_state.pt"
grep -h '^CONTINUE_COEFFICIENT_LR=' "$LOG" | tail -1
echo "WARM_EXIT=$rc"; exit $rc
