#!/bin/bash
# R1 training pilot of adaptive RIFT-SAS on AirSAS Armadillo5k (docs/RIFT_SAS_Train.md, A12/A13) on one PVC card.
# The production argument list of scripts_pvc_sas/run_airsas_comparison_pvc.sh (PROFILE=full) plus the R1
# representation (64^3 points on a 128^3 raster) and per-arm RIFT knobs; refinement and band probes are off in
# this pilot (A13). Training stops cooperatively after PILOT_TRAIN_SECONDS (SIGTERM -> checkpoint_latest.pt,
# status.json reason=signal), far before step 26,000, so the trainer's final TEST evaluation is never reached.
# A copy of checkpoint_latest.pt is taken at half time; both are read out on TRAIN64/VAL64 by elevation group and
# band (rift_sas_checkpoint_readout.py, never TEST), and the Lambertian/transmittance traces are extracted.
# Resume mode: PILOT_RESUME_RUN=<earlier run dir> continues that run in its original root with the same arm knobs
# (the trainer reconciles the saved recipe and restores model, calibration, optimizer and RNG state) via an
# explicit --resume of its checkpoint_latest.pt; the previous checkpoint_mid.pt/latest read-outs are kept under
# a resume-specific name and the new read-outs use checkpoint_resume_start_<job> (copy of the checkpoint resumed
# from, named per job so repeated resumes keep every start state) and checkpoint_latest.
# Knobs travel in --export (no commas in values):
#   sbatch --export=ALL,PILOT_TAG=P1,PILOT_COEFFICIENT_LR=1e-3,PILOT_INIT_SCALE=1e-2 scripts_pvc_sas/rift_sas_pilot_pvc.sbatch
#   sbatch --export=ALL,PILOT_TAG=P1_lr1e-3_init1e-2,PILOT_COEFFICIENT_LR=1e-3,PILOT_INIT_SCALE=1e-2,PILOT_RESUME_RUN=<run dir> scripts_pvc_sas/rift_sas_pilot_pvc.sbatch
#SBATCH --job-name=pvc-sas-r1-pilot
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
PKG=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG
LOG=$PKG/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log
CACHE="${AIRSAS_CACHE_ROOT:?}/armadillo_5khz_trainonly_120rings_v1_20260912"
TAG="${PILOT_TAG:?set PILOT_TAG}"
if [ -n "${PILOT_RESUME_RUN:-}" ]; then
  RUN="${PILOT_RESUME_RUN%/}"; OUT=$(dirname "$RUN"); NAME=$(basename "$RUN")
  [ "$NAME" = "airsas_armadillo5k_adaptive_rift_sas_r1pilot_${TAG}" ] || { echo "PILOT_TAG does not match $RUN"; echo "PILOT_EXIT=2"; exit 2; }
  [ -s "$RUN/checkpoint_latest.pt" ] || { echo "no checkpoint_latest.pt in $RUN"; echo "PILOT_EXIT=2"; exit 2; }
else
  OUT=$PKG/rift_sas_train/pilot/${TAG}_${SLURM_JOB_ID}
  NAME=airsas_armadillo5k_adaptive_rift_sas_r1pilot_${TAG}
  RUN=$OUT/$NAME
fi
BOUND=${PILOT_TRAIN_SECONDS:-2700}
mkdir -p "$OUT" || exit 1
hostname; date; python -c "from rift_pvc import accelerator as a; print(a.describe())"
python scripts/validate_airsas_5k_cache.py "$CACHE" || { echo "PILOT_EXIT=2"; exit 2; }

ARGS=(
  --cache "$CACHE" --model adaptive_rift_sas
  --checkpoint-root "$OUT" --checkpoint-name "$NAME"
  --require-explicit-splits --sh-degree 3 --num-rays 4900 --max-bins 110 --grad-clip 1.0
  --beamwidth-deg 30 --sh-direction rx_to_point --seed 42 --profile full
  # R1 representation (A12): 64^3 points splatted on a 128^3 raster
  --initial-granularity 64 --granularity 128 --adaptive-capacity 262144 --max-active 262144
  # pilot: refinement events and next-band probes off (A13)
  --refine-every 1000000000 --probe-every 1000000000
  --checkpoint-every "${PILOT_CHECKPOINT_EVERY:-500}"
  # per-arm RIFT knobs
  --coefficient-lr "${PILOT_COEFFICIENT_LR:-1e-3}" --position-lr "${PILOT_POSITION_LR:-1e-4}"
  --init-scale "${PILOT_INIT_SCALE:-1e-2}" --lambertian-ratio "${PILOT_LAMBERTIAN_RATIO:-0.0}"
  --opacity-scale "${PILOT_OPACITY_SCALE:-500}"
)
READOUTS="checkpoint_mid checkpoint_latest"
if [ -n "${PILOT_RESUME_RUN:-}" ]; then
  SUFFIX=resume_${SLURM_JOB_ID}
  START=checkpoint_resume_start_${SLURM_JOB_ID}   # job-named: a later resume must not overwrite an earlier start state
  cp "$RUN/checkpoint_latest.pt" "$RUN/$START.pt" || exit 1
  mv "$RUN/status.json" "$RUN/status_before_${SUFFIX}.json" 2>/dev/null   # a stale reason=signal must not pass the stop check
  ARGS+=(--resume "$RUN/checkpoint_latest.pt")
  READOUTS="$START checkpoint_latest"
  echo "resume mode: continuing $RUN from step $(python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['step'])" "$RUN/$START.pt")"
  MID=""
else
  SUFFIX=fresh
  ( sleep $((BOUND / 2)); cp "$RUN/checkpoint_latest.pt" "$RUN/checkpoint_mid.pt" 2>/dev/null \
      && echo "mid-run copy of checkpoint_latest.pt taken at $(date '+%T')" ) &
  MID=$!
fi
echo "pilot $TAG: ${ARGS[*]}"
start=$SECONDS
timeout --preserve-status --signal=TERM --kill-after=300s "${BOUND}s" python -u train_sas_pvc.py "${ARGS[@]}"
rc=$?; echo "=== training: exit=$rc wall=$((SECONDS - start))s ==="
[ -n "$MID" ] && { kill $MID 2>/dev/null; wait $MID 2>/dev/null; }
cat "$RUN/status.json" 2>/dev/null; echo
grep -q '"reason": "signal"' "$RUN/status.json" 2>/dev/null || { echo "no cooperative stop recorded"; rc=1; }

for ckpt in $READOUTS; do
  [ -s "$RUN/$ckpt.pt" ] || { echo "missing $ckpt.pt"; continue; }
  echo "=== read-out: $ckpt (TRAIN64/VAL64 by elevation group and band; never TEST) ==="
  python -u scripts_pvc_sas/rift_sas_checkpoint_readout.py "$RUN/$ckpt.pt" --device xpu \
    --train-pings 64 --val-pings 64 --json "$OUT/readout_${ckpt}_${SUFFIX}.json" || rc=1
done
echo "=== traces ==="
python scripts_pvc_sas/rift_sas_pilot_traces.py "$LOG" --csv "$OUT/traces_${SUFFIX}.csv" || true
cat "$RUN/history.csv" 2>/dev/null
date
n=$(grep -c 'fallback from XPU to CPU' "$LOG" || true); echo "XPU->CPU fallback lines in this log: $n"
[ "$n" != "0" ] && rc=1
echo "PILOT_EXIT=$rc"; exit $rc
