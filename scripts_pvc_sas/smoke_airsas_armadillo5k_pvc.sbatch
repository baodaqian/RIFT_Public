#!/bin/bash
# Acceptance 7c of the sonar PVC port (Package G): bounded real-data smoke of
# one AirSAS Armadillo5k comparison arm on one PVC card through the PVC twin of
# the production launcher (scripts_pvc_sas/run_airsas_comparison_pvc.sh,
# PROFILE=full: the production argument list unaltered, checkpoint name
# suffixed _pvcsmoke). Phase 1 runs until SIGTERM after PVC_TERM_AFTER seconds
# (cooperative stop: checkpoint_latest.pt + status.json reason=signal), phase 2
# is the identical command with RESUME_CHECKPOINT set (explicit --resume).
# Optional phase 3 (PVC_VALIDATION_READOUT=1): --eval-only on the validation
# role only; the test role is never evaluated by a smoke. Requires the staged
# cache (prerequisite P0; scripts_pvc_sas/prepare_airsas_cache_pvc.sbatch).
# Resume-only mode: PVC_RESUME_FROM_RUN=<run dir of an earlier phase 1> skips
# phase 1 and resumes that run (PVC_PHASE1_EXIT / PVC_PHASE1_WALL carry the
# earlier phase's outcome into the report; status_phase1.json must exist).
#   sbatch --export=ALL,MODEL=adaptive_rift_sas,AIRSAS_CACHE_ROOT=... scripts_pvc_sas/smoke_airsas_armadillo5k_pvc.sbatch
#SBATCH --job-name=pvc-smoke-airsas-arm5k
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
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
MODEL=${MODEL:-adaptive_rift_sas}
case "$MODEL" in adaptive_rift_sas|sh_sas) ;; *) echo "MODEL must be adaptive_rift_sas or sh_sas"; exit 2 ;; esac
PKG=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG
LOG=$PKG/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log
: "${AIRSAS_CACHE_ROOT:?set AIRSAS_CACHE_ROOT (see .env.datasets.sh) to the directory holding the staged caches}"
CACHE=${AIRSAS_CACHE:-$AIRSAS_CACHE_ROOT/armadillo_5khz_trainonly_120rings_v1_20260912}
OUT=${PVC_OUTPUT_ROOT:-$PKG/outputs/airsas_armadillo5k_${MODEL}_${SLURM_JOB_ID}}
NAME=airsas_armadillo5k_${MODEL}_full_v1_pvcsmoke
TERM_AFTER=${PVC_TERM_AFTER:-1500}; RESUME_AFTER=${PVC_RESUME_AFTER:-900}
hostname; date; echo "model: $MODEL  cache: $CACHE  output: $OUT"
python -c "from rift_pvc import accelerator as a; print(a.describe())"
[ -s "$CACHE/manifest.json" ] || { echo "cache is not staged: $CACHE (prerequisite P0)"; echo "SMOKE_EXIT=2"; exit 2; }
mkdir -p "$OUT" || exit 1

run_stage () {   # run_stage <label> <seconds> <cmd...>: SIGTERM to the trainer after <seconds>
  local label=$1 bound=$2; shift 2
  echo "=== ${label}: $(date '+%F %T'), SIGTERM after ${bound}s ==="
  local start=$SECONDS
  timeout --preserve-status --signal=TERM --kill-after=600s "${bound}s" "$@"
  local rc=$? elapsed=$((SECONDS-start))
  echo "=== ${label}: exit=${rc} wall=${elapsed}s ==="
  LAST_WALL=$elapsed
  return $rc
}
if [ -n "${PVC_RESUME_FROM_RUN:-}" ]; then
  RUN="$PVC_RESUME_FROM_RUN"; OUT=$(dirname "$RUN"); NAME=$(basename "$RUN")
  [ -s "$RUN/checkpoint_latest.pt" ] && [ -s "$RUN/status_phase1.json" ] || { echo "resume-only mode needs checkpoint_latest.pt and status_phase1.json in $RUN"; echo "SMOKE_EXIT=2"; exit 2; }
  rc1=${PVC_PHASE1_EXIT:?set PVC_PHASE1_EXIT}; wall1=${PVC_PHASE1_WALL:?set PVC_PHASE1_WALL}
  echo "resume-only mode: phase 1 taken from $RUN (exit $rc1, wall ${wall1}s)"
else
  export SCENE=armadillo MODEL PROFILE=full CACHE CHECKPOINT_ROOT="$OUT" CHECKPOINT_NAME="$NAME"
  run_stage "phase 1: fresh run (production launcher twin)" "$TERM_AFTER" bash scripts_pvc_sas/run_airsas_comparison_pvc.sh; rc1=$?; wall1=$LAST_WALL
  RUN="$OUT/$NAME"
  cp "$RUN/status.json" "$RUN/status_phase1.json" 2>/dev/null || true
fi
export SCENE=armadillo MODEL PROFILE=full CACHE CHECKPOINT_ROOT="$OUT" CHECKPOINT_NAME="$NAME"
ls -la "$RUN" 2>/dev/null
RESUME_CHECKPOINT="$RUN/checkpoint_latest.pt" run_stage "phase 2: identical command with RESUME_CHECKPOINT" "$RESUME_AFTER" \
  bash scripts_pvc_sas/run_airsas_comparison_pvc.sh; rc2=$?; wall2=$LAST_WALL
date
python scripts_pvc_sas/smoke_report_pvc.py --run-dir "$RUN" --log "$LOG" --model "$MODEL" \
  --phase1-exit "$rc1" --phase2-exit "$rc2" --phase1-wall "$wall1" --phase2-wall "$wall2" \
  --status-phase1 "$RUN/status_phase1.json" --expected-steps 26000 --output "$OUT/smoke_report_${SLURM_JOB_ID}.json"
rc=$?
if [ "${PVC_VALIDATION_READOUT:-0}" = "1" ] && [ $rc -eq 0 ]; then
  echo "=== phase 3: validation-only read-out on checkpoint_latest (never the test role) ==="
  python -u train_sas_pvc.py --cache "$CACHE" --model "$MODEL" --checkpoint-root "$OUT" --checkpoint-name "$NAME" \
    --resume "$RUN/checkpoint_latest.pt" --eval-only --evaluation-role validation --eval-pings 0 --eval-bins 0 \
    --require-explicit-splits --sh-degree 3 --num-rays 4900 --max-bins 110 --grad-clip 1.0 --beamwidth-deg 30 \
    --sh-direction rx_to_point --seed 42; rc3=$?
  echo "PHASE3_EXIT=$rc3"; [ $rc3 -ne 0 ] && rc=$rc3
  cat "$RUN/validation_eval.json" 2>/dev/null | head -20
fi
echo "PHASE1_EXIT=$rc1 PHASE2_EXIT=$rc2 SMOKE_EXIT=$rc"; exit $rc
