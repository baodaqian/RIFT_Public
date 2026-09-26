#!/bin/bash
# Synthetic pre-smoke of train_sas_pvc.py on one PVC card (Package G): the
# Armadillo5k production argument list (--profile full: 26000 steps, 4900 rays,
# 110 loss bins) on a synthetic cache of the Armadillo5k shape (120 rings x 360
# pings, 556 bins; identity synthetic_sas_rings_v1, never an AirSAS identity),
# bounded by wall clock: SIGTERM after PVC_TERM_AFTER seconds (the trainer
# saves checkpoint_latest.pt after the current step and writes status.json
# reason=signal), then the IDENTICAL command with --resume for PVC_RESUME_AFTER
# seconds. Reports s/step, refinement events and XPU->CPU fallbacks. Not a
# production run and not the AirSAS smoke (that one needs the staged cache).
#   sbatch --export=ALL,MODEL=adaptive_rift_sas scripts_pvc_sas/presmoke_synthetic_pvc.sbatch
#SBATCH --job-name=pvc-presmoke-sas-synth
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
unset RIFT_ACCELERATOR RIFT_PVC_ALLOW_BACKEND   # never inherit a forced backend from the submitting shell
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
MODEL=${MODEL:-adaptive_rift_sas}
case "$MODEL" in adaptive_rift_sas|sh_sas|rift_sas) ;; *) echo "MODEL must be adaptive_rift_sas, sh_sas or rift_sas"; exit 2 ;; esac
PKG=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG
LOG=$PKG/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log
CACHE=${SAS_SYNTH_CACHE:-$PKG/sas_data_synthetic/synthetic_rings120_v1}
OUT=${PVC_OUTPUT_ROOT:-$PKG/outputs/presmoke_synth_${MODEL}_${SLURM_JOB_ID}}
NAME=synthetic_rings120_${MODEL}_full_v1
TERM_AFTER=${PVC_TERM_AFTER:-900}; RESUME_AFTER=${PVC_RESUME_AFTER:-480}
hostname; date; echo "model: $MODEL  cache: $CACHE  output: $OUT"
python -c "from rift_pvc import accelerator as a; print(a.describe())"
mkdir -p "$OUT" || exit 1
python -u rift_pvc_sas/synthetic.py --output "$CACHE" --production-shape --seed 0 || { echo "SMOKE_EXIT=1"; exit 1; }
python -c "import json,sys; m=json.load(open(sys.argv[1])); print('cache identity', m['dataset_identity'], m['num_pings'], 'pings', m['num_bins'], 'bins', m['split_contract']['counts'])" "$CACHE/manifest.json"
CMD=(python -u train_sas_pvc.py --cache "$CACHE" --model "$MODEL" --checkpoint-root "$OUT" --checkpoint-name "$NAME"
     --require-explicit-splits --sh-degree 3 --num-rays 4900 --max-bins 110 --grad-clip 1.0 --beamwidth-deg 30
     --sh-direction rx_to_point --seed 42 --profile full)
echo "command: ${CMD[*]}"

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

run_stage "phase 1: fresh run" "$TERM_AFTER" "${CMD[@]}"; rc1=$?; wall1=$LAST_WALL
RUN="$OUT/$NAME"
cp "$RUN/status.json" "$RUN/status_phase1.json" 2>/dev/null || true
ls -la "$RUN" 2>/dev/null
run_stage "phase 2: identical command with --resume" "$RESUME_AFTER" "${CMD[@]}" --resume "$RUN/checkpoint_latest.pt"; rc2=$?; wall2=$LAST_WALL
date
python scripts_pvc_sas/smoke_report_pvc.py --run-dir "$RUN" --log "$LOG" --model "$MODEL" \
  --phase1-exit "$rc1" --phase2-exit "$rc2" --phase1-wall "$wall1" --phase2-wall "$wall2" \
  --status-phase1 "$RUN/status_phase1.json" --expected-steps 26000 --output "$OUT/smoke_report_${SLURM_JOB_ID}.json"
rc=$?
echo "PHASE1_EXIT=$rc1 PHASE2_EXIT=$rc2 SMOKE_EXIT=$rc"; exit $rc
