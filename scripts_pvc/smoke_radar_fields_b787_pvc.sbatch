#!/bin/bash
# Gate 3 of the Radar Fields PVC port: bounded real-data smoke on one PVC card.
# B787, the production argument list of train_rift_dataset_pvc.py --method
# radar_fields (source-adapted-v3, 2400 train views, 1t1r, 960 steps, --device
# rewritten to xpu, --checkpoint-name suffixed _pvcsmoke). source-adapted-v3 fixes
# --steps at parse time, so the run is bounded by wall clock: SIGTERM after
# PVC_TERM_AFTER seconds (the trainer saves checkpoint_latest after the current
# step and exits 0), then the IDENTICAL command with --resume runs again for
# PVC_RESUME_AFTER seconds. Reports s/step and XPU->CPU fallbacks. Not a
# production run.
#SBATCH --job-name=pvc-smoke-b787-radar-fields
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
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageE/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT || exit 1
source .local-setup/activate-pvc.sh || exit 1
export RIFT_ACCELERATOR=xpu
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 WANDB_MODE=offline PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
LOG=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageE/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log
OUT=${PVC_OUTPUT_ROOT:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageE/outputs/radar_fields_smoke_${SLURM_JOB_ID}}
TERM_AFTER=${PVC_TERM_AFTER:-1200}; RESUME_AFTER=${PVC_RESUME_AFTER:-600}
PLAN="$OUT/plan_${SLURM_JOB_ID}.json"
hostname; date; echo "output root: $OUT"
python -c "from rift_pvc import accelerator as a; assert a.backend() == 'xpu' and a.device_count() == 1; print(a.describe())" || exit 1
mkdir -p "$OUT" || exit 1
python -u scripts_pvc/plan_radar_fields_pvc.py --dataset-root "$RIFT_DATA_ROOT" --output-root "$OUT" \
  --object b787 --num-train 2400 --num-tx 1 --num-rx 1 > "$PLAN" || { echo "SMOKE_EXIT=1"; exit 1; }
cat "$PLAN"
python -c "import json,sys;[print(a) for a in json.load(open(sys.argv[1]))['command']]" "$PLAN" > "$OUT/command_${SLURM_JOB_ID}.txt" || exit 1
mapfile -t CMD < "$OUT/command_${SLURM_JOB_ID}.txt" || exit 1
CKDIR=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['output_dir'])" "$PLAN") || exit 1

run_stage () {   # run_stage <label> <seconds> <cmd...>: SIGTERM to the trainer after <seconds>
  local label=$1 bound=$2; shift 2
  echo "=== ${label}: $(date '+%F %T'), SIGTERM after ${bound}s ==="
  local start=$SECONDS
  timeout --preserve-status --signal=TERM --kill-after=900s "${bound}s" "$@"
  local rc=$? elapsed=$((SECONDS-start))
  echo "=== ${label}: exit=${rc} wall=${elapsed}s ==="
  return $rc
}

FIRST="$OUT/fresh_${SLURM_JOB_ID}.json"
FRESH_LOG="$OUT/fresh_${SLURM_JOB_ID}.log"
RESUME_LOG="$OUT/resume_${SLURM_JOB_ID}.log"
run_stage "phase 1: fresh run" "$TERM_AFTER" "${CMD[@]}" 2>&1 | tee "$FRESH_LOG"
phase_codes=("${PIPESTATUS[@]}")
rc1=${phase_codes[0]}; [ "${phase_codes[1]}" -eq 0 ] || exit 1
python scripts_pvc/radar_fields_smoke_report.py --dataset rift --root "$CKDIR" --log "$FRESH_LOG" \
  --exit-code "$rc1" --output "$FIRST" || exit 1
COMPLETE=$(python -c "import json,sys;print(int(json.load(open(sys.argv[1]))['complete']))" "$FIRST") || exit 1
if [ "$COMPLETE" -eq 1 ]; then
  echo "SMOKE_EXIT=0 COMPLETE_BEFORE_RESUME=true RECOVERY_EXERCISED=false"; exit 0
fi
CK=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint'])" "$FIRST") || exit 1
run_stage "phase 2: identical command with --resume" "$RESUME_AFTER" "${CMD[@]}" --resume "$CK" 2>&1 | tee "$RESUME_LOG"
phase_codes=("${PIPESTATUS[@]}")
rc2=${phase_codes[0]}; [ "${phase_codes[1]}" -eq 0 ] || exit 1
python scripts_pvc/radar_fields_smoke_report.py --dataset rift --root "$CKDIR" --log "$RESUME_LOG" \
  --exit-code "$rc2" --previous "$FIRST" --output "$OUT/resume_${SLURM_JOB_ID}.json" || exit 1
echo "PHASE1_EXIT=$rc1 PHASE2_EXIT=$rc2 SMOKE_EXIT=0 RECOVERY_EXERCISED=true"
