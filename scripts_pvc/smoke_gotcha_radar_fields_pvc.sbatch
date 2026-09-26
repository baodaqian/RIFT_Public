#!/bin/bash
# Bounded real-data smoke of the native GOTCHA Radar Fields twin on one PVC card:
# Camry, HH, eight passes, 1500 training pass-sectors, the campaign's command
# (train_gotcha_dataset_pvc.py --method radar_fields --config <campaign config,
# an empty mapping = source-adapted-v3 defaults>), stopped by SIGTERM after
# PVC_SECONDS (default 1500 s) so the twin writes its interrupted-status
# checkpoint (rng_xpu, accelerator_backend, tcnn_shim). Set
# PVC_RESUME=<checkpoint_latest.pt> and PVC_OUTPUT_ROOT=<same root> to resume.
#SBATCH --job-name=pvc-smoke-gotcha-radar-fields
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
OUT=${PVC_OUTPUT_ROOT:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageE/outputs/gotcha_radar_fields_smoke_${SLURM_JOB_ID}}
CONFIG=${PVC_RF_CONFIG:-/scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645/configs/gotcha_radar_fields.json}
hostname; date; echo "output root: $OUT  config: $CONFIG"
cat "$CONFIG" || exit 1
python -c "from rift_pvc import accelerator as a; assert a.backend() == 'xpu' and a.device_count() == 1; print(a.describe())" || exit 1
mkdir -p "$OUT" || exit 1
TRAIN_LOG="$OUT/trainer_${SLURM_JOB_ID}.log"
RESUME_ARGS=(); REPORT_ARGS=()
if [ -n "${PVC_RESUME:-}" ]; then
  BEFORE="$OUT/before_${SLURM_JOB_ID}.json"
  python scripts_pvc/radar_fields_smoke_report.py --dataset gotcha --snapshot "$PVC_RESUME" --output "$BEFORE" || exit 1
  RESUME_ARGS=(--resume "$PVC_RESUME"); REPORT_ARGS=(--previous "$BEFORE")
fi
timeout --preserve-status -s TERM -k 300 "${PVC_SECONDS:-1500}" python -u train_gotcha_dataset_pvc.py --dataset-root "$GOTCHA_DATA_ROOT" --output-root "$OUT" \
  --region camry --polarizations hh --passes 1 2 3 4 5 6 7 8 --method radar_fields --num-train 1500 --num-tx 1 --num-rx 1 \
  --config "$CONFIG" "${RESUME_ARGS[@]}" 2>&1 | tee "$TRAIN_LOG"
phase_codes=("${PIPESTATUS[@]}")
rc=${phase_codes[0]}; [ "${phase_codes[1]}" -eq 0 ] || exit 1
python scripts_pvc/radar_fields_smoke_report.py --dataset gotcha --root "$OUT" --log "$TRAIN_LOG" \
  --exit-code "$rc" "${REPORT_ARGS[@]}" --output "$OUT/report_${SLURM_JOB_ID}.json" || exit 1
echo "TRAINER_EXIT=$rc SMOKE_EXIT=0 (validated checkpoint and cooperative outcome)"
