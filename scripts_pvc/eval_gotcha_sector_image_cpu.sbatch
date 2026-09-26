#!/bin/bash
# GOTCHA common sector-image column (scripts_pvc/eval_gotcha_sector_image_pvc.py) on the CPU partition (no GPU cap).
# Usage: export METHOD=spinr CAMPAIGN_ROOT=... TASK=camry-spinr-full CHECKPOINT=... TARGETS=.../radarsplat/hh/targets
#               OUT=... LABEL=... [MAX_SECTORS=5]      (METHOD=zero|observed take no CHECKPOINT)
#        sbatch --export=ALL --job-name eval-sector-$LABEL --output $OUT/$LABEL-%j.log scripts_pvc/eval_gotcha_sector_image_cpu.sbatch
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
set -eo pipefail
: "${METHOD:?}" "${CAMPAIGN_ROOT:?}" "${TASK:?}" "${TARGETS:?}" "${OUT:?}" "${LABEL:?}"
case "$METHOD" in zero|observed) ;; *) : "${CHECKPOINT:?}" ;; esac
cd "${SOURCE_ROOT:-/home/u.db364833/RIFT}"
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu RIFT_PVC_ALLOW_BACKEND=cpu RIFT_PVC_TCNN_SHIM=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
EXTRA=()
case "$METHOD" in zero|observed) ;; *) EXTRA+=(--checkpoint "$CHECKPOINT") ;; esac
[ -n "${MAX_SECTORS:-}" ] && EXTRA+=(--max-sectors "$MAX_SECTORS")
mkdir -p "$OUT"
hostname; date
python -u scripts_pvc/eval_gotcha_sector_image_pvc.py --method "$METHOD" --campaign-root "$CAMPAIGN_ROOT" --task "$TASK" \
  --radarsplat-targets "$TARGETS" --label "$LABEL" --out-dir "$OUT" --device cpu "${EXTRA[@]}"
date
