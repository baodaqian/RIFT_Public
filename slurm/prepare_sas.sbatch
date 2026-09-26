#!/bin/bash
#SBATCH --job-name=rift_sas_prep
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=embers
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

ROOT="/storage/project/r-jromberg3-0/dbao31/RIFT"
cd "$ROOT"
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT

PROFILE="${PROFILE:-smoke}"
SCENE="${SCENE:-buddha}"
case "$SCENE" in
  armadillo|buddha|bunny|xyz_dragon) ;;
  *) echo "Unsupported reportable SH-SAS scene: $SCENE" >&2; exit 2 ;;
esac
MAX_PINGS=0
SUFFIX="author20k_20db"
if [[ "$PROFILE" == "smoke" ]]; then
  MAX_PINGS=8
  SUFFIX="smoke"
elif [[ "$PROFILE" != "full" ]]; then
  echo "PROFILE must be smoke or full" >&2
  exit 2
fi

OUT="datasets/sh_sas_reed/cache/${SCENE}_${SUFFIX}"
if [[ -f "$OUT/manifest.json" ]] && grep -q '"complete": true' "$OUT/manifest.json"; then
  echo "Complete cache exists: $OUT"
  exit 0
fi

python -u scripts/prepare_sh_sas_sim.py \
  --system-data datasets/sh_sas_reed/simulated/system_data.pik \
  --transients "datasets/sh_sas_reed/simulated/${SCENE}/data_full.npy" \
  --output "$OUT" \
  --scene "$SCENE" \
  --max-pings "$MAX_PINGS" \
  --batch-size 32 \
  --bin-upsample 20 \
  --bounces 2 \
  --bandwidth-khz 20 \
  --snr-db 20 \
  --seed 0 \
  --device cuda
