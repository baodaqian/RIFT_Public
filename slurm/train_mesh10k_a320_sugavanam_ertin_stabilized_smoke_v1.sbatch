#!/bin/bash
#SBATCH --job-name=rift_a320_sevz_s1
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=embers
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=Report-%j.out
#SBATCH --export=NONE

# Prepared launcher only.  Do not submit or register without a new explicit
# authorization.  This identity never reads or writes the preserved raw SDF.
set -euo pipefail

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
STAGE1=/storage/scratch1/1/dbao31/rift_mesh10k_baselines_20260830_v1/a320/se/scatter/checkpoint_final.pth.tar
OUTPUT=/storage/scratch1/1/dbao31/rift_mesh10k_a320_se_validzero_smoke_v1

cd "$PROJECT"
if [[ ! -f "$STAGE1" ]]; then
  echo "missing exact retained A320 Stage-1 final checkpoint: $STAGE1" >&2
  exit 3
fi
if [[ -e "$OUTPUT/checkpoint_final.pth.tar" || -e "$OUTPUT/surface_reconstruction.npz" || -e "$OUTPUT/run_summary.json" ]]; then
  echo "A320 stabilized smoke already has terminal artifacts at $OUTPUT" >&2
  exit 4
fi
if [[ -e "$OUTPUT/checkpoint_projection_failure.pth.tar" ]]; then
  echo "A320 stabilized smoke has a preserved terminal projection failure at $OUTPUT" >&2
  exit 5
fi
if [[ -d "$OUTPUT" && ! -f "$OUTPUT/checkpoint_latest.pth.tar" ]] && compgen -G "$OUTPUT/*" >/dev/null; then
  echo "A320 stabilized smoke has non-resumable artifacts; refusing to overwrite them" >&2
  exit 6
fi

module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
RIFT_SITE="$CONDA_PREFIX/lib/python3.10/site-packages"
EXPORT_SITE=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/sensor_fusion/lib/python3.10/site-packages
export PYTHONPATH="$PROJECT:$RIFT_SITE:$EXPORT_SITE"
python -B -c 'import numpy, scipy, torch, skimage; from skimage.measure import marching_cubes'

RESUME=()
if [[ -f "$OUTPUT/checkpoint_latest.pth.tar" ]]; then
  python -B -c '
import sys
import torch

try:
    state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(sys.argv[1], map_location="cpu")
raise SystemExit(0 if state.get("resume_allowed") is True else 7)
' "$OUTPUT/checkpoint_latest.pth.tar" || {
    echo "latest A320 checkpoint is not a manager-safe clean interruption" >&2
    exit 7
  }
  RESUME=(--resume "$OUTPUT/checkpoint_latest.pth.tar")
  echo "resuming only the A320 stabilized smoke's own latest checkpoint"
fi

python -B -u train_sugavanam_ertin_smoke.py --recipe legacy-a320-stabilized \
  --output-dir "$OUTPUT" \
  --steps 120 \
  --init-steps 200 --init-batch 2048 --init-log-every 25 \
  --batch-on 512 --batch-off 512 --batch-iso 256 --batch-signed 512 \
  --batch-boundary 256 --batch-inner 128 --n-iso 1024 \
  --off-warmup 20 --off-ramp 40 \
  --iso-start 60 --iso-refresh 30 \
  --gate-every 10 --gate-grid 32 --grid-chunk 32768 \
  --boundary-shell-resolution 24 --inner-anchor-count 1024 \
  --projection-oversample 2.0 \
  --mesh-grid 64 --mesh-points 5000 \
  --log-every 5 --save-every 20 \
  --scatter-threshold 0.15 --max-scatter-points 20000 \
  --n-fourier 9 --fourier-scale 2.0 --hidden-dim 512 --n-layers 8 \
  --radius-quantile 0.5 --radius-cap-fraction 0.65 \
  --boundary-margin 1e-4 --signed-offset-pitches 1.0 \
  --seed 42 --device cuda \
  "${RESUME[@]}"

echo "A320 Sugavanam--Ertin valid-zero stabilized derivative smoke complete at $OUTPUT"
