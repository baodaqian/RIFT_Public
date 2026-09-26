#!/bin/bash
#SBATCH --job-name=rift_se2_v2p1
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=embers
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=Report-%j.out
#SBATCH --export=NONE

set -euo pipefail

cd /storage/project/r-jromberg3-0/dbao31/RIFT
module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Keep the original RIFT Torch/NumPy/SciPy stack first, and borrow only missing
# export-time packages (notably scikit-image) from the compatible Python-3.10
# sensor_fusion environment. The ordered paths were import- and marching-cubes-tested.
RIFT_SITE="$CONDA_PREFIX/lib/python3.10/site-packages"
EXPORT_SITE=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/sensor_fusion/lib/python3.10/site-packages
export PYTHONPATH="$RIFT_SITE:$EXPORT_SITE"
python -B -c 'import numpy, scipy, torch, skimage; from skimage.measure import marching_cubes'

MODE=${1:-smoke}
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "usage: sbatch slurm/train_sugavanam_ertin_validzero.sbatch {smoke|full}" >&2
  exit 2
fi

SCATTER=training_checkpoints/b787_sugavanam_ertin_scatter/checkpoint_final.pth.tar
if [[ ! -f "$SCATTER" ]]; then
  echo "missing retained Stage-1 source: $SCATTER" >&2
  exit 3
fi

if [[ "$MODE" == "smoke" ]]; then
  OUTPUT=training_checkpoints/b787_sugavanam_ertin_validzero_v2p1_smoke
  RUN_ARGS=(
    --run-kind smoke
    --steps 120
    --init-steps 200 --init-batch 2048 --init-log-every 25
    --batch-on 512 --batch-off 512 --batch-iso 256
    --batch-boundary 256 --batch-inner 128 --n-iso 1024
    --off-warmup 20 --off-ramp 40
    --iso-start 60 --iso-refresh 30
    --gate-every 10 --gate-grid 32 --grid-chunk 32768
    --mesh-grid 64 --mesh-points 5000
    --log-every 5 --save-every 20
  )
else
  OUTPUT=training_checkpoints/b787_sugavanam_ertin_validzero_v2p1
  RUN_ARGS=(--run-kind full)
fi

if [[ -f "$OUTPUT/checkpoint_final.pth.tar" && -f "$OUTPUT/surface_reconstruction.npz" && -f "$OUTPUT/run_summary.json" ]]; then
  echo "SE2 $MODE already complete at $OUTPUT"
  exit 0
fi

RESUME=()
if [[ -f "$OUTPUT/checkpoint_latest.pth.tar" ]]; then
  RESUME=(--resume "$OUTPUT/checkpoint_latest.pth.tar")
  echo "SE2 $MODE resuming its own latest checkpoint"
fi

python train_sugavanam_ertin_validzero.py \
  --scatter-checkpoint "$SCATTER" \
  --output-dir "$OUTPUT" \
  --manager-identity rift_sugertin_validzero_v2p1 \
  --artifact-identity b787_sugavanam_ertin_validzero_v2p1 \
  --scatter-threshold 0.15 --max-scatter-points 20000 \
  --n-fourier 9 --fourier-scale 2.0 --hidden-dim 512 --n-layers 8 \
  --boundary-margin 1e-4 \
  --projection-tolerance 1e-4 --projection-iterations 24 \
  --projection-min-acceptance 0.5 --projection-oversample 2.0 \
  --radius-quantile 0.5 --radius-cap-fraction 0.65 \
  --seed 42 --device cuda \
  "${RUN_ARGS[@]}" "${RESUME[@]}"

echo "Sugavanam--Ertin SE2 $MODE complete at $OUTPUT"
