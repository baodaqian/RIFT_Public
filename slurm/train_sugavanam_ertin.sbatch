#!/bin/bash
#SBATCH --job-name=rift_sugertin
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=embers
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
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
export PYTHONUNBUFFERED=1

MODE=${1:-full}
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "usage: sbatch slurm/train_sugavanam_ertin.sbatch {smoke|full}" >&2
  exit 2
fi

DATA=data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz
if [[ "$MODE" == "smoke" ]]; then
  RUN=b787_sugavanam_ertin_smoke
  SCATTER_EPOCHS=2
  SDF_ARGS=(--steps 3 --smoke)
  EVAL_ARGS=(--gt-grid 32 --n-surface 1000 --n-volume 1000)
else
  RUN=b787_sugavanam_ertin
  SCATTER_EPOCHS=150
  SDF_ARGS=(--steps 5000 --n-fourier 9 --mesh-grid 192 --mesh-points 50000)
  EVAL_ARGS=(--gt-grid 240 --n-surface 20000 --n-volume 50000)
fi
SCATTER_RUN=${RUN}_scatter
SCATTER_DIR=training_checkpoints/$SCATTER_RUN
SDF_DIR=training_checkpoints/$RUN

echo "Sugavanam--Ertin pipeline mode=$MODE; geometry truth is post-hoc only"

if [[ ! -f "$SCATTER_DIR/checkpoint_final.pth.tar" ]]; then
  RESUME=()
  if [[ -f "$SCATTER_DIR/checkpoint_latest.pth.tar" ]]; then
    RESUME=(--resume "$SCATTER_DIR/checkpoint_latest.pth.tar")
    echo "Stage 1 resuming from checkpoint_latest"
  fi
  python train.py \
    --data-format npz --npz-path "$DATA" \
    --checkpoint-name "$SCATTER_RUN" \
    --scene-repr grid --forward-operator range --range-model product --compute-dtype float64 \
    --extent 0.15 --granularity 48 --l1-weight 3e-7 \
    --num-train 1800 --num-val 200 --num-test 0 --num-freq-wanted 600 \
    --epochs "$SCATTER_EPOCHS" --bp-init 100 --phase-sign -1.0 --lr 3e-3 \
    --seed 42 --val-from-tail --checkpoint-metric train \
    "${RESUME[@]}"
fi

if [[ ! -f "$SCATTER_DIR/checkpoint_final.pth.tar" ]]; then
  echo "Stage 1 ended without checkpoint_final (likely SIGTERM); resume the same Slurm command." >&2
  exit 75
fi

if [[ ! -f "$SDF_DIR/checkpoint_final.pth.tar" ]]; then
  RESUME=()
  if [[ -f "$SDF_DIR/checkpoint_latest.pth.tar" ]]; then
    RESUME=(--resume "$SDF_DIR/checkpoint_latest.pth.tar")
    echo "Stage 2 resuming from checkpoint_latest"
  fi
  python train_sugavanam_ertin_reference.py \
    --scatter-checkpoint "$SCATTER_DIR/checkpoint_final.pth.tar" \
    --output-dir "$SDF_DIR" \
    --scatter-threshold 0.15 --max-scatter-points 20000 \
    --seed 42 "${SDF_ARGS[@]}" "${RESUME[@]}"
fi

if [[ ! -f "$SDF_DIR/surface_reconstruction.npz" ]]; then
  echo "Stage 2 ended without surface_reconstruction.npz; hold for inspection." >&2
  exit 76
fi

COMPARE=(
  --sdf-surfaces "$SDF_DIR/surface_reconstruction.npz"
  --sdf-labels "$RUN"
  --npz-path "$DATA" --stl data/B787.stl --extent 0.15
  --csv "$SDF_DIR/b787_geometry_metrics.csv"
  "${EVAL_ARGS[@]}"
)
RIFT_COMPARE_CKPT=${RIFT_COMPARE_CKPT:-training_checkpoints/b787_r5_prune_target20k/checkpoint_best.pth.tar}
if [[ -f "$RIFT_COMPARE_CKPT" ]]; then
  COMPARE+=(--rift-checkpoints "$RIFT_COMPARE_CKPT" --rift-labels b787_r5_prune_target20k)
else
  echo "WARNING: matched RIFT checkpoint absent at $RIFT_COMPARE_CKPT; writing baseline-only row"
fi
python scripts/compare_sugavanam_ertin_geometry.py "${COMPARE[@]}"
echo "Sugavanam--Ertin $MODE pipeline complete"
