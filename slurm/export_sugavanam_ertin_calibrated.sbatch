#!/bin/bash
#SBATCH --job-name=rift_sugertin_cal
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

# This launcher is deliberately export/evaluate only.  It never invokes
# train.py or train_sugavanam_ertin_reference.py and never writes the retained SE1
# checkpoint/history directory.
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "usage: sbatch slurm/export_sugavanam_ertin_calibrated.sbatch" >&2
  exit 2
fi

cd /storage/project/r-jromberg3-0/dbao31/RIFT
module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
export PYTHONUNBUFFERED=1

RUN=b787_sugavanam_ertin
CAL_RUN=b787_sugavanam_ertin_stage1median_v1
LABEL=b787_sugavanam_ertin_stage1median_v1
SCATTER_DIR=training_checkpoints/${RUN}_scatter
SDF_DIR=training_checkpoints/$RUN
CAL_DIR=training_checkpoints/$CAL_RUN
DATA=data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz
SCATTER_CKPT=$SCATTER_DIR/checkpoint_final.pth.tar
SDF_CKPT=$SDF_DIR/checkpoint_final.pth.tar
EXPECTED_SDF_SHA=a6589e5c1e4f3ee762674823075e59a721a72a330955262b17bf28f981399afe
EXPECTED_SCATTER_SHA=3b19daa58d30c541da2c0f273daa44793b0eab9ac64581277801b106cd595f12

for required in "$SCATTER_CKPT" "$SDF_CKPT" "$SDF_DIR/checkpoint_best.pth.tar" \
                "$SDF_DIR/checkpoint_latest.pth.tar" "$SDF_DIR/sdf_history.csv"; do
  [[ -f "$required" ]] || { echo "REFUSE missing retained SE1 artifact: $required" >&2; exit 74; }
done

source_before=$(sha256sum "$SDF_CKPT" | awk '{print $1}')
scatter_before=$(sha256sum "$SCATTER_CKPT" | awk '{print $1}')
[[ "$source_before" == "$EXPECTED_SDF_SHA" && "$scatter_before" == "$EXPECTED_SCATTER_SHA" ]] || {
  echo "REFUSE retained SE1 source hash drift: source=$source_before scatter=$scatter_before" >&2
  exit 75
}
echo "SE1 calibration-only export from checkpoint_final; source=$source_before scatter=$scatter_before"

python scripts/export_sugavanam_ertin_calibrated.py \
  --scatter-checkpoint "$SCATTER_CKPT" \
  --sdf-checkpoint "$SDF_CKPT" \
  --output-dir "$CAL_DIR" \
  --scatter-threshold 0.15 --max-scatter-points 20000 \
  --expected-step 5000 --mesh-grid 192 --mesh-points 50000 --seed 42

source_after=$(sha256sum "$SDF_CKPT" | awk '{print $1}')
scatter_after=$(sha256sum "$SCATTER_CKPT" | awk '{print $1}')
[[ "$source_after" == "$EXPECTED_SDF_SHA" && "$scatter_after" == "$EXPECTED_SCATTER_SHA" &&
   "$source_after" == "$source_before" && "$scatter_after" == "$scatter_before" ]] || {
  echo "FATAL retained SE1 source artifact changed during calibration" >&2
  exit 80
}
[[ -f "$CAL_DIR/surface_reconstruction.npz" && -f "$CAL_DIR/run_summary.json" ]] || {
  echo "FATAL calibration did not publish surface/summary" >&2
  exit 81
}

COMPARE=(
  --sdf-surfaces "$CAL_DIR/surface_reconstruction.npz"
  --sdf-labels "$LABEL"
  --npz-path "$DATA" --stl data/B787.stl --extent 0.15
  --gt-grid 240 --n-surface 20000 --n-volume 50000
  --csv "$CAL_DIR/b787_geometry_metrics.csv"
)
RIFT_COMPARE_CKPT=${RIFT_COMPARE_CKPT:-training_checkpoints/b787_r5_prune_target20k/checkpoint_best.pth.tar}
[[ -f "$RIFT_COMPARE_CKPT" ]] || {
  echo "REFUSE matched RIFT comparison checkpoint missing: $RIFT_COMPARE_CKPT" >&2
  exit 82
}
COMPARE+=(--rift-checkpoints "$RIFT_COMPARE_CKPT" --rift-labels b787_r5_prune_target20k)
python scripts/compare_sugavanam_ertin_geometry.py "${COMPARE[@]}"
echo "Sugavanam--Ertin calibrated final export complete; retained source SHA unchanged: $source_after"
