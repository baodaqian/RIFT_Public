#!/bin/bash
#SBATCH --job-name=rift_sh_sas
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=embers
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
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

PROFILE="${1:-full}"
if [[ $# -gt 0 ]]; then shift; fi
DATA=data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz

case "$PROFILE" in
  smoke)
    NAME=b787_sh_sas_smoke
    PROFILE_ARGS=(
      --num-train 2 --num-val 1 --steps 2 --views-per-step 1
      --granularity 4 --num-freq-wanted 4 --train-tx 1 --train-rx 1
      --eval-freq-wanted 4 --eval-tx 1 --eval-rx 1 --eval-max-views 1
      --eval-every 1 --checkpoint-every 1
      --hash-log2-size 12 --query-chunk 256
      --pair-chunk 1 --point-chunk 256 --occlusion-point-chunk 256
    )
    ;;
  full)
    NAME=b787_sh_sas
    PROFILE_ARGS=(
      --num-train 1800 --num-val 200 --steps 1000 --views-per-step 1
      --granularity 48 --num-freq-wanted 100
      --eval-every 50 --checkpoint-every 10
      --sh-degree 3 --hidden-dim 32
      --hash-levels 16 --hash-features 2 --hash-base-resolution 16
      --hash-final-resolution 4096 --hash-log2-size 19
      --query-chunk 32768 --pair-chunk 16 --point-chunk 16384
      --occlusion-point-chunk 16384
    )
    ;;
  eval)
    NAME=b787_sh_sas
    PROFILE_ARGS=()
    ;;
  *)
    echo "usage: sbatch slurm/train_sh_sas.sbatch {smoke|full|eval} [extra train_sh_sas.py args]" >&2
    exit 2
    ;;
esac

OUT="training_checkpoints/$NAME"
mkdir -p "$OUT"

if [[ "$PROFILE" == "eval" ]]; then
  CKPT="$OUT/checkpoint_best.pth.tar"
  [[ -f "$CKPT" ]] || CKPT="$OUT/checkpoint_final.pth.tar"
  [[ -f "$CKPT" ]] || { echo "no SH-SAS checkpoint available for eval" >&2; exit 3; }
  python train_sh_sas.py --npz-path "$DATA" --checkpoint-name "$NAME" \
    --resume "$CKPT" --eval-only "$@"
  exit 0
fi

RESUME_ARGS=()
if [[ -f "$OUT/checkpoint_final.pth.tar" ]]; then
  echo "SH-SAS $PROFILE training already complete; preserving checkpoint_final"
elif [[ -f "$OUT/checkpoint_latest.pth.tar" ]]; then
  RESUME_ARGS=(--resume "$OUT/checkpoint_latest.pth.tar")
elif [[ -f "$OUT/checkpoint_best.pth.tar" ]]; then
  RESUME_ARGS=(--resume "$OUT/checkpoint_best.pth.tar")
fi

if [[ ! -f "$OUT/checkpoint_final.pth.tar" ]]; then
  echo "SH-SAS $PROFILE training: ${RESUME_ARGS[*]:-fresh}"
  python train_sh_sas.py \
    --npz-path "$DATA" --checkpoint-name "$NAME" \
    --seed 42 --val-from-tail --extent 0.15 \
    --opacity-key dc --opacity-scale 0.1 --opacity-normalize \
    --lambertian --occlusion --phase-sign 1 --range-model sum2 \
    --compute-dtype fp64 "${PROFILE_ARGS[@]}" "${RESUME_ARGS[@]}" "$@"
fi

if [[ "$PROFILE" == "full" ]]; then
  CKPT="$OUT/checkpoint_best.pth.tar"
  [[ -f "$CKPT" ]] || CKPT="$OUT/checkpoint_final.pth.tar"
  python scripts/eval_b787_geometry_metrics.py \
    --checkpoints "$CKPT" --labels SH-SAS --extent 0.15 \
    --csv "$OUT/b787_geometry_metrics.csv"
fi

echo "SH-SAS $PROFILE pipeline finished"
