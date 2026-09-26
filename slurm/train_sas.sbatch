#!/bin/bash
#SBATCH --job-name=rift_sas
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

MODEL="${MODEL:-sh_sas}"
PROFILE="${PROFILE:-smoke}"
SCENE="${SCENE:-buddha}"
case "$MODEL" in rift_sas|sh_sas) ;; *) echo "MODEL must be rift_sas or sh_sas" >&2; exit 2 ;; esac
case "$SCENE" in armadillo|buddha|bunny|xyz_dragon) ;; *) echo "Unsupported scene" >&2; exit 2 ;; esac

if [[ "$PROFILE" == "smoke" ]]; then
  CACHE="datasets/sh_sas_reed/cache/${SCENE}_smoke"
  NAME="${SCENE}_${MODEL}_smoke"
  EXTRA=(--steps 2 --granularity 8 --num-rays 16 --max-bins 4 --eval-every 1 \
         --eval-pings 1 --eval-bins 4 --checkpoint-every 1 --log-every 1 \
         --hash-levels 2 --hash-base-resolution 2 --hash-final-resolution 8 \
         --hash-log2-size 8 --hidden-dim 8 --query-chunk 4096 --max-pings 8)
elif [[ "$PROFILE" == "full" ]]; then
  CACHE="datasets/sh_sas_reed/cache/${SCENE}_author20k_20db"
  NAME="${SCENE}_${MODEL}_author50k"
  EXTRA=(--steps 50000 --granularity 64 --num-rays 4900 --max-bins 110 \
         --eval-every 500 --eval-pings 8 --eval-bins 32 --checkpoint-every 100 \
         --log-every 10 --hash-levels 16 --hash-base-resolution 16 \
         --hash-final-resolution 4096 --hash-log2-size 19 --hidden-dim 32)
else
  echo "PROFILE must be smoke or full" >&2
  exit 2
fi

if [[ ! -f "$CACHE/manifest.json" ]] || ! grep -q '"complete": true' "$CACHE/manifest.json"; then
  echo "Required complete cache missing: $CACHE" >&2
  exit 3
fi
if [[ -f "training_checkpoints/$NAME/status.json" ]] && grep -q '"done": true' "training_checkpoints/$NAME/status.json"; then
  echo "Completed training identity exists: $NAME"
  exit 0
fi

stop_requested=0
train_pid=""
forward_training_signal() {
  stop_requested=1
  if [[ -n "$train_pid" ]] && kill -0 "$train_pid" 2>/dev/null; then
    echo "Forwarding batch signal to training PID $train_pid"
    kill -TERM "$train_pid"
  fi
}
trap forward_training_signal TERM INT

set +e
python -u train_sas.py \
  --cache "$CACHE" \
  --model "$MODEL" \
  --checkpoint-name "$NAME" \
  --lr 1e-3 \
  --sh-degree 3 \
  --opacity-scale 500 \
  --lambertian-ratio 0 \
  --normal-step 0.0032 \
  --signal-scale 10 \
  "${EXTRA[@]}" &
train_pid=$!

# Bash's wait returns immediately when the batch-shell trap runs, while Python is still finishing
# its current optimizer step and atomically writing checkpoint_latest.pt. Keep waiting until the
# child is actually reaped; otherwise the shell can exit and Slurm can kill Python mid-checkpoint.
while true; do
  wait "$train_pid"
  train_rc=$?
  if ! kill -0 "$train_pid" 2>/dev/null; then
    break
  fi
done
set -e
trap - TERM INT

if (( stop_requested )); then
  echo "Training exited after forwarded signal with code $train_rc; skipping geometry evaluation"
  exit "$train_rc"
fi
if (( train_rc != 0 )); then
  exit "$train_rc"
fi

if [[ "$PROFILE" == "full" ]]; then
  python scripts/eval_sas_geometry.py \
    --density "training_checkpoints/$NAME/density.npy" \
    --geometry "$CACHE/geometry.npz" \
    --gt-root datasets/sh_sas_reed/simulated/gt_meshes \
    --scene "$SCENE" \
    --threshold 0.2 \
    --output "training_checkpoints/$NAME/geometry_metrics"
fi
