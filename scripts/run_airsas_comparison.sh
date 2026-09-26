#!/usr/bin/env bash
set -euo pipefail

# Research-owned orchestration only: no scheduler submission is performed.
: "${SCENE:?set SCENE=armadillo}"
: "${MODEL:?set MODEL=adaptive_rift_sas or sh_sas}"
: "${CACHE:?set CACHE to a complete prepared Armadillo5k cache}"
[[ "$SCENE" == "armadillo" ]] || { echo "this comparison is fixed to Armadillo5k" >&2; exit 2; }
PROFILE="${PROFILE:-smoke}"
PROJECT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT/training_checkpoints}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-airsas_armadillo5k_${MODEL}_${PROFILE}_v1}"
[[ "$CHECKPOINT_NAME" == airsas_armadillo5k_* ]] || {
  echo "CHECKPOINT_NAME must retain the Armadillo5k identity" >&2
  exit 2
}

case "$MODEL" in
  adaptive_rift_sas|sh_sas) ;;
  *) echo "MODEL must be adaptive_rift_sas or sh_sas" >&2; exit 2 ;;
esac
case "$PROFILE" in
  smoke|full) ;;
  *) echo "PROFILE must be smoke or full" >&2; exit 2 ;;
esac

python "$PROJECT/scripts/validate_airsas_5k_cache.py" "$CACHE"

ARGS=(
  --cache "$CACHE"
  --model "$MODEL"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --checkpoint-name "$CHECKPOINT_NAME"
  --require-explicit-splits
  --sh-degree 3
  --num-rays 4900
  --max-bins 110
  --grad-clip 1.0
  --beamwidth-deg 30
  --sh-direction rx_to_point
  --seed 42
)
if [[ "$PROFILE" == "full" ]]; then
  ARGS+=(--profile full)
else
  ARGS+=(--steps 4 --num-rays 49 --max-bins 8 --eval-every 2 --checkpoint-every 2 --eval-pings 2 --eval-bins 8)
fi
exec python "$PROJECT/train_sas.py" "${ARGS[@]}"
