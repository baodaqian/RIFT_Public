#!/usr/bin/env bash
set -euo pipefail

# Research-owned orchestration only: this script never submits to a scheduler.
# It can run Reed's official AirSAS 5 kHz frontend in an already allocated shell,
# then converts the resulting weights into the shared cache.
: "${SCENE:?set SCENE=armadillo or bunny}"
: "${SYSTEM_DATA:?set SYSTEM_DATA to a supported AirSAS5k system-data file}"
: "${OUTPUT_CACHE:?set OUTPUT_CACHE to a new AirSAS5k cache directory}"
case "$SCENE" in
  armadillo)
    EXPECTED_SYSTEM_DATA_NAME="system_data_arma_5k.pik"
    REED_SCENE="arma_5k"
    FRONTEND_SPARSITY="1e-1"
    ;;
  bunny)
    EXPECTED_SYSTEM_DATA_NAME="system_data_bunny_5k.pik"
    REED_SCENE="bunny_5k"
    FRONTEND_SPARSITY="1e-2"
    ;;
  *)
    echo "unsupported AirSAS5k scene: $SCENE" >&2
    exit 2
    ;;
esac
[[ "$(basename "$SYSTEM_DATA")" == "$EXPECTED_SYSTEM_DATA_NAME" ]] || {
  echo "SYSTEM_DATA must be $EXPECTED_SYSTEM_DATA_NAME" >&2
  exit 2
}
PROJECT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
REED_ROOT="${REED_ROOT:-$PROJECT/external/Reed_SAS_reference}"
NORMALIZATION_MODE="${NORMALIZATION_MODE:-train_only}"
[[ "$NORMALIZATION_MODE" == "train_only" ]] || {
  echo "this AirSAS5k comparison requires NORMALIZATION_MODE=train_only" >&2
  exit 2
}
if [[ "${RUN_REED_FRONTEND:-0}" == "1" ]]; then
  : "${WEIGHTS_DIR:?RUN_REED_FRONTEND=1 requires a new explicit WEIGHTS_DIR; do not reuse the Reed example/global-normalized directory}"
  FRONTEND_OUTPUT_DIR="$(dirname "$WEIGHTS_DIR")"
  if [[ -e "$FRONTEND_OUTPUT_DIR/commandline_args.txt" || ( -d "$FRONTEND_OUTPUT_DIR" && -n "$(find "$FRONTEND_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ) ]]; then
    echo "refusing non-empty frontend output directory: $FRONTEND_OUTPUT_DIR" >&2
    exit 2
  fi
  if [[ "$NORMALIZATION_MODE" != "train_only" ]]; then
    echo "RUN_REED_FRONTEND=1 requires NORMALIZATION_MODE=train_only" >&2
    exit 2
  fi
else
  if [[ -z "${WEIGHTS_DIR:-}" ]]; then
    EXAMPLE_WEIGHTS="$REED_ROOT/scenes/airsas/${REED_SCENE}/example_outputs/deconvolved_measurements/numpy"
    if [[ -d "$EXAMPLE_WEIGHTS" ]]; then
      WEIGHTS_DIR="$EXAMPLE_WEIGHTS"
    else
      WEIGHTS_DIR="$REED_ROOT/scenes/airsas/${REED_SCENE}/deconvolved_measurements/numpy"
    fi
  fi
fi

if [[ "${RUN_REED_FRONTEND:-0}" == "1" ]]; then
  TRAIN_SCALE="${NORMALIZATION_SCALE:-$(python "$PROJECT/scripts/compute_airsas_train_scale.py" \
    --system-data "$SYSTEM_DATA" --reed-root "$REED_ROOT")}"
  [[ -n "$TRAIN_SCALE" ]] || { echo "failed to compute a positive train-only scale" >&2; exit 2; }
  SCENE_DIR="$REED_ROOT/scenes/airsas/${REED_SCENE}"
  OFFICIAL_FRONTEND_SCRIPT="$SCENE_DIR/pulse_deconvolve.sh"
  [[ -s "$OFFICIAL_FRONTEND_SCRIPT" ]] || { echo "missing official Reed $SCENE 5k frontend: $OFFICIAL_FRONTEND_SCRIPT" >&2; exit 2; }
  [[ -d "$SCENE_DIR" ]] || { echo "missing Reed scene directory: $SCENE_DIR" >&2; exit 2; }
  mkdir -p "$FRONTEND_OUTPUT_DIR"
  python "$REED_ROOT/inr_reconstruction/deconvolve_measurements.py" \
    --inr_config "$SCENE_DIR/pulse_deconvolve.json" \
    --system_data "$SYSTEM_DATA" \
    --output_dir "$FRONTEND_OUTPUT_DIR" \
    --learning_rate 1e-3 \
    --num_trans_per_inr 360 \
    --number_iterations 1000 \
    --info_every 999 \
    --sparsity "$FRONTEND_SPARSITY" \
    --load_wfm "$REED_ROOT/data/wfm/5khz_bw_lfm.npy" \
    --phase_loss 1e-4 \
    --normalization_mode train_only \
    --normalization_scale "$TRAIN_SCALE" \
    --max_transmissions 43200
fi

python "$PROJECT/scripts/prepare_airsas_cache.py" \
  --scene "$SCENE" \
  --system-data "$SYSTEM_DATA" \
  --weights-dir "$WEIGHTS_DIR" \
  --output "$OUTPUT_CACHE" \
  --reed-root "$REED_ROOT" \
  --bandwidth-khz 5 \
  --normalization-mode "$NORMALIZATION_MODE" \
