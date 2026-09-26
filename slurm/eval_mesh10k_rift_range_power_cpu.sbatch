#!/bin/bash
# Validate one mesh10k RIFT checkpoint_best in Radar Fields' normalized
# range-power domain. The selected checkpoint is snapshotted once so a running
# training job cannot change the model under a resumed evaluation.
#SBATCH --job-name=rift_m10k_power_v1
#SBATCH --account=gts-jromberg3
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=10G
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/mgr_slurm_logs/Report-%j.out

set -euo pipefail

[[ $# -eq 1 ]] || {
  echo 'usage: eval_mesh10k_rift_range_power_cpu.sbatch {a320|x59|firetruck|racecar|loader}' >&2
  exit 94
}
[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo 'ERROR: this evaluator requires a Slurm allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == embers && "${SLURM_JOB_PARTITION:-}" == cpu-small ]] || {
  echo 'ERROR: mesh10k power validation requires Embers/cpu-small' >&2
  exit 94
}

SCENE=$1
PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
DATA_ROOT=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/mesh_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k_data/objects
EXPERIMENT_ROOT=/storage/scratch1/1/dbao31/rift_mesh10k_baselines_20260830_v1

case "$SCENE" in
  a320) NPZ="$DATA_ROOT/airliner_a320_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz" ;;
  x59) NPZ="$DATA_ROOT/supersonic_x59_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz" ;;
  firetruck) NPZ="$DATA_ROOT/firetruck_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz" ;;
  racecar) NPZ="$DATA_ROOT/race_car_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz" ;;
  loader) NPZ="$DATA_ROOT/loader_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz" ;;
  *) echo "ERROR: unknown mesh10k scene: $SCENE" >&2; exit 94 ;;
esac

SOURCE_CHECKPOINT="$EXPERIMENT_ROOT/$SCENE/rift/model/checkpoint_best.pth.tar"
STATS="$EXPERIMENT_ROOT/$SCENE/rf/model/radar_fields_power_stats.json"
RUN_ROOT="$EXPERIMENT_ROOT/$SCENE/rift/range_power_validation_v1"
SNAPSHOT="$RUN_ROOT/checkpoint_best_snapshot.pth.tar"
SNAPSHOT_SHA="$RUN_ROOT/checkpoint_best_snapshot.sha256"
LABEL="${SCENE}_rift_checkpoint_best_snapshot"

for required in \
  "$PROJECT_ROOT/scripts/eval_b787_range_power.py" \
  "$SOURCE_CHECKPOINT" \
  "$NPZ" \
  "$STATS"; do
  [[ -s "$required" ]] || {
    echo "ERROR: missing or empty validation input: $required" >&2
    exit 94
  }
done

mkdir -p "$RUN_ROOT"
if [[ ! -s "$SNAPSHOT" ]]; then
  temporary="$SNAPSHOT.tmp.${SLURM_JOB_ID}"
  trap 'rm -f "${temporary:-}"' EXIT
  before=$(stat -c '%s:%Y' "$SOURCE_CHECKPOINT")
  cp "$SOURCE_CHECKPOINT" "$temporary"
  after=$(stat -c '%s:%Y' "$SOURCE_CHECKPOINT")
  [[ "$before" == "$after" ]] || {
    echo 'ERROR: checkpoint_best changed while it was being snapshotted' >&2
    exit 75
  }
  mv "$temporary" "$SNAPSHOT"
  trap - EXIT
  sha256sum "$SNAPSHOT" > "$SNAPSHOT_SHA"
else
  [[ -s "$SNAPSHOT_SHA" ]] || {
    echo "ERROR: existing snapshot lacks its SHA-256 record: $SNAPSHOT_SHA" >&2
    exit 94
  }
  (cd "$RUN_ROOT" && sha256sum -c "$(basename "$SNAPSHOT_SHA")")
fi

module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$PROJECT_ROOT"
python -B -u scripts/eval_b787_range_power.py \
  --checkpoint "$SNAPSHOT" \
  --label "$LABEL" \
  --npz-path "$NPZ" \
  --stats "$STATS" \
  --out-dir "$RUN_ROOT" \
  --num-train 3200 \
  --num-val 1000 \
  --seed 42 \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --save-every 5 \
  --resume

SUMMARY="$RUN_ROOT/${LABEL}_metrics.json"
CACHE="$RUN_ROOT/${LABEL}_per_view.npz"
python -B - "$SUMMARY" "$CACHE" <<'PY'
import json
import math
import sys

import numpy as np

summary_path, cache_path = sys.argv[1:]
with open(summary_path, "r", encoding="utf-8") as handle:
    result = json.load(handle)
assert result["views_complete"] == result["views_total"] == 1000
for key in (
    "normalized_range_power_rel_mse",
    "normalized_range_power_rmse",
    "normalized_range_power_psnr_db",
    "linear_range_power_rel_mse",
    "coherent_complex_rel_mse",
    "checkpoint_val_rel_mse",
):
    assert math.isfinite(float(result[key])), (key, result[key])
assert float(result["normalized_range_power_target_squared_norm"]) > 0.0
assert int(result["normalized_range_power_element_count"]) > 0
reference = float(result["checkpoint_val_rel_mse"])
recomputed = float(result["coherent_complex_rel_mse"])
assert math.isclose(recomputed, reference, rel_tol=1.0e-3, abs_tol=1.0e-7), (
    "coherent checkpoint parity failed", reference, recomputed
)
with np.load(cache_path) as cache:
    assert np.isfinite(cache["range_power_rel_mse"]).all()
    assert np.isfinite(cache["coherent_rel_mse"]).all()
    assert cache["roi_mask"].any(axis=1).all()
print("mesh10k RIFT range-power validation gates passed")
PY
