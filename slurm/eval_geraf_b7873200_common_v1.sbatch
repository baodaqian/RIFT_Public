#!/bin/bash
# Production GeRaF B7873200 common-signal evaluation.
# The manager supplies the allocation and owns submission; this launcher only
# validates the allocation and runs the fixed evaluation command.
# Do not use this file to submit directly from a login node.
#SBATCH --job-name=rift_geraf_common
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=10G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo 'ERROR: GeRaF common evaluation requires a Slurm allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_ACCOUNT:-}" == "gts-jromberg3-ece" ]] || {
  echo 'ERROR: evaluation requires account gts-jromberg3-ece' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == "inferno" && "${SLURM_JOB_PARTITION:-}" == "gpu-h100" ]] || {
  echo 'ERROR: evaluation requires Embers/gpu-h100; Inferno is not permitted' >&2
  exit 94
}

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
PYTHON=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/RIFT/bin/python
CHECKPOINT=/storage/scratch1/1/dbao31/rift_b7873200_geraf_fullscale_v1/fit/checkpoint_best.pth.tar
NPZ=/storage/project/r-jromberg3-0/dbao31/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
ROLE_MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json
CACHE_ROOT=/storage/scratch1/1/dbao31/rift_b7873200_geraf_fullscale_v1/prepared_targets
POWER_STATS=/storage/scratch1/1/dbao31/rift_b7873200_radar_fields_production_v1/attempt-12941397/rf_b7873200_production_v1/radar_fields_power_stats.json
OUT_ROOT=/storage/scratch1/1/dbao31/rift_b7873200_geraf_fullscale_v1/common_signal_evaluation_v1
SMOKE_ROOT=/storage/scratch1/1/dbao31/rift_b7873200_geraf_fullscale_v1/common_signal_evaluation_smoke_v1
SMOKE_SUMMARY="$SMOKE_ROOT/geraf_b7873200_common_metrics.json"
SMOKE_CACHE="$SMOKE_ROOT/geraf_b7873200_common_per_view.npz"
SMOKE_IDENTITY="$SMOKE_ROOT/geraf_b7873200_common_per_view_identity.json"

for required in "$PROJECT/scripts/eval_geraf_complex_response.py" "$CHECKPOINT" "$NPZ" "$ROLE_MANIFEST" "$CACHE_ROOT/geraf_b7873200_acquisition.npz" "$POWER_STATS"; do
  [[ -s "$required" ]] || {
    echo "ERROR: missing evaluation input: $required" >&2
    exit 94
  }
done

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$OUT_ROOT"
cd "$PROJECT"
"$PYTHON" -B scripts/validate_eval_geraf_b7873200_common.py

if [[ ! -e "$SMOKE_SUMMARY" && ! -e "$SMOKE_CACHE" && ! -e "$SMOKE_IDENTITY" ]]; then
  "$PYTHON" -B -u scripts/eval_geraf_complex_response.py \
    --checkpoint "$CHECKPOINT" \
    --npz-path "$NPZ" \
    --role-manifest "$ROLE_MANIFEST" \
    --cache-root "$CACHE_ROOT" \
    --power-stats "$POWER_STATS" \
    --out-dir "$SMOKE_ROOT" \
    --device cuda \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --max-views 1 \
    --save-every 1
elif [[ -s "$SMOKE_SUMMARY" && -s "$SMOKE_CACHE" && -s "$SMOKE_IDENTITY" ]]; then
  "$PYTHON" -B - "$SMOKE_SUMMARY" "$SMOKE_CACHE" "$SMOKE_IDENTITY" "$CHECKPOINT" <<'PY'
import json
import math
import os
import sys

import numpy as np

summary_path, cache_path, identity_path, checkpoint_path = sys.argv[1:]
with open(summary_path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)
assert summary.get("status") == "incomplete_or_smoke"
assert summary.get("views_complete") == summary.get("views_total") == 1
selection = summary.get("checkpoint_selection", {})
assert selection.get("step") == 37_000
assert selection.get("path") == os.path.abspath(checkpoint_path)
for key in ("coherent_complex_rel_mse", "normalized_range_power_rel_mse"):
    assert math.isfinite(float(summary[key]))
with np.load(cache_path, allow_pickle=False) as cache:
    assert np.isfinite(cache["coherent_rel_mse"]).all()
    assert np.isfinite(cache["range_power_rel_mse"]).all()
with open(identity_path, "r", encoding="utf-8") as handle:
    identity = json.load(handle).get("resume_identity", {})
assert identity.get("checkpoint") == os.path.abspath(checkpoint_path)
assert identity.get("checkpoint_step") == 37_000
print("GERAF_B7873200_COMMON_EVAL_SMOKE_REUSE_PASS")
PY
else
  echo "ERROR: smoke output is partial or inconsistent under $SMOKE_ROOT" >&2
  exit 94
fi

"$PYTHON" -B -u scripts/eval_geraf_complex_response.py \
  --checkpoint "$CHECKPOINT" \
  --npz-path "$NPZ" \
  --role-manifest "$ROLE_MANIFEST" \
  --cache-root "$CACHE_ROOT" \
  --power-stats "$POWER_STATS" \
  --out-dir "$OUT_ROOT" \
  --device cuda \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --save-every 5 \
  --resume

"$PYTHON" -B - "$OUT_ROOT/geraf_b7873200_common_metrics.json" "$OUT_ROOT/geraf_b7873200_common_per_view.npz" <<'PY'
import json
import math
import sys

import numpy as np

summary_path, cache_path = sys.argv[1:]
with open(summary_path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)
assert summary["status"] == "production_complete"
assert summary["views_complete"] == summary["views_total"] == 1000
assert summary["coherent_sample_count"] == 153_600_000
assert summary["normalized_range_power_element_count"] == 3_328_000
assert math.isclose(
    float(summary["normalized_range_power_target_squared_norm"]),
    411925.32444763184,
    rel_tol=0.0,
    abs_tol=1.0e-4,
)
for key in ("coherent_complex_rel_mse", "normalized_range_power_rel_mse"):
    assert math.isfinite(float(summary[key]))
with np.load(cache_path, allow_pickle=False) as cache:
    assert np.isfinite(cache["coherent_rel_mse"]).all()
    assert np.isfinite(cache["range_power_rel_mse"]).all()
print("GERAF_B7873200_COMMON_EVAL_POSTFLIGHT_PASS")
PY
