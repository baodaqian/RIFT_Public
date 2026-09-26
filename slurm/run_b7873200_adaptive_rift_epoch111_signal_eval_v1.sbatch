#!/bin/bash
# Evaluation-only signal audit of the selected Adaptive RIFT B787 checkpoint.
# This launcher never trains, resumes training, or writes to the checkpoint root.
#SBATCH --job-name=mgr_b7873200_adaptive_rift_epoch111_signal_eval_v1_20260910
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=12G
#SBATCH --time=12:00:00
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0,atl1-1-01-002-8-0,atl1-1-02-006-34-0
#SBATCH --export=NONE
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/b7873200_full_model_matrix_v1_20260906/mgr_slurm_logs/adaptive-signal-eval-%j.out
#SBATCH --error=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/b7873200_full_model_matrix_v1_20260906/mgr_slurm_logs/adaptive-signal-eval-%j.err

set -euo pipefail
umask 077

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
CHECKPOINT=/storage/scratch1/1/dbao31/rift_b7873200_adaptive_fullscale_v1/b78710k_adaptive_rift_fullscale_v1/checkpoint_best.pth.tar
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
STATS=/storage/scratch1/1/dbao31/rift_b7873200_radar_fields_production_v1/attempt-12941397/rf_b7873200_production_v1/radar_fields_power_stats.json
OUT_DIR=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/b7873200_full_model_matrix_v1_20260906/adaptive_rift_epoch111_signal_eval_v1/attempt_001
LABEL=adaptive_rift_epoch111

[[ "${SLURM_JOB_QOS:-}" == inferno ]] || { echo 'ERROR: Inferno QoS required' >&2; exit 94; }
[[ "${SLURM_JOB_PARTITION:-}" == gpu-h200 ]] || { echo 'ERROR: gpu-h200 partition required' >&2; exit 94; }
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo 'ERROR: one non-array Slurm allocation required' >&2; exit 94; }

for required in "$PROJECT/scripts/eval_b787_range_power.py" "$CHECKPOINT" "$NPZ" "$STATS"; do
  [[ -s "$required" ]] || { echo "ERROR: missing or empty required input: $required" >&2; exit 94; }
done

checkpoint_before="$(stat -c '%s:%Y' "$CHECKPOINT")"

module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT

cd "$PROJECT"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -B - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('ERROR: exactly one CUDA device is required')
print('RIFT_B787_ADAPTIVE_SIGNAL_EVAL_GPU=' + torch.cuda.get_device_name(0), flush=True)
PY

python -B -u scripts/eval_b787_range_power.py \
  --checkpoint "$CHECKPOINT" \
  --label "$LABEL" \
  --npz-path "$NPZ" \
  --stats "$STATS" \
  --out-dir "$OUT_DIR" \
  --num-train 3200 \
  --num-val 1000 \
  --seed 42 \
  --device cuda \
  --pair-chunk 64 \
  --point-chunk 262144 \
  --threads "$SLURM_CPUS_PER_TASK" \
  --save-every 5 \
  --resume

checkpoint_after="$(stat -c '%s:%Y' "$CHECKPOINT")"
[[ "$checkpoint_before" == "$checkpoint_after" ]] || {
  echo 'ERROR: selected checkpoint metadata changed during evaluation' >&2
  exit 95
}

SUMMARY="$OUT_DIR/${LABEL}_metrics.json"
CACHE="$OUT_DIR/${LABEL}_per_view.npz"
python -B - "$SUMMARY" "$CACHE" "$CHECKPOINT" "$NPZ" "$STATS" <<'PY'
import json
import math
import os
import sys

import numpy as np

summary_path, cache_path, checkpoint, npz_path, stats_path = sys.argv[1:]
with open(summary_path, 'r', encoding='utf-8') as handle:
    result = json.load(handle)

assert result['views_complete'] == result['views_total'] == 1000, result
assert result['checkpoint_epoch'] == 111, result['checkpoint_epoch']
assert result['scene_repr'] == 'point_sh', result['scene_repr']
assert result['device'].startswith('cuda'), result['device']
assert os.path.realpath(result['checkpoint']) == os.path.realpath(checkpoint)
assert os.path.realpath(result['npz_path']) == os.path.realpath(npz_path)
assert os.path.realpath(result['stats_path']) == os.path.realpath(stats_path)
for key in ('coherent_complex_rel_mse', 'normalized_range_power_rel_mse'):
    assert math.isfinite(float(result[key])), (key, result[key])
with np.load(cache_path) as cache:
    assert len(cache['view_indices']) == 1000
    assert np.isfinite(cache['coherent_rel_mse']).all()
    assert np.isfinite(cache['range_power_rel_mse']).all()
print(json.dumps({
    'coherent_complex_rel_mse': result['coherent_complex_rel_mse'],
    'normalized_range_power_rel_mse': result['normalized_range_power_rel_mse'],
    'views_complete': result['views_complete'],
    'checkpoint_epoch': result['checkpoint_epoch'],
}, sort_keys=True), flush=True)
PY

if find "$OUT_DIR" -type f \( -name '*.pth' -o -name '*.pth.tar' -o -name '*.ckpt' \) -print -quit | grep -q .; then
  echo 'ERROR: evaluation output contains a checkpoint-like file' >&2
  exit 95
fi

echo RIFT_B787_ADAPTIVE_SIGNAL_EVAL_V1_PASS
