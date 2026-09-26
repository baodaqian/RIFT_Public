#!/bin/bash
# One new real-B787 optimizer update and one validation view; not a fit or
# production-prefix schedule. Manager supplies Embers RTX6000/6CPU/32G/12G/1h.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE
set -euo pipefail
umask 077
[[ "$#" -eq 0 ]] || { echo 'ERROR: no arguments accepted' >&2; exit 64; }
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 94
[[ "${SLURM_JOB_QOS:-}" == embers && "${SLURM_MEM_PER_NODE:-}" == 32768 ]] || exit 94
PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json
for source in train_radar_fields.py rift/radar_fields.py rift/radar_fields_dataset.py; do
  [[ -f "$PROJECT/$source" ]] || { echo "ERROR: missing $source" >&2; exit 94; }
done
[[ -f "$NPZ" && -f "$MANIFEST" && -x /usr/bin/time ]] || exit 94
module purge
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT
cd "$PROJECT"
export PYTHONPATH=. PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
CAMPAIGN=/storage/scratch1/1/dbao31/rift_b7873200_all_model_smoke_v1/radar_fields
mkdir -p -- "$CAMPAIGN"
ATTEMPT="$CAMPAIGN/attempt-$SLURM_JOB_ID"
# Preserve prior attempts. No resume or reuse of an existing output directory.
mkdir -- "$ATTEMPT"
/usr/bin/time -v python -B -u - "$ATTEMPT" "$NPZ" "$MANIFEST" <<'PY'
import csv
import json
import math
from pathlib import Path
import resource
import runpy
import sys
import time

import numpy as np
import torch

attempt, npz, manifest = sys.argv[1:]
if not torch.cuda.is_available():
    raise RuntimeError('RF smoke requires an allocated CUDA device; no CPU fallback')
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
sys.argv = [
    'train_radar_fields.py', '--npz-path', npz,
    '--sealed-protocol', '--sealed-split-manifest', manifest,
    '--num-train', '3200', '--num-val', '1000', '--num-test', '1000',
    '--seed', '42', '--checkpoint-root', attempt,
    '--checkpoint-name', 'rf_real_update_v1', '--device', 'cuda',
    '--extent', '0.15', '--granularity', '48', '--steps', '1',
    '--view-batch', '1', '--train-pairs', '0', '--val-pairs', '0',
    '--eval-every', '1', '--checkpoint-every', '1', '--eval-max-views', '1',
    '--stats-max-views', '0', '--query-chunk', '32768', '--pair-chunk', '8',
]
runpy.run_path('train_radar_fields.py', run_name='__main__')
torch.cuda.synchronize()
elapsed = time.perf_counter() - started
output = Path(attempt) / 'rf_real_update_v1'
checkpoint = torch.load(output / 'checkpoint_final.pth.tar', map_location='cpu', weights_only=True)
if checkpoint['step'] != 1 or checkpoint['scene_repr'] != 'radar_fields':
    raise RuntimeError('RF smoke did not complete exactly one native update')
args = checkpoint['args']
for key, expected in {
    'sealed_protocol': True, 'num_train': 3200, 'num_val': 1000, 'num_test': 1000,
    'steps': 1, 'view_batch': 1, 'train_pairs': 0, 'val_pairs': 0,
    'eval_max_views': 1, 'stats_max_views': 0,
}.items():
    if args.get(key) != expected:
        raise RuntimeError(f'Unexpected RF smoke configuration: {key}')
partition = np.random.Generator(np.random.PCG64(42)).permutation(10000)
expected_roles = {'train': partition[:3200].tolist(), 'val': partition[9000:].tolist(),
                  'test': partition[8000:9000].tolist()}
split = checkpoint['split_provenance']
if split['role_ids'] != expected_roles or split['test_payload_materialized'] is not False:
    raise RuntimeError('RF smoke did not preserve canonical sealed roles')
if not checkpoint.get('sealed_protocol_contract'):
    raise RuntimeError('Missing sealed response contract')
stats = checkpoint['power_stats']
if (stats['train_view_indices'] != expected_roles['train']
        or stats['normalization_scan_view_indices'] != expected_roles['train']
        or not math.isfinite(stats['peak_power']) or stats['peak_power'] <= 0):
    raise RuntimeError('RF smoke normalization was not full, positive, train-only')
for key in ('radar_fields_state_dict', 'model_state_dict'):
    for name, value in checkpoint[key].items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f'Nonfinite saved state: {key}/{name}')
moments = checkpoint['optimizer_state_dict']['state']
nonzero_update_evidence = False
for state in moments.values():
    for name, value in state.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise RuntimeError(f'Nonfinite optimizer state: {name}')
    if 'exp_avg' in state:
        if int(state['step'].item()) != 1:
            raise RuntimeError('Unexpected optimizer step')
        nonzero_update_evidence |= bool(torch.count_nonzero(state['exp_avg']).item())
if not nonzero_update_evidence:
    raise RuntimeError('No nonzero native optimizer moment was saved')
with (output / 'radar_fields_history.csv').open(newline='') as handle:
    history = list(csv.DictReader(handle))
if len(history) != 1 or int(history[0]['step']) != 1:
    raise RuntimeError('Missing completed one-view validation record')
for key in ('train_loss', 'train_rel_mse', 'val_rel_mse', 'val_rmse'):
    if not math.isfinite(float(history[0][key])):
        raise RuntimeError(f'Nonfinite RF smoke metric: {key}')
gpu_peak = torch.cuda.max_memory_allocated()
gpu_total = torch.cuda.get_device_properties(0).total_memory
host_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
if gpu_peak >= 0.8 * gpu_total or host_peak >= 0.8 * 32 * 1024**3:
    raise RuntimeError('RF smoke exceeded the 80 percent memory headroom gate')
summary = {
    'kind': 'radar_fields_b7873200_one_update_engineering_v1',
    'production_ready': False, 'small_fit_demonstrated': False,
    'native_metric_domain': history[0]['metric_domain'],
    'training_updates': 1, 'validation_views': 1,
    'nonzero_optimizer_moment': nonzero_update_evidence,
    'elapsed_seconds': elapsed, 'host_peak_bytes': host_peak,
    'gpu_peak_allocated_bytes': gpu_peak, 'gpu_total_bytes': gpu_total,
    'history': history,
}
with (Path(attempt) / 'smoke_summary.json').open('x') as handle:
    json.dump(summary, handle, indent=2, allow_nan=False)
    handle.write('\n')
print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
print('RF_B7873200_REAL_UPDATE_V1_PASS', flush=True)
PY
