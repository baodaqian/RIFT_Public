#!/bin/bash
# CGLS least-squares refinement DRY RUN v2. CPU only, no GPU, outside the smoke count.
# 2h wall, up from 1h: job 13258713 hit a 1h wall mid-iteration-1 because the CGLS script
# inherits torch.set_num_threads(1) from diagnose_sas_forward_control (line 39) at import.
# Sonnet 1 added _configure_thread_count() to the run path; with 4 threads the first
# "[cgls] iteration 1" line is expected within ~25 min. If it is not, read the printed thread count.
# TWO invocations sequentially in one job: the second exercises the RESUME path on CPU
# before it is needed on a ~3h GPU job, where resume is the contingency for a wall-cap cut.
# No --d3-traces-npz in dry-run mode, per spec.
# Must NOT create or touch experiment_state/airsas_bunny5k_cgls288_ungated_v1_20260916,
# which is the slot 7 GPU job's output directory and must be absent when that job runs.
# Staging only; account, QoS, slurm log path and submission are supplied at sbatch time.
# Remote destination: /storage/project/r-jromberg3-0/dbao31/RIFT/slurm/run_sas_cgls_dryrun_v2_20260916.sbatch

#SBATCH --job-name=mgr_sas_cgls_dryrun_v2_20260916
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --tmp=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE

set -euo pipefail
umask 077

SRC=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_sources/airsas_armadillo5k_fullgrid150x150x120_smoke_v1_20260914
PY=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/RIFT/bin/python
CACHE=/storage/project/r-jromberg3-0/dbao31/RIFT/datasets/sh_sas_reed/cache/bunny_5khz_trainonly_120rings_v1_20260913
D3SUM=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_forward_control_d3b_20260915/forward_control_summary.json
OUT=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_bunny5k_cgls288_dryrun_20260916

cd "$SRC"
export PYTHONPATH="$SRC"
export PYTHONUNBUFFERED=1
# four cores used deliberately: the single-thread pin was removed from this script.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "=== INVOCATION 1: --dry-run --dry-run-k 2 ==="
"$PY" -B -u "$SRC/scripts/diagnose_sas_cgls.py" \
  --cache "$CACHE" \
  --d3-summary-json "$D3SUM" \
  --output "$OUT" \
  --device cpu \
  --dry-run \
  --dry-run-k 2

echo ""
echo "=== INVOCATION 2: --dry-run --dry-run-k 3 --resume (exercises the resume path) ==="
"$PY" -B -u "$SRC/scripts/diagnose_sas_cgls.py" \
  --cache "$CACHE" \
  --d3-summary-json "$D3SUM" \
  --output "$OUT" \
  --device cpu \
  --dry-run \
  --dry-run-k 3 \
  --resume "$OUT/cgls_state.pt"
