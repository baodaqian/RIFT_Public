#!/bin/bash
# Sonar warm-start fit, JOB 1 of 2: DRY RUN. CPU only, no GPU, writes no stage checkpoint.
# Exits before the first optimizer update, after computing the update-0 precondition and V0.
# --checkpoint-root is passed ABSOLUTELY on purpose: the driver's default is the relative
# "training_checkpoints" (parser line 73), which against this launcher's cwd would write into
# the isolated source tree instead of the project checkpoint tree.
# Staging only; account, QoS, slurm log path and submission are supplied at sbatch time.
# Remote destination: /storage/project/r-jromberg3-0/dbao31/RIFT/slurm/run_sas_warmstart_fit_dryrun_20260916.sbatch

#SBATCH --job-name=mgr_sas_warmstart_fit_dryrun_20260916
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --tmp=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE

set -euo pipefail
umask 077

SRC=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_sources/airsas_armadillo5k_fullgrid150x150x120_smoke_v1_20260914
PY=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/RIFT/bin/python
D3=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_forward_control_d3b_20260915

cd "$SRC"
export PYTHONPATH="$SRC"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

exec "$PY" -B -u "$SRC/scripts/run_sas_warmstart_fit.py" \
  --cache /storage/project/r-jromberg3-0/dbao31/RIFT/datasets/sh_sas_reed/cache/bunny_5khz_trainonly_120rings_v1_20260913 \
  --init-field-npz "$D3/forward_control_traces.npz" \
  --d3-summary-json "$D3/forward_control_summary.json" \
  --checkpoint-root /storage/project/r-jromberg3-0/dbao31/RIFT/training_checkpoints \
  --device cpu \
  --dry-run
