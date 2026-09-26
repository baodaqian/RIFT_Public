#!/bin/bash
# Sonar warm-start fit, JOB 2 of 2: THE FIT. Single L40S GPU.
# Spends sonar smoke slot 6 of 11. Approved by Fable (rift-3b) under Daqian's delegation,
# conditional on RIFT-Coord's technical release on the 13248777 update-0 numbers.
# --checkpoint-root is passed ABSOLUTELY on purpose: the driver's default is relative and
# against this launcher's cwd would write into the isolated source tree.
# Staging only; account, QoS, slurm log path and submission are supplied at sbatch time.
# Remote destination: /storage/project/r-jromberg3-0/dbao31/RIFT/slurm/run_sas_warmstart_fit_gpu_20260916.sbatch

#SBATCH --job-name=mgr_sas_warmstart_fit_gpu_20260916
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --tmp=12G
#SBATCH --time=04:00:00
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
  --device cuda
