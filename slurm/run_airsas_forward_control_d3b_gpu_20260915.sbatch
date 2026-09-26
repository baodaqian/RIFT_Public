#!/bin/bash
# AirSAS D3 TRAIN-only backprojection diagnostic (single GPU, no training, no checkpoint).
# Real GPU run after dry run 13245557 verified selection: 288 pings, 36 per ring, d3b dir untouched.
# Named by diagnostic rather than version number, deliberately.
# Staging only; account, QoS, exclusions, slurm log path and submission are supplied at sbatch time.
# Remote destination: /storage/project/r-jromberg3-0/dbao31/RIFT/slurm/run_airsas_forward_control_d3b_gpu_20260915.sbatch

#SBATCH --job-name=mgr_airsas_forward_control_d3b_gpu_20260915
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
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

cd "$SRC"
export PYTHONPATH="$SRC"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

exec "$PY" -B -u "$SRC/scripts/diagnose_sas_forward_control.py" \
  --cache /storage/project/r-jromberg3-0/dbao31/RIFT/datasets/sh_sas_reed/cache/armadillo_5khz_trainonly_120rings_v1_20260912 \
  --bunny-cache /storage/project/r-jromberg3-0/dbao31/RIFT/datasets/sh_sas_reed/cache/bunny_5khz_trainonly_120rings_v1_20260913 \
  --reed-bunny-image /storage/project/r-jromberg3-0/dbao31/RIFT/external/Reed_SAS_reference/scenes/airsas/bunny_5k/bp/scene.npy \
  --reed-arma20k-image /storage/project/r-jromberg3-0/dbao31/RIFT/external/Reed_SAS_reference/scenes/airsas/arma_20k/bp/scene.npy \
  --output /storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_forward_control_d3b_20260915 \
  --device cuda \
  --seed 0 \
  --run-d3 \
  --no-run-d1b-v2
