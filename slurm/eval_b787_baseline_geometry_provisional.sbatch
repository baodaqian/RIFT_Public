#!/bin/bash
# Provisional geometry extraction for completed B787 baseline checkpoints.
# The output directory is unique and intentionally retained as an audit artifact.
# Submit from /storage/project/r-jromberg3-0/dbao31/RIFT.
#SBATCH --job-name=rift_b787_basegeom
#SBATCH --account=gts-jromberg3-ece
#SBATCH --partition=cpu-gnr
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/scratch1/1/dbao31/rift_power_baselines_20260813_v2/evaluation/provisional_geometry_20260821_cpu_retry2/slurm-%j.out
#SBATCH --error=/storage/scratch1/1/dbao31/rift_power_baselines_20260813_v2/evaluation/provisional_geometry_20260821_cpu_retry2/slurm-%j.err

set -euo pipefail

cd /storage/project/r-jromberg3-0/dbao31/RIFT
PYTHON=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/RIFT/bin/python

"$PYTHON" scripts/eval_b787_baseline_geometry_provisional.py \
    --out-dir /storage/scratch1/1/dbao31/rift_power_baselines_20260813_v2/evaluation/provisional_geometry_20260821_cpu_retry2
