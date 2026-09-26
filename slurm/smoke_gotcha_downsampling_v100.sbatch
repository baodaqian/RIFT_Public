#!/bin/bash
# One explicitly authorized bounded smoke; never launches production trainers.
#SBATCH --job-name=gotcha-downsample-smoke
#SBATCH --account=gts-jromberg3
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=16G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --export=NONE

set -euo pipefail
ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
BATCH="$ROOT/experiment_state/gotcha_downsampling_v100_smoke_20260921"
RUN="$BATCH/$SLURM_JOB_ID"
module purge
module load anaconda3/2023.03 nvhpc-cuda/12.4 gcc/12.3.0
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/RIFT
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg WANDB_MODE=offline SLURM_EXPORT_ENV=ALL
export TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR="$RUN/extensions"
# Reuse known original SM70 dependencies; copies isolate the JIT build cache.
# No downloads, environment changes or writes to the completed production run.
ORIGINAL="$ROOT/experiment_state/rift_production_radarsplat_20260920"
mkdir -p "$RUN/extensions"
cp -a "$ORIGINAL/extensions/13401732/gsplat_cuda" "$RUN/extensions/"
export PYTHONPATH="$ORIGINAL/dependencies/13401732:$ROOT"
cd "$ROOT"
date -Is
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
exec srun python -B scripts/smoke_gotcha_downsampling.py --output "$RUN/report.json"
