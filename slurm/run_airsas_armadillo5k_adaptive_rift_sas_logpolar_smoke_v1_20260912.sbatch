#!/bin/bash
# AirSAS Armadillo5kHz Adaptive RIFT-SAS log-polar smoke launcher.
# Staged for Astra review only; it does not submit itself.
#SBATCH --job-name=mgr_rift_airsas_arm5k_adaptive_logpolar_smoke_v1_20260912
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --tmp=12G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0,atl1-1-01-002-8-0,atl1-1-02-006-34-0

set -euo pipefail
umask 077

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT

module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT

cd "$PROJECT"
export PYTHONPATH="$PROJECT"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python -B /storage/project/r-jromberg3-0/dbao31/RIFT/train_sas.py --cache /storage/project/r-jromberg3-0/dbao31/RIFT/datasets/sh_sas_reed/cache/armadillo_5khz_trainonly_120rings_v1_20260912 --model adaptive_rift_sas --calibration-mode log_polar --checkpoint-root /storage/project/r-jromberg3-0/dbao31/RIFT/training_checkpoints --checkpoint-name airsas_armadillo5k_adaptive_rift_sas_logpolar_smoke_v1_20260912 --require-explicit-splits --sh-degree 3 --num-rays 4900 --max-bins 110 --grad-clip 1.0 --beamwidth-deg 30 --sh-direction rx_to_point --seed 42 --steps 1100 --eval-every 100 --eval-pings 8 --eval-bins 0 --checkpoint-every 100 --log-every 10
