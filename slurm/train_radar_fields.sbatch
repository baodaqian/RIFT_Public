#!/bin/bash
#SBATCH --job-name=rift_radar_fields
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=embers
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=Report-%j.out
#SBATCH --export=NONE

set -euo pipefail

cd /storage/project/r-jromberg3-0/dbao31/RIFT

module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
export PYTHONUNBUFFERED=1

echo "Radar Fields training: $*"
python train_radar_fields.py "$@"
echo "Radar Fields training finished"
