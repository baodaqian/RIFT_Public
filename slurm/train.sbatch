#!/bin/bash
#SBATCH --job-name=RIFT                        # Job name
#SBATCH --account=gts-jromberg3-coda20         # charge account
#SBATCH --export=NONE
#SBATCH -N1 --gres=gpu:H200:1                  # Number of nodes and cores per node required
#SBATCH -C gpu-h200
#SBATCH -t3-00:00:00                           # Duration of the job
#SBATCH -qinferno                              # QOS Name
#SBATCH -oReport-%j.out                        # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL             # Mail preferences
#SBATCH --mail-user=dbao31@gatech.edu          # E-mail address for notifications
#
# Replaces 10281.sbatch ... 10286.sbatch (six near-duplicate files, one per
# hardcoded Train_RIFT_1028X_Y.py hyperparameter combination). All the run
# knobs that used to live inline in each .py file are now CLI flags on
# train.py -- pass them through when submitting, e.g.:
#
#   sbatch slurm/train.sbatch --data-dir data/ansysRCS/AEDT_Sphere_Repeat_CSV \
#       --checkpoint-name match_AEDT_sphere_w2000 --w2 2000
#
cd /storage/project/r-jromberg3-0/dbao31/RIFT

module purge
module load anaconda3/2023.03

CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
export PYTHONUNBUFFERED=1

echo "RIFT training: $@"
python train.py "$@"
echo "RIFT training finished"
