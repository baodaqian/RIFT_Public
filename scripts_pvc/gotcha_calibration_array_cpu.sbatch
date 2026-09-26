#!/bin/bash
# GOTCHA system constant from the dataset's calibration trihedrals (Casteel et al. 2007, Table 1),
# HH, all eight passes, production-split TRAIN sectors only (scripts/gotcha_calibration_array.py).
#SBATCH --job-name=gotcha-calibration-array
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_calibration/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
hostname; date; git -C "$PWD" rev-parse HEAD
python scripts/gotcha_calibration_array.py --dataset-root /scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined \
  --polarization hh --workers 8 --out /scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_calibration/gotcha_calibration_hh.json
rc=$?; date; echo "EXIT=$rc"; exit $rc
