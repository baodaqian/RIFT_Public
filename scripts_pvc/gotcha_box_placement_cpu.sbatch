#!/bin/bash
# Data-driven placement evidence for the smaller GOTCHA Camry box (scripts_pvc/gotcha_box_placement.py):
# multi-height footprint images and a height-sharpness sweep, HH, TRAIN sectors only, all native pulses and bins.
#SBATCH --job-name=gotcha-box-placement
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_box_placement_20260923/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export GOTCHA_DATA_ROOT=/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined
hostname; date; git -C "$PWD" rev-parse HEAD
python scripts_pvc/gotcha_box_placement.py --out-dir /scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_box_placement_20260923 \
  --azimuth-step 5 --half 8 --step 0.1 --heights -0.5 0.0 0.5 1.0 1.5 --sweep-half 4 --sweep-step 0.1 \
  --sweep-z -1.5 2.5 0.1 --workers 16 --threads 2
rc=$?; date; echo "EXIT=$rc"; exit $rc
