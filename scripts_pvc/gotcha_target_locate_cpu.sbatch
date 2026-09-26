#!/bin/bash
# New-target step 1 (scripts_pvc/gotcha_target_locate.py): locate vehicles from TRAIN data for new target boxes.
# Usage: export LOCATE_OUT=<dir> LOCATE_TARGETS="name:x:y name:x:y ..."; sbatch --export=ALL --output <log> THIS
#SBATCH --job-name=gotcha-target-locate
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -eo pipefail
: "${LOCATE_OUT:?}" "${LOCATE_TARGETS:?}"
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
hostname; date; git rev-parse HEAD
for t in $LOCATE_TARGETS; do
  IFS=: read -r name x y <<< "$t"
  python -W ignore -u scripts_pvc/gotcha_target_locate.py --name "$name" --x "$x" --y "$y" --output-dir "$LOCATE_OUT" --threads $SLURM_CPUS_PER_TASK
done
date
