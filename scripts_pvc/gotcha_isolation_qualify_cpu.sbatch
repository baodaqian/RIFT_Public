#!/bin/bash
# Injection qualification of per-sector box isolation (scripts_pvc/gotcha_isolation_qualify.py), HH TRAIN geometry.
# Box and grid are passed through ISO_ARGS (space-free tokens joined by commas are not supported; export ISO_ARGS
# in the submitting shell and use --export=ALL).
#SBATCH --job-name=gotcha-isolation-qualify
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_isolation_qualify_20260923/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu PYTHONUNBUFFERED=1
export GOTCHA_DATA_ROOT=/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined
hostname; date; git -C "$PWD" rev-parse HEAD; echo "ISO_ARGS=$ISO_ARGS"
python scripts_pvc/gotcha_isolation_qualify.py $ISO_ARGS --threads 32 --out-dir "${ISO_OUT:?}"
rc=$?; date; echo "EXIT=$rc"; exit $rc
