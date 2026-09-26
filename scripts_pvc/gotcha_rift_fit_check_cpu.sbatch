#!/bin/bash
# CPU run of scripts_pvc/gotcha_rift_fit_check.py (fit, Adam regime, optional Hessian curvature) on one checkpoint.
#   FIT_CKPT=<checkpoint.pt> FIT_OUT=<report.json> FIT_ARGS="--sectors 24 --curvature 6 ..." \
#   sbatch --export=ALL scripts_pvc/gotcha_rift_fit_check_cpu.sbatch
# (FIT_ARGS may contain spaces because it travels in the exported environment, not in --export.)
#SBATCH --job-name=gotcha-rift-fit-check
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/rift_camry_box_v2_smoke/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" RIFT_ACCELERATOR=cpu OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
hostname; date; git -C "$PWD" rev-parse HEAD; echo "checkpoint: $FIT_CKPT"; echo "args: $FIT_ARGS"; echo "output: $FIT_OUT"
python -W ignore -u scripts_pvc/gotcha_rift_fit_check.py "$FIT_CKPT" --threads 16 --output "$FIT_OUT" $FIT_ARGS
rc=$?; date; echo "EXIT=$rc"; exit $rc
