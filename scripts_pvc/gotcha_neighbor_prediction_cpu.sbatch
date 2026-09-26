#!/bin/bash
# CPU run of scripts_pvc/gotcha_neighbor_prediction.py (held-out sector predictability ceiling, A28).
#   NB_OUT=<report.json> NB_ARGS="--roles validation --targets 10 ..." sbatch --export=ALL scripts_pvc/gotcha_neighbor_prediction_cpu.sbatch
# (NB_ARGS may contain spaces because it travels in the exported environment, not in --export.)
#SBATCH --job-name=gotcha-neighbor-prediction
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/rift_camry_box_v2_smoke/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" RIFT_ACCELERATOR=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} PYTHONUNBUFFERED=1
hostname; date; git -C "$PWD" rev-parse HEAD; echo "args: $NB_ARGS"; echo "output: $NB_OUT"
python -W ignore -u scripts_pvc/gotcha_neighbor_prediction.py --threads ${SLURM_CPUS_PER_TASK:-16} --output "$NB_OUT" $NB_ARGS
rc=$?; date; echo "EXIT=$rc"; exit $rc
