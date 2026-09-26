#!/bin/bash
# CPU run of scripts_pvc/gotcha_pass_phase_closure.py: cross-pass phase at the car from TRAIN units only
# (docs/RIFT_GOTCHA_Tune.md B54/B55). Summarize afterwards on the login node with --summarize.
#   CLOSURE_OUT=<report.json> CLOSURE_ARGS="..." sbatch --export=ALL scripts_pvc/gotcha_pass_phase_closure_cpu.sbatch
#SBATCH --job-name=gotcha-phase-closure
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_unit_split_floor_20260923/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" RIFT_ACCELERATOR=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} PYTHONUNBUFFERED=1
SHARDS=/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_filtered_shards_20260923/camry_box_v2_provisional/keep6/New_Transfer/shards
hostname; date; git -C "$PWD" rev-parse HEAD; echo "args: $CLOSURE_ARGS"; echo "output: $CLOSURE_OUT"
python -W ignore -u scripts_pvc/gotcha_pass_phase_closure.py --shard-root "$SHARDS" --threads ${SLURM_CPUS_PER_TASK:-16} --output "$CLOSURE_OUT" $CLOSURE_ARGS
rc=$?; date; echo "EXIT=$rc"; exit $rc
