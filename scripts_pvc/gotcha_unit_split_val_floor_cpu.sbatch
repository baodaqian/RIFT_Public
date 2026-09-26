#!/bin/bash
# CPU run of scripts_pvc/gotcha_unit_split_val_floor.py: re-rendering floors for VAL on the pass-held-out unit
# split (docs/RIFT_GOTCHA_Tune.md B48 iv, B52 c). One role per job; combine with --summarize on the login node.
#   FLOOR_OUT=<report.json> FLOOR_ARGS="--roles validation ..." sbatch --export=ALL scripts_pvc/gotcha_unit_split_val_floor_cpu.sbatch
# (FLOOR_ARGS may contain spaces because it travels in the exported environment, not in --export.)
# FLOOR_SHARDS overrides the shard root (default: the uncorrected keep6 shards).
#SBATCH --job-name=gotcha-unit-floor
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_unit_split_floor_20260923/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" RIFT_ACCELERATOR=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} PYTHONUNBUFFERED=1
SHARDS=${FLOOR_SHARDS:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_filtered_shards_20260923/camry_box_v2_provisional/keep6/New_Transfer/shards}
hostname; date; git -C "$PWD" rev-parse HEAD; echo "args: $FLOOR_ARGS"; echo "output: $FLOOR_OUT"
python -W ignore -u scripts_pvc/gotcha_unit_split_val_floor.py --shard-root "$SHARDS" --threads ${SLURM_CPUS_PER_TASK:-16} --output "$FLOOR_OUT" $FLOOR_ARGS
rc=$?; date; echo "EXIT=$rc"; exit $rc
