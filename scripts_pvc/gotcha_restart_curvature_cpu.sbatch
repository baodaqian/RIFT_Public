#!/bin/bash
# Reviewer check (docs/RIFT_GOTCHA_Tune.md B60): stability number S = (base lr / eps) lambda_max at a densify arm's
# next cosine restart, re-measured on the last event's units with the trainer's update_curvature. CPU, TRAIN units.
#   RC_CKPT=<checkpoint.pt> RC_SHARDS=<the run's shard root> RC_OUT=<report.json> sbatch --export=ALL <this file>
#SBATCH --job-name=gotcha-restart-curv
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_unit_split_floor_20260923/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" RIFT_ACCELERATOR=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} PYTHONUNBUFFERED=1
hostname; date; git -C "$PWD" rev-parse HEAD; echo "checkpoint: $RC_CKPT shards: $RC_SHARDS output: $RC_OUT"
python -W ignore -u scripts_pvc/gotcha_restart_curvature.py "$RC_CKPT" --shard-root "$RC_SHARDS" --output "$RC_OUT" \
  --threads ${SLURM_CPUS_PER_TASK:-16}
rc=$?; date; echo "EXIT=$rc"; exit $rc
