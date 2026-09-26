#!/bin/bash
# Box-isolated, decimated GOTCHA shard for one pass (scripts_pvc/gotcha_filtered_shard.py): TRAIN/VAL sectors
# filtered at the frozen operating point (docs/RIFT_GOTCHA_Tune.md section 7); TEST rows zero, never read.
# Submit with --export=ALL,SHARD_PASS=<1..8> (and optionally SHARD_ROOT).
#SBATCH --job-name=gotcha-filtered-shard
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_filtered_shards_20260923/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu PYTHONUNBUFFERED=1
export GOTCHA_DATA_ROOT=/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined
ROOT=${SHARD_ROOT:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_filtered_shards_20260923/camry_box_v2_provisional}
hostname; date; git -C "$PWD" rev-parse HEAD; echo "pass ${SHARD_PASS:?} -> $ROOT"
python scripts_pvc/gotcha_filtered_shard.py --pass-id "$SHARD_PASS" --out-root "$ROOT" --keep-every 1 6 --verify 4 --threads 32
rc=$?; date; echo "EXIT=$rc"; exit $rc
