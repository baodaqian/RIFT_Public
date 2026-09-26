#!/bin/bash
# New-target step 3 (scripts_pvc/gotcha_target_filtered_shard.py): box-isolated, decimated shard for one pass of a
# named target region. TRAIN/VAL sectors filtered at the Camry's frozen operating point; TEST rows zero, never read.
# Usage: export TARGET_REGION=<name> TARGET_REGION_CONFIG=<json> TARGET_BOX=<box name> TARGET_OUT=<root> SHARD_PASS=<1..8>
#        sbatch --export=ALL --output <log> THIS
#SBATCH --job-name=gotcha-target-shard
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -o pipefail
: "${TARGET_REGION:?}" "${TARGET_REGION_CONFIG:?}" "${TARGET_BOX:?}" "${TARGET_OUT:?}" "${SHARD_PASS:?}"
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu PYTHONUNBUFFERED=1
export GOTCHA_DATA_ROOT=/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined
hostname; date; git -C "$PWD" rev-parse HEAD; echo "pass $SHARD_PASS region $TARGET_REGION box $TARGET_BOX -> $TARGET_OUT"
python scripts_pvc/gotcha_target_filtered_shard.py --pass-id "$SHARD_PASS" --out-root "$TARGET_OUT" \
  --region "$TARGET_REGION" --region-config "$TARGET_REGION_CONFIG" --box-name "$TARGET_BOX" \
  --keep-every 1 6 --verify 4 --threads 32
rc=$?; date; echo "EXIT=$rc"; exit $rc
