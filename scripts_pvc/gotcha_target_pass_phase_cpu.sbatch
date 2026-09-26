#!/bin/bash
# New-target step 4a (scripts_pvc/gotcha_target_pass_phase.py): leave-one-pass-out per-pass phase at a new target from
# TRAIN units only (full unit split; every k-th all-TRAIN sector ID), then its summary. Separate from the Camry tool.
# Usage: export TARGET_SHARDS=<keep6 shard root> TARGET_REGION=<name> TARGET_REGION_CONFIG=<json> PHASE_OUT=<report.json>
#        sbatch --export=ALL --output <log> THIS        (the summary is written to ${PHASE_OUT%.json}_summary.json)
#SBATCH --job-name=gotcha-target-phase
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -o pipefail
: "${TARGET_SHARDS:?}" "${TARGET_REGION:?}" "${TARGET_REGION_CONFIG:?}" "${PHASE_OUT:?}"
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" RIFT_ACCELERATOR=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} PYTHONUNBUFFERED=1
export GOTCHA_DATA_ROOT=/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined
hostname; date; git -C "$PWD" rev-parse HEAD; echo "shards $TARGET_SHARDS region $TARGET_REGION -> $PHASE_OUT ${PHASE_ARGS:-}"
python -W ignore -u scripts_pvc/gotcha_target_pass_phase.py --loo --shard-root "$TARGET_SHARDS" --region "$TARGET_REGION" \
  --region-config "$TARGET_REGION_CONFIG" --threads ${SLURM_CPUS_PER_TASK:-16} --output "$PHASE_OUT" ${PHASE_ARGS:-} || exit $?
python -W ignore -u scripts_pvc/gotcha_target_pass_phase.py --summarize-loo "$PHASE_OUT" --region "$TARGET_REGION" \
  --region-config "$TARGET_REGION_CONFIG" --output "${PHASE_OUT%.json}_summary.json"
rc=$?; date; echo "EXIT=$rc"; exit $rc
