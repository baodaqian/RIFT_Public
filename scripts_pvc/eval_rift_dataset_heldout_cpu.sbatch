#!/bin/bash
# Held-out signal scoring (scripts_pvc/eval_rift_dataset_heldout_pvc.py) on the CPU partition.
# No GPU: does not count against the PVC GRES cap. Resumable: rerunning with the same
# LABEL/OUT continues from the per-view cache (--resume is always passed).
#
# Usage: export METHOD=geraf OBJECT=a320 SCENE=airliner_a320 ROLE=validation \
#               CHECKPOINT=... OUT=... LABEL=a320_geraf_val [TEST_OPT_IN=1]
#        sbatch --export=ALL --job-name eval-heldout-$LABEL --output $OUT/$LABEL-%j.log \
#               scripts_pvc/eval_rift_dataset_heldout_cpu.sbatch
# ROLE=test requires TEST_OPT_IN=1 (adds --allow-reserved-test).
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -eo pipefail
: "${METHOD:?}" "${OBJECT:?}" "${SCENE:?}" "${ROLE:?}" "${CHECKPOINT:?}" "${OUT:?}" "${LABEL:?}"
cd "${SOURCE_ROOT:-/home/u.db364833/RIFT}"
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu RIFT_PVC_ALLOW_BACKEND=cpu RIFT_PVC_TCNN_SHIM=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
D=/scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645/inputs/RIFT_dataset
P=/scratch/group/p.cis261724.000/RIFT_pvc_runs/production_20260921_35jobs/outputs/rift/train2400/1t1r_2010b2bbe725
Q=/scratch/group/p.cis261724.000/RIFT_pvc_runs/production_20260921_rf_rs_14jobs/outputs/rift/train2400/1t1r_2010b2bbe725
MANIFEST=${MANIFEST:-$P/$SCENE/role_manifest.json}
[ "$METHOD" = radar_fields ] && MANIFEST=${MANIFEST_RF:-$Q/$SCENE/role_manifest.json}
EXTRA=()
if [ "$ROLE" = test ]; then
  [ "${TEST_OPT_IN:-0}" = 1 ] || { echo "ROLE=test needs TEST_OPT_IN=1"; exit 2; }
  EXTRA+=(--allow-reserved-test)
fi
[ -n "${MAX_VIEWS:-}" ] && EXTRA+=(--max-views "$MAX_VIEWS")      # debug/diagnostic cap
mkdir -p "$OUT"
hostname; date
python -u scripts_pvc/eval_rift_dataset_heldout_pvc.py --method "$METHOD" --object "$OBJECT" --dataset-root "$D" \
  --role-manifest "$MANIFEST" --checkpoint "$CHECKPOINT" --stats "$Q/$SCENE/radar_fields/radar_fields_power_stats.json" \
  --label "$LABEL" --out-dir "$OUT" --role "$ROLE" --device cpu --save-every 50 --resume "${EXTRA[@]}"
date
