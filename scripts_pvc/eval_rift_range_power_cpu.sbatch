#!/bin/bash
# RIFT held-out coherent + MF range-power scoring (scripts_pvc/eval_b787_range_power_pvc.py) on the CPU partition.
# MF power is computed from RIFT's own predicted coherent spectrum (|IFFT_f S|^2), as the RIFT evaluator defines it.
# Usage: export OBJECT=a320 SCENE=airliner_a320 ROLE=validation|test CHECKPOINT=... OUT=... LABEL=... [TEST_OPT_IN=1 MAX_VIEWS=]
#        sbatch --export=ALL --job-name eval-rift-$LABEL --output $OUT/$LABEL-%j.log scripts_pvc/eval_rift_range_power_cpu.sbatch
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
set -eo pipefail
: "${OBJECT:?}" "${SCENE:?}" "${ROLE:?}" "${CHECKPOINT:?}" "${OUT:?}" "${LABEL:?}"
cd "${SOURCE_ROOT:-/home/u.db364833/RIFT}"
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu RIFT_PVC_ALLOW_BACKEND=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
D=/scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645/inputs/RIFT_dataset
P=/scratch/group/p.cis261724.000/RIFT_pvc_runs/production_20260921_35jobs/outputs/rift/train2400/1t1r_2010b2bbe725
Q=/scratch/group/p.cis261724.000/RIFT_pvc_runs/production_20260921_rf_rs_14jobs/outputs/rift/train2400/1t1r_2010b2bbe725
EXTRA=()
if [ "$ROLE" = test ]; then
  [ "${TEST_OPT_IN:-0}" = 1 ] || { echo "ROLE=test needs TEST_OPT_IN=1"; exit 2; }
  EXTRA+=(--allow-reserved-test)
fi
[ -n "${MAX_VIEWS:-}" ] && EXTRA+=(--max-views "$MAX_VIEWS")
mkdir -p "$OUT"
hostname; date
python -u scripts_pvc/eval_b787_range_power_pvc.py --object "$OBJECT" --dataset-root "$D" \
  --role-manifest "$P/$SCENE/role_manifest.json" --checkpoint "$CHECKPOINT" \
  --stats "$Q/$SCENE/radar_fields/radar_fields_power_stats.json" --label "$LABEL" --out-dir "$OUT" \
  --role "$ROLE" --device cpu --threads "$SLURM_CPUS_PER_TASK" --save-every 25 --resume "${EXTRA[@]}"
date
