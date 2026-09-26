#!/bin/bash
# TRAIN + VALIDATION complex responses on the view sphere (scripts_pvc/export_rift_dataset_sphere_response_pvc.py),
# CPU partition. Resumable per method (an existing <method>.npz is skipped).
#
# Usage: export OBJECT=b787 SCENE=b787 OUT=/scratch/user/u.db364833/RIFT_runs/sphere_response_20260924 \
#               [OBSERVABLE=complex|power METHODS=... MAX_VIEWS=]
#        sbatch --export=ALL --job-name sphere-$OBJECT --output $OUT/logs/$OBJECT-%j.log \
#               scripts_pvc/export_rift_dataset_sphere_response_cpu.sbatch
# OBJECT is the dataset object id (b787, airliner_a320, supersonic_x59, firetruck, race_car, loader);
# SCENE is its run directory name (the same ids).
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
set -eo pipefail
: "${OBJECT:?}" "${SCENE:?}" "${OUT:?}"
cd "${SOURCE_ROOT:-/home/u.db364833/RIFT}"
source .local-setup/activate-pvc.sh
set -u
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu RIFT_PVC_ALLOW_BACKEND=cpu RIFT_PVC_TCNN_SHIM=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
D=/scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645/inputs/RIFT_dataset
P=/scratch/group/p.cis261724.000/RIFT_pvc_runs/production_20260921_35jobs/outputs/rift/train2400/1t1r_2010b2bbe725
Q=/scratch/group/p.cis261724.000/RIFT_pvc_runs/production_20260921_rf_rs_14jobs/outputs/rift/train2400/1t1r_2010b2bbe725
E=/scratch/user/u.db364833/RIFT_runs/final_eval_20260923/signal
EXTRA=()
[ -n "${METHODS:-}" ] && EXTRA+=(--methods "$METHODS")
[ -n "${MAX_VIEWS:-}" ] && EXTRA+=(--max-views "$MAX_VIEWS")
mkdir -p "$OUT"
hostname; date
python -u scripts_pvc/export_rift_dataset_sphere_response_pvc.py --object "$OBJECT" --dataset-root "$D" \
  --role-manifest "$P/$SCENE/role_manifest.json" --eval-signal-dir "$E" --out-dir "$OUT" \
  --observable "${OBSERVABLE:-complex}" --stats "$Q/$SCENE/radar_fields/radar_fields_power_stats.json" \
  --rs-eval-dir /scratch/user/u.db364833/RIFT_runs/eval_heldout_checks --device cpu "${EXTRA[@]}"
date
