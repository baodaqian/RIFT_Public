#!/bin/bash
# Model-free baseline (scripts_pvc/eval_rift_dataset_model_free_pvc.py) for one RIFT-dataset scene on the CPU partition.
# METHOD=mfbp is the train-only coherent matched-filter backprojection on the 48^3 evaluator lattice.
# Usage: export OBJECT=a320 SCENE=airliner_a320 OUT=.../backprojection [METHOD=mfbp] [RESUME=1]
#        sbatch --export=ALL --job-name mfbp-$OBJECT --output $OUT/$OBJECT-%j.log scripts_pvc/eval_rift_dataset_model_free_cpu.sbatch
# Writes $OUT/$OBJECT/{model_free_identity.json,accumulator_latest.npz,matched_filter.npz}; RESUME=1 continues the accumulator.
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
: "${OBJECT:?}" "${SCENE:?}" "${OUT:?}"
cd "${SOURCE_ROOT:-/home/u.db364833/RIFT}"
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu RIFT_PVC_ALLOW_BACKEND=cpu RIFT_PVC_TCNN_SHIM=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
D=/scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645/inputs/RIFT_dataset
P=/scratch/group/p.cis261724.000/RIFT_pvc_runs/production_20260921_35jobs/outputs/rift/train2400/1t1r_2010b2bbe725
EXTRA=()
[ "${RESUME:-0}" = 1 ] && EXTRA+=(--resume)
hostname; date
python -u scripts_pvc/eval_rift_dataset_model_free_pvc.py --object "$OBJECT" --dataset-root "$D" \
  --role-manifest "$P/$SCENE/role_manifest.json" --method "${METHOD:-mfbp}" --output-dir "$OUT/$OBJECT" \
  --device cpu "${EXTRA[@]}"
date
