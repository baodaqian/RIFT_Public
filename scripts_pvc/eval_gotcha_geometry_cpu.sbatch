#!/bin/bash
# Camry 3D geometry (scripts_pvc/eval_gotcha_geometry_pvc.py) on the CPU partition (no GPU cap).
# Right-sized 09-24 (rift-eb, B83): sweeps measure 5-55 s and <=3 GB RSS, and 16 CPU / 64G / 1 h could not backfill.
# Usage: export MESH_DIR=data/meshes/camry_xv20[_data_frame] OUTPUT=OUT/NAME.json RUNS="rift=ROOT:TASK:CKPT spinr=... geraf=..."
#        sbatch --export=ALL --job-name eval-gotcha-geometry --output OUT/NAME-%j.log scripts_pvc/eval_gotcha_geometry_cpu.sbatch
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -eo pipefail
: "${MESH_DIR:?}" "${OUTPUT:?}" "${RUNS:?}"
cd "${SOURCE_ROOT:-/home/u.db364833/RIFT}"
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu RIFT_PVC_ALLOW_BACKEND=cpu RIFT_PVC_TCNN_SHIM=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
ARGS=()
for run in $RUNS; do ARGS+=(--run "$run"); done
hostname; date
python -u scripts_pvc/eval_gotcha_geometry_pvc.py --mesh-dir "$MESH_DIR" --output "$OUTPUT" "${ARGS[@]}" ${EXTRA_ARGS:-}
date
