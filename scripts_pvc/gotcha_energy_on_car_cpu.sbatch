#!/bin/bash
# Energy of adaptive-RIFT GOTCHA checkpoints relative to the Camry solid (scripts_pvc/gotcha_energy_on_car_pvc.py, A58).
# Usage: export OUTPUT=OUT.json RUNS="LABEL=CKPT LABEL=CKPT ..." [FIELDS="LABEL=rift:CKPT LABEL=spinr:CKPT LABEL=npz:PATH:KEY ..."] [MESH_DIR=...]
#        (RUNS: RIFT point energies; FIELDS: any method on the scorer lattice, A67; give either or both)
#        sbatch --export=ALL --output OUT-%j.log scripts_pvc/gotcha_energy_on_car_cpu.sbatch
#SBATCH --job-name=gotcha-energy-on-car
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
: "${OUTPUT:?}"
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu RIFT_PVC_ALLOW_BACKEND=cpu RIFT_PVC_TCNN_SHIM=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ARGS=()
for run in ${RUNS:-}; do ARGS+=(--run "$run"); done
for field in ${FIELDS:-}; do ARGS+=(--field "$field"); done
hostname; date; git rev-parse HEAD
python -u scripts_pvc/gotcha_energy_on_car_pvc.py --mesh-dir "${MESH_DIR:-data/meshes/camry_xv20_data_frame_box_v2}" --output "$OUTPUT" "${ARGS[@]}"
date
