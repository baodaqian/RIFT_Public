#!/bin/bash
# Provisional GOTCHA geometry (Table 9 protocol) and registered view panels for one new vehicle (CPU), after
# render_gotcha_newcar_panels_cpu.sbatch has staged the checkpoints and the TRAIN backprojection under $OUT/$KEY.
# Usage: export KEY=santafe TARGET=santafe_2004 RIFT_RUN=SANTAFE_F5gA_ep40 SPINR_RUN=santafe_full_e150 \
#               OUT=/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_newcar_provisional_20260924
#        sbatch --export=ALL --job-name newcar-eval-$KEY --output $OUT/logs/eval-$KEY-%j.log scripts_pvc/eval_gotcha_newcar_cpu.sbatch
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
set -eo pipefail
: "${KEY:?}" "${TARGET:?}" "${RIFT_RUN:?}" "${SPINR_RUN:?}" "${OUT:?}"
cd "${SOURCE_ROOT:-/home/u.db364833/RIFT}"
source .local-setup/activate-pvc.sh
set -u
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu RIFT_PVC_ALLOW_BACKEND=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
G=/scratch/group/p.cis261724.000/RIFT_pvc_runs
C=$OUT/$KEY/checkpoints
BP=$OUT/$KEY/backprojection_fullsplit_carphase_v2_all.npz
E=$OUT/$KEY/geometry_$(date +%H%M)
hostname; date
python -u scripts_pvc/eval_gotcha_newcar_geometry_pvc.py --target "$TARGET" --mesh-root "$G/gotcha_new_targets_20260924/meshes" \
  --backprojection "$BP" --output-dir "$E" --run "rift=rift:$C/rift_${RIFT_RUN}_best.pt" \
  --run "spinr=spinr:$C/spinr_${SPINR_RUN}_best.pt" --run "backprojection=npz:$BP:data_energy"
python -u scripts_pvc/render_gotcha_newcar_panels_pvc.py --target "$TARGET" --key "$KEY" \
  --mesh-root "$G/gotcha_new_targets_20260924/meshes" --backprojection "$BP" --registration "$E/registration.json" \
  --rift "$C/rift_${RIFT_RUN}_best.pt" --spinr "$C/spinr_${SPINR_RUN}_best.pt" --output-dir "$E/render"
date
