#!/bin/bash
# SE Stage-2 surface geometry (scripts_pvc/eval_surface_geometry_pvc.py) for several collection scenes, CPU partition.
# Usage: export SE_ROOT=.../outputs/rift/train2400/1t1r_2010b2bbe725 OUT=... PAIRS="a320:airliner_a320 b787:b787 ..."
#        sbatch --export=ALL --job-name eval-se-surfaces --output OUT/se_surfaces-%j.log scripts_pvc/eval_surface_geometry_cpu.sbatch
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -eo pipefail
: "${SE_ROOT:?}" "${OUT:?}" "${PAIRS:?}"
cd "${SOURCE_ROOT:-/home/u.db364833/RIFT}"
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu RIFT_PVC_ALLOW_BACKEND=cpu PYTHONUNBUFFERED=1
D=/scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645/inputs/RIFT_dataset
P=/scratch/group/p.cis261724.000/RIFT_pvc_runs/production_20260921_35jobs/outputs/rift/train2400/1t1r_2010b2bbe725
mkdir -p "$OUT"
hostname; date
status=0
for pair in $PAIRS; do
  O=${pair%%:*}; S=${pair#*:}; R=$SE_ROOT/$S/sugavanam_ertin_spgl1
  echo "== $O ($S)"
  python -u scripts_pvc/eval_surface_geometry_pvc.py --object "$O" --dataset-root "$D" \
    --role-manifest "$P/$S/role_manifest.json" --surface "$R/surface.npz" --checkpoint "$R/checkpoint_final.pt" \
    --output "$OUT/${O}_se_surface.json" || { echo "FAILED $O"; status=1; }
done
date
exit $status
