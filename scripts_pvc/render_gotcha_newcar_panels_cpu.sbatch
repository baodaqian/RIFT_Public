#!/bin/bash
# Provisional GOTCHA view-figure panels for one new vehicle (CPU): snapshot the validation-selected RIFT and SpINR-style
# checkpoints of the running jobs, backproject the measured TRAIN units (full split), then render the panels
# (scripts_pvc/render_gotcha_newcar_panels_pvc.py). Nothing here reads VALIDATION or TEST responses.
#
# Usage: export KEY=sentra TARGET=sentra_b15 REGION=sentra_box_v1 IDENT=907e3262a7273720 RIFT_RUN=SENTRA_F5gA_ep40 \
#               SPINR_RUN=sentra_full_e150 OUT=/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_newcar_provisional_20260924
#        sbatch --export=ALL --job-name newcar-$KEY --output $OUT/logs/$KEY-%j.log scripts_pvc/render_gotcha_newcar_panels_cpu.sbatch
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
set -eo pipefail
: "${KEY:?}" "${TARGET:?}" "${REGION:?}" "${IDENT:?}" "${RIFT_RUN:?}" "${SPINR_RUN:?}" "${OUT:?}"
cd "${SOURCE_ROOT:-/home/u.db364833/RIFT}"
source .local-setup/activate-pvc.sh
set -u
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu RIFT_PVC_ALLOW_BACKEND=cpu RIFT_PVC_TCNN_SHIM=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
G=/scratch/group/p.cis261724.000/RIFT_pvc_runs
SUB=$IDENT/frequency_stride2_all_roles_v2
STAGE=$OUT/$KEY/checkpoints
mkdir -p "$STAGE"
hostname; date
# snapshot: the trainers rewrite these files; copy, then read only the copies
cp -p "$G/rift_newtargets_20260924/$RIFT_RUN/$REGION/$SUB/rift_full_native/checkpoint_best.pt" "$STAGE/rift_${RIFT_RUN}_best.pt"
cp -p "$G/baselines_newtargets_20260924/spinr/$SPINR_RUN/$REGION/$SUB/spinr/checkpoint_best.pt" "$STAGE/spinr_${SPINR_RUN}_best.pt"
ls -l "$STAGE"
BP=$OUT/$KEY/backprojection_fullsplit_carphase_v2_all.npz
[ -f "$BP" ] || python -W ignore -u scripts_pvc/gotcha_data_coherent_image.py \
  --shard-root "$G/gotcha_filtered_shards_20260924_newtargets/${REGION}_provisional_carphase_v2/keep6/New_Transfer/shards" \
  --region "$REGION" --region-config rift_pvc/regions/gotcha_new_targets.json --all-pulses \
  --split-stride 1 --heldout-fraction 0.1 --label "$KEY carphase_v2 all pulses, full split" --output "$BP" --threads 16
python -u scripts_pvc/render_gotcha_newcar_panels_pvc.py --target "$TARGET" --key "$KEY" \
  --mesh-root "$G/gotcha_new_targets_20260924/meshes" --backprojection "$BP" \
  --rift "$STAGE/rift_${RIFT_RUN}_best.pt" --spinr "$STAGE/spinr_${SPINR_RUN}_best.pt" \
  --output-dir "$OUT/$KEY/render_$(date +%H%M)"
date
