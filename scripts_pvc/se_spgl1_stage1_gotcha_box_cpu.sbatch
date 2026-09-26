#!/bin/bash
# SE SPGL1 Stage 1 for GOTCHA Camry in the RIFT tuning campaign's data setup (docs/RIFT_GOTCHA_Tune.md,
# BASELINES_CAMRY_BOX.md): box-isolated filtered keep6 shards with the per-pass phase constants (carphase_v2),
# region camry_box_v2 (6 m cube, G40 = 15 cm), passes 1-8, HH, all 20 native pulses per unit, frequency
# stride 2, the pass-held-out unit split. Planned by train_gotcha_dataset_pvc.py itself
# (scripts_pvc/se_spgl1_stage1.py --gotcha); SE recipe protocols/se_g40_readout48.json (initialization_std 0.05).
# Dense Eq. 2 operators: split A (stride 4) 2-67 GB per sub-aperture (70 of them); split B (stride 1) about 4x.
# Knobs: SE_COMMAND (solve|assemble), SE_GROUPS (solve only; UNSET for assemble), SE_ROOT (output root; code runs
#        from $SE_ROOT/source unless SE_SOURCE is set),
#        SE_SPLIT_STRIDE (4), SE_HELDOUT_FRACTION (0.5), SE_SHARDS (carphase_v2 shards), SE_VERBOSITY (SPGL1 log
#        level, 1; 2 prints every 10th iteration).
# Usage: export SE_COMMAND=solve SE_GROUPS=0-6 SE_ROOT=...; sbatch --export=ALL --exclude=ac091,ac023 THIS
#SBATCH --job-name=se-camry-box
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=08:00:00
set -eo pipefail
: "${SE_ROOT:?set SE_ROOT}" "${SE_COMMAND:?set SE_COMMAND}"
# Run from the root's own source snapshot (identity re-checked by every solve/assemble/stage2).
cd "${SE_SOURCE:-$SE_ROOT/source}"
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
STRIDE=${SE_SPLIT_STRIDE:-4}
HELDOUT=${SE_HELDOUT_FRACTION:-0.5}
SHARDS=${SE_SHARDS:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_filtered_shards_20260923/camry_box_v2_provisional_carphase_v2/keep6/New_Transfer/shards}
OUT=$SE_ROOT/split_stride${STRIDE}_heldout${HELDOUT}/sugavanam_ertin_spgl1
hostname; date; git rev-parse HEAD 2>/dev/null || echo "snapshot of $(cat GIT_HEAD 2>/dev/null) + uncommitted ($(cat SNAPSHOT_TIME 2>/dev/null))"
echo "command: $SE_COMMAND groups: ${SE_GROUPS:-none} output: $OUT shards: $SHARDS stride: $STRIDE held-out: $HELDOUT"
grep -q '"initialization_std": 0.05' protocols/se_g40_readout48.json || { echo "SE config lost initialization_std 0.05"; exit 3; }
rc=0
python -u scripts_pvc/se_spgl1_stage1.py "$SE_COMMAND" --output "$OUT" ${SE_GROUPS:+--groups $SE_GROUPS} --verbosity "${SE_VERBOSITY:-1}" \
  --gotcha --dataset-root /scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined \
  --shard-root "$SHARDS" --region camry_box_v2 --region-config rift_pvc/regions/camry_box_v2.json \
  --polarizations hh --passes 1 2 3 4 5 6 7 8 --num-tx 1 --num-rx 1 --pulses-per-sector 0 --frequency-stride 2 \
  --unit-split-stride "$STRIDE" --unit-split-heldout-pass 4 --unit-split-heldout-fraction "$HELDOUT" \
  --method sugavanam_ertin --config protocols/se_g40_readout48.json --output-root "$SE_ROOT/frontend_unused" || rc=$?
date; echo "EXIT=$rc"; exit $rc
