#!/bin/bash
# NEW-TARGET COPY of scripts_pvc/run_gotcha_spinr_box_pvc.sbatch (docs/RIFT_GOTCHA_Tune.md A73-A75): identical except that the region
# is a required knob: TARGET_REGION (e.g. sentra_box_v1, prizm_box_v1) and TARGET_REGION_CONFIG
# (rift_pvc/regions/gotcha_new_targets.json). There is no Camry default, and the shard root must be given explicitly.
# SpINR baseline on GOTCHA Camry in the RIFT tuning campaign's data setup (BASELINES_CAMRY_BOX.md): the box-isolated
# filtered keep6 shards of camry_box_v2 with the adopted per-pass phase constant (carphase_v2, B60), all 20 stored rows
# per unit, stride-2 frequencies, and the pass-held-out unit split (rift_pvc/gotcha_unit_split.py). The SpINR recipe is
# the released adapter's (protocols/gotcha_spinr_g48_midpoint.json: G48 midpoint, 150 epochs, 1024-pulse batches,
# Adam 1e-4 cosine to 1e-5); only the data, region and split differ from the 2026-09-22 Camry run (2156166).
# Launched by rift-cc (one owner per job; job ID and outcome go to its ledger), except pre-authorized smokes.
# Knobs (values without spaces; lists separated by ':' so they travel in sbatch --export=ALL,VAR=...):
#   SPINR_SHARDS (carphase_v2 keep6 box shards)  shard root
#   SPINR_PASSES (1:2:3:4:5:6:7:8)
#   SPINR_SPLIT_STRIDE (4)          unit split: every Nth unsealed sector ID (1 = all 305, the F5full split)
#   SPINR_HELDOUT_FRACTION (0.5)    unit split: seeded fraction of those IDs held out in pass 4 (F5full: 0.1)
#   SPINR_HELDOUT_PASS (4)
#   SPINR_EPOCHS (150)              150 = the comparison budget; fewer = development_only (smokes)
#   SPINR_VALIDATION_EVERY (5)
#   SPINR_OUT_ROOT                  output root (default a new one per job); keep it to resume
#   SPINR_RESUME                    <run dir>/checkpoint_latest.pt of that output root (same knobs)
# Production (user decision 2026-09-23 ~22:00): the FULL split, like F5full: SPINR_SPLIT_STRIDE=1
# SPINR_HELDOUT_FRACTION=0.1 (2,410 TRAIN / 30 VAL units). Legs are 12 h; TERM at 300 s saves checkpoint_latest,
# and the next leg resumes it with the same SPINR_OUT_ROOT and SPINR_RESUME=<run dir>/checkpoint_latest.pt.
#SBATCH --job-name=pvc-spinr-target-box
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/baselines_camry_box_v2/spinr/logs/%x-%j.log
set -o pipefail
: "${TARGET_REGION:?set TARGET_REGION}" "${TARGET_REGION_CONFIG:?set TARGET_REGION_CONFIG}"
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
RUNS=/scratch/group/p.cis261724.000/RIFT_pvc_runs
SHARDS=${SPINR_SHARDS:?set SPINR_SHARDS to the target keep6 shards}
PASSES=${SPINR_PASSES:-1:2:3:4:5:6:7:8}
STRIDE=${SPINR_SPLIT_STRIDE:-4}
HELDOUT=${SPINR_HELDOUT_FRACTION:-0.5}
HELDOUT_PASS=${SPINR_HELDOUT_PASS:-4}
EPOCHS=${SPINR_EPOCHS:-150}
VALID=${SPINR_VALIDATION_EVERY:-5}
OUT=${SPINR_OUT_ROOT:-$RUNS/baselines_camry_box_v2/spinr/outputs_${SLURM_JOB_ID}}
RESUME=()
[ -n "${SPINR_RESUME:-}" ] && RESUME=(--resume "$SPINR_RESUME")
mkdir -p "$OUT"
# The released Camry config (G48 midpoint, 1024-pulse batches, 150-epoch cosine) with only the epoch count and the
# validation cadence as knobs; written into the output root so a resume reads the identical file.
CONFIG="$OUT/spinr_config_e${EPOCHS}_v${VALID}.json"
[ -f "$CONFIG" ] || printf '{"spinr": {"grid_size": 48, "nodes_per_cell": 1, "epochs": %d, "cosine_epochs": 150, "pulse_batch_size": 1024, "validation_every": %d}}\n' "$EPOCHS" "$VALID" > "$CONFIG"
hostname; date; git -C "$PWD" rev-parse HEAD; git -C "$PWD" status --short | head -40
echo "output root: $OUT shards: $SHARDS passes: $PASSES split stride: $STRIDE held-out pass: $HELDOUT_PASS held-out fraction: $HELDOUT epochs: $EPOCHS validation every: $VALID config: $(cat "$CONFIG") resume: ${SPINR_RESUME:-none}"
for p in ${PASSES//[,:]/ }; do [ -f "$SHARDS/pass${p}_hh.npz" ] || { echo "missing $SHARDS/pass${p}_hh.npz"; exit 3; }; done
echo "trainer start: $(date +%H:%M:%S)"
python -W always -u train_gotcha_dataset_pvc.py \
  --dataset-root /scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined \
  --shard-root "$SHARDS" --region "$TARGET_REGION" --region-config "$TARGET_REGION_CONFIG" \
  --output-root "$OUT" --polarizations hh --passes ${PASSES//[,:]/ } --num-tx 1 --num-rx 1 \
  --pulses-per-sector 0 --frequency-stride 2 \
  --unit-split-stride "$STRIDE" --unit-split-heldout-pass "$HELDOUT_PASS" --unit-split-heldout-fraction "$HELDOUT" \
  --method spinr --method-config "$CONFIG" --device xpu "${RESUME[@]}" >> "$OUT/train.log" 2>&1 &
PID=$!
trap 'echo "TERM at $(date +%H:%M:%S): forwarding to trainer $PID"; kill -TERM $PID 2>/dev/null' TERM
wait $PID; rc=$?
if [ "$rc" -gt 128 ]; then wait $PID; rc=$?; fi
echo "trainer end: $(date +%H:%M:%S)"; echo "RUN_EXIT=$rc"
echo "XPU_CPU_FALLBACK_LINES=$(grep -c 'Aten Op fallback' "$OUT/train.log")"
grep -E 'SpINR GOTCHA epoch' "$OUT/train.log" | tail -20
exit $rc
