#!/bin/bash
# NEW-TARGET COPY of scripts_pvc/run_gotcha_rift_densify_pvc.sbatch (docs/RIFT_GOTCHA_Tune.md A73-A75): identical except that the region
# is a required knob: TARGET_REGION (e.g. sentra_box_v1, prizm_box_v1) and TARGET_REGION_CONFIG
# (rift_pvc/regions/gotcha_new_targets.json). There is no Camry default, and the shard root must be given explicitly.
# RIFT on GOTCHA Camry HH, tuning campaign A41 (docs/RIFT_GOTCHA_Tune.md): dense box support, Plenoxel-style
# trilinear densify events (rift_pvc/gotcha_densify.py), pass-held-out unit split (rift_pvc/gotcha_unit_split.py:
# every 4th unsealed sector ID in all 8 passes, pass-4 units of a seeded half held out), full-native loss,
# B787 optimizer at a reduced learning rate, TRAIN fit scored every RIFT_DENSE_TRAIN_EVAL_EVERY epochs.
# Launched by rift-cc (one owner per job); job ID and outcome go to RIFT_PVC_Adaptation.md.
# Knobs (values without spaces; lists separated by ':' or ',' -- use ':' with sbatch --export=ALL,VAR=...,
# whose own separator is the comma):
#   RIFT_DENSE_LR (3e-4)            lr and pos-lr
#   RIFT_DENSE_EPOCHS (24)
#   RIFT_DENSE_DENSIFY (4:10:16)    1-based epochs after which to densify (0.125 -> 0.0625 -> 0.03125 -> 0.0156 m);
#                                   10 puts the second level on the cosine restart at epoch 11 (B52)
#   RIFT_DENSE_LR_TARGET (1.0)      curvature rule: stability-number ceiling after each densify event
#   RIFT_DENSE_LR_FACTOR (1.0)      fixed rule only (not used by the default curvature rule)
#   RIFT_DENSE_SHARDS (keep6 box shards)  shard root; the car-phase corrected copy (A52) is
#                                   .../camry_box_v2_provisional_carphase_b56/keep6/New_Transfer/shards
#   RIFT_DENSE_PASSES (1:2:3:4:5:6:7:8)  passes to load (':' or ','); the held-out pass 4 must stay in
#   RIFT_DENSE_SPLIT_STRIDE (4)     unit split: every Nth unsealed sector ID (1 = all 305)
#   RIFT_DENSE_HELDOUT_FRACTION (0.5)  unit split: seeded fraction of those IDs held out in pass 4
#   RIFT_DENSE_INIT (trilinear)     child start: trilinear (Plenoxels) or inherit (render-preserving)
#   RIFT_DENSE_NORMALIZE (volume)   trilinear children: volume (x1/8, A41) or energy
#   RIFT_DENSE_MAX_POINTS (524288)  slot capacity = active cap
#   RIFT_DENSE_INIT_DEGREE (0)      initial SH degree (max 4)
#   RIFT_DENSE_TRAIN_EVAL_EVERY (2)
#   RIFT_DENSE_OUT_ROOT             output root (default a new one per job); keep it to resume
#   RIFT_DENSE_RESUME               checkpoint to resume from (same knobs and output root)
#   RIFT_DENSE_EXTEND_FROM          completed run directory (.../rift_full_native) to extend to RIFT_DENSE_EPOCHS:
#                                   scripts_pvc/gotcha_extend_epochs.py writes it into a NEW RIFT_DENSE_OUT_ROOT with
#                                   only recipe epochs changed, then the job resumes it (same knobs otherwise). To
#                                   restart an interrupted extension, set RIFT_DENSE_RESUME instead (it wins).
#   RIFT_DENSE_EXTEND_PAST_RESTART  1 to let the extension train after restarts/refinement events the guard does not
#                                   cover (declare the curvature measurement, B53/B59)
#   RIFT_DENSE_GAIN_WARM_START      opt-in (A62): pooled = least-squares gain over every start pulse (default first pulse)
#   RIFT_DENSE_GAIN_LR              opt-in (A62): the gain's own AdamW group at this rate (eps 1e-8), no curvature cuts
#   RIFT_DENSE_CURVATURE_START      opt-in (A62): 1 = curvature rule once at the start, base set to S = target (may raise)
#   RIFT_DENSE_CURVATURE_GUARD      opt-in curvature guard (A57), ':' list of growth and/or restart: re-apply the
#                                   curvature rule after SH growth / warm restarts (a recipe key: keep it on resume;
#                                   with RIFT_DENSE_EXTEND_FROM it is added to the extended run's recipe)
#SBATCH --job-name=pvc-rift-densify
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/rift_camry_densify/logs/%x-%j.log
set -o pipefail
: "${TARGET_REGION:?set TARGET_REGION}" "${TARGET_REGION_CONFIG:?set TARGET_REGION_CONFIG}"
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
LR=${RIFT_DENSE_LR:-3e-4}
EPOCHS=${RIFT_DENSE_EPOCHS:-24}
DENSIFY=${RIFT_DENSE_DENSIFY:-4:10:16}
LR_FACTOR=${RIFT_DENSE_LR_FACTOR:-1.0}
LR_TARGET=${RIFT_DENSE_LR_TARGET:-1.0}
STRIDE=${RIFT_DENSE_SPLIT_STRIDE:-4}
PASSES=${RIFT_DENSE_PASSES:-1:2:3:4:5:6:7:8}
HELDOUT=${RIFT_DENSE_HELDOUT_FRACTION:-0.5}
INIT=${RIFT_DENSE_INIT:-trilinear}
NORMALIZE=${RIFT_DENSE_NORMALIZE:-volume}
MAXP=${RIFT_DENSE_MAX_POINTS:-524288}
INIT_DEGREE=${RIFT_DENSE_INIT_DEGREE:-0}
TRAIN_EVAL=${RIFT_DENSE_TRAIN_EVAL_EVERY:-2}
VARIANT=()
[ -n "${RIFT_DENSE_GAIN_WARM_START:-}" ] && VARIANT+=(--gain-warm-start "$RIFT_DENSE_GAIN_WARM_START")
[ -n "${RIFT_DENSE_GAIN_LR:-}" ] && VARIANT+=(--gain-lr "$RIFT_DENSE_GAIN_LR")
[ "${RIFT_DENSE_CURVATURE_START:-0}" = 1 ] && VARIANT+=(--densify-curvature-start)
GUARD=(); [ -n "${RIFT_DENSE_CURVATURE_GUARD:-}" ] && GUARD=(--densify-curvature-guard ${RIFT_DENSE_CURVATURE_GUARD//[,:]/ })
OUT=${RIFT_DENSE_OUT_ROOT:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/rift_camry_densify/outputs_${SLURM_JOB_ID}}
SHARDS=${RIFT_DENSE_SHARDS:?set RIFT_DENSE_SHARDS to the target keep6 shards}
RESUME=()
[ -n "${RIFT_DENSE_RESUME:-}" ] && RESUME=(--resume "$RIFT_DENSE_RESUME")
mkdir -p "$OUT"
if [ -n "${RIFT_DENSE_EXTEND_FROM:-}" ] && [ -z "${RIFT_DENSE_RESUME:-}" ]; then
  PAST=(); [ "${RIFT_DENSE_EXTEND_PAST_RESTART:-0}" = 1 ] && PAST=(--past-restart)
  EXT_GUARD=(); [ ${#GUARD[@]} -gt 0 ] && EXT_GUARD=(--curvature-guard "${GUARD[@]:1}")
  python scripts_pvc/gotcha_extend_epochs.py "$RIFT_DENSE_EXTEND_FROM" "$OUT" --epochs "$EPOCHS" "${PAST[@]}" "${EXT_GUARD[@]}" \
    --note "slurm job $SLURM_JOB_ID" > "$OUT/extension_$SLURM_JOB_ID.json" || { echo "extension refused"; exit 4; }
  cat "$OUT/extension_$SLURM_JOB_ID.json"
  RESUME=(--resume "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["resume"])' "$OUT/extension_$SLURM_JOB_ID.json")")
fi
hostname; date; git -C "$PWD" rev-parse HEAD; git -C "$PWD" status --short | head -40
echo "output root: $OUT lr: $LR epochs: $EPOCHS densify: $DENSIFY lr target: $LR_TARGET lr factor: $LR_FACTOR shards: $SHARDS passes: $PASSES split stride: $STRIDE held-out fraction: $HELDOUT init: $INIT normalize: $NORMALIZE max points: $MAXP init degree: $INIT_DEGREE train eval every: $TRAIN_EVAL resume: ${RESUME[1]:-none} extend from: ${RIFT_DENSE_EXTEND_FROM:-none} curvature guard: ${RIFT_DENSE_CURVATURE_GUARD:-none} variant: ${VARIANT[*]:-none}"
for p in ${PASSES//[,:]/ }; do [ -f "$SHARDS/pass${p}_hh.npz" ] || { echo "missing $SHARDS/pass${p}_hh.npz"; exit 3; }; done
echo "trainer start: $(date +%H:%M:%S)"
python -W always -u train_gotcha_dataset_pvc.py \
  --dataset-root /scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined \
  --shard-root "$SHARDS" --region "$TARGET_REGION" --region-config "$TARGET_REGION_CONFIG" \
  --output-root "$OUT" --polarizations hh --passes ${PASSES//[,:]/ } --num-tx 1 --num-rx 1 \
  --pulses-per-sector 0 --frequency-stride 2 --method rift --optimizer b787 --adam-eps 1.490932099939119 \
  --lr "$LR" --pos-lr "$LR" --loss-domain full_native \
  --support-box 3.0 1.5 1.25 --support-pitch 0.125 --sh-degree 4 --sh-init-degree "$INIT_DEGREE" \
  --max-points "$MAXP" --densify-epochs ${DENSIFY//[,:]/ } --densify-max-active "$MAXP" \
  --densify-lr-target "$LR_TARGET" --densify-lr-factor "$LR_FACTOR" --densify-init "$INIT" --densify-normalize "$NORMALIZE" --densify-max-level 3 --train-eval-every "$TRAIN_EVAL" "${GUARD[@]}" "${VARIANT[@]}" \
  --unit-split-stride "$STRIDE" --unit-split-heldout-pass 4 --unit-split-heldout-fraction "$HELDOUT" \
  --device xpu --point-chunk 16384 --checkpoint-every 50 --epochs "$EPOCHS" "${RESUME[@]}" >> "$OUT/train.log" 2>&1 &
PID=$!
trap 'echo "TERM at $(date +%H:%M:%S): forwarding to trainer $PID"; kill -TERM $PID 2>/dev/null' TERM
wait $PID; rc=$?
if [ "$rc" -gt 128 ]; then wait $PID; rc=$?; fi
echo "trainer end: $(date +%H:%M:%S)"; echo "RUN_EXIT=$rc"
echo "XPU_CPU_FALLBACK_LINES=$(grep -c 'Aten Op fallback' "$OUT/train.log")"
grep -E '^\{"method"' "$OUT/train.log" | tail -40
exit $rc
