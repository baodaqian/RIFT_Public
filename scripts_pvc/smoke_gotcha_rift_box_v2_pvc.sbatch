#!/bin/bash
# First adaptive-RIFT check on the box-isolated Camry target (docs/RIFT_GOTCHA_Tune.md section 7 step 0d; B11/B14).
# Filtered shard root (keep6: every 6th native pulse, all 424 bins, TRAIN/VAL rows filtered, TEST rows zero, status
# provisional), region camry_box_v2 from rift_pvc/regions/, box support 6 x 3 x 2.5 m at 0.125 m pitch, SH degree 2
# trainable from update one (max 4), plain full-native complex loss, B787 optimizer in arm-B units (adam eps 1.4909).
# Short smoke: RIFT_SMOKE_EPOCHS epochs (default 3). Output root is a smoke root, never a campaign root.
# Launched by rift-cc under the AGENTS.md smoke rule; job ID and outcome go to RIFT_PVC_Adaptation.md (target provisional).
# Second version after 2157094 stalled on ac023 (the node with the recorded Intel JIT stalls, RIFT_PVC_Adaptation.md node
# caveat), now excluded: 2 h limit, TERM forwarded to the trainer 3 min
# before the limit (it stops at the next update and writes its history). The periodic stack-dump wrapper used by
# 2157134/2157139/2157140 (scripts_pvc/run_with_stack_dumps.py) is removed: it located the ac023 stall, and 2157140
# segfaulted right after a dump (rift-4b: faulthandler walks frames without the GIL). For a silent phase use a one-shot
# faulthandler.register(SIGUSR1) or py-spy from an srun step instead.
#SBATCH --job-name=pvc-smoke-rift-box-v2
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/rift_camry_box_v2_smoke/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
EPOCHS=${RIFT_SMOKE_EPOCHS:-3}
INIT_DEGREE=${RIFT_SMOKE_INIT_DEGREE:-2}   # A/B arm: 2 (B7 start) or 0 (arm-B style, bands unlock at refinement)
EXTRA=()   # optional optimizer arms (values without spaces): RIFT_SMOKE_LR, RIFT_SMOKE_POS_LR, RIFT_SMOKE_ADAM_EPS
[ -n "${RIFT_SMOKE_LR:-}" ] && EXTRA+=(--lr "$RIFT_SMOKE_LR")
[ -n "${RIFT_SMOKE_POS_LR:-}" ] && EXTRA+=(--pos-lr "$RIFT_SMOKE_POS_LR")
# Sparsity arms (A31/B29; values without spaces): train.py's target-mode pruning and a mu1 multiple.
#   RIFT_SMOKE_PRUNE_TARGET (enables --prune-every RIFT_SMOKE_PRUNE_EVERY, default 2), RIFT_SMOKE_PRUNE_START
#   (default 2), RIFT_SMOKE_PRUNE_END (default 8), min-active = target; RIFT_SMOKE_MU1_SCALE (default 1).
if [ -n "${RIFT_SMOKE_PRUNE_TARGET:-}" ]; then
  EXTRA+=(--prune-every "${RIFT_SMOKE_PRUNE_EVERY:-2}" --prune-target-active "$RIFT_SMOKE_PRUNE_TARGET"
          --prune-start-epoch "${RIFT_SMOKE_PRUNE_START:-2}" --prune-end-epoch "${RIFT_SMOKE_PRUNE_END:-8}"
          --prune-min-active "$RIFT_SMOKE_PRUNE_TARGET")
fi
[ -n "${RIFT_SMOKE_MU1_SCALE:-}" ] && EXTRA+=(--mu1-scale "$RIFT_SMOKE_MU1_SCALE")
# train.py --step-every N (gradients of N pass-sectors summed per step); pair with RIFT_SMOKE_LR/POS_LR = 0.003/N.
[ -n "${RIFT_SMOKE_STEP_EVERY:-}" ] && EXTRA+=(--step-every "$RIFT_SMOKE_STEP_EVERY")
ADAM_EPS=${RIFT_SMOKE_ADAM_EPS:-1.490932099939119}
SHARDS=${RIFT_SMOKE_SHARDS:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_filtered_shards_20260923/camry_box_v2_provisional/keep6/New_Transfer/shards}
OUT=/scratch/group/p.cis261724.000/RIFT_pvc_runs/rift_camry_box_v2_smoke/outputs_${SLURM_JOB_ID}
mkdir -p "$OUT"
hostname; date; git -C "$PWD" rev-parse HEAD; git -C "$PWD" status --short; echo "output root: $OUT shards: $SHARDS epochs: $EPOCHS init degree: $INIT_DEGREE adam eps: $ADAM_EPS extra: ${EXTRA[*]}"
# Shard guard: runs after allocation (a missing shard costs the card a few seconds and ends with exit 3).
for p in 1 2 3 4 5 6 7 8; do [ -f "$SHARDS/pass${p}_hh.npz" ] || { echo "missing $SHARDS/pass${p}_hh.npz"; exit 3; }; done
echo "trainer start: $(date +%H:%M:%S)"
python -W always -u train_gotcha_dataset_pvc.py \
  --dataset-root /scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined \
  --shard-root "$SHARDS" --region camry_box_v2 --region-config rift_pvc/regions/camry_box_v2.json \
  --output-root "$OUT" --polarizations hh --passes 1 2 3 4 5 6 7 8 --num-train 1500 --num-tx 1 --num-rx 1 \
  --pulses-per-sector 0 --frequency-stride 2 --method rift --optimizer b787 --adam-eps "$ADAM_EPS" "${EXTRA[@]}" \
  --loss-domain full_native --support-box 3.0 1.5 1.25 --support-pitch 0.125 --sh-degree 4 --sh-init-degree "$INIT_DEGREE" \
  --device xpu --point-chunk 16384 --epochs "$EPOCHS" > "$OUT/train.log" 2>&1 &
PID=$!
trap 'echo "TERM at $(date +%H:%M:%S): forwarding to trainer $PID"; kill -TERM $PID 2>/dev/null' TERM
wait $PID; rc=$?
# A trapped TERM makes the first wait return >128 while the trainer is still stopping; wait again for its real status (rift-cc).
if [ "$rc" -gt 128 ]; then wait $PID; rc=$?; fi
echo "trainer end: $(date +%H:%M:%S)"; echo "SMOKE_EXIT=$rc"
echo "XPU_CPU_FALLBACK_LINES=$(grep -c 'Aten Op fallback' "$OUT/train.log")"
echo "--- train.log without JSON lines and stack dumps ---"
grep -v '^ *"\|^ *[]}{[]\+,\?$\|^  File \|^    \|^Thread \|^Current thread\|^=== stack dump' "$OUT/train.log" | tail -60
find "$OUT" -name recipe.json -exec python -c "import json,sys;r=json.load(open(sys.argv[1]));print({k:r.get(k) for k in ('optimizer','adam_eps','loss_domain','support','sh_degree','sh_init_degree','range_model','initialization','priors','pulses_per_sector','frequency_stride')})" {} \;
find "$OUT" -name dataset.json -exec python -c "import json,sys;d=json.load(open(sys.argv[1]));s=d['summary'];print({'identity':s.get('identity'),'viewpoints':s.get('viewpoints'),'pulses':s.get('pulses_by_polarization'),'training_selection':s.get('training_selection')})" {} \;
find "$OUT" -name initialization.json -exec cat {} \; | head -30
# Which sigma^2 the loss used (rift-cc): the TRAIN mean power stored with the checkpoint, computed from the shard root in use.
find "$OUT" -name checkpoint_latest.pt -exec python -c "import torch,sys;c=torch.load(sys.argv[1],map_location='cpu',weights_only=False);s=c['training_statistics'];print({p:{k:v for k,v in st.items() if k in ('loss_domain','mean_power','projected_mean_power','count')} for p,st in s.items()}, 'dataset identity', c.get('dataset_identity'))" {} \;
find "$OUT" -name history.json -exec python -c "import json,sys;[print(e['epoch'],e['validation'].get('full_native_complex_rel_mse'),e['validation'].get('pooled_rel_mse'),e.get('priors'),e.get('optimizer')) for e in json.load(open(sys.argv[1]))]" {} \;
exit $rc
