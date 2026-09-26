#!/bin/bash
# Real-data smoke of the GOTCHA adaptive RIFT with the RIFT-dataset B787
# optimizer and schedule (default --optimizer b787: AdamW lr/pos-lr 0.003,
# eps 1e-8; cosine warm restarts per epoch; B787 refinement cadence and
# thresholds; one fixed training order), on top of the RIFT-dataset start and
# priors and --range-model sum2, on one PVC card: the pinned Camry production
# command (HH, passes 1-8, 1500 train, cap 16, stride 2, batched, point chunk
# 16384) bounded to 2 epochs. Compare epoch-1/2 validation with smoke 2156186
# (earlier optimizer: 59.4 and 12.2). No refinement event falls in 2 epochs
# (the first is after epoch 10). Any "Aten Op fallback" line is a defect.
#SBATCH --job-name=pvc-smoke-gotcha-rift-b787-optimizer
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/rift_b787_optimizer_smoke/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
# Optimizer-diagnostic arms (2026-09-22): space-free values only (sbatch --export splits on commas and spaces).
#   RIFT_SMOKE_EPOCHS (default 2), RIFT_SMOKE_ARM (label), RIFT_SMOKE_LR / RIFT_SMOKE_POS_LR / RIFT_SMOKE_ADAM_EPS
#   (unset = the recipe's B787 values). Adam is invariant to (objective x c, eps) <-> (objective, eps / c), so
#   RIFT_SMOKE_ADAM_EPS=1.4909323... is exactly the objective in B787 units (sigma^2 = 6.707e-9) at eps 1e-8.
EPOCHS=${RIFT_SMOKE_EPOCHS:-2}
ARM=${RIFT_SMOKE_ARM:-b787}
EXTRA=()
[ -n "${RIFT_SMOKE_LR:-}" ] && EXTRA+=(--lr "$RIFT_SMOKE_LR")
[ -n "${RIFT_SMOKE_POS_LR:-}" ] && EXTRA+=(--pos-lr "$RIFT_SMOKE_POS_LR")
[ -n "${RIFT_SMOKE_ADAM_EPS:-}" ] && EXTRA+=(--adam-eps "$RIFT_SMOKE_ADAM_EPS")
OUT=/scratch/group/p.cis261724.000/RIFT_pvc_runs/rift_b787_optimizer_smoke/outputs_${ARM}_${SLURM_JOB_ID}
mkdir -p "$OUT"
hostname; date; git -C "$PWD" rev-parse HEAD; git -C "$PWD" status --short; echo "output root: $OUT arm: $ARM epochs: $EPOCHS extra: ${EXTRA[*]}"
python -W always -u train_gotcha_dataset_pvc.py --dataset-root /scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined \
  --output-root "$OUT" --region camry --polarizations hh --passes 1 2 3 4 5 6 7 8 --num-train 1500 --num-tx 1 --num-rx 1 \
  --pulses-per-sector 16 --frequency-stride 2 --method rift --optimizer b787 --device xpu --point-chunk 16384 --epochs "$EPOCHS" "${EXTRA[@]}" 2>&1 \
  | tee "$OUT/train.log" | grep -v '^ *"\|^ *[]}{[]\+,\?$'
rc=$?; date; echo "SMOKE_EXIT=$rc"
echo "XPU_CPU_FALLBACK_LINES=$(grep -c 'Aten Op fallback' "$OUT/train.log")"
find "$OUT" -name recipe.json -exec python -c "import json,sys;r=json.load(open(sys.argv[1]));print({k:r.get(k) for k in ('optimizer','lr','pos_lr','adam_eps','refine_every','probe_every','max_level','range_model','initialization','priors')})" {} \;
find "$OUT" -name initialization.json -exec cat {} \; | head -40
find "$OUT" -name history.json -exec python -c "import json,sys;[print(e['epoch'],e['validation']['pooled_rel_mse'],e['validation']['full_native_complex_rel_mse'],e.get('priors'),e.get('optimizer')) for e in json.load(open(sys.argv[1]))]" {} \;
exit $rc
