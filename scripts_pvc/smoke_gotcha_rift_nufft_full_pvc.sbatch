#!/bin/bash
# Real-data smoke of the GOTCHA adaptive-RIFT control (RIFT-dataset NUFFT forward
# + full-native complex loss; rift_pvc/gotcha_nufft.py) on one PVC card.
#   1. scripts_pvc/bench_gotcha_nufft_pvc.py: per-update cost of the control
#      against the default recipe over the same 40 TRAIN pass-sectors.
#   2. The pinned Camry production command (HH, passes 1-8, 1500 train, cap 16,
#      stride 2, batched, point chunk 16384) with --forward-evaluation nufft
#      --loss-domain full_native, bounded to 2 epochs.
# Any "Aten Op fallback" line is a defect. Runs from a source snapshot taken at
# submission ($SNAPSHOT), never from the shared working tree.
# Usage: export SNAPSHOT=/scratch/.../rift_nufft_control_smoke/source_<tag>
#        sbatch --export=ALL scripts_pvc/smoke_gotcha_rift_nufft_full_pvc.sbatch
#SBATCH --job-name=pvc-smoke-gotcha-rift-nufft-full
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/rift_nufft_control_smoke/logs/%x-%j.log
set -o pipefail
: "${SNAPSHOT:?set SNAPSHOT}"
cd "$SNAPSHOT"
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
OUT=/scratch/group/p.cis261724.000/RIFT_pvc_runs/rift_nufft_control_smoke/outputs_${SLURM_JOB_ID}
hostname; date; echo "snapshot: $SNAPSHOT"; echo "output root: $OUT"
stamp() { while IFS= read -r line; do printf '%s %s\n' "$(date +%T)" "$line"; done; }

python -W always -u scripts_pvc/bench_gotcha_nufft_pvc.py --updates 40 --device xpu 2>&1 | stamp
echo "BENCH_EXIT=${PIPESTATUS[0]}"

# B: delivers TERM to this shell only; forward it so the trainer checkpoints and stops.
python -W always -u train_gotcha_dataset_pvc.py --dataset-root /scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined \
  --output-root "$OUT" --region camry --polarizations hh --passes 1 2 3 4 5 6 7 8 --num-train 1500 --num-tx 1 --num-rx 1 \
  --pulses-per-sector 16 --frequency-stride 2 --method rift --device xpu --point-chunk 16384 --epochs 2 \
  --forward-evaluation nufft --loss-domain full_native > "$OUT.train.log" 2>&1 &
PID=$!
trap 'echo "forwarding TERM to $PID"; kill -TERM "$PID" 2>/dev/null' TERM
wait "$PID"; rc=$?
if [ "$rc" -gt 128 ]; then wait "$PID"; rc=$?; fi
grep -v '^ *"\|^ *[]}{[]\+,\?$' "$OUT.train.log"
date; echo "SMOKE_EXIT=$rc"
echo "ATEN_FALLBACK_LINES=$(grep -c 'Aten Op fallback' "$OUT.train.log")"
find "$OUT" -name recipe.json -exec python -c "import json,sys;r=json.load(open(sys.argv[1]));print({k:r.get(k) for k in ('forward_evaluation','loss_domain','loss_normalization','checkpoint_selection','optimizer','range_model','initialization','priors')})" {} \;
find "$OUT" -name initialization.json -exec cat {} \; | head -40
find "$OUT" -name history.json -exec python -c "import json,sys;[print(e['epoch'],'full',e['validation']['full_native_complex_rel_mse'],'projected',e['validation']['pooled_rel_mse'],e.get('priors')) for e in json.load(open(sys.argv[1]))]" {} \;
exit $rc
