#!/bin/bash
# Real-data smoke of the GOTCHA adaptive RIFT with the RIFT-dataset start and
# priors (defaults: --initialization backprojection --priors rift_dataset,
# --range-model sum2) on one PVC card: the pinned Camry production command
# (HH, passes 1-8, 1500 train, cap 16, stride 2, batched, point chunk 16384)
# bounded to 2 epochs. Checks for XPU->CPU fallback, the backprojection-start
# cost, and epoch-1/2 validation against 16.9 (sum2, random start) and
# about 21 (unit). Any "Aten Op fallback" line is a defect.
#SBATCH --job-name=pvc-smoke-gotcha-rift-dataset-recipe
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
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/rift_dataset_recipe_smoke/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
OUT=/scratch/group/p.cis261724.000/RIFT_pvc_runs/rift_dataset_recipe_smoke/outputs_${SLURM_JOB_ID}
hostname; date; git -C "$PWD" rev-parse HEAD; echo "output root: $OUT"
python -W always -u train_gotcha_dataset_pvc.py --dataset-root /scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined \
  --output-root "$OUT" --region camry --polarizations hh --passes 1 2 3 4 5 6 7 8 --num-train 1500 --num-tx 1 --num-rx 1 \
  --pulses-per-sector 16 --frequency-stride 2 --method rift --device xpu --point-chunk 16384 --epochs 2 2>&1 \
  | grep -v '^ *"\|^ *[]}{[]\+,\?$'
rc=$?; date; echo "SMOKE_EXIT=$rc"
find "$OUT" -name initialization.json -exec cat {} \; | head -40
find "$OUT" -name history.json -exec python -c "import json,sys;[print(e['epoch'],e['validation']['pooled_rel_mse'],e['validation']['full_native_complex_rel_mse'],e.get('priors')) for e in json.load(open(sys.argv[1]))]" {} \;
exit $rc
