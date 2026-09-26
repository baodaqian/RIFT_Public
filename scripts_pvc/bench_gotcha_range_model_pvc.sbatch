#!/bin/bash
# Card check of the GOTCHA amplitude law (range_model sum2 vs unit) on the PVC
# batched lane: on-device agreement with autograd, sum2 == scaled unit, and
# forward+backward seconds at production shapes (Camry TRAIN sector, 16 pulses,
# stride 2, G48 grid, point chunk 16384). Any "Aten Op fallback" line is a defect.
#SBATCH --job-name=pvc-smoke-gotcha-range-model
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/range_model_smoke/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
hostname; date; git -C "$PWD" rev-parse HEAD
python -W always -u scripts_pvc/bench_gotcha_range_model_pvc.py \
  --dataset-root /scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined
rc=$?; date; echo "SMOKE_EXIT=$rc"; exit $rc
