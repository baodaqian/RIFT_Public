#!/bin/bash
# CPU run of scripts_pvc/gotcha_coherent_readout_pvc.py (A52): coherent backprojection readout of one checkpoint.
#   COH_CKPT=<checkpoint.pt> COH_OUT=<out.npz> COH_ARGS="--no-data ..." sbatch --export=ALL scripts_pvc/gotcha_coherent_readout_cpu.sbatch
#SBATCH --job-name=gotcha-coherent-readout
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/rift_camry_densify/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" RIFT_ACCELERATOR=cpu OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
SHARDS=${COH_SHARDS:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_filtered_shards_20260923/camry_box_v2_provisional/keep6/New_Transfer/shards}
hostname; date; echo "checkpoint: $COH_CKPT out: $COH_OUT shards: $SHARDS args: $COH_ARGS"
python -W ignore -u scripts_pvc/gotcha_coherent_readout_pvc.py "$COH_CKPT" --output "$COH_OUT" --shard-root "$SHARDS" \
  --region-config rift_pvc/regions/camry_box_v2.json --threads 16 $COH_ARGS
rc=$?; date; echo "EXIT=$rc"; exit $rc
