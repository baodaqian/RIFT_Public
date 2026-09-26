#!/bin/bash
#SBATCH --job-name=b787_bp_planform
#SBATCH --account=gts-jromberg3-coda20
#SBATCH --export=NONE
#SBATCH -N1 --gres=gpu:H200:1
#SBATCH -C gpu-h200
#SBATCH -t2:00:00
#SBATCH -qinferno
#SBATCH -oReport-%j.out
#
# B787 backprojection planform diagnostic -- see scripts/b787_bp_planform.py
# and EXPERIMENT_MANAGER_HANDOFF.md 2026-07-03 ("New queued diagnostic").
# Submit with --qos=embers --account=gts-jromberg3-ece overrides (see
# scripts/watch_pace_job.ps1 on the local experiment-manager side).

cd /storage/project/r-jromberg3-0/dbao31/RIFT

module purge
module load anaconda3/2023.03

CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

echo "B787 backprojection planform diagnostic: $@"
python scripts/b787_bp_planform.py \
    --data-dirs data/AEDT_B787_Sample/zone1_CSV data/AEDT_B787_Sample/zone3_CSV \
    --num-viewpoints 14 --num-freq 200 \
    --extent 35.0 --pitch 0.015 --chunk 500000 \
    --arr-dist 50 --num-rx 15 --num-tx 16 \
    "$@"
echo "B787 backprojection planform diagnostic finished"
