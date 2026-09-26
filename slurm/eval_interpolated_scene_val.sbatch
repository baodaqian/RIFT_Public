#!/bin/bash
#SBATCH --job-name=eval_interp_val
#SBATCH --account=gts-jromberg3-coda20
#SBATCH --export=NONE
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH -C gpu-a100
#SBATCH --mem=64G
#SBATCH -t2:00:00
#SBATCH -qinferno
#SBATCH -oReport-%j.out
#
# Interpolated-scene validation eval -- EXPERIMENT_MANAGER_HANDOFF.md 2026-07-13
# ("Interpolated-scene validation eval: GPU run requested"). Submit with
# --qos=embers --account=gts-jromberg3-ece overrides (see
# scripts/watch_pace_job.ps1 on the local experiment-manager side).

cd /storage/project/r-jromberg3-0/dbao31/RIFT

module purge
module load anaconda3/2023.03

CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

echo "Interpolated-scene validation eval: $@"
python scripts/eval_interpolated_scene_val.py --device cuda "$@"
rc=$?
echo "Interpolated-scene validation eval finished (exit $rc)"
exit $rc
