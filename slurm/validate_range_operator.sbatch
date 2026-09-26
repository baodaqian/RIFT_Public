#!/bin/bash
#SBATCH --job-name=validate_range_operator
#SBATCH --account=gts-jromberg3-coda20
#SBATCH --export=NONE
#SBATCH -N1 --gres=gpu:H200:1
#SBATCH -C gpu-h200
#SBATCH -t2:00:00
#SBATCH -qinferno
#SBATCH -oReport-%j.out
#
# Phase 1 GPU-only gates -- see scripts/validate_range_operator.py and
# EXPERIMENT_MANAGER_HANDOFF.md 2026-07-04 ("Range-operator GPU validation:
# stages B + E"). Submit with --qos=embers --account=gts-jromberg3-ece
# overrides (see scripts/watch_pace_job.ps1 on the local experiment-manager
# side). Exit code 0 = all selected gates passed.

cd /storage/project/r-jromberg3-0/dbao31/RIFT

module purge
module load anaconda3/2023.03

CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

echo "Range-operator GPU validation: $@"
python scripts/validate_range_operator.py --stages B,E --device cuda "$@"
rc=$?
echo "Range-operator GPU validation finished (exit $rc)"
exit $rc
