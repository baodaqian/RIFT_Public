#!/bin/bash
# Package H's subset of G0: unchanged hash field, G48 occlusion and fp64 range.
#SBATCH --job-name=pvc-sh-sas-probe
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageH/logs/%x-%j.log
set -eo pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export RIFT_ACCELERATOR=xpu
python -m pytest rift_pvc/tests/test_sh_sas_xpu.py -q -s -p no:cacheprovider \
  --basetemp "/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageH/probes/job_${SLURM_JOB_ID}" \
  -k "${PYTEST_SELECTOR:-operator_process}"
