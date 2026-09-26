#!/bin/bash
# Run rift_pvc/tests (or PYTEST_TARGET) on one PVC card in the RIFT-PVC env.
#   sbatch [--export=ALL,PYTEST_TARGET="rift_pvc/tests/test_x.py"] scripts_pvc/run_tests_pvc.sbatch
#SBATCH --job-name=pvc-tests
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageD/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 WANDB_MODE=offline PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
hostname; date; python -c "from rift_pvc import accelerator as a; print(a.describe())"
python -m pytest -p rift_pvc.pytest_policy ${PYTEST_TARGET:-rift_pvc/tests} -q -rs -p no:cacheprovider ${PYTEST_ARGS:-}
rc=$?; date; echo "TESTS_EXIT=$rc"; exit $rc
