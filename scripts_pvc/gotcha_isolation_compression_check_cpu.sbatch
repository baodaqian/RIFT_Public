#!/bin/bash
# Filter-then-decimate qualification and data acceptance for GOTCHA box isolation
# (scripts_pvc/gotcha_isolation_compression_check.py; review items B3/B4 in docs/RIFT_GOTCHA_Tune.md).
# Box and grid come through CHECK_ARGS exported in the submitting shell (--export=ALL); CHECK_OUT is the output root.
#SBATCH --job-name=gotcha-isolation-compress
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_isolation_compress_20260923/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export GOTCHA_DATA_ROOT=/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined
hostname; date; git -C "$PWD" rev-parse HEAD; echo "CHECK_ARGS=$CHECK_ARGS"
python scripts_pvc/gotcha_isolation_compression_check.py $CHECK_ARGS --threads 8 --out-dir "${CHECK_OUT:?}"
rc=$?; date; echo "EXIT=$rc"; exit $rc
