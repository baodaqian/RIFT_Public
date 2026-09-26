#!/bin/bash
# Gate G0 of the sonar PVC port: XPU op probe on one PVC card (section 5 of
# RIFT_SAS_PVC_Adaptation.md). Any XPU->CPU fallback line fails the job.
#SBATCH --job-name=pvc-sas-ops-probe
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
unset RIFT_ACCELERATOR RIFT_PVC_ALLOW_BACKEND   # never inherit a forced backend from the submitting shell
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
LOG=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log
OUT=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG/probes
hostname; date; python -c "from rift_pvc import accelerator as a; print(a.describe())"
python -u scripts_pvc_sas/ops_probe_pvc.py --workdir "$OUT/work_${SLURM_JOB_ID}" \
  --output "$OUT/ops_probe_${SLURM_JOB_ID}.json" --spread-steps "${PROBE_SPREAD_STEPS:-30}"
rc=$?; date
n=$(grep -c 'fallback from XPU to CPU' "$LOG" || true); echo "XPU->CPU fallback lines in this log: $n"
[ "$n" != "0" ] && rc=1
echo "PROBE_EXIT=$rc"; exit $rc
