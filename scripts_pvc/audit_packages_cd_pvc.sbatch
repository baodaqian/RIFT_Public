#!/bin/bash
# PVC default; override --partition=gpu_debug --gres=gpu:h100:1 and export
# AUDIT_BACKEND=cuda for the H100 compatibility leg.
#SBATCH --job-name=pvc-audit-cd
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageD/logs/%x-%j.log
set -eo pipefail
cd /home/u.db364833/RIFT
if [[ ${AUDIT_BACKEND:-xpu} == cuda ]]; then
    source .local-setup/activate.sh
else
    source .local-setup/activate-pvc.sh
fi
export RIFT_ACCELERATOR=${AUDIT_BACKEND:-xpu}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=offline
AUDIT_ROOT=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageD/audit_cd_${SLURM_JOB_ID}
mkdir -p "$AUDIT_ROOT"
export MPLCONFIGDIR="$AUDIT_ROOT/matplotlib" XDG_CACHE_HOME="$AUDIT_ROOT/cache"
export TMPDIR="$AUDIT_ROOT/tmp"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME/torch/kernels"
date -Is
hostname
git rev-parse HEAD
python -c 'from rift_pvc import accelerator as a; print(a.describe()); assert a.is_available()'
set +e
python -B -m pytest -q -rs -p no:cacheprovider --basetemp="$AUDIT_ROOT/existing_tests" \
    rift_pvc/tests/test_accelerator.py rift_pvc/tests/test_train_pvc_wrapper.py \
    rift_pvc/tests/test_frontends_pvc.py rift_pvc/tests/test_sugavanam_ertin_pvc.py \
    rift_pvc/tests/test_sugavanam_ertin_xpu.py rift_pvc/tests/test_se_smoke_acceptance_pvc.py \
    2>&1 | tee "$AUDIT_ROOT/existing_tests.log"
SUITE_RC=${PIPESTATUS[0]}
python -B -m pytest -v -rs -p no:cacheprovider --basetemp="$AUDIT_ROOT/audit_tests" \
    scripts_pvc/audit_packages_cd_pvc.py 2>&1 | tee "$AUDIT_ROOT/audit_tests.log"
AUDIT_RC=${PIPESTATUS[0]}
FALLBACK_RC=0
if grep -in 'fallback from XPU to CPU' "$AUDIT_ROOT/existing_tests.log" "$AUDIT_ROOT/audit_tests.log"; then
    FALLBACK_RC=1
fi
printf 'EXISTING_SUITE_EXIT=%s AUDIT_EXIT=%s FALLBACK_FAILURE=%s\n' "$SUITE_RC" "$AUDIT_RC" "$FALLBACK_RC"
date -Is
if (( SUITE_RC != 0 || AUDIT_RC != 0 || FALLBACK_RC != 0 )); then exit 1; fi
