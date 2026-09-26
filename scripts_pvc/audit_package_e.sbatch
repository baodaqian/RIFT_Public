#!/bin/bash
# Independent Package E audit. Findings produce a nonzero job outcome.
#SBATCH --job-name=pvc-audit-package-e
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
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageE/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT || exit 1
source .local-setup/activate-pvc.sh || exit 1
export RIFT_ACCELERATOR=xpu
unset RIFT_PVC_TCNN_HALF RIFT_PVC_TCNN_WEIGHT_GRAD
export PYTHONPATH="$PWD:$PWD/tests" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 WANDB_MODE=offline
AUDIT_ROOT=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageE/independent_audit_${SLURM_JOB_ID}
mkdir -p "$AUDIT_ROOT/runtime" || exit 1
export TMPDIR="$AUDIT_ROOT/runtime" MPLCONFIGDIR="$AUDIT_ROOT/runtime/matplotlib" XDG_CACHE_HOME="$AUDIT_ROOT/runtime/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python -c 'from rift_pvc import accelerator as a; import torch; assert a.backend() == "xpu" and torch.xpu.device_count() == 1; print(a.describe())' || exit 1
python -m pytest -p rift_pvc.pytest_policy -q -rs -s -p no:cacheprovider \
  rift_pvc/tests/test_tcnn_torch.py rift_pvc/tests/test_radar_fields_pvc.py \
  rift_pvc/tests/test_radar_fields_recovery_pvc.py \
  rift_pvc/tests/test_radar_fields_xpu.py rift_pvc/tests/test_tcnn_parity.py \
  -k 'not determinism_report and not candidates_report and not bmm_report' \
  --basetemp "$AUDIT_ROOT/pytest" --junitxml "$AUDIT_ROOT/pytest.xml" > "$AUDIT_ROOT/tests.log" 2>&1
tests_rc=$?
cat "$AUDIT_ROOT/tests.log"
python -u scripts_pvc/audit_package_e.py --device xpu --output "$AUDIT_ROOT/probes" > "$AUDIT_ROOT/probes.log" 2>&1
audit_rc=$?
cat "$AUDIT_ROOT/probes.log"
fallback_rc=0
if grep -E 'Aten Op fallback from XPU to CPU' "$AUDIT_ROOT/tests.log" "$AUDIT_ROOT/probes.log"; then fallback_rc=1; fi
printf 'TESTS_EXIT=%s AUDIT_EXIT=%s FALLBACK_EXIT=%s\n' "$tests_rc" "$audit_rc" "$fallback_rc"
if [ "$tests_rc" -ne 0 ] || [ "$audit_rc" -ne 0 ] || [ "$fallback_rc" -ne 0 ]; then exit 2; fi
