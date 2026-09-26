#!/bin/bash
# Submit with explicit --partition, --gres, --output and AUDIT_BACKEND/AUDIT_ROOT.
# Synthetic fixtures only; one accelerator per job, at most fifteen minutes.
#SBATCH --account=158648339640
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
set -eo pipefail
cd /home/u.db364833/RIFT
if [[ "$AUDIT_BACKEND" == cuda ]]; then
  source .local-setup/activate.sh
else
  source .local-setup/activate-pvc.sh
fi
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export RIFT_ACCELERATOR="$AUDIT_BACKEND" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
export WANDB_MODE=offline
export MPLCONFIGDIR="$AUDIT_ROOT/runtime/$AUDIT_BACKEND/mpl"
export XDG_CACHE_HOME="$AUDIT_ROOT/runtime/$AUDIT_BACKEND/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$AUDIT_ROOT/$AUDIT_BACKEND"
AUDIT_EXTRA=()
if [[ "${AUDIT_REQUIRE_CUDA_PARITY:-0}" == 1 ]]; then
  AUDIT_EXTRA+=(--require-cuda-parity)
fi
python -u scripts_pvc/audit_ab_devices.py --backend "$AUDIT_BACKEND" \
  --output "$AUDIT_ROOT/$AUDIT_BACKEND" --reference "${AUDIT_REFERENCE:-$AUDIT_ROOT/cuda/numerics.pt}" "${AUDIT_EXTRA[@]}"
if [[ "$AUDIT_BACKEND" == xpu ]]; then
  python -m pytest -p rift_pvc.pytest_policy -q -rs -p no:cacheprovider \
    --basetemp="$AUDIT_ROOT/xpu/pytest" -W 'error:Aten Op fallback from XPU to CPU' \
    rift_pvc/tests/test_spinr_xpu.py rift_pvc/tests/test_geraf_pvc.py rift_pvc/tests/test_geraf_xpu.py \
    rift_pvc/tests/test_geraf_smoke.py rift_pvc/tests/test_geraf_readout_pvc.py rift_pvc/tests/test_ab_audit.py
else
  # Historical source inventories are documentation, never acceptance gates.
  python -m pytest -p rift_pvc.pytest_policy -q -rs -p no:cacheprovider --basetemp="$AUDIT_ROOT/cuda/pytest" \
    tests/test_spinr_direct.py tests/test_spinr_fidelity.py tests/test_spinr_gotcha.py \
    tests/test_spinr_quadrature_audit.py tests/test_geraf_source.py \
    tests/test_geraf_target_sampling.py tests/test_geraf_v1_hardening.py
fi
echo AUDIT_TESTS_PASSED
