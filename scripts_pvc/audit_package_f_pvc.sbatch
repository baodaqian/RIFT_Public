#!/bin/bash
# Independent Package F checks using an existing CUDA dump, never raw radar responses.
#SBATCH --job-name=pvc-audit-package-f
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
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageF/logs/%x-%j.log
set -eo pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
set -u
export RIFT_ACCELERATOR=xpu
unset RIFT_PVC_ALLOW_BACKEND RIFT_PVC_RADARSPLAT_ALLOW_CUDA_RESUME
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 WANDB_MODE=offline
AUDIT_DIR=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageF/audit_${SLURM_JOB_ID}
mkdir -p "$AUDIT_DIR"
export TMPDIR="$AUDIT_DIR/tmp" MPLCONFIGDIR="$AUDIT_DIR/matplotlib" XDG_CACHE_HOME="$AUDIT_DIR/cache"
mkdir -p "$TMPDIR"
python -c 'from rift_pvc import accelerator as a; import torch; print(a.describe()); assert a.backend() == "xpu" and not torch.cuda.is_available()'
python -m pip check > "$AUDIT_DIR/pip-check.log" 2>&1
# Preserve the author's replay report: the audit has its own symlink and output.
ln -s /scratch/group/p.cis261724.000/RIFT_pvc_runs/packageF/parity/radarsplat_parity_h100_2153858.pt "$AUDIT_DIR/reference.pt"
export RIFT_PVC_RADARSPLAT_PARITY_DUMP="$AUDIT_DIR/reference.pt"
set +e
RIFT_ACCELERATOR=cpu python -B -m pytest -q -rs -p no:cacheprovider -p rift_pvc.pytest_policy \
  tests/test_radarsplat_release.py tests/test_radarsplat_collection.py tests/test_radarsplat_gotcha.py \
  > "$AUDIT_DIR/original-tests.log" 2>&1
original_rc=$?
python -B -m pytest -q -rs -p no:cacheprovider -p rift_pvc.pytest_policy \
  rift_pvc/tests/test_fused_ssim_torch.py rift_pvc/tests/test_gsplat_torch_ops.py \
  rift_pvc/tests/test_radarsplat_pvc.py rift_pvc/tests/test_radarsplat_xpu.py \
  > "$AUDIT_DIR/device-tests.log" 2>&1
device_rc=$?
python -B -m pytest -q -rs -p no:cacheprovider rift_pvc/tests/test_radarsplat_parity.py \
  > "$AUDIT_DIR/parity-tests.log" 2>&1
parity_rc=$?
python -u scripts_pvc/audit_package_f_pvc.py --dump "$AUDIT_DIR/reference.pt" \
  --output "$AUDIT_DIR/diagnostics.json" > "$AUDIT_DIR/diagnostics.log" 2>&1
diagnostic_rc=$?
grep -n 'fallback from XPU to CPU' "$AUDIT_DIR/"*.log > "$AUDIT_DIR/fallbacks.txt"
fallback_rc=$?
printf 'original_tests=%s device_tests=%s parity_tests=%s diagnostics=%s fallback_scan=%s\n' \
  "$original_rc" "$device_rc" "$parity_rc" "$diagnostic_rc" "$fallback_rc" | tee "$AUDIT_DIR/status.txt"
echo "AUDIT_DIR=$AUDIT_DIR"
[[ $original_rc == 0 && $device_rc == 0 && $parity_rc == 0 && $diagnostic_rc == 0 && $fallback_rc == 1 ]]
