# Usage from this checkout: source .local-setup/activate-public-pvc.sh
# Public packaging helper. The original ACES activate-pvc.sh is preserved.
if ! declare -F conda >/dev/null 2>&1; then
    if ! command -v conda >/dev/null 2>&1; then
        printf '%s\n' 'Conda is required; initialize it before sourcing this file.' >&2
        return 1
    fi
    rift_public_conda_base=$(conda info --base) || return
    source "$rift_public_conda_base/etc/profile.d/conda.sh" || return
    unset rift_public_conda_base
fi
conda activate "${RIFT_PVC_ENV:-RIFT-PVC}" || return
rift_public_source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || return
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export SYCL_CACHE_PERSISTENT=1
export PYTORCH_DEBUG_XPU_FALLBACK=1
export RIFT_ACCELERATOR=xpu
export SYCL_CACHE_DIR="${SYCL_CACHE_DIR:-${RIFT_PVC_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/rift-public}/sycl}"
if [ -f "$rift_public_source_root/.env.datasets.sh" ]; then
    source "$rift_public_source_root/.env.datasets.sh"
fi
unset rift_public_source_root
