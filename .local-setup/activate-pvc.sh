# Usage: source /home/u.db364833/RIFT/.local-setup/activate-pvc.sh
# Intel PVC (Data Center GPU Max 1100) environment. Load NO CUDA or Intel
# compiler modules: the XPU torch wheel bundles its own oneAPI runtime and the
# cluster's intel-compilers module shadows it with an older UR loader.
source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate /scratch/group/p.cis261724.000/envs/RIFT-PVC || return
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
# Persistent SYCL kernel cache: first-use JIT compiles are cached across jobs.
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=/scratch/group/p.cis261724.000/sycl_cache
# Report any op that silently falls back from XPU to CPU.
export PYTORCH_DEBUG_XPU_FALLBACK=1
source /home/u.db364833/RIFT/.env.datasets.sh
