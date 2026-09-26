#!/bin/bash
#SBATCH --job-name=rift-pvc-verify
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/u.db364833/RIFT/.local-setup/pvc-verify-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
export WANDB_MODE=offline
hostname; date; echo "ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"; module list 2>&1 | tail -2
python -c "import torch; print('torch', torch.__version__, '| xpu', torch.xpu.is_available(), '|', torch.xpu.get_device_name(0))"
echo "=== pip check ==="; python -m pip check
echo "=== verify_environment_pvc.py ==="; python -u .local-setup/verify_environment_pvc.py || { echo "VERIFY_FAILED"; exit 1; }
echo "=== tests/test_training_architectures.py (CPU) ==="; python -m pytest tests/test_training_architectures.py -q -p no:cacheprovider || { echo "TESTS_FAILED"; exit 1; }
echo "=== scripts/validate_range_operator.py (stages A/C/D on CPU; B/E/F need CUDA or PACE data) ==="
timeout 600 python -u scripts/validate_range_operator.py --stages A,C,D --device cpu 2>&1 | tail -25 || echo "validate_range_operator.py exited nonzero (see above)"
date; echo VERIFY_DONE
