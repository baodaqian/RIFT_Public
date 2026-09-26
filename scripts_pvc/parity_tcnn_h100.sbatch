#!/bin/bash
# Gate 2 of the Radar Fields PVC port: dump the real tiny-cuda-nn modules on an
# H100 (campaign build, CUDA env) and replay them through the torch shim in the
# same job (shim on CUDA); the PVC replay runs separately via run_tests_pvc.sbatch
# with RIFT_PVC_TCNN_PARITY_NPZ pointing at the NPZ written here.
#SBATCH --job-name=h100-tcnn-parity
#SBATCH --account=158648339640
#SBATCH --partition=gpu_debug
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageE/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
module load GCCcore/12.3.0 CUDA/12.4.1
source .local-setup/activate.sh
SNAP=/scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645
export PYTHONPATH="$SNAP/runtime/b787_radar_fields/site:$PWD" OMP_NUM_THREADS=8 WANDB_MODE=offline PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
OUT=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageE/parity/tcnn_parity_h100_${SLURM_JOB_ID}.npz
hostname; date; nvidia-smi -L
python -c "import torch, tinycudann as t; print('torch', torch.__version__, torch.version.cuda, 'tinycudann', t.__file__)" || { echo "PARITY_EXIT=3"; exit 3; }
python -u scripts_pvc/parity_tcnn_dump.py --output "$OUT" --points 4096; rc=$?
echo "DUMP_EXIT=$rc"; [ $rc -ne 0 ] && { echo "PARITY_EXIT=$rc"; exit $rc; }
export RIFT_PVC_TCNN_PARITY_NPZ="$OUT"
echo "=== shim replay on the H100 (CUDA) ==="
python -m pytest rift_pvc/tests/test_tcnn_parity.py -q -rs -s -p no:cacheprovider; rc=$?
date; echo "PARITY_EXIT=$rc"; exit $rc
