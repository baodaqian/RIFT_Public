#!/bin/bash
# Controlled RadarSplat B787 sweep over the occupancy-target threshold only.
# The completed S0 result is immutable; every cell writes to its own new run directory.
# Usage: sbatch ... train_radarsplat_occupancy_sweep.sbatch <cell> <fresh|resume>
# Cells: occ050, occ020, occ010, occ005, occ001.
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=embers
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --tmp=1G
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@300
#SBATCH --export=NONE
set -euo pipefail

[[ "$#" -eq 2 ]] || {
    echo 'ERROR: expected <cell> <fresh|resume>' >&2
    exit 2
}
CELL="$1"
MODE="$2"
case "$CELL" in
    occ050) OCCUPANCY_THRESHOLD=0.05 ;;
    occ020) OCCUPANCY_THRESHOLD=0.02 ;;
    occ010) OCCUPANCY_THRESHOLD=0.01 ;;
    occ005) OCCUPANCY_THRESHOLD=0.005 ;;
    occ001) OCCUPANCY_THRESHOLD=0.001 ;;
    *)
        echo "ERROR: unknown sweep cell: $CELL" >&2
        exit 2
        ;;
esac
[[ "$MODE" == fresh || "$MODE" == resume ]] || {
    echo 'ERROR: mode must be fresh or resume' >&2
    exit 2
}

ROOT=/storage/scratch1/1/dbao31/rift_power_baselines_impl_20260813_v2
DATA=/storage/project/r-jromberg3-0/dbao31/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz
CACHE=/storage/scratch1/1/dbao31/rift_power_baselines_20260813_v2/cache/b787_mf_power_targets_e32a32r32
SWEEP_ROOT=/storage/scratch1/1/dbao31/rift_radarsplat_occupancy_sweep_20260820
CHECKPOINT="$SWEEP_ROOT/$CELL/checkpoints"

resume_flag=--resume
if [[ "$MODE" == fresh ]]; then
    resume_flag=--no-resume
    [[ ! -e "$CHECKPOINT" ]] || {
        echo "ERROR: fresh cell already has a checkpoint directory: $CHECKPOINT" >&2
        exit 3
    }
else
    [[ -f "$CHECKPOINT/checkpoint_latest.pth.tar" ]] || {
        echo "ERROR: resume cell has no latest checkpoint: $CHECKPOINT" >&2
        exit 3
    }
fi

cd "$ROOT"
module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
source scripts/activate_exact_numpy.sh
export NO_FAST_MATH=0

python -B -c 'import importlib.metadata as m, torch; from torch.utils.cpp_extension import CUDA_HOME; assert torch.__version__ == "2.6.0+cu124", torch.__version__; assert torch.version.cuda == "12.4"; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability(0) == (9,0); assert ("H100" in torch.cuda.get_device_name(0) or "H200" in torch.cuda.get_device_name(0)); assert m.version("jaxtyping") == "0.2.19"; assert m.version("ninja") == "1.11.1.3"; assert m.version("rich") == "13.9.4"; assert CUDA_HOME; print(CUDA_HOME, torch.cuda.get_device_name(0), flush=True)'
CUDA_HOME=$(python -B -c 'from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)')
[[ "$(readlink -f "$(command -v nvcc)")" == "$(readlink -f "$CUDA_HOME/bin/nvcc")" ]] || {
    echo 'ERROR: PATH nvcc is not CUDA_HOME/bin/nvcc' >&2
    exit 2
}
nvcc --version | grep -F 'release 12.4' >/dev/null

train_pid=''
forward_signal() {
    if [[ -n "$train_pid" ]] && kill -0 "$train_pid" 2>/dev/null; then
        kill -TERM "$train_pid"
    fi
}
trap forward_signal TERM INT
set +e
python -B -u train_radarsplat.py \
    --profile paper_b787 --npz-path "$DATA" --cache-root "$CACHE" \
    --checkpoint-dir "$CHECKPOINT" --seed 42 \
    --n-azimuth 32 --n-elevation 32 --n-range 32 \
    --azimuth-start-deg -14.4 --azimuth-span-deg 28.8 \
    --elevation-start-deg -14.4 --elevation-span-deg 28.8 \
    --output-azimuth-resolution-deg 0.9 --elevation-resolution-deg 0.9 \
    --intermediate-azimuth-resolution-deg 0.09 \
    --azimuth-beamwidth-deg 6.768541419812125 \
    --spectral-leakage-width-m 0.19986163866666667 \
    --range-span-m 0.3 --eps2d-pixels 0.3 --gaussian-chunk-size 1024 \
    --max-raster-candidate-pairs 150000000 --raster-backend official_cuda \
    --init-num-gaussians 20000 --init-extent 1 --init-scale 0.002999736023229956 \
    --init-opacity 0.1 --init-noise-probability 0.1 --requested-sh-level 10 \
    --sh-degree 4 --sh-degree-interval 600 --planar-initialization \
    --scene-scale-margin 1 --global-scale 1 --use-noise-probability \
    --use-spectral-leakage --use-azimuth-antenna-gain --no-use-multipath \
    --output-floor 1e-6 --output-ceiling 1 --steps 3000 \
    --means-lr-base 1.6e-4 --scales-lr 5e-3 --quaternions-lr 1e-3 \
    --opacity-lr 5e-2 --noise-probability-lr 5e-2 --sh0-lr 2.5e-3 \
    --shN-lr 2.5e-3 --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-15 \
    --means-final-lr-factor 0.01 --ssim-weight 0.2 --occupancy-weight 5 \
    --occupancy-threshold "$OCCUPANCY_THRESHOLD" \
    --max-size-weight 100 --opacity-noise-weight 100 \
    --max-scale 0.005999472046459912 --refine-start 500 --refine-stop 15000 \
    --refine-every 100 --reset-every 3000 --pause-refine-after-reset 0 \
    --grow-grad2d 2e-4 --grow-scale3d 0.01 --grow-scale2d 0.05 \
    --prune-opacity 0.0005 --prune-scale3d 0.1 --prune-scale2d 0.15 \
    --refine-scale2d-stop 0 --no-revised-opacity --production-raster-smoke \
    --validation-every 1000 --checkpoint-every 100 --checkpoint-seconds 600 \
    --log-every 10 "$resume_flag" --device cuda &
train_pid=$!
while true; do
    wait "$train_pid"
    train_rc=$?
    kill -0 "$train_pid" 2>/dev/null || break
done
set -e
trap - TERM INT
exit "$train_rc"
