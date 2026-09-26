#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/storage/project/r-jromberg3-0/dbao31/RIFT}"
SHROOT="$PROJECT/external/SH_SAS_reference"
ENV_PREFIX="/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/RIFT_SAS_CU124_L40S"
CACHE="$PROJECT/datasets/sh_sas_reed/cache/armadillo_5khz_trainonly_120rings_v1_20260912"
OUTPUT_DIR="$PROJECT/training_checkpoints"
EXP_NAME="airsas_armadillo5k_official_shsas_smoke_v1_20260912"
RUN="$OUTPUT_DIR/$EXP_NAME"

: "${SLURM_JOB_ID:?must run in a Slurm allocation}"

if ! type module >/dev/null 2>&1; then
  for module_init in /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
    if [[ -r "$module_init" ]]; then
      # shellcheck disable=SC1090
      source "$module_init"
      break
    fi
  done
fi
type module >/dev/null 2>&1 || { echo "PACE environment modules are not initialized" >&2; exit 2; }
module load anaconda3/2023.03
module load nvhpc-cuda/12.4

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

export CUDA_HOME=/usr/local/pace-apps/manual/packages/nvhpc/24.5/Linux_x86_64/24.5/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
export PYQTGRAPH_QT_LIB=PyQt6
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$SHROOT/inr_reconstruction:$SHROOT:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"

if ! python -c 'from torch.utils.tensorboard import SummaryWriter'; then
  python -m pip install --no-deps --only-binary=:all: \
    tensorboard==2.19.0 absl-py==2.2.2 grpcio==1.71.0 tensorboard-data-server==0.7.2
fi

if ! python -c 'import eval_sh'; then
  (
    cd "$SHROOT/inr_reconstruction/eval-sh"
    python -m pip install --no-build-isolation --no-deps .
  )
fi

python -B "$PROJECT/scripts/validate_official_shsas_cuda.py" kernel
mkdir -p "$RUN"
cd "$SHROOT/scenes/airsas/arma"

python -B "$SHROOT/inr_reconstruction/reconstruct_scene_dir_sh.py" \
  --scene_inr_config "$SHROOT/scenes/airsas/arma/nbp_config.json" \
  --system_data "$PROJECT/datasets/sh_sas_reed/airsas_processed/system_data_files/system_data_arma_5k.pik" \
  --fit_folder /storage/scratch1/1/dbao31/airsas_armadillo5k_frontend_trainonly_v1_20260912/deconvolved_measurements \
  --output_dir "$OUTPUT_DIR" \
  --expname "$EXP_NAME" \
  --rift_cache "$CACHE" \
  --rift_seed 42 \
  --plot_thresh 2 \
  --learning_rate 1e-3 \
  --num_epochs 6 \
  --num_rays 5000 \
  --info_every 1 \
  --scene_every 1000000 \
  --export_model_every 5 \
  --accum_grad 5 \
  --scale_factor 30 \
  --max_weights 200 \
  --use_up_to 120 \
  --sampling_distribution_uniformity 1 \
  --lambertian_ratio 0 \
  --occlusion \
  --occlusion_scale 500 \
  --num_layers 2 \
  --num_neurons 32 \
  --reg_start 0 \
  --thresh 0 \
  --smooth_loss 50 \
  --smooth_delta 1 \
  --sparsity 10 \
  --point_at_center \
  --transmit_from_tx \
  --normalize_scene_dims \
  --beamwidth 30 \
  --phase_loss 0.1 \
  --sh_levels 3 \
  2>&1 | tee -a "$RUN/native_train.log"

python -B "$PROJECT/scripts/validate_official_shsas_cuda.py" checkpoints --run "$RUN" --cache "$CACHE"
