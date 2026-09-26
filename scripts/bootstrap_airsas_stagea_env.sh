#!/usr/bin/env bash
set -euo pipefail

# Research-owned StageA bootstrap for an already allocated PACE GPU shell.
# It never submits jobs, edits manager files, or removes an existing target.

die() {
  echo "bootstrap_airsas_stagea_env: $*" >&2
  exit 2
}

PROJECT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
REED_ROOT="${REED_ROOT:-$PROJECT/external/Reed_SAS_reference}"
CONDA_ROOT="${ANACONDA_ROOT:-/usr/local/pace-apps/manual/packages/anaconda3/2023.03}"
CUDA_HOME="${CUDA_HOME:-/usr/local/pace-apps/manual/packages/nvhpc/24.5/Linux_x86_64/24.5/cuda}"
BASE_ENV="${RIFT_BASE_ENV:-/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/RIFT}"
ENV_PREFIX="${RIFT_SAS_ENV_PREFIX:-/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/RIFT_SAS_CU124}"
PIP_PACKAGE_ARCHIVE="${RIFT_SAS_PIP_PACKAGE_ARCHIVE:-/storage/home/hcoda1/1/dbao31/.conda/pkgs/pip-25.2-pyhc872135_1.conda}"
BUILD_ROOT="${TCNN_BUILD_ROOT:-${TMPDIR:-/tmp}/rift_sas_tcnn_v2.0}"
BUILD_JOBS="${TCNN_BUILD_JOBS:-4}"
RUN_FRONTEND_AFTER_SMOKE="${RUN_FRONTEND_AFTER_SMOKE:-0}"
RESUME_EXISTING="${RIFT_SAS_RESUME_EXISTING:-0}"
REED_GUI_PACKAGES=(
  "pyqtgraph==0.13.7"
  "PyQt6==6.7.1"
  "PyOpenGL==3.1.7"
  "PyMCubes==0.1.6"
)
TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-90}"
case "$TCNN_CUDA_ARCHITECTURES" in
  90) TCNN_EXPECTED_CAPABILITY="9,0" ;;
  89) TCNN_EXPECTED_CAPABILITY="8,9" ;;
  *) die "TCNN_CUDA_ARCHITECTURES must be 90 or 89" ;;
esac
export TCNN_CUDA_ARCHITECTURES TCNN_EXPECTED_CAPABILITY

TCNN_ARCHIVE_URL="${TCNN_ARCHIVE_URL:-https://codeload.github.com/NVlabs/tiny-cuda-nn/tar.gz/refs/tags/v2.0}"
CUTLASS_ARCHIVE_URL="${CUTLASS_ARCHIVE_URL:-https://codeload.github.com/NVIDIA/cutlass/tar.gz/1eb6355}"
FMT_ARCHIVE_URL="${FMT_ARCHIVE_URL:-https://codeload.github.com/fmtlib/fmt/tar.gz/b0c8263}"

[[ -d "$PROJECT" ]] || die "project directory is missing: $PROJECT"
[[ -d "$REED_ROOT" ]] || die "Reed reference is missing: $REED_ROOT"
[[ -d "$BASE_ENV" ]] || die "base RIFT environment is missing: $BASE_ENV"
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || die "TCNN_BUILD_JOBS must be a positive integer"
[[ "$RUN_FRONTEND_AFTER_SMOKE" == "0" || "$RUN_FRONTEND_AFTER_SMOKE" == "1" ]] || \
  die "RUN_FRONTEND_AFTER_SMOKE must be 0 or 1"
[[ "$RESUME_EXISTING" == "0" || "$RESUME_EXISTING" == "1" ]] || \
  die "RIFT_SAS_RESUME_EXISTING must be 0 or 1"
if [[ "$RUN_FRONTEND_AFTER_SMOKE" == "1" ]]; then
  [[ -s "$PROJECT/scripts/run_airsas_frontend.sh" ]] || die "frontend launcher is missing"
fi

for command_name in curl tar python; do
  command -v "$command_name" >/dev/null 2>&1 || die "required command is missing: $command_name"
done

if ! type module >/dev/null 2>&1; then
  for module_init in /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
    if [[ -r "$module_init" ]]; then
      # shellcheck disable=SC1090
      source "$module_init"
      break
    fi
  done
fi
type module >/dev/null 2>&1 || die "PACE environment modules are not initialized"
module load anaconda3/2023.03
module load nvhpc-cuda/12.4

[[ -r "$CONDA_ROOT/etc/profile.d/conda.sh" ]] || die "conda initialization is missing: $CONDA_ROOT"
# shellcheck disable=SC1090
source "$CONDA_ROOT/etc/profile.d/conda.sh"
command -v conda >/dev/null 2>&1 || die "conda is unavailable after module setup"

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
export PYQTGRAPH_QT_LIB=PyQt6

if [[ -e "$ENV_PREFIX" ]]; then
  [[ "$RESUME_EXISTING" == "1" ]] || die "refusing to overwrite existing environment target: $ENV_PREFIX"
  [[ -x "$ENV_PREFIX/bin/python" ]] || die "existing environment target has no Python: $ENV_PREFIX"
  [[ -s "$ENV_PREFIX/conda-meta/history" ]] || die "existing environment target has no conda history: $ENV_PREFIX"
else
  [[ "$RESUME_EXISTING" == "0" ]] || die "cannot resume missing environment target: $ENV_PREFIX"
  conda create -y --prefix "$ENV_PREFIX" --clone "$BASE_ENV"
fi
[[ -s "$PIP_PACKAGE_ARCHIVE" ]] || die "cached pip package is missing: $PIP_PACKAGE_ARCHIVE"
conda install -y --offline --force-reinstall --no-deps --prefix "$ENV_PREFIX" "$PIP_PACKAGE_ARCHIVE"
conda activate "$ENV_PREFIX"

python - <<'PY'
import pip
from pip._vendor.resolvelib.structs import RequirementInformation

assert pip.__version__ == "25.2", pip.__version__
assert RequirementInformation.__name__ == "RequirementInformation"
PY

python - <<'PY'
import os

import torch

assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torch.version.cuda == "12.4", torch.version.cuda
assert torch.cuda.is_available(), "StageA bootstrap requires an allocated CUDA device"
expected_capability = tuple(int(value) for value in os.environ["TCNN_EXPECTED_CAPABILITY"].split(","))
assert torch.cuda.get_device_capability() == expected_capability, torch.cuda.get_device_capability()
PY

if ! python -c 'import commentjson' >/dev/null 2>&1; then
  python -m pip install --no-input --disable-pip-version-check commentjson==0.9.0
fi
python -m pip install --no-input --disable-pip-version-check "${REED_GUI_PACKAGES[@]}"
command -v ninja >/dev/null 2>&1 || die "ninja is required in the cloned RIFT environment"

TCNN_REUSE=0
if [[ "$RESUME_EXISTING" == "1" ]]; then
  if python - <<'PY'
import importlib.metadata as metadata
import os

import torch
import tinycudann

assert metadata.version("tinycudann") == "2.0"
assert torch.cuda.is_available()
expected_capability = tuple(int(value) for value in os.environ["TCNN_EXPECTED_CAPABILITY"].split(","))
assert torch.cuda.get_device_capability() == expected_capability
PY
  then
    TCNN_REUSE=1
  fi
fi

if [[ "$TCNN_REUSE" == "0" ]]; then
  [[ ! -e "$BUILD_ROOT" ]] || die "refusing to reuse existing build target: $BUILD_ROOT"
  mkdir -p "$BUILD_ROOT/tiny-cuda-nn-v2.0"
  TCNN_ROOT="$BUILD_ROOT/tiny-cuda-nn-v2.0"

  download_and_extract() {
    local url="$1"
    local archive="$2"
    local destination="$3"
    curl -fL --output "$archive" "$url"
    tar -xzf "$archive" --strip-components=1 -C "$destination"
  }

  download_and_extract \
    "$TCNN_ARCHIVE_URL" \
    "$BUILD_ROOT/tiny-cuda-nn-v2.0.tar.gz" \
    "$TCNN_ROOT"

  mkdir -p "$TCNN_ROOT/dependencies/cutlass" "$TCNN_ROOT/dependencies/fmt"
  download_and_extract \
    "$CUTLASS_ARCHIVE_URL" \
    "$BUILD_ROOT/cutlass.tar.gz" \
    "$TCNN_ROOT/dependencies/cutlass"
  download_and_extract \
    "$FMT_ARCHIVE_URL" \
    "$BUILD_ROOT/fmt.tar.gz" \
    "$TCNN_ROOT/dependencies/fmt"

  [[ -s "$TCNN_ROOT/bindings/torch/setup.py" ]] || die "tiny-cuda-nn Torch setup.py was not extracted"
  [[ -s "$TCNN_ROOT/dependencies/cutlass/include/cutlass/cutlass.h" ]] || die "CUTLASS headers are incomplete"
  [[ -s "$TCNN_ROOT/dependencies/fmt/src/format.cc" ]] || die "fmt sources are incomplete"

  pushd "$TCNN_ROOT/bindings/torch" >/dev/null
  export MAX_JOBS="$BUILD_JOBS"
  python setup.py bdist_wheel
  mapfile -t TCNN_WHEELS < <(find dist -maxdepth 1 -type f -name 'tinycudann-*.whl' -print | sort)
  [[ "${#TCNN_WHEELS[@]}" -eq 1 ]] || die "expected exactly one tinycudann wheel, found ${#TCNN_WHEELS[@]}"
  python -m pip install --no-input --disable-pip-version-check --no-deps "${TCNN_WHEELS[0]}"
  popd >/dev/null
fi

export PYTHONPATH="$REED_ROOT:$REED_ROOT/inr_reconstruction:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export ARM5K_CONFIG="$REED_ROOT/scenes/airsas/arma_5k/pulse_deconvolve.json"
[[ -s "$ARM5K_CONFIG" ]] || die "Arm5k config is missing: $ARM5K_CONFIG"

python - <<'PY'
import os

import commentjson
import mcubes
import numpy as np
import OpenGL
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import QApplication
import scipy
import torch
import tinycudann as tcnn

import deconvolve_measurements  # noqa: F401
import network  # noqa: F401

config_path = os.environ["ARM5K_CONFIG"]
with open(config_path, encoding="utf-8") as handle:
    config = commentjson.load(handle)
expected_capability = tuple(int(value) for value in os.environ["TCNN_EXPECTED_CAPABILITY"].split(","))

assert config["encoding"]["otype"] == "HashGrid"
assert config["encoding"]["n_levels"] == 20
assert config["encoding"]["n_features_per_level"] == 2
assert config["encoding"]["log2_hashmap_size"] == 15
assert config["encoding"]["base_resolution"] == 16
assert config["encoding"]["per_level_scale"] == 1.5
assert config["network"]["otype"] == "FullyFusedMLP"
assert config["network"]["n_neurons"] == 128
assert config["network"]["n_hidden_layers"] == 3

assert torch.cuda.is_available()
assert torch.version.cuda == "12.4"
assert torch.cuda.get_device_capability() == expected_capability
model = tcnn.NetworkWithInputEncoding(
    n_input_dims=2,
    n_output_dims=1,
    encoding_config=config["encoding"],
    network_config=config["network"],
).cuda()
coordinates = torch.rand((256, 2), device="cuda", requires_grad=True)
prediction = model(coordinates)
assert prediction.shape == (256, 1), prediction.shape
loss = prediction.square().mean()
assert torch.isfinite(loss), loss
loss.backward()
torch.cuda.synchronize()
assert coordinates.grad is not None
assert torch.isfinite(coordinates.grad).all()
parameter_gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
assert parameter_gradients
assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in parameter_gradients)
assert np.isfinite(loss.detach().cpu().item())
print(
    "RIFT_SAS_STAGEA_ENV_SMOKE_PASS",
    torch.__version__,
    torch.version.cuda,
    tcnn.__file__,
    scipy.__version__,
)
PY

if [[ "$RUN_FRONTEND_AFTER_SMOKE" == "1" ]]; then
  exec bash "$PROJECT/scripts/run_airsas_frontend.sh"
fi
echo "RIFT_SAS_STAGEA_ENV_SMOKE_PASS"
