#!/bin/bash
# Fresh v2 allocated CPU-only environment and engineering gate for the
# isolated corrected B7873200 Sugavanam--Ertin Stage-2 lane.  It runs only
# synthetic validators in a job-owned temporary directory.  It never opens a
# B787 archive, response payload, baseline checkpoint, or scientific output.
#
# V1 remains historical evidence.  V2 changes only the scoped interpreter
# search path needed for scikit-image's marching-cubes import: RIFT supplies
# NumPy/SciPy/PyTorch first, and the established sensor_fusion site supplies
# scikit-image last.  No environment is installed, changed, or persisted.
# The experiment manager supplies partition, account, resources, time, job
# name, and durable report destination.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE

set -euo pipefail
umask 077

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: B7873200 SE CPU validation v2 takes no arguments' >&2
  exit 64
}
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: B7873200 SE CPU validation v2 requires one non-array allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == "embers" ]] || {
  echo "ERROR: B7873200 SE CPU validation v2 requires Embers, got ${SLURM_JOB_QOS:-<unset>}" >&2
  exit 94
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
for path in \
  "$PROJECT_ROOT/train.py" \
  "$PROJECT_ROOT/train_sugavanam_ertin_stage1.py" \
  "$PROJECT_ROOT/train_sugavanam_ertin_stage2.py" \
  "$PROJECT_ROOT/rift/calibration.py" \
  "$PROJECT_ROOT/rift/npz_dataset.py" \
  "$PROJECT_ROOT/rift/range_operator.py" \
  "$PROJECT_ROOT/rift/sparse_scene.py" \
  "$PROJECT_ROOT/rift/sugavanam_ertin.py" \
  "$PROJECT_ROOT/rift/sugavanam_ertin_a320_stabilized.py" \
  "$PROJECT_ROOT/rift/sugavanam_ertin_b7873200_stage1.py" \
  "$PROJECT_ROOT/rift/sugavanam_ertin_b7873200_stage2_v1.py" \
  "$PROJECT_ROOT/rift/sugavanam_ertin_stage2_runtime_v1.py" \
  "$PROJECT_ROOT/rift/sugavanam_ertin_validzero.py" \
  "$PROJECT_ROOT/scripts/validate_sugavanam_ertin_b7873200_stage1.py" \
  "$PROJECT_ROOT/scripts/validate_sugavanam_ertin_b7873200_stage1_operator.py" \
  "$PROJECT_ROOT/scripts/validate_sugavanam_ertin_b7873200_stage2_static.py" \
  "$PROJECT_ROOT/scripts/validate_sugavanam_ertin_b7873200_stage2_v1.py" \
  "$PROJECT_ROOT/scripts/validate_sugavanam_ertin_b7873200_stage2_environment_v2.py"; do
  [[ -f "$path" ]] || { echo "ERROR: missing reviewed source: $path" >&2; exit 94; }
done

module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

RIFT_SITE="$CONDA_PREFIX/lib/python3.10/site-packages"
EXPORT_SITE=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/sensor_fusion/lib/python3.10/site-packages
[[ -d "$RIFT_SITE" ]] || { echo "ERROR: missing RIFT site-packages: $RIFT_SITE" >&2; exit 94; }
[[ -d "$EXPORT_SITE" ]] || { echo "ERROR: missing scoped scikit-image site-packages: $EXPORT_SITE" >&2; exit 94; }
export SE_STAGE2_RIFT_SITE="$RIFT_SITE"
export SE_STAGE2_EXPORT_SITE="$EXPORT_SITE"
export PYTHONPATH="$PROJECT_ROOT:$RIFT_SITE:$EXPORT_SITE"

# Prove the mixed search path before model/test imports.  The Python check
# verifies module origins, prints versions and paths, and calls marching_cubes
# on a tiny synthetic field to expose an import/ABI failure before lifecycle
# validators create fixtures.
python -B - <<'PY'
import json
import os
from pathlib import Path
import sys

import numpy
import scipy
import torch
import skimage
from skimage.measure import marching_cubes

rift_site = Path(os.environ["SE_STAGE2_RIFT_SITE"]).resolve()
export_site = Path(os.environ["SE_STAGE2_EXPORT_SITE"]).resolve()
search_path = [str(Path(entry).resolve()) for entry in sys.path if entry]
try:
    rift_position = search_path.index(str(rift_site))
    export_position = search_path.index(str(export_site))
except ValueError as exc:
    raise RuntimeError("scoped dependency sites are absent from sys.path") from exc
if rift_position >= export_position:
    raise RuntimeError("RIFT site-packages must precede the scoped scikit-image site")

def require_origin(module, root, name):
    location = getattr(module, "__file__", None)
    if not location:
        raise RuntimeError(f"{name} has no import location")
    resolved = Path(location).resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"{name} resolved outside expected site: {resolved}")
    return str(resolved)

origins = {
    "numpy": require_origin(numpy, rift_site, "numpy"),
    "scipy": require_origin(scipy, rift_site, "scipy"),
    "torch": require_origin(torch, rift_site, "torch"),
    "skimage": require_origin(skimage, export_site, "skimage"),
}
field = numpy.indices((2, 2, 2), dtype=numpy.float32)[0] - numpy.float32(0.5)
vertices, faces, _normals, _values = marching_cubes(field, level=0.0)
if vertices.size == 0 or faces.size == 0:
    raise RuntimeError("scikit-image marching_cubes returned an empty synthetic surface")
print(
    "SE_B7873200_STAGE2_ENVIRONMENT_V2_PASS "
    + json.dumps(
        {
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "skimage": skimage.__version__,
            "origins": origins,
            "rift_site_index": rift_position,
            "export_site_index": export_position,
            "synthetic_vertices": int(len(vertices)),
            "synthetic_faces": int(len(faces)),
        },
        sort_keys=True,
    ),
    flush=True,
)
PY

validation_parent="${SLURM_TMPDIR:-/tmp}"
[[ -d "$validation_parent" && -w "$validation_parent" && -x "$validation_parent" ]] || {
  echo "ERROR: no usable validation temporary parent: $validation_parent" >&2
  exit 94
}
validation_parent="$(cd -P -- "$validation_parent" && pwd -P)"
[[ "$validation_parent" != "/" ]] || {
  echo 'ERROR: refusing root as the validation temporary parent' >&2
  exit 94
}
validation_tmp="$(mktemp -d -- "$validation_parent/rift-se-b7873200-cpu-v2.${SLURM_JOB_ID}.XXXXXX")"
validation_tmp="$(cd -P -- "$validation_tmp" && pwd -P)"
case "$validation_tmp" in
  "$validation_parent"/rift-se-b7873200-cpu-v2."${SLURM_JOB_ID}".*) ;;
  *)
    echo "ERROR: unexpected validation temporary path: $validation_tmp" >&2
    exit 94
    ;;
esac
[[ "$(stat -c '%u' -- "$validation_tmp")" == "$(id -u)" ]] || {
  echo "ERROR: validation temporary child is not job-user owned: $validation_tmp" >&2
  exit 94
}
chmod 700 -- "$validation_tmp"
cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "${validation_tmp:-}" && -d "$validation_tmp" \
        && "$(dirname -- "$validation_tmp")" == "$validation_parent" \
        && "$(stat -c '%u' -- "$validation_tmp")" == "$(id -u)" ]]; then
    rm -rf -- "$validation_tmp" || echo "ERROR: failed to clean $validation_tmp" >&2
  fi
  exit "$status"
}
trap cleanup EXIT
export RIFT_VALIDATION_TMPDIR="$validation_tmp"
export TMPDIR="$validation_tmp"
export TEMP="$validation_tmp"
export TMP="$validation_tmp"

printf 'SE_B7873200_CPU_CONTRACT_V2_JOB=%s\n' "$SLURM_JOB_ID"
printf 'SE_B7873200_CPU_CONTRACT_V2_TEMP=%s\n' "$validation_tmp"

run_validator() {
  local label="$1"
  local marker="$2"
  local script="$3"
  local log="$validation_tmp/$label.log"
  printf 'VALIDATOR_START=%s\n' "$label"
  python -B -u "$script" 2>&1 | tee "$log"
  grep -Fq "$marker" "$log" || {
    echo "ERROR: validator $label omitted terminal marker: $marker" >&2
    exit 95
  }
  printf 'VALIDATOR_PASS=%s\n' "$label"
}

run_validator environment_static \
  'SE_B7873200_STAGE2_ENVIRONMENT_STATIC_V2_PASS:' \
  scripts/validate_sugavanam_ertin_b7873200_stage2_environment_v2.py
run_validator stage1_static \
  'SE_B7873200_STAGE1_STATIC_PASS:' \
  scripts/validate_sugavanam_ertin_b7873200_stage1.py
run_validator stage1_operator \
  'SE_B7873200_STAGE1_OPERATOR_PASS:' \
  scripts/validate_sugavanam_ertin_b7873200_stage1_operator.py
run_validator stage2_static \
  'SE_B7873200_STAGE2_STATIC_PASS:' \
  scripts/validate_sugavanam_ertin_b7873200_stage2_static.py
run_validator stage2_lifecycle \
  'SE_B7873200_STAGE2_CPU_CONTRACT_PASS:' \
  scripts/validate_sugavanam_ertin_b7873200_stage2_v1.py

echo 'SE_B7873200_ALLOCATED_CPU_CONTRACT_V2_PASS'
