#!/bin/bash
# One combined real-data B787 Sugavanam--Ertin engineering smoke.
# The experiment manager is the only submitter.  This is not a production
# run, a 3,200-view comparison, or evidence of convergence or geometry quality.
# Stage 1 and Stage 2 execute in this one allocation and share the validated
# Stage-1 smoke bundle; no separate GPU stage is launched.
# AutoResume=false: continuation is literal and operator-selected until a
# cross-attempt interruption test has actually run.
#SBATCH --job-name=mgr_se_b7873200_real_smoke_v4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=inferno
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --tmp=12G
#SBATCH --time=01:00:00
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0,atl1-1-01-002-8-0,atl1-1-02-006-34-0
#SBATCH --export=NONE

set -euo pipefail
umask 077

case "$#" in
  0) resume_args=() ;;
  1)
    [[ "$1" == "resume" ]] || {
      echo 'ERROR: SE real smoke accepts zero arguments or the literal resume argument' >&2
      exit 64
    }
    resume_args=(--resume)
    ;;
  *)
    echo 'ERROR: SE real smoke accepts zero arguments or the literal resume argument' >&2
    exit 64
    ;;
esac

[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo 'ERROR: one non-array Slurm allocation is required' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == "inferno" && "${SLURM_JOB_PARTITION:-}" == "gpu-rtx6000" ]] || {
  echo 'ERROR: SE real smoke requires the Inferno RTX6000 allocation' >&2
  exit 94
}
[[ "${SLURM_CPUS_PER_TASK:-}" == "6" && "${SLURM_MEM_PER_NODE:-}" == "32768" ]] || {
  echo 'ERROR: SE real smoke requires 6 CPUs and 32 GiB RAM' >&2
  exit 94
}

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
PARENT_MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json
SMOKE_ROOT=/storage/scratch1/1/dbao31/rift_b7873200_sugavanam_ertin_real_smoke_v4
SMOKE_RUN_NAME=b78710k_sugavanam_ertin_stage1_stage2_engineering_smoke_v3
STAGE1_CHECKPOINT_NAME=sugavanam_ertin_b7873200_stage1_engineering_smoke_v3
RUN_ROOT="$SMOKE_ROOT/$SMOKE_RUN_NAME"
REPORT="$RUN_ROOT/real_smoke_report.json"

for source in \
  "$PROJECT/train.py" \
  "$PROJECT/rift/npz_dataset.py" \
  "$PROJECT/rift/sugavanam_ertin_b7873200_real_smoke.py" \
  "$PROJECT/train_sugavanam_ertin_smoke.py" \
  "$PROJECT/train_sugavanam_ertin_stage2.py" \
  "$PROJECT/rift/sugavanam_ertin_b7873200_stage2_v1.py" \
  "$PROJECT/rift/sugavanam_ertin_stage2_runtime_v1.py" \
  "$PROJECT/scripts/validate_sugavanam_ertin_b7873200_real_smoke_source.py" \
  "$PROJECT/scripts/validate_sugavanam_ertin_b7873200_real_smoke_runtime.py" \
  "$NPZ" \
  "$PARENT_MANIFEST" \
  /usr/bin/time; do
  [[ -f "$source" ]] || { echo "ERROR: missing reviewed smoke input: $source" >&2; exit 94; }
done

if [[ "$#" -eq 0 ]]; then
  [[ ! -e "$RUN_ROOT" ]] || {
    echo 'ERROR: fresh SE real smoke requires a new output identity' >&2
    exit 94
  }
else
  [[ ! -d "$RUN_ROOT/stage2/complete" ]] || {
    echo 'ERROR: completed SE real smoke is terminal and cannot resume' >&2
    exit 94
  }
  [[ -f "$RUN_ROOT/$STAGE1_CHECKPOINT_NAME/checkpoint_latest.pth.tar" || -f "$RUN_ROOT/stage2/checkpoint_latest.pth.tar" ]] || {
    echo 'ERROR: literal resume requires this smoke identity latest checkpoint' >&2
    exit 94
  }
fi

module purge
module load anaconda3/2023.03
module load nvhpc-cuda/12.4
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT
cd "$PROJECT"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6

RIFT_SITE="$CONDA_PREFIX/lib/python3.10/site-packages"
EXPORT_SITE=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/sensor_fusion/lib/python3.10/site-packages
[[ -d "$RIFT_SITE" && -d "$EXPORT_SITE" ]] || {
  echo 'ERROR: required RIFT/scikit-image dependency sites are unavailable' >&2
  exit 94
}
export SE_REAL_SMOKE_RIFT_SITE="$RIFT_SITE"
export SE_REAL_SMOKE_EXPORT_SITE="$EXPORT_SITE"
export PYTHONPATH="$PROJECT:$RIFT_SITE:$EXPORT_SITE"

command -v nvidia-smi >/dev/null 2>&1 || {
  echo 'ERROR: nvidia-smi is unavailable on the allocated GPU node' >&2
  exit 94
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -B - <<'PY'
import os
from pathlib import Path
import sys
import torch
import skimage
from skimage.measure import marching_cubes

rift_site = Path(os.environ["SE_REAL_SMOKE_RIFT_SITE"]).resolve()
export_site = Path(os.environ["SE_REAL_SMOKE_EXPORT_SITE"]).resolve()
search_path = [str(Path(entry).resolve()) for entry in sys.path if entry]
if search_path.index(str(rift_site)) >= search_path.index(str(export_site)):
    raise SystemExit("ERROR: RIFT site-packages must precede the scoped scikit-image site")
if rift_site not in Path(torch.__file__).resolve().parents:
    raise SystemExit("ERROR: Torch resolved outside the RIFT site")
if export_site not in Path(skimage.__file__).resolve().parents:
    raise SystemExit("ERROR: scikit-image resolved outside the scoped export site")
field = torch.zeros((2, 2, 2), dtype=torch.float32)
field[1, :, :] = 1.0
vertices, faces, _normals, _values = marching_cubes(field.numpy(), level=0.5)
if len(vertices) == 0 or len(faces) == 0:
    raise SystemExit("ERROR: marching_cubes dependency preflight returned an empty surface")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("ERROR: SE real smoke requires exactly one visible CUDA device")
print("SE_B7873200_REAL_SMOKE_GPU=" + torch.cuda.get_device_name(0))
print("SE_B7873200_REAL_SMOKE_GPU_TOTAL_MIB=" + str(torch.cuda.get_device_properties(0).total_memory // 2**20))
print("SE_B7873200_REAL_SMOKE_SKIMAGE=" + skimage.__version__)
PY

echo 'SE_B7873200_REAL_SMOKE_RUNTIME_PREFLIGHT_START'
python -B -u scripts/validate_sugavanam_ertin_b7873200_real_smoke_runtime.py
echo 'SE_B7873200_REAL_SMOKE_RUNTIME_PREFLIGHT_PASS'

echo 'SE_B7873200_REAL_SMOKE_SOURCE_PREFLIGHT_START'
python -B -u scripts/validate_sugavanam_ertin_b7873200_real_smoke_source.py
echo 'SE_B7873200_REAL_SMOKE_SOURCE_PREFLIGHT_PASS'

if [[ "$#" -eq 1 ]]; then
  resume_target=''
  if [[ -f "$RUN_ROOT/stage2/checkpoint_latest.pth.tar" ]]; then
    resume_target="$RUN_ROOT/stage2/checkpoint_latest.pth.tar"
  else
    resume_target="$RUN_ROOT/$STAGE1_CHECKPOINT_NAME/checkpoint_latest.pth.tar"
  fi
  resume_args+=("$resume_target")
fi

mkdir -p -- "$SMOKE_ROOT/launcher_logs"
time_log="$SMOKE_ROOT/launcher_logs/${SLURM_JOB_ID}.time.txt"
driver_log="$SMOKE_ROOT/launcher_logs/${SLURM_JOB_ID}.driver.log"
set +e
/usr/bin/time -v -o "$time_log" python -B -u train_sugavanam_ertin_smoke.py \
  --checkpoint-root "$SMOKE_ROOT" \
  --npz-path "$NPZ" \
  --parent-role-manifest "$PARENT_MANIFEST" \
  --device cuda \
  "${resume_args[@]}" 2>&1 | tee "$driver_log"
pipeline_status=("${PIPESTATUS[@]}")
driver_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
set -e
if (( driver_status != 0 )); then
  echo "ERROR: SE real smoke driver exited $driver_status; retaining partial evidence" >&2
  exit "$driver_status"
fi
if (( tee_status != 0 )); then
  echo 'ERROR: SE real smoke log capture failed; retaining partial evidence' >&2
  exit "$tee_status"
fi
[[ -f "$REPORT" ]] || {
  echo 'ERROR: SE real smoke did not produce its terminal report' >&2
  exit 95
}
grep -Fxq 'SE_B7873200_REAL_SMOKE_LIFECYCLE_COMPLETE' "$driver_log" || {
  echo 'ERROR: SE real smoke lacked its lifecycle-complete marker' >&2
  exit 95
}
echo 'SE_B7873200_REAL_SMOKE_LIFECYCLE_PASS'
