#!/bin/bash
# Gate 0: inventory all 11,520 GOTCHA paths and payload-audit only the exact
# 10,368 train/validation files. The 1,152 test MAT payloads remain unopened.
# This job never converts responses, applies autofocus, or trains.
#SBATCH --job-name=rift_gotcha_j8_g0
#SBATCH --account=gts-jromberg3
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=20G
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/mgr_slurm_logs/gotcha-joint8-gate0-%j.out

set -euo pipefail

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: GOTCHA Joint-8 Gate 0 takes no arguments' >&2
  exit 2
}
[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo 'ERROR: GOTCHA Joint-8 Gate 0 requires a Slurm allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == embers && "${SLURM_JOB_PARTITION:-}" == cpu-small ]] || {
  echo 'ERROR: GOTCHA Joint-8 Gate 0 requires Embers/cpu-small' >&2
  exit 94
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
DATASET_ROOT=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814
RAW_GOTCHA_ROOT="$DATASET_ROOT/extracted/data/GOTCHA"
RUN_ROOT="$PROJECT_ROOT/experiment_state/public_radar_tuning/gotcha_joint8_fullpol_v1"
GATE0_JSON="$RUN_ROOT/joint8_fullpol_dataaudit_v1.json"

for path in \
  "$PROJECT_ROOT/scripts/audit_gotcha_joint_fullpol_v1.py" \
  "$PROJECT_ROOT/rift/gotcha_joint_fullpol.py"; do
  [[ -f "$path" ]] || {
    echo "ERROR: missing Gate-0 authored source: $path" >&2
    exit 94
  }
done
for directory in \
  "$RAW_GOTCHA_ROOT/GOTCHA-CP_Disc1/DATA" \
  "$RAW_GOTCHA_ROOT/GOTCHA-CP_Disc2/DATA"; do
  [[ -d "$directory" ]] || {
    echo "ERROR: missing frozen GOTCHA raw directory: $directory" >&2
    exit 94
  }
done

mkdir -p "$RUN_ROOT"
cd "$PROJECT_ROOT"
module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

gate0_postflight() {
python -B - "$GATE0_JSON" "$DATASET_ROOT" <<'PY'
import json
from pathlib import Path
import sys

from scripts.audit_gotcha_joint_fullpol_v1 import validate_gate0_inventory_manifest

path = Path(sys.argv[1])
expected_dataset_root = Path(sys.argv[2]).resolve()
with path.open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)
summary = validate_gate0_inventory_manifest(
    manifest,
    expected_dataset_root=expected_dataset_root,
    require_source_files_exist=True,
)
compatibility = manifest.get("contract_compatibility", {})
if compatibility.get("passed") is not True:
    raise SystemExit("Gate-0 manifest lacks a passing contract-compatibility record")
print(json.dumps({"gate0_postflight": "passed", **summary}, sort_keys=True))
PY
}

shopt -s nullglob
gate0_staging=("$RUN_ROOT"/.joint8_fullpol_dataaudit_v1.json.*.tmp)
shopt -u nullglob
if (( ${#gate0_staging[@]} != 0 )); then
  echo "ERROR: partial/extraneous Gate-0 staging artifact exists: ${gate0_staging[*]}" >&2
  exit 94
fi

if [[ -e "$GATE0_JSON" || -L "$GATE0_JSON" ]]; then
  [[ -f "$GATE0_JSON" && ! -L "$GATE0_JSON" ]] || {
    echo "ERROR: existing Gate-0 artifact is not a regular non-symlink file: $GATE0_JSON" >&2
    exit 94
  }
  gate0_postflight
  echo '{"gate0_idempotent_existing_artifact":true}'
  exit 0
fi

python -B "$PROJECT_ROOT/scripts/audit_gotcha_joint_fullpol_v1.py" \
  --dataset-root "$DATASET_ROOT" \
  --output "$GATE0_JSON"

gate0_postflight
