#!/bin/bash
# Gate 1: convert only train/validation payloads into 32 native, ragged shards;
# all 1,152 test payloads stay sealed until the frozen-checkpoint evaluation.
# Gate 0 is checked as a scientific prerequisite inside this job; deliberately
# no Slurm/manager dependency directive is encoded in this additive wrapper.
#SBATCH --job-name=rift_gotcha_j8_g1
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
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/mgr_slurm_logs/gotcha-joint8-gate1-%j.out

set -euo pipefail

[[ "$#" -eq 0 ]] || {
  echo 'ERROR: GOTCHA Joint-8 Gate 1 takes no arguments' >&2
  exit 2
}
[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo 'ERROR: GOTCHA Joint-8 Gate 1 requires a Slurm allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == embers && "${SLURM_JOB_PARTITION:-}" == cpu-small ]] || {
  echo 'ERROR: GOTCHA Joint-8 Gate 1 requires Embers/cpu-small' >&2
  exit 94
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
DATASET_ROOT=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814
RUN_ROOT="$PROJECT_ROOT/experiment_state/public_radar_tuning/gotcha_joint8_fullpol_v1"
GATE0_JSON="$RUN_ROOT/joint8_fullpol_dataaudit_v1.json"
OUTPUT_ROOT="$DATASET_ROOT/converted_v3_joint8_fullpol"

for path in \
  "$PROJECT_ROOT/scripts/materialize_gotcha_joint_fullpol_v1.py" \
  "$PROJECT_ROOT/rift/gotcha_joint_fullpol.py"; do
  [[ -f "$path" ]] || {
    echo "ERROR: missing Gate-1 authored source: $path" >&2
    exit 94
  }
done
[[ -f "$GATE0_JSON" && ! -L "$GATE0_JSON" ]] || {
  echo "ERROR: missing regular Gate-0 manifest: $GATE0_JSON" >&2
  exit 94
}

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

# Fail before conversion unless the on-disk Gate-0 artifact is the complete,
# unopened, correction-free manifest produced by the frozen Gate-0 contract.
gate0_prerequisite() {
python -B - "$GATE0_JSON" "$DATASET_ROOT" <<'PY'
import json
from pathlib import Path
import sys

from scripts.materialize_gotcha_joint_fullpol_v1 import (
    adapt_gate0_inventory,
    validate_authentic_gate0,
)

path = Path(sys.argv[1])
expected_dataset_root = Path(sys.argv[2]).resolve()
with path.open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)
summary = validate_authentic_gate0(
    manifest,
    gate0_inventory_path=path,
    enforce_production_paths=True,
)
if Path(manifest["dataset_root"]).resolve() != expected_dataset_root:
    raise SystemExit("Gate-0 prerequisite names the wrong frozen dataset root")
records = adapt_gate0_inventory(manifest)
for record in records:
    source = Path(record.source_path)
    if source.is_symlink() or not source.is_file():
        raise SystemExit(f"Gate-0 source is missing or not regular: {source}")
print(json.dumps({"gate0_prerequisite": "passed", **summary}, sort_keys=True))
PY
}

gate1_postflight() {
python -B - "$OUTPUT_ROOT" "$GATE0_JSON" <<'PY'
import json
from pathlib import Path
import sys

from scripts.materialize_gotcha_joint_fullpol_v1 import (
    _validate_materialization_manifest,
    adapt_gate0_inventory,
    validate_authentic_gate0,
)

root = Path(sys.argv[1])
gate0_path = Path(sys.argv[2])
manifest_path = root / "manifest.json"
with manifest_path.open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)
with gate0_path.open("r", encoding="utf-8") as stream:
    gate0 = json.load(stream)
validate_authentic_gate0(
    gate0,
    gate0_inventory_path=gate0_path,
    enforce_production_paths=True,
)
records = adapt_gate0_inventory(gate0)
summary = _validate_materialization_manifest(
    manifest,
    output_root=root,
    records=records,
    gate0_payload=gate0,
    gate0_inventory_path=gate0_path,
    require_published_manifest=True,
)
if manifest.get("corrections_applied") is not False:
    raise SystemExit("Gate 1 must preserve corrections_applied=false")
if manifest.get("inventoried_source_file_count") != 11520:
    raise SystemExit("Gate 1 inventoried source count is not 11,520")
if manifest.get("payload_audited_source_file_count") != 10368:
    raise SystemExit("Gate 1 payload-audited source count is not 10,368")
if manifest.get("sealed_test_source_file_count") != 1152:
    raise SystemExit("Gate 1 sealed-test source count is not 1,152")
print(json.dumps({"gate1_postflight": "passed", **summary}, sort_keys=True))
PY
}

gate0_prerequisite

if [[ -e "$OUTPUT_ROOT" || -L "$OUTPUT_ROOT" ]]; then
  [[ -d "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
    echo "ERROR: existing Gate-1 output is not a regular non-symlink directory: $OUTPUT_ROOT" >&2
    exit 94
  }
fi

python -B "$PROJECT_ROOT/scripts/materialize_gotcha_joint_fullpol_v1.py" \
  --gate0-json "$GATE0_JSON" \
  --output-root "$OUTPUT_ROOT"

gate1_postflight
