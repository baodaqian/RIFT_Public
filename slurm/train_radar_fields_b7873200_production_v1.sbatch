#!/bin/bash
# Proposed full-scale B787 Radar Fields recipe.  This is a release package,
# not launch authorization; the config keeps production_clearance=false.
# Manager envelope: Inferno / slow / RTX6000 / 6 CPU / 32G / 12G tmp / 12h,
# charged to gts-jromberg3-ece.  The scientific recipe is unchanged.
# Training keeps the sealed test and unused roles unavailable.  The postflight
# readout opens val and test only after the final checkpoint exists.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --tmp=12G
#SBATCH --time=12:00:00
#SBATCH --signal=TERM@300
#SBATCH --export=NONE
set -euo pipefail
umask 077

MODE=fresh
RESUME_ARGS=()
RESUME_AUTO=0
READOUT_ONLY_CHECKPOINT=''
if [[ "$#" -eq 0 ]]; then
  RESUME_ARGS=()
elif [[ "$#" -eq 1 && "$1" == "--resume" ]]; then
  # The manager supplies a pathless resume request.  Resolve it from the newest
  # preserved clean-stop checkpoint so every continuation advances from the
  # latest completed attempt rather than replaying an older checkpoint.
  RESUME_ARGS=(--resume)
  RESUME_AUTO=1
elif [[ "$#" -eq 2 && "$1" == "--resume" && -f "$2" ]]; then
  RESUME_ARGS=(--resume "$2")
elif [[ "$#" -eq 2 && "$1" == "--readout-only" && -f "$2" && "$(basename "$2")" == "checkpoint_final.pth.tar" ]]; then
  MODE=readout_only
  READOUT_ONLY_CHECKPOINT="$2"
else
  echo 'ERROR: expected no arguments, --resume CHECKPOINT, or --readout-only CHECKPOINT' >&2
  exit 64
fi

[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 94
[[ "${SLURM_JOB_QOS:-}" == inferno ]] || exit 94
[[ "${SLURM_JOB_ACCOUNT:-}" == gts-jromberg3-ece ]] || exit 94
[[ "${SLURM_MEM_PER_NODE:-}" == 32768 ]] || exit 94

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
CONFIG="$PROJECT/protocols/radar_fields_b7873200_production_v1.json"
NPZ=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz
MANIFEST=/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json
CAMPAIGN=/storage/scratch1/1/dbao31/rift_b7873200_radar_fields_production_v1
CHECKPOINT_NAME=rf_b7873200_production_v1

for source in \
  train_radar_fields.py \
  rift/radar_fields.py \
  rift/radar_fields_dataset.py \
  protocols/radar_fields_b7873200_production_v1.json \
  scripts/readout_radar_fields_b7873200_native.py \
  scripts/validate_radar_fields_production_config.py \
  scripts/validate_radar_fields_native_readout.py \
  scripts/inspect_radar_fields_training_resource.py; do
  [[ -f "$PROJECT/$source" ]] || { echo "ERROR: missing $source" >&2; exit 94; }
done
[[ -f "$NPZ" && -f "$MANIFEST" ]] || exit 94

module purge
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT
cd "$PROJECT"
command -v srun >/dev/null 2>&1 || { echo 'ERROR: srun is required for the RF job steps' >&2; exit 94; }
export PYTHONPATH=. PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
# --export=NONE is retained at allocation time; make the activated runtime
# and the explicit srun steps inherit this job environment.
export SLURM_EXPORT_ENV=ALL

python -B scripts/validate_radar_fields_production_config.py \
  --config "$CONFIG" \
  --launcher "$PROJECT/slurm/train_radar_fields_b7873200_production_v1.sbatch" \
  --readout "$PROJECT/scripts/readout_radar_fields_b7873200_native.py" \
  --readout-validator "$PROJECT/scripts/validate_radar_fields_native_readout.py" \
  --resource-inspector "$PROJECT/scripts/inspect_radar_fields_training_resource.py"

mkdir -p -- "$CAMPAIGN"

if [[ "$RESUME_AUTO" -eq 1 ]]; then
  RESUME_SOURCE=''
  RESUME_SOURCE_MTIME=0
  while IFS= read -r candidate; do
    candidate_attempt="$(dirname "$(dirname "$candidate")")"
    [[ -f "$candidate_attempt/training_status.json" ]] || continue
    grep -Eq '"status"[[:space:]]*:[[:space:]]*"clean_stop"' \
      "$candidate_attempt/training_status.json" || continue
    candidate_mtime="$(stat -c '%Y' "$candidate")"
    if [[ "$candidate_mtime" -gt "$RESUME_SOURCE_MTIME" ]]; then
      RESUME_SOURCE="$candidate"
      RESUME_SOURCE_MTIME="$candidate_mtime"
    fi
  done < <(find "$CAMPAIGN" -mindepth 3 -maxdepth 3 -type f -name checkpoint_latest.pth.tar -print)
  if [[ -z "$RESUME_SOURCE" ]]; then
    echo 'ERROR: --resume requested but no preserved clean-stop checkpoint was found' >&2
    exit 94
  fi
  RESUME_ARGS=(--resume "$RESUME_SOURCE")
  echo "RF resume source selected: $RESUME_SOURCE"
fi

if [[ "$MODE" == readout_only ]]; then
  RECOVERY="$CAMPAIGN/readout-recovery-$SLURM_JOB_ID"
  mkdir -- "$RECOVERY"
  SOURCE_ATTEMPT="$(dirname "$(dirname "$READOUT_ONLY_CHECKPOINT")")"
  SOURCE_RESOURCE="$SOURCE_ATTEMPT/training_resource.json"
  set +e
  SOURCE_RESOURCE_STATUS_JSON=$(python -B scripts/inspect_radar_fields_training_resource.py \
    --path "$SOURCE_RESOURCE" --reject-failed)
  SOURCE_RESOURCE_EXIT=$?
  set -e
  if [[ "$SOURCE_RESOURCE_EXIT" -ne 0 ]]; then
    echo 'ERROR: refusing readout-only recovery because prior training resource evidence recorded a failed gate' >&2
    echo "$SOURCE_RESOURCE_STATUS_JSON" >&2
    exit "$SOURCE_RESOURCE_EXIT"
  fi
  srun --kill-on-bad-exit=1 --ntasks=1 --nodes=1 --cpu-bind=none \
    python -B scripts/readout_radar_fields_b7873200_native.py \
    --config "$CONFIG" \
    --checkpoint "$READOUT_ONLY_CHECKPOINT" \
    --npz-path "$NPZ" \
    --sealed-split-manifest "$MANIFEST" \
    --output "$RECOVERY/native_readout.json" \
    --device cuda \
    --roles val test
  python -B scripts/validate_radar_fields_native_readout.py \
    --config "$CONFIG" \
    --output "$RECOVERY/native_readout.json"
  python -B - "$RECOVERY/recovery_summary.json" "$READOUT_ONLY_CHECKPOINT" "$SOURCE_RESOURCE_STATUS_JSON" <<'PY'
import json
import sys
from pathlib import Path

output, checkpoint, source_resource_status_json = sys.argv[1:]
source_resource_status = json.loads(source_resource_status_json)
record = {
    "kind": "radar_fields_completed_checkpoint_readout_recovery_v1",
    "checkpoint": str(Path(checkpoint).resolve()),
    "source_training_resource": source_resource_status,
    "training_rerun": False,
}
with open(output, "x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
  echo "RF readout-only recovery: $RECOVERY"
  echo 'RF_B7873200_READOUT_ONLY_V1_PASS'
  exit 0
fi

ATTEMPT="$CAMPAIGN/attempt-$SLURM_JOB_ID"
# Every attempt gets a fresh output directory; an optional resume source is
# read-only input and is never overwritten.
mkdir -- "$ATTEMPT"

if [[ "${#RESUME_ARGS[@]}" -eq 2 ]]; then
  RESUME_SOURCE="${RESUME_ARGS[1]}"
  RESUME_SOURCE_ATTEMPT="$(dirname "$(dirname "$RESUME_SOURCE")")"
  RESUME_SOURCE_RESOURCE="$RESUME_SOURCE_ATTEMPT/training_resource.json"
  RESUME_RESOURCE_STATUS_JSON=$(python -B scripts/inspect_radar_fields_training_resource.py \
    --path "$RESUME_SOURCE_RESOURCE")
  python -B - "$ATTEMPT/resume_summary.json" "$RESUME_SOURCE" "$RESUME_RESOURCE_STATUS_JSON" <<'PY'
import json
import sys
from pathlib import Path

output, checkpoint, resource_status_json = sys.argv[1:]
record = {
    "kind": "radar_fields_resume_source_assessment_v1",
    "source_checkpoint": str(Path(checkpoint).resolve()),
    "source_training_resource": json.loads(resource_status_json),
    "current_attempt_acceptance_scope": "current_attempt_only; prior acceptance remains separately evidenced",
}
with open(output, "x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
fi

set +e
srun --kill-on-bad-exit=1 --ntasks=1 --nodes=1 --cpu-bind=none \
  python -B - "$CONFIG" "$ATTEMPT" "$NPZ" "$MANIFEST" "${RESUME_ARGS[@]}" <<'PY'
import json
import runpy
import resource
import sys
from pathlib import Path

import torch

config_path, attempt, npz_path, manifest_path, *resume = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    config = json.load(handle)
training = config["training"]
if not torch.cuda.is_available():
    raise RuntimeError("RF production training requires the allocated CUDA device")
torch.cuda.reset_peak_memory_stats()

argv = [
    "train_radar_fields.py",
    "--npz-path", npz_path,
    "--sealed-protocol",
    "--sealed-split-manifest", manifest_path,
    "--checkpoint-root", attempt,
    "--checkpoint-name", training["checkpoint_name"],
]
for key in (
    "seed", "num_train", "num_val", "num_test", "extent", "granularity",
    "steps", "view_batch", "train_pairs", "val_pairs", "eval_every",
    "checkpoint_every", "eval_max_views", "stats_max_views", "lr",
    "weight_fft", "weight_occ", "weight_bimodal", "occupancy_threshold",
    "dynamic_range_db", "range_margin", "range_law", "intensity_offset",
    "intensity_scaler", "hidden_dim", "feature_dim", "sh_degree",
    "sigmoid_tightness", "hash_levels", "hash_features", "hash_base_resolution",
    "hash_final_resolution", "hash_log2_size", "query_chunk", "pair_chunk",
    "device",
):
    value = training[key]
    argv.extend([f"--{key.replace('_', '-')}", str(value)])
if training["val_from_tail"]:
    argv.append("--val-from-tail")
if training["no_batch_norm"]:
    argv.append("--no-batch-norm")
if resume:
    if resume != ["--resume", resume[-1]]:
        raise RuntimeError(f"unexpected resume arguments: {resume!r}")
    argv.extend(resume)
sys.argv = argv
runpy.run_path("train_radar_fields.py", run_name="__main__")

torch.cuda.synchronize()
gpu_peak = int(torch.cuda.max_memory_allocated())
gpu_total = int(torch.cuda.get_device_properties(0).total_memory)
host_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
host_budget = int(float(config["resources"]["memory_gb"]) * 1024**3)
resource_report = {
    "scope": "training Python process only",
    "host_peak_rss_bytes": host_peak,
    "host_budget_bytes": host_budget,
    "gpu_peak_allocated_bytes": gpu_peak,
    "gpu_total_bytes": gpu_total,
    "headroom_fraction": 0.8,
    "host_headroom_pass": host_peak < 0.8 * host_budget,
    "gpu_headroom_pass": gpu_peak < 0.8 * gpu_total,
    "headroom_gate_pass": host_peak < 0.8 * host_budget and gpu_peak < 0.8 * gpu_total,
}
attempt_path = Path(attempt)
with (attempt_path / "training_resource.json").open("x", encoding="utf-8") as handle:
    json.dump(resource_report, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
if not resource_report["headroom_gate_pass"]:
    raise RuntimeError("RF production training exceeded the 80 percent resource headroom gate")

checkpoint_dir = attempt_path / training["checkpoint_name"]
final = checkpoint_dir / "checkpoint_final.pth.tar"
latest = checkpoint_dir / "checkpoint_latest.pth.tar"
steps = int(training["steps"])
if final.is_file():
    header = torch.load(final, map_location="cpu", weights_only=True)
    if int(header.get("step", -1)) != steps or header.get("scene_repr") != "radar_fields":
        raise RuntimeError("RF final checkpoint header is not the requested completed recipe")
    status = "completed"
    checkpoint_path = str(final)
elif latest.is_file():
    header = torch.load(latest, map_location="cpu", weights_only=True)
    step = int(header.get("step", -1))
    if not 0 < step < steps or header.get("scene_repr") != "radar_fields":
        raise RuntimeError("RF latest checkpoint is not a usable clean-stop checkpoint")
    status = "clean_stop"
    checkpoint_path = str(latest)
else:
    status = "no_checkpoint"
    checkpoint_path = None
with (attempt_path / "training_status.json").open("x", encoding="utf-8") as handle:
    json.dump({"status": status, "checkpoint": checkpoint_path}, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(f"RF training status: {status}", flush=True)
PY
TRAIN_EXIT=$?
set -e

if [[ "$TRAIN_EXIT" -ne 0 ]]; then
  echo "ERROR: RF trainer failed with exit $TRAIN_EXIT; preserving attempt $ATTEMPT" >&2
  exit "$TRAIN_EXIT"
fi

CHECKPOINT="$ATTEMPT/$CHECKPOINT_NAME/checkpoint_final.pth.tar"
LATEST_CHECKPOINT="$ATTEMPT/$CHECKPOINT_NAME/checkpoint_latest.pth.tar"
[[ -f "$ATTEMPT/training_resource.json" && -f "$ATTEMPT/training_status.json" ]] || {
  echo 'ERROR: RF trainer returned without its required status/resource evidence' >&2
  exit 95
}
if [[ "${#RESUME_ARGS[@]}" -eq 2 ]]; then
  [[ -f "$ATTEMPT/resume_summary.json" ]] || {
    echo 'ERROR: resumed RF attempt lacks its prior-evidence assessment' >&2
    exit 95
  }
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  if [[ -f "$LATEST_CHECKPOINT" ]]; then
    echo 'RF trainer stopped cleanly with checkpoint_latest; no production marker was emitted' >&2
    echo 'Use a reviewed --resume CHECKPOINT or wait for manager policy; this launcher does not auto-continue.' >&2
    exit 143
  fi
  echo 'ERROR: final RF checkpoint was not published and no usable clean-stop checkpoint exists' >&2
  exit 95
fi

srun --kill-on-bad-exit=1 --ntasks=1 --nodes=1 --cpu-bind=none \
  python -B scripts/readout_radar_fields_b7873200_native.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --npz-path "$NPZ" \
  --sealed-split-manifest "$MANIFEST" \
  --output "$ATTEMPT/native_readout.json" \
  --device cuda \
  --roles val test

python -B scripts/validate_radar_fields_native_readout.py \
  --config "$CONFIG" \
  --output "$ATTEMPT/native_readout.json"

echo "RF production attempt: $ATTEMPT"
echo 'RF_B7873200_PRODUCTION_V1_PASS'
