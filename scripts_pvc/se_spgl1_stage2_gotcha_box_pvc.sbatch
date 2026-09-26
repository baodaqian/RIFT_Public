#!/bin/bash
# SE SPGL1 Stage 2 for GOTCHA Camry in the RIFT tuning campaign's data setup, on one PVC card
# (scripts_pvc/se_spgl1_stage1.py stage2 --gotcha). Resumes the unchanged paper-v1 workflow from the
# Stage-1 checkpoint that `assemble` wrote (scripts_pvc/se_spgl1_stage1_gotcha_box_cpu.sbatch). The source
# flags below MUST match that sbatch byte for byte (the acquisition and recipe identity are re-checked),
# and the code runs from $SE_ROOT/source. Resuming is re-running this identical script.
# Exit codes: 0 complete, 2 incomplete (status.json) or partition refusal (rerun on another node),
# 75 interrupted -> resume; 143 + STAGE2_RESULT=error is also an interrupt.
# Usage: export SE_ROOT=... [SE_SPLIT_STRIDE=4 SE_HELDOUT_FRACTION=0.5 SE_SHARDS=...]
#        sbatch --export=ALL --output $SE_ROOT/logs/%x-%j.log THIS
#SBATCH --job-name=se-camry-box-stage2
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023,ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@300
set -eo pipefail
: "${SE_ROOT:?set SE_ROOT}"
cd "${SE_SOURCE:-$SE_ROOT/source}"
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=xpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
STRIDE=${SE_SPLIT_STRIDE:-4}
HELDOUT=${SE_HELDOUT_FRACTION:-0.5}
SHARDS=${SE_SHARDS:-/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_filtered_shards_20260923/camry_box_v2_provisional_carphase_v2/keep6/New_Transfer/shards}
OUT=$SE_ROOT/split_stride${STRIDE}_heldout${HELDOUT}/sugavanam_ertin_spgl1
RUNTIME=$SE_ROOT/runtime/stage2_stride${STRIDE}
export MPLCONFIGDIR="$RUNTIME/mpl" XDG_CACHE_HOME="$RUNTIME/cache" TMPDIR="$RUNTIME/tmp"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TMPDIR"
date; echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME output=$OUT shards=$SHARDS stride=$STRIDE held-out=$HELDOUT"
grep -q '"initialization_std": 0.05' protocols/se_g40_readout48.json || { echo "SE config lost initialization_std 0.05"; exit 3; }
# The run's own recipe (written by assemble) must carry std 0.05 too; print both for an independent check.
echo "SE_CONFIG=$PWD/protocols/se_g40_readout48.json $(tr -d '\n ' < protocols/se_g40_readout48.json)"
python -c "import json,sys; r=json.load(open(sys.argv[1])); r=r.get('recipe', r); print('RUN_RECIPE', json.dumps({k: r.get(k) for k in ('initialization_std', 'granularity', 'export_grid', 'stage2_steps', 'stage1_solver')})); sys.exit(0 if r.get('initialization_std') == 0.05 else 3)" "$OUT/recipe.json" || { echo "run recipe initialization_std is not 0.05"; exit 3; }
set +e
python -u scripts_pvc/se_spgl1_stage1.py stage2 --output "$OUT" \
  --gotcha --dataset-root /scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined \
  --shard-root "$SHARDS" --region camry_box_v2 --region-config rift_pvc/regions/camry_box_v2.json \
  --polarizations hh --passes 1 2 3 4 5 6 7 8 --num-tx 1 --num-rx 1 --pulses-per-sector 0 --frequency-stride 2 \
  --unit-split-stride "$STRIDE" --unit-split-heldout-pass 4 --unit-split-heldout-fraction "$HELDOUT" \
  --method sugavanam_ertin --config protocols/se_g40_readout48.json --output-root "$SE_ROOT/frontend_unused" &
PID=$!
trap 'echo "forwarding TERM to $PID"; kill -TERM "$PID" 2>/dev/null' TERM
wait "$PID"; STATUS=$?
if [ "$STATUS" -gt 128 ]; then wait "$PID"; STATUS=$?; fi
set -e
echo "STAGE2_EXIT=$STATUS"
case $STATUS in
  0)  echo "STAGE2_RESULT=complete" ;;
  75) echo "STAGE2_RESULT=interrupted_resumable (re-run this identical script)" ;;
  2)  echo "STAGE2_RESULT=incomplete_or_partition_refusal (see the log and $OUT/status.json)" ;;
  *)  echo "STAGE2_RESULT=error" ;;
esac
grep -c 'Aten Op fallback' "$SE_ROOT/logs/"*"$SLURM_JOB_ID"*.log 2>/dev/null | sed 's/^/ATEN_FALLBACK_LINES=/' || true
[ -f "$OUT/status.json" ] && echo "STATUS_JSON $(python -c "import json,sys; s=json.load(open(sys.argv[1])); print(json.dumps({k: s.get(k) for k in ('phase', 'sdf_step')}))" "$OUT/status.json")"
[ -f "$OUT/surface.npz" ] && echo "SURFACE $OUT/surface.npz"
date; exit $STATUS
