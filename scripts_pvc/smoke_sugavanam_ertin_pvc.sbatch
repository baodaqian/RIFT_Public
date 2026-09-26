#!/bin/bash
# Package C bounded real-data smoke: Sugavanam--Ertin paper-v1 on one PVC card.
#
#   sbatch scripts_pvc/smoke_sugavanam_ertin_pvc.sbatch
#
# What it does, all through the _pvc entry points and the *unchanged* production
# recipe (protocols/se_g40_readout48.json, initialization_std 0.05):
#   0. accelerator identity + Stage-1 operator timing probe (ms per view)
#   1. gate-integrity reference: the literal paper std=1.0 must still be refused
#   2. fresh production run, bounded by wall clock; SIGTERM makes the trainer
#      take its cooperative clean-interruption path and checkpoint
#   3. resume from that checkpoint and continue
#   4. report the Stage-1 loss curve, per-step wall time and any XPU fallback
#
# Boundedness comes from the wall clock, never from altering the recipe.
#SBATCH --job-name=pvcC-se-smoke
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageC/logs/%x-%j.log
set -eo pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 WANDB_MODE=offline PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"

DATASET_ROOT=/scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645/inputs/RIFT_dataset
BASE=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageC
OUT=$BASE/smoke_${SLURM_JOB_ID}
PROTOCOL=protocols/se_g40_readout48.json
NPZ=$DATASET_ROOT/objects/b787.npz
MANIFEST=$OUT/train2400/1t1r_2010b2bbe725/b787/role_manifest.json
RUN_ROOT=$OUT/train2400/1t1r_2010b2bbe725/b787/sugavanam_ertin
FRESH_SECONDS=${FRESH_SECONDS:-900}
RESUME_SECONDS=${RESUME_SECONDS:-600}

echo "host=$(hostname) job=${SLURM_JOB_ID} date=$(date -Is)"
mkdir -p "$OUT"

echo; echo "=============== 0a. accelerator ==============="
python -c "from rift_pvc import accelerator as a; import json; print(json.dumps(a.describe(), indent=2))"

echo; echo "=============== 0b. materialize the production role manifest ==============="
# --check-initialization writes the derived manifest and runs the bounded CPU
# probe; it starts no fitting. This is the production frontend, PVC entry points.
python train_rift_dataset_pvc.py --object b787 --method sugavanam_ertin \
    --num-train 2400 --num-tx 1 --num-rx 1 \
    --dataset-root "$DATASET_ROOT" --output-root "$OUT" \
    --se-recipe paper-v1 --se-config "$PROTOCOL" --check-initialization \
    > "$OUT/frontend_check_initialization.json" 2> "$OUT/frontend_check_initialization.err"
echo "frontend --check-initialization exit=$?"
python scripts_pvc/se_pvc_smoke_report.py --gate "$OUT/frontend_check_initialization.json" \
    --label "production gate (std=0.05)"
ls -la "$(dirname "$MANIFEST")" || true

echo; echo "=============== 0c. Stage-1 operator timing probe ==============="
python scripts_pvc/se_pvc_timing_probe.py --npz-path "$NPZ" \
    --parent-role-manifest "$MANIFEST" --config "$PROTOCOL" --views 3 \
    2>&1 | tee "$OUT/timing_probe.json"
echo "timing probe exit=${PIPESTATUS[0]}"

echo; echo "=============== 1. gate integrity: literal paper std=1.0 must be refused ==============="
GATE_EXIT=0
python train_sugavanam_ertin_pvc.py --recipe paper-v1 \
    --npz-path "$NPZ" --parent-role-manifest "$MANIFEST" \
    --config protocols/se_paper_std1.json --check-initialization \
    --checkpoint-root "$OUT/gate_reference" > "$OUT/gate_std1.json" 2>&1 || GATE_EXIT=$?
if [[ $GATE_EXIT != 2 ]]; then
    echo "FAIL: std=1.0 probe returned $GATE_EXIT; expected 2" >&2
    exit 1
fi
python scripts_pvc/se_pvc_smoke_report.py --gate "$OUT/gate_std1.json" \
    --label "gate-integrity reference (std=1.0)" --expect-degenerate

echo; echo "=============== 2. fresh production run, ${FRESH_SECONDS}s then SIGTERM ==============="
date -Is
FRESH_START=$SECONDS
set +e
timeout --preserve-status --signal=TERM --kill-after=120 "$FRESH_SECONDS" \
    python train_sugavanam_ertin_pvc.py --recipe paper-v1 \
        --npz-path "$NPZ" --parent-role-manifest "$MANIFEST" \
        --config "$PROTOCOL" --checkpoint-root "$RUN_ROOT" \
    2>&1 | tee "$OUT/run_fresh.log"
FRESH_CODES=("${PIPESTATUS[@]}")
set -e
FRESH_ELAPSED=$(( SECONDS - FRESH_START ))
if [[ ${FRESH_CODES[0]} != 75 || ${FRESH_CODES[1]} != 0 ]]; then
    echo "FAIL: fresh run must cleanly interrupt (75), with a written log; got ${FRESH_CODES[*]}" >&2
    exit 1
fi
date -Is
python scripts_pvc/se_pvc_smoke_report.py --run-root "$RUN_ROOT" \
    --logs "$OUT/run_fresh.log" --elapsed-seconds "$FRESH_ELAPSED" | tee "$OUT/fresh_report.json"

echo; echo "=============== 3. resume from checkpoint_latest.pt, ${RESUME_SECONDS}s ==============="
RESUME_START=$SECONDS
set +e
  timeout --preserve-status --signal=TERM --kill-after=120 "$RESUME_SECONDS" \
      python train_sugavanam_ertin_pvc.py --recipe paper-v1 \
          --npz-path "$NPZ" --parent-role-manifest "$MANIFEST" \
          --config "$PROTOCOL" --checkpoint-root "$RUN_ROOT" \
          --resume "$RUN_ROOT/checkpoint_latest.pt" \
      2>&1 | tee "$OUT/run_resume.log"
RESUME_CODES=("${PIPESTATUS[@]}")
set -e
RESUME_ELAPSED=$(( SECONDS - RESUME_START ))
if [[ ( ${RESUME_CODES[0]} != 75 && ${RESUME_CODES[0]} != 0 ) || ${RESUME_CODES[1]} != 0 ]]; then
    echo "FAIL: resume must cleanly interrupt (75) or complete (0); got ${RESUME_CODES[*]}" >&2
    exit 1
fi
date -Is

echo; echo "=============== 4. report ==============="
python scripts_pvc/se_pvc_smoke_report.py --run-root "$RUN_ROOT" \
    --elapsed-seconds $(( FRESH_ELAPSED + RESUME_ELAPSED )) \
    --previous-report "$OUT/fresh_report.json" \
    --logs "$OUT/run_fresh.log" "$OUT/run_resume.log" | tee "$OUT/smoke_report.json"

echo; echo "=============== fallback scan ==============="
SCAN_EXIT=0
grep -in "fallback from XPU to CPU" "$OUT"/*.log "$OUT"/*.json "$OUT"/*.err || SCAN_EXIT=$?
if [[ $SCAN_EXIT != 1 ]]; then
    echo "FAIL: operator fallback found or log scan failed" >&2
    exit 1
fi
echo "OK: no XPU->CPU operator fallback in any log"
echo "done $(date -Is); outputs under $OUT"
