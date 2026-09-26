#!/bin/bash
# SONAR SLOT 7 OF 11: CGLS least-squares refinement curve on D3's 288 Bunny 5 kHz pings, K=12.
# THE FIT. Single L40S GPU. Spends slot 7; slots 8-10 are the production releases, 11 is last.
#
# APPROVED by Fable (rift-3b) under Daqian's standing delegation, in the form on file:
# slot 7 of 11, spec tmp/fable_sonar_slot7_ls_refinement_spec_20260916.md, K=12 via Coord-Opus,
# resting on dry run 13261908 (COMPLETED 0:0, 26:42, two invocations against fresh state) with
# 13260039 as the earlier behavioural run reproduced to the digit. Daqian's own words on the
# slot, relayed in that approval: "Run now". Submitted on that approval plus Coord-Opus's
# technical release. Paid-Inferno does not need a separate ask here: GPU fits WITHIN the sonar
# slots are the one carved-out exception where Fable approves rather than Daqian.
#
# WALL CAP IS DELIBERATELY DOUBLE-LAYERED AND THE TWO NUMBERS MUST NOT BE EQUALISED.
#   Slurm  --time=04:15:00
#   script  WALL_CAP_SECONDS_DEFAULT = 3h30m  (line 215; NOT passed, so the default applies)
# The 45-minute gap is the point: the script cuts itself at 3h30 and still has time to write
# its final report and state. Raising the script cap to the Slurm limit would turn a clean
# self-cut into a TIMEOUT kill with no report -- which is exactly how 13258713 was lost.
#
# FRESH RUN, DECLARED. Intent is a flag, never inferred from disk. The wall-cap contingency is
# a SEPARATE later submission that would declare FRESH_RUN=0 and pass --resume; that is why
# this guard cannot block the designed resume the way an inferred output-dir-absent check would
# (Coord-Opus caught exactly that failure in an earlier draft of mine).
#
# Flag names verified against the script's own parser at lines 953-977, not against anyone's
# description of it. Script identity confirmed on PACE: 61,859 bytes, 2026-09-16 14:02:04.
#
# Staging only; account, QoS and slurm log path are supplied at sbatch time.
# Remote destination: /storage/project/r-jromberg3-0/dbao31/RIFT/slurm/run_sas_cgls288_ungated_slot7_20260916.sbatch

#SBATCH --job-name=mgr_sas_cgls288_ungated_slot7_20260916
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --tmp=12G
#SBATCH --time=04:15:00
#SBATCH --no-requeue
#SBATCH --export=NONE

set -euo pipefail
umask 077

SRC=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_sources/airsas_armadillo5k_fullgrid150x150x120_smoke_v1_20260914
PY=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/RIFT/bin/python
D3=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_forward_control_d3b_20260915
CACHE=/storage/project/r-jromberg3-0/dbao31/RIFT/datasets/sh_sas_reed/cache/bunny_5khz_trainonly_120rings_v1_20260913
OUT=/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/airsas_bunny5k_cgls288_ungated_v1_20260916

FRESH_RUN=1

: "${SLURM_JOB_ID:?must run in a Slurm allocation}"

if [[ "$FRESH_RUN" -eq 1 && -e "$OUT" ]]; then
  echo "ABORT: fresh run declared but $OUT already exists. Not overwriting; nothing is deleted here." >&2
  exit 3
fi

cd "$SRC"
export PYTHONPATH="$SRC"
export PYTHONUNBUFFERED=1
# The script sets its own thread count via _configure_thread_count(); these match it.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "=== SLOT 7: CGLS 288-ping ungated refinement curve, K=12, device cuda ==="
echo "    slurm limit 04:15:00 against the script's own 3h30m cap (default, not passed)"
nvidia-smi || true
echo ""

exec "$PY" -B -u "$SRC/scripts/diagnose_sas_cgls.py" \
  --cache "$CACHE" \
  --d3-summary-json "$D3/forward_control_summary.json" \
  --d3-traces-npz "$D3/forward_control_traces.npz" \
  --output "$OUT" \
  --device cuda \
  --k 12
