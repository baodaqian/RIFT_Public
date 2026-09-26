#!/bin/bash
#SBATCH --job-name=rift_sas_sigprobe
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --signal=B:TERM@90
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/logs/rift_sas_sigprobe_%j.out

set -euo pipefail

out="/storage/project/r-jromberg3-0/dbao31/RIFT/logs/rift_sas_sigprobe_${SLURM_JOB_ID}"
mkdir -p "$out"

stop_requested=0
train_pid=""
forward_training_signal() {
  stop_requested=1
  if [[ -n "$train_pid" ]] && kill -0 "$train_pid" 2>/dev/null; then
    echo "Forwarding batch signal to training PID $train_pid"
    kill -TERM "$train_pid"
  fi
}
trap forward_training_signal TERM INT

set +e
python3 -u - "$out" <<'PY' &
import json
import pathlib
import signal
import sys
import time

root = pathlib.Path(sys.argv[1])
stop = False

def handler(signum, _frame):
    global stop
    stop = True
    print(f"Received signal {signum}; checkpointing", flush=True)

signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)
while not stop:
    time.sleep(0.05)
time.sleep(0.2)
(root / "checkpoint_latest.ok").write_text("atomic checkpoint complete\n", encoding="utf-8")
(root / "status.json").write_text(json.dumps({"done": False, "reason": "signal"}), encoding="utf-8")
PY
train_pid=$!

while true; do
  wait "$train_pid"
  train_rc=$?
  if ! kill -0 "$train_pid" 2>/dev/null; then
    break
  fi
done
set -e
trap - TERM INT

if (( stop_requested )); then
  echo "Training exited after forwarded signal with code $train_rc; skipping geometry evaluation"
  exit "$train_rc"
fi
if (( train_rc != 0 )); then
  exit "$train_rc"
fi
touch "$out/geometry_ran"
