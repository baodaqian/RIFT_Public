#!/usr/bin/env python3
"""Verify the GeRaF v2 timed srun wrapper survives its warning signal.

The launcher deliberately makes GNU ``time`` ignore ``SIGTERM`` while GNU
``env --default-signal=TERM`` restores the default disposition before Python
starts.  This probe verifies that starting disposition, then sends TERM to the
direct timed wrapper: the wrapper must keep its timing child alive long enough
to write an RSS record while this child handles the forwarded TERM and exits
normally.  It contains no Torch, radar data, or training work.
"""

from __future__ import annotations

import os
from pathlib import Path
import signal
import time


received = False
_TIMED_WRAPPER_READY_ENV = "GERAF_TIMED_WRAPPER_READY_FILE"
_TIMED_WRAPPER_READY_MARKER = "GERAF_TIMED_WRAPPER_CHILD_READY"


def _on_term(_signum: int, _frame: object) -> None:
    global received
    received = True


def _publish_ready() -> None:
    raw_path = os.environ.get(_TIMED_WRAPPER_READY_ENV)
    if raw_path is None:
        raise RuntimeError("signal-path probe requires a timed-wrapper ready-marker path")
    ready_path = Path(raw_path)
    if not ready_path.parent.is_dir():
        raise RuntimeError("signal-path probe ready-marker parent is missing")
    try:
        with ready_path.open("x", encoding="utf-8") as handle:
            handle.write(_TIMED_WRAPPER_READY_MARKER + "\n")
    except FileExistsError as exc:
        raise RuntimeError("signal-path probe ready marker already exists") from exc


def main() -> None:
    wrapper_pid_text = os.environ.get("GERAF_TIMED_WRAPPER_PID")
    if wrapper_pid_text is None or not wrapper_pid_text.isdecimal():
        raise RuntimeError("signal-path probe requires a numeric timed-wrapper PID")
    wrapper_pid = int(wrapper_pid_text)
    if wrapper_pid <= 1 or wrapper_pid == os.getpid():
        raise RuntimeError("signal-path probe received an invalid timed-wrapper PID")
    if signal.getsignal(signal.SIGTERM) != signal.SIG_DFL:
        raise RuntimeError(
            "signal-path probe did not start with the default SIGTERM disposition"
        )
    signal.signal(signal.SIGTERM, _on_term)
    _publish_ready()
    # Signal the direct Slurm task.  Its wrapper must forward TERM to the
    # isolated group containing GNU time (which ignores it) and this Python
    # child (which has just installed its clean handler).
    os.kill(wrapper_pid, signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while not received and time.monotonic() < deadline:
        time.sleep(0.01)
    if not received:
        raise RuntimeError("signal-path probe child did not receive wrapper-forwarded SIGTERM")
    print("GERAF_B78716_SMALLFIT_SIGNAL_PATH_PASS", flush=True)


if __name__ == "__main__":
    main()
