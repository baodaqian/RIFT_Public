#!/usr/bin/env python3
"""Run one GeRaF v2 child under GNU time while forwarding a Slurm warning.

This is deliberately an additive v2 launcher helper.  It is the direct Slurm
step task, so a normal ``#SBATCH --signal=TERM@...`` warning reaches this
process even when Slurm does not signal descendants.  The helper keeps GNU
``time`` alive for its RSS record, resets the Python child's TERM disposition
with GNU ``env``, and forwards a received TERM to the isolated child process
group.  The child then owns the established GeRaF clean-stop protocol.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import NoReturn, Sequence


_READY_ENV = "GERAF_TIMED_WRAPPER_READY_FILE"
_READY_MARKER = "GERAF_TIMED_WRAPPER_CHILD_READY"


def _parse_args(argv: Sequence[str] | None = None) -> tuple[Path, Path | None, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-log", required=True, type=Path)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    command = list(parsed.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a child command is required after --")
    return parsed.time_log, parsed.ready_file, command


def _terminal(message: str, status: int = 96) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(status)


def main(argv: Sequence[str] | None = None) -> int:
    time_log, ready_file, command = _parse_args(argv)
    if signal.getsignal(signal.SIGTERM) != signal.SIG_DFL:
        _terminal("timed wrapper did not start with the default SIGTERM disposition")
    if time_log.exists():
        _terminal(f"refusing to overwrite an existing GNU time record: {time_log}", 94)
    if not time_log.parent.is_dir():
        _terminal(f"missing existing time-log directory: {time_log.parent}", 94)
    if ready_file is not None:
        if ready_file.exists():
            _terminal(f"refusing to overwrite an existing ready marker: {ready_file}", 94)
        if not ready_file.parent.is_dir():
            _terminal(f"missing existing ready-marker directory: {ready_file.parent}", 94)

    print("GERAF_TIMED_WRAPPER_DIRECT_TERM_DEFAULT", flush=True)
    child: subprocess.Popen[str] | None = None
    pending_term = False
    forward_failed = False

    def request_term(_signum: int, _frame: object) -> None:
        nonlocal pending_term
        pending_term = True

    signal.signal(signal.SIGTERM, request_term)
    child_environment = os.environ.copy()
    child_environment["GERAF_TIMED_WRAPPER_PID"] = str(os.getpid())
    child_environment["GERAF_TIMED_WRAPPER_PROTOCOL"] = "v1"
    if ready_file is not None:
        child_environment[_READY_ENV] = str(ready_file)
    child_argv = [
        "/bin/bash",
        "-c",
        'trap "" TERM; exec "$@"',
        "geraf-timed-child",
        "/usr/bin/time",
        "-v",
        "-o",
        str(time_log),
        "/usr/bin/env",
        "--default-signal=TERM",
        "--",
        *command,
    ]
    try:
        child = subprocess.Popen(
            child_argv,
            env=child_environment,
            start_new_session=True,
        )
    except OSError as exc:
        _terminal(f"could not start timed child: {exc}", 95)

    print(f"GERAF_TIMED_WRAPPER_CHILD_START child_pgid={child.pid}", flush=True)

    def abort_before_ready(message: str) -> NoReturn:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        while True:
            try:
                child.wait()
                break
            except InterruptedError:
                continue
        _terminal(message)

    def child_is_ready() -> bool:
        if ready_file is None:
            return True
        try:
            marker = ready_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            return False
        expected = _READY_MARKER + "\n"
        if marker == expected:
            return True
        if expected.startswith(marker):
            return False
        _terminal(f"timed child wrote an invalid ready marker: {ready_file}", 95)

    ready_observed = ready_file is None
    term_forwarded = False
    while True:
        if not ready_observed:
            ready_observed = child_is_ready()
            if ready_observed:
                print("GERAF_TIMED_WRAPPER_CHILD_READY", flush=True)
        if pending_term and not term_forwarded:
            # A probe (or a just-ready driver) may close its marker immediately
            # before signalling this wrapper, after the poll above.  Re-read
            # before classifying the warning as a before-ready terminal stop.
            if not ready_observed:
                ready_observed = child_is_ready()
                if ready_observed:
                    print("GERAF_TIMED_WRAPPER_CHILD_READY", flush=True)
            if not ready_observed:
                abort_before_ready(
                    "timed wrapper received TERM before its Python child published signal readiness"
                )
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                forward_failed = True
                term_forwarded = True
                print(
                    "ERROR: timed wrapper received TERM after its child process group disappeared",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                term_forwarded = True
                print(
                    f"GERAF_TIMED_WRAPPER_FORWARD_TERM child_pgid={child.pid}",
                    flush=True,
                )
        try:
            child_status = child.wait(timeout=0.1)
            break
        except (InterruptedError, subprocess.TimeoutExpired):
            continue

    if forward_failed:
        _terminal("timed wrapper could not forward its TERM to the child process group")
    if not ready_observed:
        ready_observed = child_is_ready()
        if ready_observed:
            print("GERAF_TIMED_WRAPPER_CHILD_READY", flush=True)
    if not ready_observed:
        abort_before_ready("timed child exited before publishing signal readiness")
    if child_status < 0:
        return 128 + (-child_status)
    return child_status


if __name__ == "__main__":
    raise SystemExit(main())
