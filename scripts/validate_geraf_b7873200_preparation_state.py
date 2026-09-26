#!/usr/bin/env python3
"""Torch-free behavioral checks for the GeRaF preparation state contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_STATE_SPEC = importlib.util.spec_from_file_location(
    "geraf_b7873200_preparation_state_under_test",
    ROOT / "rift" / "geraf_b7873200_preparation_state.py",
)
if _STATE_SPEC is None or _STATE_SPEC.loader is None:
    raise RuntimeError("cannot load the Torch-free GeRaF preparation state contract")
_STATE = importlib.util.module_from_spec(_STATE_SPEC)
_STATE_SPEC.loader.exec_module(_STATE)
PREPARATION_STATE_SCHEMA = _STATE.PREPARATION_STATE_SCHEMA
build_preparation_state = _STATE.build_preparation_state
complete_preparation_phase = _STATE.complete_preparation_phase
validate_preparation_state = _STATE.validate_preparation_state


class Gates:
    def __init__(self) -> None:
        self.count = 0

    def check(self, condition: bool, detail: str) -> None:
        if not condition:
            raise AssertionError(detail)
        self.count += 1
        print(f"PASS {self.count:02d}: {detail}", flush=True)


def _reject(gates: Gates, payload: dict[str, object], detail: str) -> None:
    try:
        validate_preparation_state(payload, expected_total=4_200)
    except ValueError:
        gates.check(True, detail)
    else:
        raise AssertionError(detail)


def main() -> None:
    gates = Gates()
    events: list[str] = []

    def fake_finalizer() -> None:
        events.append("finalized")

    def stop_during_finalization() -> bool:
        events.append("stop-checked")
        return True

    status, exit_code = complete_preparation_phase(fake_finalizer, stop_during_finalization)
    gates.check(
        events == ["finalized", "stop-checked"]
        and status == "complete_clean_stop_before_fit"
        and exit_code == 143,
        "a mocked TERM observed after successful finalization becomes a clean before-fit stop rather than ordinary completion",
    )
    normal_status, normal_exit = complete_preparation_phase(lambda: None, lambda: False)
    gates.check(
        normal_status == "complete" and normal_exit is None,
        "a successful finalization with no stop remains an ordinary complete preparation",
    )
    running = build_preparation_state(
        status="running",
        completed_count=0,
        total_count=4_200,
        last_completed_view=None,
    )
    partial = build_preparation_state(
        status="partial_clean_stop",
        completed_count=1_237,
        total_count=4_200,
        last_completed_view=9_999,
        stop_signal=15,
    )
    complete_before_fit = build_preparation_state(
        status="complete_clean_stop_before_fit",
        completed_count=4_200,
        total_count=4_200,
        last_completed_view=7_001,
        stop_signal=15,
    )
    complete = build_preparation_state(
        status="complete",
        completed_count=4_200,
        total_count=4_200,
        last_completed_view=7_001,
    )
    gates.check(
        running["status"] == "running"
        and partial["completed_count"] == 1_237
        and complete_before_fit["stop_signal"] == 15
        and complete["stop_signal"] is None,
        "preparation lifecycle records running, partial-clean-stop, complete-before-fit, and complete states",
    )
    gates.check(
        partial["schema"] == PREPARATION_STATE_SCHEMA
        and partial["total_count"] == 4_200
        and partial["last_completed_view"] == 9_999,
        "clean preparation state retains the sealed total and the last durably completed target",
    )
    _reject(
        gates,
        {**partial, "completed_count": 4_200},
        "a partial-clean-stop record cannot claim every target is complete",
    )
    _reject(
        gates,
        {**complete_before_fit, "stop_signal": None},
        "a complete-before-fit record must retain the cooperative stop signal",
    )
    _reject(
        gates,
        {**complete, "total_count": 4_199},
        "a preparation record with the wrong sealed total is rejected",
    )
    _reject(
        gates,
        {**running, "status": "failed"},
        "an unknown preparation status cannot be treated as resumable",
    )
    print(f"GERAF_B7873200_PREPARATION_STATE_PASS gates={gates.count}", flush=True)


if __name__ == "__main__":
    main()
