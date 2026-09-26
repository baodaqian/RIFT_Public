"""Pure state contract for resumable GeRaF B7873200 target preparation.

This module intentionally has no NumPy, PyTorch, archive, or scheduler
dependency.  The preparer writes these records after each durable target, and
the manager-side launcher can distinguish a clean partial preparation from a
raw process failure without trusting an exit code alone.
"""

from __future__ import annotations

from typing import Callable, Mapping


PREPARATION_STATE_FILENAME = "geraf_b7873200_preparation_state.json"
PREPARATION_STATE_SCHEMA = "rift_geraf_b7873200_preparation_state_v1"
CLEAN_STOP_EXIT_CODE = 143
VALID_PREPARATION_STATUSES = frozenset(
    {
        "running",
        "partial",
        "partial_clean_stop",
        "complete_clean_stop_before_fit",
        "complete",
    }
)


def complete_preparation_phase(
    finalizer: Callable[[], object], stop_requested: Callable[[], bool]
) -> tuple[str, int | None]:
    """Run finalization, then classify a stop observed during finalization.

    The finalizer is deliberately outside any exception handler: a failed
    finalization remains a real failure.  The stop callback is evaluated only
    after the finalizer returns successfully, which covers a TERM delivered
    while the complete cache is being checked and sealed.
    """

    finalizer()
    if stop_requested():
        return "complete_clean_stop_before_fit", CLEAN_STOP_EXIT_CODE
    return "complete", None


def build_preparation_state(
    *,
    status: str,
    completed_count: int,
    total_count: int,
    last_completed_view: int | None,
    stop_signal: int | None = None,
) -> dict[str, object]:
    """Build and validate one durable preparation-phase record."""

    payload = {
        "schema": PREPARATION_STATE_SCHEMA,
        "status": status,
        "completed_count": completed_count,
        "total_count": total_count,
        "last_completed_view": last_completed_view,
        "stop_signal": stop_signal if status in {"partial_clean_stop", "complete_clean_stop_before_fit"} else None,
    }
    validate_preparation_state(payload)
    return payload


def validate_preparation_state(
    payload: Mapping[str, object], *, expected_total: int | None = None
) -> None:
    """Reject malformed or semantically impossible preparation state."""

    if payload.get("schema") != PREPARATION_STATE_SCHEMA:
        raise ValueError("GeRaF preparation state has the wrong schema")
    status = payload.get("status")
    if status not in VALID_PREPARATION_STATUSES:
        raise ValueError(f"GeRaF preparation state has an unknown status: {status!r}")
    total = payload.get("total_count")
    completed = payload.get("completed_count")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
        or isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 0
        or completed > total
    ):
        raise ValueError("GeRaF preparation state has invalid completed/total counts")
    if expected_total is not None and total != int(expected_total):
        raise ValueError("GeRaF preparation state total does not match the sealed lane")
    last_view = payload.get("last_completed_view")
    if last_view is not None and (
        isinstance(last_view, bool) or not isinstance(last_view, int) or last_view < 0
    ):
        raise ValueError("GeRaF preparation state has an invalid last completed view")
    stop_signal = payload.get("stop_signal")
    if stop_signal is not None and (
        isinstance(stop_signal, bool) or not isinstance(stop_signal, int)
    ):
        raise ValueError("GeRaF preparation state has an invalid stop signal")
    if status == "partial_clean_stop" and completed >= total:
        raise ValueError("a partial clean stop cannot report a complete target count")
    if status in {"complete_clean_stop_before_fit", "complete"} and completed != total:
        raise ValueError("a complete preparation state must report every target")
    if status in {"partial_clean_stop", "complete_clean_stop_before_fit"} and stop_signal is None:
        raise ValueError("a clean preparation stop must retain its signal number")
