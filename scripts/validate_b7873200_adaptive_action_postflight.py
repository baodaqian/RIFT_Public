#!/usr/bin/env python3
"""Pure-Python branch checks for the B787 action-gate postflight adapter.

This deliberately does not import PyTorch or open an acquisition archive.  It
only verifies that the release adapter distinguishes a valid measurement from
an over-budget one and fails closed when a CUDA/process measurement is absent,
zero, or malformed.
"""
from __future__ import annotations

from typing import Any

from postflight_b7873200_adaptive_action_gate import STATUS, build_postflight


def _report(**changes: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "engineering_status": STATUS,
        "pass": True,
        "process_max_rss_kib": 10 * 1024 * 1024,
        "peak_torch_allocated_bytes": 1024 * 1024 * 1024,
        "peak_torch_reserved_bytes": 2 * 1024 * 1024 * 1024,
        "initial_metrics": {"loss": 1.0},
        "final_metrics": {"loss": 0.5},
        "scene_parameter_change_l2": 1.0,
        "gain_parameter_change_l2": 1.0,
    }
    report.update(changes)
    return report


def _postflight(
    report: dict[str, Any],
    *,
    process_rss_kib: Any = 10 * 1024 * 1024,
    host_limit_gib: Any = 32,
    gpu_total_mib: Any = 24 * 1024,
) -> dict[str, Any]:
    return build_postflight(
        report,
        time_process_rss_kib=process_rss_kib,
        host_limit_gib=host_limit_gib,
        gpu_total_mib=gpu_total_mib,
        whole_job_rss_raw=None,
    )


def _expect_invalid(label: str, report: dict[str, Any], **kwargs: Any) -> None:
    try:
        _postflight(report, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"{label}: invalid measurement produced release evidence")


def main() -> int:
    positive = _postflight(_report())
    assert positive["action_gate_pass"] is True
    assert positive["memory_acceptance"] is True
    assert positive["overall_release_pass"] is True

    over_budget = _postflight(_report(), process_rss_kib=30 * 1024 * 1024)
    assert over_budget["action_gate_pass"] is True
    assert over_budget["memory_acceptance"] is False
    assert over_budget["overall_release_pass"] is False

    _expect_invalid("zero /usr/bin/time RSS", _report(), process_rss_kib=0)
    _expect_invalid("noninteger /usr/bin/time RSS", _report(), process_rss_kib=1.25)
    _expect_invalid("zero report RSS", _report(process_max_rss_kib=0))
    _expect_invalid("missing report RSS", _report(process_max_rss_kib=None))
    _expect_invalid("zero allocated CUDA bytes", _report(peak_torch_allocated_bytes=0))
    _expect_invalid("missing allocated CUDA bytes", _report(peak_torch_allocated_bytes=None))
    _expect_invalid("zero reserved CUDA bytes", _report(peak_torch_reserved_bytes=0))
    _expect_invalid("missing reserved CUDA bytes", _report(peak_torch_reserved_bytes=None))
    _expect_invalid("noninteger allocator bytes", _report(peak_torch_allocated_bytes=1.25))
    _expect_invalid(
        "reserved CUDA bytes below allocated bytes",
        _report(peak_torch_allocated_bytes=2, peak_torch_reserved_bytes=1),
    )
    _expect_invalid("zero host envelope", _report(), host_limit_gib=0)
    _expect_invalid("zero GPU envelope", _report(), gpu_total_mib=0)
    print("RIFT_B7873200_ACTION_GATE_POSTFLIGHT_BRANCHES_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
