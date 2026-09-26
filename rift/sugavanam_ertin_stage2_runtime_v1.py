"""Generic durable lifecycle primitives for corrected Sugavanam--Ertin Stage 2.

The historical A320 trainer contains useful geometry math but not a recoverable
artifact lifecycle.  This module intentionally supplies only lifecycle,
checkpoint, finite-JSON, and atomic-publication machinery.  A scene-specific
entrypoint provides model/data geometry while this module makes interruption
and completion semantics explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import signal
import tempfile
from typing import Any, Callable, Mapping

import numpy as np
import torch


RUNTIME_SCHEMA = "rift_sugavanam_ertin_stage2_runtime_v1"
CLEAN_INTERRUPTION_EXIT_CODE = 143
COMPLETION_FILENAMES = (
    "checkpoint_final.pth.tar",
    "surface_reconstruction.npz",
    "run_summary.json",
    "status.json",
)


class RuntimeContractError(ValueError):
    """A run, resume, or output artifact does not satisfy the lifecycle contract."""


class LifecyclePhase(str, Enum):
    INITIALIZATION = "initialization"
    TRAINING = "training"
    EXPORT_PENDING = "export_pending"
    COMPLETE = "complete"


class LifecycleState(str, Enum):
    RUNNING = "running"
    CLEAN_INTERRUPTED = "clean_interrupted"
    FAILED = "failed"
    COMPLETE = "complete"


@dataclass(frozen=True)
class LifecycleRecord:
    phase: LifecyclePhase
    state: LifecycleState
    step: int
    max_steps: int
    resume_allowed: bool
    checkpoint_path: str | None
    exit_code: int | None
    contract: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema"] = RUNTIME_SCHEMA
        payload["phase"] = self.phase.value
        payload["state"] = self.state.value
        return payload


_STOP_REQUESTED = False


def reset_stop_request() -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False


def stop_requested() -> bool:
    return bool(_STOP_REQUESTED)


def install_stop_handlers() -> None:
    def request_stop(_signal_number: int, _frame: object) -> None:
        global _STOP_REQUESTED
        _STOP_REQUESTED = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)


def _resolved(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def ensure_child(root: str | os.PathLike[str], child: str | os.PathLike[str]) -> Path:
    root_path = _resolved(root)
    child_path = _resolved(child)
    try:
        child_path.relative_to(root_path)
    except ValueError as exc:
        raise RuntimeContractError(f"artifact path escapes its run root: {child_path}") from exc
    return child_path


def _json_value(value: object, label: str = "payload") -> object:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)) and not isinstance(value, (bool, np.bool_)):
        value = float(value)
        if not math.isfinite(value):
            raise RuntimeContractError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item, f"{label}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{label}[]") for item in value]
    raise RuntimeContractError(f"{label} is not finite JSON data: {type(value).__name__}")


def atomic_json_dump(payload: Mapping[str, object], path: str | os.PathLike[str]) -> None:
    value = _json_value(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json_mapping(path: str | os.PathLike[str], label: str) -> dict[str, object]:
    location = Path(path)
    if not location.is_file():
        raise FileNotFoundError(f"missing {label}: {location}")
    with location.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeContractError(f"{label} must be a JSON object")
    _json_value(payload, label)
    return payload


def atomic_torch_save(payload: Mapping[str, object], path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_torch_mapping(path: str | os.PathLike[str], label: str) -> dict[str, object]:
    location = Path(path)
    if not location.is_file():
        raise FileNotFoundError(f"missing {label}: {location}")
    try:
        payload = torch.load(location, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        payload = torch.load(location, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeContractError(f"{label} must contain a mapping")
    return dict(payload)


def same_value(value_a: object, value_b: object) -> bool:
    """Exact semantic equality that also handles NumPy/Torch arrays."""

    if torch.is_tensor(value_a) or torch.is_tensor(value_b):
        if not (torch.is_tensor(value_a) and torch.is_tensor(value_b)):
            return False
        return bool(
            value_a.dtype == value_b.dtype
            and value_a.shape == value_b.shape
            and torch.isfinite(value_a).all()
            and torch.isfinite(value_b).all()
            and torch.equal(value_a.detach().cpu(), value_b.detach().cpu())
        )
    if isinstance(value_a, np.ndarray) or isinstance(value_b, np.ndarray):
        if not (isinstance(value_a, np.ndarray) and isinstance(value_b, np.ndarray)):
            return False
        return bool(
            value_a.dtype == value_b.dtype
            and value_a.shape == value_b.shape
            and np.isfinite(value_a).all()
            and np.isfinite(value_b).all()
            and np.array_equal(value_a, value_b)
        )
    if isinstance(value_a, Mapping) and isinstance(value_b, Mapping):
        return value_a.keys() == value_b.keys() and all(
            same_value(value_a[key], value_b[key]) for key in value_a
        )
    if isinstance(value_a, (list, tuple)) and isinstance(value_b, (list, tuple)):
        return len(value_a) == len(value_b) and all(same_value(left, right) for left, right in zip(value_a, value_b))
    if isinstance(value_a, (float, np.floating)) or isinstance(value_b, (float, np.floating)):
        try:
            left, right = float(value_a), float(value_b)
        except (TypeError, ValueError):
            return False
        return bool(math.isfinite(left) and math.isfinite(right) and left == right)
    return value_a == value_b


def make_lifecycle_record(
    *,
    phase: LifecyclePhase,
    state: LifecycleState,
    step: int,
    max_steps: int,
    resume_allowed: bool,
    checkpoint_path: str | None,
    exit_code: int | None,
    contract: Mapping[str, object],
) -> dict[str, object]:
    if step < 0 or max_steps <= 0 or step > max_steps:
        raise RuntimeContractError("lifecycle step/max_steps is invalid")
    if phase is LifecyclePhase.COMPLETE:
        if state is not LifecycleState.COMPLETE or resume_allowed or checkpoint_path is None:
            raise RuntimeContractError("complete lifecycle record has invalid resume semantics")
    elif state is LifecycleState.COMPLETE:
        raise RuntimeContractError("only complete phase may carry complete state")
    if state is LifecycleState.CLEAN_INTERRUPTED:
        if not resume_allowed or exit_code != CLEAN_INTERRUPTION_EXIT_CODE or checkpoint_path is None:
            raise RuntimeContractError("clean interruption must name a resumable checkpoint and exit 143")
    elif resume_allowed:
        raise RuntimeContractError("only clean interruptions may permit resume")
    return LifecycleRecord(
        phase=phase,
        state=state,
        step=int(step),
        max_steps=int(max_steps),
        resume_allowed=bool(resume_allowed),
        checkpoint_path=checkpoint_path,
        exit_code=exit_code,
        contract=dict(contract),
    ).as_dict()


def validate_lifecycle_record(
    payload: Mapping[str, object], *, expected_contract: Mapping[str, object], max_steps: int
) -> dict[str, object]:
    if not isinstance(payload, Mapping) or payload.get("schema") != RUNTIME_SCHEMA:
        raise RuntimeContractError("lifecycle record schema changed")
    if not isinstance(payload.get("step"), int) or isinstance(payload.get("step"), bool):
        raise RuntimeContractError("lifecycle step must be an integer")
    if not isinstance(payload.get("max_steps"), int) or isinstance(payload.get("max_steps"), bool):
        raise RuntimeContractError("lifecycle max_steps must be an integer")
    if not isinstance(payload.get("resume_allowed"), bool):
        raise RuntimeContractError("lifecycle resume_allowed must be boolean")
    if payload.get("checkpoint_path") is not None and not isinstance(payload.get("checkpoint_path"), str):
        raise RuntimeContractError("lifecycle checkpoint_path must be a string or null")
    if payload.get("exit_code") is not None and (
        not isinstance(payload.get("exit_code"), int) or isinstance(payload.get("exit_code"), bool)
    ):
        raise RuntimeContractError("lifecycle exit_code must be an integer or null")
    if not isinstance(payload.get("contract"), Mapping):
        raise RuntimeContractError("lifecycle contract must be a mapping")
    try:
        phase = LifecyclePhase(payload.get("phase"))
        state = LifecycleState(payload.get("state"))
    except ValueError as exc:
        raise RuntimeContractError("lifecycle phase/state is invalid") from exc
    expected = make_lifecycle_record(
        phase=phase,
        state=state,
        step=int(payload.get("step", -1)),
        max_steps=int(payload.get("max_steps", -1)),
        resume_allowed=bool(payload.get("resume_allowed")),
        checkpoint_path=payload.get("checkpoint_path"),
        exit_code=payload.get("exit_code"),
        contract=payload.get("contract", {}),
    )
    if expected["max_steps"] != int(max_steps):
        raise RuntimeContractError("lifecycle max_steps changed")
    if not same_value(expected["contract"], expected_contract):
        raise RuntimeContractError("lifecycle contract changed")
    if set(payload) != set(expected):
        raise RuntimeContractError("lifecycle record has missing or unexpected fields")
    return expected


def write_lifecycle(
    path: str | os.PathLike[str],
    record: Mapping[str, object],
    *,
    expected_contract: Mapping[str, object],
    max_steps: int,
) -> None:
    checked = validate_lifecycle_record(record, expected_contract=expected_contract, max_steps=max_steps)
    atomic_json_dump(checked, path)


def prepare_run_root(
    root: str | os.PathLike[str],
    *,
    resume: str | os.PathLike[str] | None,
    expected_contract: Mapping[str, object],
    max_steps: int,
) -> str:
    """Validate entry state without deleting, overwriting, or fabricating evidence.

    Returns ``"complete"``, ``"fresh"``, or ``"resume"``.  Completed
    packages are checked by the caller because it owns the model-specific
    surface/checkpoint schema.
    """

    root_path = Path(root)
    complete = root_path / "complete"
    latest = root_path / "checkpoint_latest.pth.tar"
    lifecycle = root_path / "lifecycle.json"
    failure = root_path / "terminal_failure.json"
    if complete.exists():
        if resume is not None:
            raise RuntimeContractError("a completed run cannot be resumed")
        return "complete"
    if failure.exists():
        raise RuntimeContractError("terminal failure evidence is preserved; do not automatically resume")
    if resume is not None:
        if _resolved(resume) != _resolved(latest):
            raise RuntimeContractError("resume must name this run's checkpoint_latest.pth.tar")
        if not latest.is_file() or not lifecycle.is_file():
            raise RuntimeContractError("resume requires both latest checkpoint and lifecycle record")
        record = validate_lifecycle_record(
            load_json_mapping(lifecycle, "lifecycle"),
            expected_contract=expected_contract,
            max_steps=max_steps,
        )
        if record["state"] != LifecycleState.CLEAN_INTERRUPTED.value or record["resume_allowed"] is not True:
            raise RuntimeContractError("only a recorded clean interruption may resume")
        if _resolved(record["checkpoint_path"]) != _resolved(latest):
            raise RuntimeContractError("lifecycle checkpoint path does not match latest checkpoint")
        return "resume"
    if root_path.exists() and any(root_path.iterdir()):
        raise RuntimeContractError("fresh run refuses a nonempty output root")
    root_path.mkdir(parents=True, exist_ok=True)
    return "fresh"


def record_terminal_failure(
    root: str | os.PathLike[str],
    *,
    phase: LifecyclePhase,
    step: int,
    max_steps: int,
    expected_contract: Mapping[str, object],
    error: BaseException,
) -> None:
    """Retain a real failure exactly once; it is intentionally non-resumable."""

    root_path = Path(root)
    destination = root_path / "terminal_failure.json"
    if destination.exists():
        return
    record = make_lifecycle_record(
        phase=phase,
        state=LifecycleState.FAILED,
        step=step,
        max_steps=max_steps,
        resume_allowed=False,
        checkpoint_path=None,
        exit_code=None,
        contract=expected_contract,
    )
    payload = {
        "lifecycle": record,
        "error_type": type(error).__name__,
        "error": str(error),
        "resume_allowed": False,
    }
    atomic_json_dump(payload, destination)


def clean_interruption_record(
    *,
    phase: LifecyclePhase,
    step: int,
    max_steps: int,
    checkpoint_path: str,
    expected_contract: Mapping[str, object],
) -> dict[str, object]:
    return make_lifecycle_record(
        phase=phase,
        state=LifecycleState.CLEAN_INTERRUPTED,
        step=step,
        max_steps=max_steps,
        resume_allowed=True,
        checkpoint_path=checkpoint_path,
        exit_code=CLEAN_INTERRUPTION_EXIT_CODE,
        contract=expected_contract,
    )


def running_record(
    *,
    phase: LifecyclePhase,
    step: int,
    max_steps: int,
    checkpoint_path: str | None,
    expected_contract: Mapping[str, object],
) -> dict[str, object]:
    return make_lifecycle_record(
        phase=phase,
        state=LifecycleState.RUNNING,
        step=step,
        max_steps=max_steps,
        resume_allowed=False,
        checkpoint_path=checkpoint_path,
        exit_code=None,
        contract=expected_contract,
    )


def new_export_staging(root: str | os.PathLike[str]) -> Path:
    """Create an isolated, invisible staging directory without touching prior debris."""

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".export-staging-", dir=root_path))


def promote_complete_package(
    staging_dir: str | os.PathLike[str],
    complete_dir: str | os.PathLike[str],
    *,
    validate: Callable[[Path], None],
) -> Path:
    """Validate a fully staged package then atomically make it visible as complete.

    A crash leaves only a hidden staging directory.  It is never silently
    deleted or reused; a later export creates another staging directory.
    """

    staging = Path(staging_dir)
    complete = Path(complete_dir)
    if not staging.is_dir():
        raise RuntimeContractError("export staging directory is missing")
    if complete.exists():
        validate(complete)
        return complete
    validate(staging)
    os.replace(staging, complete)
    validate(complete)
    return complete


def validate_complete_filenames(directory: str | os.PathLike[str]) -> dict[str, Path]:
    root = Path(directory)
    if not root.is_dir():
        raise RuntimeContractError("completion package directory is missing")
    actual = {path.name for path in root.iterdir()}
    expected = set(COMPLETION_FILENAMES)
    if actual != expected:
        raise RuntimeContractError(
            f"completion package must contain exactly {sorted(expected)}, found {sorted(actual)}"
        )
    return {name: root / name for name in COMPLETION_FILENAMES}
