"""Data-free, structural-only contract for the two paper-local GOTCHA ROIs.

Version 1 deliberately cannot release measured extraction or fitting. It
accepts only the exact unresolved/default declaration, so native placement,
observation IDs, calibration, spotlighting, padding, and PSF/leakage remain
unresolved. A future measured extractor must bind actual ``NativeShard``
identity and role validation before response materialization; that work must
arrive in a separately reviewed contract version with the loader integrated.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "rift_gotcha_step3_two_target_roi_protocol_v1"
TARGET_IDS = ("tophat", "toyota_camry")
ROLE_SPLIT = {"train": 288, "validation": 36, "test": 36}


def _duplicate_rejecting_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _close_triplet(actual: Any, expected: tuple[float, float, float], *, label: str) -> None:
    try:
        values = tuple(float(value) for value in actual)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain three numeric values") from error
    if len(values) != 3 or any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(values, expected)):
        raise ValueError(f"{label} must be {list(expected)!r}; got {actual!r}")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_unresolved(item: Mapping[str, Any], label: str, expected: Mapping[str, Any]) -> None:
    if set(item) != set(expected):
        raise ValueError(f"{label} must have exactly the data-free v1 fields")
    for key, value in expected.items():
        if item.get(key) != value:
            raise ValueError(f"{label}.{key} must remain exactly {value!r} in data-free v1")


@dataclass(frozen=True)
class GridContract:
    target_id: str
    frame: str
    lower_edge_m: tuple[float, float, float]
    upper_edge_exclusive_m: tuple[float, float, float]
    spacing_m: tuple[float, float, float]
    shape: tuple[int, int, int]
    final_sample_m: tuple[float, float, float]

    @property
    def point_count(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    def sample(self, index: int) -> tuple[float, float, float]:
        if not 0 <= int(index) < max(self.shape):
            raise IndexError(index)
        return tuple(self.lower_edge_m[axis] + self.spacing_m[axis] * index for axis in range(3))

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "frame": self.frame,
            "lower_edge_m": list(self.lower_edge_m),
            "upper_edge_exclusive_m": list(self.upper_edge_exclusive_m),
            "spacing_m": list(self.spacing_m),
            "shape": list(self.shape),
            "final_sample_m": list(self.final_sample_m),
            "point_count": self.point_count,
        }


@dataclass(frozen=True)
class TargetContract:
    target_id: str
    grid: GridContract
    declaration: Mapping[str, Any]


@dataclass(frozen=True)
class TwoRoiContract:
    schema: str
    stage: str
    role_split: Mapping[str, int]
    targets: Mapping[str, TargetContract]
    measurement_prediction_contract: str

    def target(self, target_id: str) -> TargetContract:
        try:
            return self.targets[str(target_id)]
        except KeyError as error:
            raise ValueError(f"unknown target_id: {target_id!r}") from error

    def structural_report(self) -> dict[str, Any]:
        target_states = {
            target_id: {
                "grid": target.grid.as_dict(),
                "extraction_allowed": False,
                "fit_allowed": False,
                "placement_status": target.declaration["p_native"]["status"],
            }
            for target_id, target in self.targets.items()
        }
        return {
            "schema": "rift_gotcha_step3_two_target_roi_structural_report_v1",
            "status": "PASS",
            "data_free": True,
            "measured_fit_release": False,
            "extraction_allowed": {target_id: False for target_id in self.targets},
            "fit_allowed": {target_id: False for target_id in self.targets},
            "role_split": dict(self.role_split),
            "test_policy": "sealed_no_selection",
            "test_sealed": False,
            "test_state": "declared_policy_only_no_future_release_claim",
            "targets": target_states,
            "measurement_prediction_contract": self.measurement_prediction_contract,
            "release_statement": "structural PASS only; data-free v1 never releases measured extraction or fit",
        }


def _validate_target(target_id: str, declaration: Mapping[str, Any]) -> TargetContract:
    target_fields = {
        "grid",
        "p_native",
        "selected_native_observation_ids",
        "frozen_phase_r0_af_calibration",
        "complex_spotlight",
        "padding_background",
        "psf_leakage",
        "extraction_allowed",
        "fit_allowed",
    }
    if set(declaration) != target_fields:
        raise ValueError(f"targets.{target_id} must have exactly the data-free v1 fields")
    expected = {
        "tophat": ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0), (40, 40, 40), (1.9, 1.9, 1.9)),
        "toyota_camry": ((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), (100, 100, 100), (4.9, 4.9, 4.9)),
    }[target_id]
    grid = _require_mapping(declaration.get("grid"), f"targets.{target_id}.grid")
    grid_fields = {
        "frame",
        "lower_edge_m",
        "upper_edge_exclusive_m",
        "spacing_m",
        "shape",
        "index_rule",
        "final_sample_m",
        "endpoint_inclusion",
        "half_voxel_shift",
        "merged_box",
        "model_specific_omission",
    }
    if set(grid) != grid_fields:
        raise ValueError(f"targets.{target_id}.grid must have exactly the data-free v1 fields")
    if grid.get("frame") != "local_target_frame":
        raise ValueError(f"targets.{target_id}.grid.frame must be local_target_frame")
    lower, upper, shape, final = expected
    _close_triplet(grid.get("lower_edge_m"), lower, label=f"targets.{target_id}.grid.lower_edge_m")
    _close_triplet(grid.get("upper_edge_exclusive_m"), upper, label=f"targets.{target_id}.grid.upper_edge_exclusive_m")
    _close_triplet(grid.get("spacing_m"), (0.1, 0.1, 0.1), label=f"targets.{target_id}.grid.spacing_m")
    if tuple(grid.get("shape", ())) != shape:
        raise ValueError(f"targets.{target_id}.grid.shape must be {list(shape)!r}")
    _close_triplet(grid.get("final_sample_m"), final, label=f"targets.{target_id}.grid.final_sample_m")
    if grid.get("index_rule") != f"lower_edge + 0.1*i for i=0..{shape[0] - 1}":
        raise ValueError(f"targets.{target_id}.grid.index_rule is not endpoint-exclusive")
    if grid.get("endpoint_inclusion") is not False or grid.get("half_voxel_shift") is not False:
        raise ValueError(f"targets.{target_id}.grid must be endpoint-exclusive with no half-voxel shift")
    if grid.get("merged_box") is not False or grid.get("model_specific_omission") is not False:
        raise ValueError(f"targets.{target_id}.grid cannot use a merged or model-specific omission")

    placement = _require_mapping(declaration.get("p_native"), f"targets.{target_id}.p_native")
    if placement.get("equation") != "p_native = R @ p_local + t":
        raise ValueError(f"targets.{target_id}.p_native.equation must be explicit")
    _require_unresolved(
        placement,
        f"targets.{target_id}.p_native",
        {
            "status": "unresolved",
            "equation": "p_native = R @ p_local + t",
            "value": None,
            "source": "unresolved",
            "transform": "unresolved",
            "frame": "unresolved",
            "height_datum": "unresolved",
            "R": None,
            "t_m": None,
            "native_frame": "unresolved",
            "source_evidence": None,
            "verification_evidence": None,
        },
    )

    selected = _require_mapping(declaration.get("selected_native_observation_ids"), f"targets.{target_id}.selected_native_observation_ids")
    _require_unresolved(selected, f"targets.{target_id}.selected_native_observation_ids", {"status": "unresolved", "values": None})

    frozen = _require_mapping(declaration.get("frozen_phase_r0_af_calibration"), f"targets.{target_id}.frozen_phase_r0_af_calibration")
    _require_unresolved(
        frozen,
        f"targets.{target_id}.frozen_phase_r0_af_calibration",
        {
            "status": "unresolved",
            "recipe": "unresolved",
            "evidence": None,
            "train_derived_calibration_policy": None,
            "correction_representation": "unresolved",
            "af_cross_channel_borrowing": False,
            "double_correction": False,
        },
    )

    spotlight = _require_mapping(declaration.get("complex_spotlight"), f"targets.{target_id}.complex_spotlight")
    _require_unresolved(
        spotlight,
        f"targets.{target_id}.complex_spotlight",
        {
            "status": "unresolved",
            "window": "unresolved",
            "taper": "unresolved",
            "operator": "unresolved",
            "aperture": "unresolved",
            "evidence": None,
            "data_prediction_application": "identical_complex_spotlight_before_magnitude_power",
            "data_prediction_equal": False,
            "role_pure": False,
        },
    )

    padding = _require_mapping(declaration.get("padding_background"), f"targets.{target_id}.padding_background")
    _require_unresolved(
        padding,
        f"targets.{target_id}.padding_background",
        {"status": "unresolved", "policy": "unresolved", "evidence": None, "hidden_padding": False},
    )

    psf = _require_mapping(declaration.get("psf_leakage"), f"targets.{target_id}.psf_leakage")
    _require_unresolved(psf, f"targets.{target_id}.psf_leakage", {"status": "unresolved", "evidence": None})

    if declaration.get("extraction_allowed") is not False or declaration.get("fit_allowed") is not False:
        raise ValueError(f"targets.{target_id} data-free v1 cannot allow extraction or fit")
    return TargetContract(
        target_id=target_id,
        grid=GridContract(target_id, "local_target_frame", lower, upper, (0.1, 0.1, 0.1), shape, final),
        declaration=declaration,
    )


def validate_declaration(value: Mapping[str, Any]) -> TwoRoiContract:
    if not isinstance(value, Mapping):
        raise ValueError("two-ROI declaration must be an object")
    root_fields = {"schema", "stage", "role_split", "test_policy", "measurement_prediction_contract", "targets"}
    if set(value) != root_fields:
        raise ValueError("declaration must have exactly the data-free v1 fields")
    if value.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")
    if value.get("stage") != "contract":
        raise ValueError("stage must be contract")
    role_split = _require_mapping(value.get("role_split"), "role_split")
    if dict(role_split) != ROLE_SPLIT:
        raise ValueError("role_split must be exactly train=288, validation=36, test=36")
    if value.get("test_policy") != "sealed_no_selection":
        raise ValueError("test_policy must be sealed_no_selection")
    measurement_contract = value.get("measurement_prediction_contract")
    if measurement_contract != "identical complex spotlight on measured data and predictions before magnitude/power transforms":
        raise ValueError("measurement_prediction_contract is not the required shared complex rule")
    targets = _require_mapping(value.get("targets"), "targets")
    if set(targets) != set(TARGET_IDS):
        raise ValueError(f"targets must contain exactly {list(TARGET_IDS)!r}")
    parsed = {target_id: _validate_target(target_id, _require_mapping(targets[target_id], f"targets.{target_id}")) for target_id in TARGET_IDS}
    return TwoRoiContract(SCHEMA, "contract", dict(ROLE_SPLIT), parsed, measurement_contract)


def load_contract(path: str | Path) -> TwoRoiContract:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=_duplicate_rejecting_pairs)
    return validate_declaration(value)


def _data_free_gate_error(target_id: str, operation: str) -> RuntimeError:
    return RuntimeError(
        f"target {target_id!r} cannot {operation}: GOTCHA step-3 v1 is data-free; "
        "a future loader-integrated, separately reviewed contract version is required"
    )


def require_target_extraction_ready(target_id: str, contract: TwoRoiContract) -> TargetContract:
    del contract
    raise _data_free_gate_error(target_id, "release measured extraction")


def require_target_fit_ready(target_id: str, contract: TwoRoiContract) -> TargetContract:
    del contract
    raise _data_free_gate_error(target_id, "release measured fit")


__all__ = [
    "GridContract",
    "ROLE_SPLIT",
    "SCHEMA",
    "TARGET_IDS",
    "TargetContract",
    "TwoRoiContract",
    "load_contract",
    "require_target_extraction_ready",
    "require_target_fit_ready",
    "validate_declaration",
]
