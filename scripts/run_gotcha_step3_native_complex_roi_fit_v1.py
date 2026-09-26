"""Run fake control or an explicit actual-archive directional-SH ROI fit.

Actual mode is deliberately separate from fake mode.  It requires an explicit
archive parent or the exact ``converted_v3_joint8_fullpol/shards`` directory,
plus a fresh output directory.  It uses the existing Gate-1 acquisition and
TopHat screen loaders, and never falls back to synthetic records.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_local(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Keep the direct-loader names in sync with the accepted measured screen path.
ACQ = _load_local("gotcha_acquisition", ROOT / "rift" / "gotcha_acquisition.py")
SOURCE_AF = _load_local("gotcha_source_af", ROOT / "rift" / "gotcha_source_af.py")
LOC = _load_local("gotcha_tophat_localization", ROOT / "rift" / "gotcha_tophat_localization.py")
ROI = _load_local("gotcha_native_complex_roi_fit", ROOT / "rift" / "gotcha_native_complex_roi_fit.py")


SCENE_ID = "gotcha_v1_joint8_fullpol"

# These are measured archive contracts, not permissive nonempty checks.  The
# identity/role counts are checked before the first Source-AF conversion, and
# the converted records are checked against the same totals afterward.
CAMRY_TRAIN_RECORD_COUNT = 117
CAMRY_VALIDATION_RECORD_COUNT = 117
CAMRY_FREQUENCY_COUNT = 424
CAMRY_TRAIN_FREQUENCY_SAMPLES = 49_608
CAMRY_VALIDATION_FREQUENCY_SAMPLES = 49_608

TOPHAT_FREQUENCY_COUNTS_BY_PASS = {1: 424, 7: 434}
TOPHAT_TRAIN_RECORD_COUNTS_BY_PANEL = {
    (1, 2): 117,
    (1, 92): 117,
    (1, 182): 117,
    (1, 272): 118,
    (7, 2): 120,
    (7, 92): 120,
    (7, 182): 120,
    (7, 272): 120,
}
TOPHAT_VALIDATION_RECORD_COUNTS_BY_PANEL = {
    (1, 1): 117,
    (1, 91): 117,
    (1, 181): 118,
    (1, 271): 117,
    (7, 1): 119,
    (7, 91): 119,
    (7, 181): 120,
    (7, 271): 120,
}
TOPHAT_TRAIN_RECORD_COUNT = 949
TOPHAT_VALIDATION_RECORD_COUNT = 947
TOPHAT_TRAIN_FREQUENCY_SAMPLES = 407_176
TOPHAT_VALIDATION_FREQUENCY_SAMPLES = 406_308


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _identity_key(identity: Any) -> tuple[int, str, int, int]:
    return (
        int(identity.pass_id),
        str(identity.polarization).lower(),
        int(identity.sector_id),
        int(identity.pulse_index),
    )


def _safe_shard_path(archive_root: str | Path, pass_id: int) -> Path:
    raw_root = Path(archive_root).expanduser()
    if raw_root.is_symlink():
        raise ValueError(f"archive root must not be a symlink: {raw_root}")
    root = raw_root.resolve()
    if not root.is_dir():
        raise ValueError(f"archive root must be an existing regular directory: {root}")
    if raw_root.name == "shards" and raw_root.parent.name == "converted_v3_joint8_fullpol":
        raw_shard_root = raw_root
    else:
        raw_shard_root = raw_root / "converted_v3_joint8_fullpol" / "shards"
    if raw_shard_root.is_symlink():
        raise ValueError(f"native shard directory must not be a symlink: {raw_shard_root}")
    shard_root = raw_shard_root.resolve()
    if not shard_root.is_dir():
        raise ValueError(f"native shard directory must be a regular directory under archive root: {shard_root}")
    raw_path = shard_root / f"pass{int(pass_id)}_hh.npz"
    if raw_path.is_symlink():
        raise ValueError(f"native shard must not be a symlink: {raw_path}")
    path = raw_path.resolve()
    try:
        path.relative_to(shard_root)
    except ValueError as exc:
        raise ValueError("resolved native shard escaped the selected shard directory") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"native shard must be a regular file under archive root: {path}")
    return path


def _validated_shard_records(shard: Any) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    identities = tuple(shard.observation_ids)
    roles = tuple(str(value).lower() for value in np.asarray(shard.role).tolist())
    if len(identities) != len(roles) or not identities:
        raise ValueError("loaded shard identity and role vectors are not aligned")
    keys = tuple(_identity_key(identity) for identity in identities)
    if len(set(keys)) != len(keys):
        raise ValueError("loaded shard contains duplicate canonical identities")
    if any(role == "test" for role in roles):
        raise ValueError("TEST role appeared in the loaded shard; actual fit is sealed closed")
    if tuple(keys) != tuple(sorted(keys)):
        raise ValueError("loaded shard identities are not in canonical order")
    return identities, roles


def _panel_record_counts(
    identities: Sequence[Any],
    roles: Sequence[str],
    *,
    role: str,
    expected: Mapping[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    observed = Counter(
        (int(identity.pass_id), int(identity.sector_id))
        for identity, observed_role in zip(identities, roles)
        if str(observed_role).lower() == str(role).lower()
        and (int(identity.pass_id), int(identity.sector_id)) in expected
    )
    normalized = {key: int(observed.get(key, 0)) for key in expected}
    if normalized != {key: int(value) for key, value in expected.items()}:
        raise ValueError(
            f"{role} panel identity counts do not match the measured contract: "
            f"expected {dict(expected)}, got {normalized}"
        )
    return normalized


def _frequency_contract(
    frequencies: Any,
    *,
    pass_id: int,
    expected_count: int,
) -> int:
    values = np.asarray(frequencies, dtype=np.float64)
    if values.ndim != 1 or int(values.size) != int(expected_count):
        raise ValueError(
            f"P{int(pass_id)} native frequency contract requires exactly "
            f"{int(expected_count)} samples; got {int(values.size)}"
        )
    if not np.isfinite(values).all() or (values.size > 1 and not np.all(np.diff(values) > 0.0)):
        raise ValueError(f"P{int(pass_id)} native frequency vector is not finite and strictly increasing")
    return int(values.size)


def _panel_contract(
    counts: Mapping[tuple[int, int], int],
    frequencies_by_pass: Mapping[int, int],
) -> dict[str, dict[str, int]]:
    return {
        f"P{pass_id}/HH/{sector_id:03d}": {
            "record_count": int(record_count),
            "frequency_count": int(frequencies_by_pass[pass_id]),
            "frequency_sample_count": int(record_count) * int(frequencies_by_pass[pass_id]),
        }
        for (pass_id, sector_id), record_count in sorted(counts.items())
    }


def _select_camry_records(
    archive_root: str | Path,
) -> tuple[tuple[Any, ...], tuple[Any, ...], dict[str, Any]]:
    path = _safe_shard_path(archive_root, 1)
    shard = ACQ.load_native_shard(
        path,
        expected_pass_id=1,
        expected_polarization="hh",
        expected_scene_id=SCENE_ID,
    )
    identities, roles = _validated_shard_records(shard)
    train_ids = tuple(
        sorted(
            (identity for identity, role in zip(identities, roles) if role == "train" and int(identity.sector_id) == 2),
            key=_identity_key,
        )
    )
    validation_ids = tuple(
        sorted(
            (identity for identity, role in zip(identities, roles) if role == "validation" and int(identity.sector_id) == 1),
            key=_identity_key,
        )
    )
    if len(train_ids) != CAMRY_TRAIN_RECORD_COUNT:
        raise ValueError(
            f"Camry P1/HH/sector002 TRAIN requires exactly {CAMRY_TRAIN_RECORD_COUNT} "
            f"records; got {len(train_ids)}"
        )
    if len(validation_ids) != CAMRY_VALIDATION_RECORD_COUNT:
        raise ValueError(
            f"Camry P1/HH/sector001 validation requires exactly {CAMRY_VALIDATION_RECORD_COUNT} "
            f"records; got {len(validation_ids)}"
        )
    frequency_count = _frequency_contract(
        shard.frequencies_hz, pass_id=1, expected_count=CAMRY_FREQUENCY_COUNT
    )
    if len(train_ids) * frequency_count != CAMRY_TRAIN_FREQUENCY_SAMPLES:
        raise ValueError("Camry TRAIN frequency-sample count disagrees with the measured contract")
    if len(validation_ids) * frequency_count != CAMRY_VALIDATION_FREQUENCY_SAMPLES:
        raise ValueError("Camry validation frequency-sample count disagrees with the measured contract")
    # All metadata/header/role/identity gates above precede the first payload
    # materialization below.
    train_observations = tuple(shard.observations(train_ids))
    validation_observations = tuple(shard.observations(validation_ids))
    train_scope = SOURCE_AF.SourceAFScope(
        pass_id=1, polarization="hh", sector_ids=(2,), role="train",
        expected_count=CAMRY_TRAIN_RECORD_COUNT, name="camry_actual_sector002_train",
    )
    validation_scope = SOURCE_AF.SourceAFScope(
        pass_id=1, polarization="hh", sector_ids=(1,), role="validation",
        expected_count=CAMRY_VALIDATION_RECORD_COUNT, name="camry_actual_sector001_validation",
    )
    train_records = SOURCE_AF.build_source_af(
        train_observations, scope=train_scope, expected_count=CAMRY_TRAIN_RECORD_COUNT
    )
    validation_records = SOURCE_AF.build_source_af(
        validation_observations, scope=validation_scope, expected_count=CAMRY_VALIDATION_RECORD_COUNT
    )
    if tuple(record.identity for record in train_records) != train_ids or tuple(record.identity for record in validation_records) != validation_ids:
        raise ValueError("source-AF conversion changed Camry canonical identity order")
    return train_records, validation_records, {
        "shards": [str(path)],
        "loaded_record_count": len(identities),
        "loaded_role_counts": dict(Counter(roles)),
        "train_selector": "P1/HH/sector002/TRAIN",
        "validation_selector": "P1/HH/sector001/validation",
        "train_record_count": len(train_records),
        "validation_record_count": len(validation_records),
        "train_frequency_counts": [int(record.frequencies_hz.size) for record in train_records],
        "validation_frequency_counts": [int(record.frequencies_hz.size) for record in validation_records],
        "train_frequency_sample_count": CAMRY_TRAIN_FREQUENCY_SAMPLES,
        "validation_frequency_sample_count": CAMRY_VALIDATION_FREQUENCY_SAMPLES,
        "source_af_selected_rows_provenance": {
            "scope_type": "SourceAFScope",
            "selected_pass_ids": [1],
            "selected_polarization": "hh",
            "selected_train_sector_ids": [2],
            "selected_validation_sector_ids": [1],
            "selected_train_record_count": CAMRY_TRAIN_RECORD_COUNT,
            "selected_validation_record_count": CAMRY_VALIDATION_RECORD_COUNT,
        },
        "train_panel_contract": _panel_contract(
            {(1, 2): CAMRY_TRAIN_RECORD_COUNT}, {1: CAMRY_FREQUENCY_COUNT}
        ),
        "validation_panel_contract": _panel_contract(
            {(1, 1): CAMRY_VALIDATION_RECORD_COUNT}, {1: CAMRY_FREQUENCY_COUNT}
        ),
        "source_af_conversion_batches": 2,
        "source_af_converted_record_count": len(train_records) + len(validation_records),
        "source_af_exactly_once_per_selected_record": True,
        "test_payload_opened": False,
    }


def _select_tophat_records(
    archive_root: str | Path,
) -> tuple[tuple[Any, ...], tuple[Any, ...], dict[str, Any]]:
    paths = tuple(_safe_shard_path(archive_root, pass_id) for pass_id in (1, 7))
    shards = tuple(
        ACQ.load_native_shard(
            path,
            expected_pass_id=pass_id,
            expected_polarization="hh",
            expected_scene_id=SCENE_ID,
        )
        for pass_id, path in zip((1, 7), paths)
    )
    train_counts_by_panel: dict[tuple[int, int], int] = {}
    validation_counts_by_panel: dict[tuple[int, int], int] = {}
    frequency_counts_by_pass: dict[int, int] = {}
    selected_validation_ids = []
    for shard in shards:
        identities, roles = _validated_shard_records(shard)
        pass_id = int(shard.pass_id)
        frequency_counts_by_pass[pass_id] = _frequency_contract(
            shard.frequencies_hz,
            pass_id=pass_id,
            expected_count=TOPHAT_FREQUENCY_COUNTS_BY_PASS[pass_id],
        )
        train_counts_by_panel.update(
            _panel_record_counts(
                identities,
                roles,
                role="train",
                expected={key: value for key, value in TOPHAT_TRAIN_RECORD_COUNTS_BY_PANEL.items() if key[0] == pass_id},
            )
        )
        validation_counts_by_panel.update(
            _panel_record_counts(
                identities,
                roles,
                role="validation",
                expected={key: value for key, value in TOPHAT_VALIDATION_RECORD_COUNTS_BY_PANEL.items() if key[0] == pass_id},
            )
        )
        selected_validation_ids.extend(
            identity
            for identity, role in zip(identities, roles)
            if role == "validation" and int(identity.sector_id) in ROI.TOPHAT_VALIDATION_SECTOR_IDS
        )
    selected_validation_ids = tuple(sorted(selected_validation_ids, key=_identity_key))
    if len(selected_validation_ids) != TOPHAT_VALIDATION_RECORD_COUNT:
        raise ValueError(
            f"TopHat validation selector requires exactly {TOPHAT_VALIDATION_RECORD_COUNT} "
            f"records; got {len(selected_validation_ids)}"
        )
    train_sample_count = sum(
        int(record_count) * int(frequency_counts_by_pass[pass_id])
        for (pass_id, _sector_id), record_count in train_counts_by_panel.items()
    )
    validation_sample_count = sum(
        int(record_count) * int(frequency_counts_by_pass[pass_id])
        for (pass_id, _sector_id), record_count in validation_counts_by_panel.items()
    )
    if train_sample_count != TOPHAT_TRAIN_FREQUENCY_SAMPLES:
        raise ValueError("TopHat TRAIN frequency-sample count disagrees with the measured contract")
    if validation_sample_count != TOPHAT_VALIDATION_FREQUENCY_SAMPLES:
        raise ValueError("TopHat validation frequency-sample count disagrees with the measured contract")
    # This accepted helper performs the complete train header gate and the one
    # multipass source-AF conversion for the 949 selected TRAIN records.
    dataset = LOC.prepare_tophat_measured_response_screen_dataset(shards)
    train_records = tuple(dataset.source_records)
    by_shard = {str(shard.shard_id): shard for shard in shards}
    validation_observations = []
    for identity in selected_validation_ids:
        validation_observations.extend(by_shard[str(identity.shard_id)].observations((identity,)))
    validation_observations.sort(key=lambda record: _identity_key(record.identity))
    validation_scope = SOURCE_AF.TophatHHTrainValidationSourceAFScope(
        expected_train_count=None,
        expected_validation_count=len(validation_observations),
        name="tophat_actual_p1_p7_validation_sectors001_091_181_271",
    )
    validation_records = SOURCE_AF.build_tophat_train_validation_source_af(
        validation_observations,
        expected_count=len(validation_observations),
        scope=validation_scope,
    )
    if len(train_records) != TOPHAT_TRAIN_RECORD_COUNT:
        raise ValueError(
            f"TopHat accepted screen selector requires exactly {TOPHAT_TRAIN_RECORD_COUNT} "
            f"TRAIN records; got {len(train_records)}"
        )
    if tuple(record.identity for record in validation_records) != selected_validation_ids:
        raise ValueError("TopHat validation source-AF conversion changed canonical identity order")
    return train_records, validation_records, {
        "shards": [str(path) for path in paths],
        "loaded_record_count": int(sum(shard.view_count for shard in shards)),
        "train_selector": "P1/P7 HH sectors 002/092/182/272 TRAIN",
        "validation_selector": "P1/P7 HH sectors 001/091/181/271 validation",
        "train_record_count": len(train_records),
        "validation_record_count": len(validation_records),
        "train_frequency_counts": [int(record.frequencies_hz.size) for record in train_records],
        "validation_frequency_counts": [int(record.frequencies_hz.size) for record in validation_records],
        "train_frequency_sample_count": TOPHAT_TRAIN_FREQUENCY_SAMPLES,
        "validation_frequency_sample_count": TOPHAT_VALIDATION_FREQUENCY_SAMPLES,
        "frequency_counts_by_pass": {str(key): int(value) for key, value in sorted(frequency_counts_by_pass.items())},
        "train_panel_contract": _panel_contract(train_counts_by_panel, frequency_counts_by_pass),
        "validation_panel_contract": _panel_contract(validation_counts_by_panel, frequency_counts_by_pass),
        "source_af_selected_rows_provenance": {
            "loader": "LOC.prepare_tophat_measured_response_screen_dataset (accepted helper, unchanged)",
            "inherited_scope_type": "MultipassHHTrainSourceAFScope",
            "inherited_scope_structural_pass_ids": list(range(1, 9)),
            "inherited_scope_polarization": "hh",
            "inherited_scope_train_sector_ids": list(ROI.TOPHAT_TRAIN_SECTOR_IDS),
            "actual_selected_pass_ids": [1, 7],
            "actual_selected_polarization": "hh",
            "actual_selected_train_sector_ids": list(ROI.TOPHAT_TRAIN_SECTOR_IDS),
            "actual_selected_train_record_count": TOPHAT_TRAIN_RECORD_COUNT,
            "actual_selected_train_frequency_sample_count": TOPHAT_TRAIN_FREQUENCY_SAMPLES,
            "structurally_permitted_but_not_selected_pass_ids": [2, 3, 4, 5, 6, 8],
            "no_p2_p8_train_records_participated": True,
        },
        "source_af_conversion_batches": 2,
        "source_af_converted_record_count": len(train_records) + len(validation_records),
        "source_af_exactly_once_per_selected_record": True,
        "screen_dataset_preparation": dataset.as_dict(),
        "panel_wise_image_pooling": False,
        "test_payload_opened": False,
    }


def _coverage(
    records: Sequence[Any],
    *,
    split: str,
    optimizer_update_count: int,
    diagnostic_steps: Sequence[int],
    solver: str = "gd",
    max_iterations: int | None = None,
    executed_iterations: int | None = None,
    selected_final_step: int | None = None,
    termination_reason: str | None = None,
) -> dict[str, Any]:
    identities = tuple(record.identity for record in records)
    panels = Counter(
        (int(identity.pass_id), str(identity.polarization).lower(), int(identity.sector_id))
        for identity in identities
    )
    is_train = str(split) == "train"
    per_panel_update_counts = (
        {f"P{key[0]}/{key[1].upper()}/{key[2]:03d}": int(value) * int(optimizer_update_count)
         for key, value in sorted(panels.items())}
        if is_train
        else None
    )
    per_record_update_counts = (
        {f"P{key[0]}/{key[1].upper()}/{key[2]:03d}": int(optimizer_update_count)
         for key in sorted(panels)}
        if is_train
        else None
    )
    coverage = {
        "unique_record_count": len(set(_identity_key(identity) for identity in identities)),
        "panel_count": len(panels),
        "records_per_panel": {
            f"P{key[0]}/{key[1].upper()}/{key[2]:03d}": int(value)
            for key, value in sorted(panels.items())
        },
        "optimizer_update_count": int(optimizer_update_count) if is_train else 0,
        "per_panel_record_update_counts": per_panel_update_counts,
        "per_panel_each_record_update_count": per_record_update_counts,
        "diagnostic_evaluation_steps": [int(value) for value in diagnostic_steps] if not is_train else [],
        "all_selected_records_covered_each_update": is_train,
        "coverage_statement": (
            "all selected TRAIN records contribute to every optimizer update; chunks are memory scheduling only"
            if str(split) == "train"
            else "validation records are evaluated only at fixed diagnostic points; no optimizer-update coverage claim"
        ),
        "spatial_frequency_chunks_are_memory_only": is_train,
    }
    if str(solver).lower() == "cgls":
        coverage.update(
            {
                "solver": "cgls",
                "max_iterations": int(max_iterations),
                "executed_iterations": int(executed_iterations),
                "selected_final_step": int(selected_final_step),
                "termination_reason": str(termination_reason),
                "cgls_all_selected_records_contributed_each_iteration": is_train,
            }
        )
    return coverage


def _canonical_id_lists(records: Sequence[Any]) -> list[list[Any]]:
    return [list(_identity_key(record.identity)) for record in records]


def _load_ragged_sidecar(path: str | Path, records: Sequence[Any]) -> ROI.RaggedComplexValues:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing native prediction sidecar: {path}")
    with np.load(path, allow_pickle=False) as payload:
        for key in ("values", "offsets", "identity_keys"):
            if key not in payload:
                raise ValueError(f"native prediction sidecar is missing {key}: {path}")
        values = np.asarray(payload["values"], dtype=np.complex128)
        offsets = np.asarray(payload["offsets"], dtype=np.int64)
        encoded_ids = np.asarray(payload["identity_keys"]).reshape(-1)
    if values.ndim != 1 or offsets.ndim != 1 or offsets.size != len(tuple(records)) + 1:
        raise ValueError(f"native prediction sidecar ragged shape is incompatible: {path}")
    if offsets[0] != 0 or offsets[-1] != values.size or np.any(np.diff(offsets) < 0):
        raise ValueError(f"native prediction sidecar offsets are incompatible: {path}")
    decoded = []
    for encoded in encoded_ids.tolist():
        if isinstance(encoded, bytes):
            encoded = encoded.decode("utf-8")
        decoded.append(tuple(json.loads(str(encoded))))
    expected = tuple(_identity_key(record.identity) for record in records)
    if tuple(decoded) != expected:
        raise ValueError(f"native prediction sidecar canonical IDs are incompatible: {path}")
    split_values = tuple(values[offsets[i] : offsets[i + 1]] for i in range(len(expected)))
    for value, record in zip(split_values, records):
        if value.shape != np.asarray(record.frequencies_hz).shape:
            raise ValueError(f"native prediction sidecar frequency shape is incompatible: {path}")
        if not np.isfinite(value.real).all() or not np.isfinite(value.imag).all():
            raise ValueError(f"native prediction sidecar contains non-finite values: {path}")
    return ROI.RaggedComplexValues(tuple(record.identity for record in records), split_values)


def _saved_native_leaf(root: str | Path, target_id: str) -> Path:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"saved-native root must be an existing directory: {root}")
    candidates = [
        root,
        root / str(target_id),
        root / str(target_id) / "attempt_001",
        root / "attempt_001",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (
            (candidate / "artifacts_final.json").is_file()
            or (candidate / "checkpoint_final.json").is_file()
            or (candidate / "predictions_final.npz").is_file()
        ):
            return candidate.resolve()
    raise ValueError(f"no launch7 native target leaf found below saved-native root: {root}")


def _metadata_compatibility(
    metadata: Mapping[str, Any],
    *,
    kind: str,
    model: Any,
    target_id: str,
    train_records: Sequence[Any],
    validation_records: Sequence[Any],
) -> list[str]:
    reasons: list[str] = []
    if metadata.get("schema") != ROI.CHECKPOINT_SCHEMA:
        reasons.append("schema/version mismatch")
    if str(kind) == "artifact":
        if metadata.get("artifact_only") is not True or metadata.get("checkpoint_name") != "final":
            reasons.append("final artifact designation mismatch")
    elif str(kind) == "checkpoint":
        if metadata.get("artifact_only") is True or metadata.get("checkpoint_name") != "final":
            reasons.append("final checkpoint designation mismatch")
    else:
        raise ValueError(f"unsupported saved-native metadata kind: {kind}")
    if metadata.get("target_id") != str(target_id):
        reasons.append("target_id mismatch")
    if int(metadata.get("step", -1)) != ROI.MAX_CGLS_ITERATIONS:
        reasons.append("selected step is not 24")
    schedule = metadata.get("schedule", {})
    if schedule.get("solver") != "cgls":
        reasons.append("saved solver is not cgls")
    if schedule.get("control_kind") != "actual_full_aperture" or schedule.get("initialization") != "zero":
        reasons.append("saved actual selector/initialization contract mismatch")
    if int(schedule.get("selected_final_step", -1)) != ROI.MAX_CGLS_ITERATIONS:
        reasons.append("saved schedule selected_final_step is not 24")
    support = metadata.get("support", {})
    if support.get("shape") != list(model.support.shape) or int(support.get("point_count", -1)) != model.point_count:
        reasons.append("support mismatch")
    placement = metadata.get("placement", {})
    if placement.get("equation") != "p_native = R @ p_local + t":
        reasons.append("placement equation/frame mismatch")
    if not np.array_equal(np.asarray(placement.get("R", []), dtype=np.float64), model.placement.rotation):
        reasons.append("placement rotation mismatch")
    if not np.array_equal(np.asarray(placement.get("t_m", []), dtype=np.float64), model.placement.translation_m):
        reasons.append("placement translation mismatch")
    if placement.get("name") != model.placement.name:
        reasons.append("placement name/frame mismatch")
    basis = metadata.get("sh_basis", {})
    if basis.get("degree") != 1 or basis.get("basis_order") != ["Y00", "Y1x", "Y1y", "Y1z"]:
        reasons.append("SH basis mismatch")
    if basis.get("convention") != ROI.SH_BASIS_CONVENTION or basis.get("coefficient_dtype") != "complex128 (real/imag field)":
        reasons.append("SH basis convention/dtype mismatch")
    source_af = metadata.get("source_af", {})
    if source_af.get("representation_tag") != ROI.SOURCE_AF_REPRESENTATION or source_af.get("formula") != ROI.SOURCE_AF_FORMULA:
        reasons.append("source-AF representation/formula mismatch")
    if source_af.get("test_payload_opened") is not False:
        reasons.append("source-AF metadata is TEST-tainted or undisclosed")
    if metadata.get("global_complex_gain") != [1.0, 0.0]:
        reasons.append("global complex gain mismatch")
    execution = metadata.get("execution_contract", {})
    if execution.get("native_geometry_dtype") != "float64" or execution.get("measurement_dtype") != "complex128":
        reasons.append("native operator dtype mismatch")
    selected = metadata.get("canonical_selected_ids", {})
    if selected.get("train") != _canonical_id_lists(train_records):
        reasons.append("TRAIN canonical IDs mismatch")
    if selected.get("validation") != _canonical_id_lists(validation_records):
        reasons.append("validation canonical IDs mismatch")
    if selected.get("identity_order") != "(pass,polarization,sector,pulse)":
        reasons.append("canonical identity ordering mismatch")
    run_provenance = metadata.get("run_provenance", {})
    if not isinstance(run_provenance, Mapping):
        reasons.append("saved run provenance is unavailable")
        run_provenance = {}
    if run_provenance.get("test_payload_opened") is not False:
        reasons.append("saved run provenance is TEST-tainted or undisclosed")
    selector_contract = run_provenance.get("selector_contract")
    if selector_contract is not None:
        expected_selector_contract = {
            "train_ids": _canonical_id_lists(train_records),
            "validation_ids": _canonical_id_lists(validation_records),
        }
        if selector_contract.get("train_ids") != expected_selector_contract["train_ids"] or selector_contract.get(
            "validation_ids"
        ) != expected_selector_contract["validation_ids"]:
            reasons.append("explicit selector provenance mismatch")
    return reasons


def _read_metadata(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"missing metadata: {path.name}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"unreadable metadata {path.name}: {exc}"
    if not isinstance(value, Mapping):
        return None, f"metadata {path.name} is not an object"
    return value, None


def _load_saved_native_arm(
    saved_native_root: str | Path,
    *,
    model: Any,
    target_id: str,
    train_records: Sequence[Any],
    validation_records: Sequence[Any],
) -> dict[str, Any]:
    """Reuse launch7 sidecars or safely fall back to a compatible checkpoint."""

    leaf = _saved_native_leaf(saved_native_root, target_id)
    rejection_reasons: list[str] = []
    artifacts_metadata, artifacts_error = _read_metadata(leaf / "artifacts_final.json")
    if artifacts_error is not None:
        rejection_reasons.append(artifacts_error)
    if artifacts_metadata is not None:
        rejection_reasons.extend(
            _metadata_compatibility(
                artifacts_metadata,
                kind="artifact",
                model=model,
                target_id=target_id,
                train_records=train_records,
                validation_records=validation_records,
            )
        )
    try:
        train_prediction = _load_ragged_sidecar(leaf / "predictions_final.npz", train_records)
        validation_prediction = _load_ragged_sidecar(leaf / "predictions_final_validation.npz", validation_records)
    except ValueError as exc:
        rejection_reasons.append(str(exc))
        train_prediction = validation_prediction = None
    if not rejection_reasons and train_prediction is not None and validation_prediction is not None:
        return {
            "native_train": train_prediction,
            "native_validation": validation_prediction,
            "source_mode": "launch7_selected_final_sidecar_reuse",
            "saved_native_leaf": str(leaf),
            "sidecar_reuse": True,
            "checkpoint_forward_fallback": False,
            "compatibility_rejection_reasons": [],
            "additional_direct_term_forecast": 0,
        }

    checkpoint_metadata, checkpoint_error = _read_metadata(leaf / "checkpoint_final.json")
    checkpoint_reasons: list[str] = []
    if checkpoint_error is not None:
        checkpoint_reasons.append(checkpoint_error)
    if checkpoint_metadata is not None:
        checkpoint_reasons.extend(
            _metadata_compatibility(
                checkpoint_metadata,
                kind="checkpoint",
                model=model,
                target_id=target_id,
                train_records=train_records,
                validation_records=validation_records,
            )
        )
    try:
        checkpoint = ROI.load_checkpoint(leaf / "checkpoint_final.npz")
    except (OSError, ValueError, KeyError) as exc:
        checkpoint_reasons.append(f"checkpoint load failed: {exc}")
        checkpoint = None
    if checkpoint is not None:
        coefficients = np.asarray(checkpoint["coefficients"], dtype=np.complex128)
        if int(checkpoint.get("step", -1)) != ROI.MAX_CGLS_ITERATIONS:
            checkpoint_reasons.append("checkpoint selected step is not 24")
        if coefficients.shape != model.coefficient_shape or not np.isfinite(coefficients.real).all() or not np.isfinite(coefficients.imag).all():
            checkpoint_reasons.append("checkpoint coefficients are incompatible")
        if checkpoint_metadata is None or checkpoint_error is not None:
            checkpoint_reasons.append("checkpoint compatibility metadata is unavailable")
    if checkpoint is None or checkpoint_reasons:
        raise ValueError(
            "saved launch7 native sidecars are incompatible and no independently compatible frozen checkpoint is available: "
            + "; ".join(dict.fromkeys(checkpoint_reasons))
        )
    coefficients = np.asarray(checkpoint["coefficients"], dtype=np.complex128)
    # This is a fresh native forward only after the support/placement/frame,
    # SH, source-AF, gain, dtype and operator-identity checks above.
    return {
        "native_train": model.forward(train_records, coefficients),
        "native_validation": model.forward(validation_records, coefficients),
        "coefficients": coefficients,
        "source_mode": "launch7_checkpoint_forward_fallback",
        "saved_native_leaf": str(leaf),
        "sidecar_reuse": False,
        "checkpoint_forward_fallback": True,
        "compatibility_rejection_reasons": list(dict.fromkeys(rejection_reasons)),
        "additional_direct_term_forecast": int(
            model.point_count * (
                sum(int(record.frequencies_hz.size) for record in train_records)
                + sum(int(record.frequencies_hz.size) for record in validation_records)
            )
        ),
    }


def _write_comparison_ragged(path: str | Path, values: ROI.RaggedComplexValues, *, domain: str) -> str:
    path = Path(path)
    payload = ROI._ragged_sidecar_payload(values)
    payload["domain"] = np.asarray([str(domain)])
    ROI._save_npz(path, **payload)
    return str(path)


def _save_comparison_coefficients(
    arm_dir: Path,
    *,
    model: Any,
    train_records: Sequence[Any],
    validation_records: Sequence[Any],
    plan: ROI.RangeSubspacePlan,
    coefficients: Any,
    step: int,
    arm_name: str,
    run_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    arm_dir.mkdir(parents=True, exist_ok=True)
    coefficients = model._validate_coefficients(coefficients)
    ROI._save_npz(arm_dir / "checkpoint_final.npz", coefficients=coefficients, step=np.asarray([int(step)], dtype=np.int64))
    restored = ROI.load_checkpoint(arm_dir / "checkpoint_final.npz")
    if not np.array_equal(restored["coefficients"], coefficients) or restored["step"] != int(step):
        raise AssertionError(f"{arm_name} checkpoint reload changed the selected coefficients")
    train_native = model.forward(train_records, restored["coefficients"])
    validation_native = model.forward(validation_records, restored["coefficients"])
    train_native_residual = ROI._range_native_residuals(train_records, train_native)
    validation_native_residual = ROI._range_native_residuals(validation_records, validation_native)
    train_range = plan.apply_forward(train_records, train_native)
    validation_range = plan.apply_forward(validation_records, validation_native)
    train_target = plan.transform_targets(train_records)
    validation_target = plan.transform_targets(validation_records)
    _write_comparison_ragged(arm_dir / "predictions_final_train.npz", train_native, domain="native_complex")
    _write_comparison_ragged(arm_dir / "predictions_final_validation.npz", validation_native, domain="native_complex")
    _write_comparison_ragged(arm_dir / "residuals_final_train.npz", train_native_residual, domain="native_complex")
    _write_comparison_ragged(arm_dir / "residuals_final_validation.npz", validation_native_residual, domain="native_complex")
    _write_comparison_ragged(arm_dir / "predictions_final.npz", train_native, domain="native_complex")
    _write_comparison_ragged(arm_dir / "residuals_final.npz", train_native_residual, domain="native_complex")
    _write_comparison_ragged(arm_dir / "range_predictions_final_train.npz", train_range, domain="B_native_train")
    _write_comparison_ragged(arm_dir / "range_predictions_final_validation.npz", validation_range, domain="B_native_validation")
    readout = model.readout(restored["coefficients"])
    diagnostics = model.diagnostics(restored["coefficients"])
    ROI._save_npz(
        arm_dir / "readout_final.npz",
        energy=readout["energy"],
        l0_real=readout["l0_real"],
        l0_imag=readout["l0_imag"],
        real_coefficient_projections=readout["real_coefficient_projections"],
    )
    ROI._save_npz(
        arm_dir / "diagnostics_final.npz",
        energy_by_voxel=diagnostics["energy_by_voxel"],
        energy_by_coefficient=diagnostics["energy_by_coefficient"],
        coefficient_l0_projection=diagnostics["coefficient_l0_projection"],
    )
    _write_json(
        arm_dir / "arm_metadata.json",
        {
            "schema": ROI.RANGE_COMPARISON_SCHEMA + ".arm",
            "arm": arm_name,
            "target_id": model.target_id,
            "step": int(step),
            "domain": "native_complex_and_B_native",
            "support": {"shape": list(model.support.shape), "point_count": model.point_count},
            "placement": ROI.placement_contract(model.placement, target_id=model.target_id),
            "source_af": {"representation": ROI.SOURCE_AF_REPRESENTATION, "formula": ROI.SOURCE_AF_FORMULA},
            "global_complex_gain": [1.0, 0.0],
            "test_payload_opened": bool(({} if run_provenance is None else run_provenance).get("test_payload_opened", False)),
            "run_provenance": {} if run_provenance is None else dict(run_provenance),
            "selected_final_reloaded_before_artifacts": True,
        },
    )
    return {
        "coefficients": restored["coefficients"],
        "native_train": train_native,
        "native_validation": validation_native,
        "range_train": train_range,
        "range_validation": validation_range,
        "train_target": train_target,
        "validation_target": validation_target,
        "checkpoint": str(arm_dir / "checkpoint_final.npz"),
        "readout": str(arm_dir / "readout_final.npz"),
        "diagnostics": str(arm_dir / "diagnostics_final.npz"),
        "selected_final_reloaded_before_artifacts": True,
    }


def _supplied_range_metrics(
    model: Any,
    records: Sequence[Any],
    native_prediction: ROI.RaggedComplexValues,
    plan: ROI.RangeSubspacePlan,
    *,
    split: str,
) -> tuple[dict[str, Any], ROI.RaggedComplexValues, ROI.RaggedComplexValues]:
    transformed_prediction = plan.apply_forward(records, native_prediction)
    target = plan.transform_targets(records)
    residual = ROI.RaggedComplexValues(
        transformed_prediction.ids,
        tuple(p - y for p, y in zip(transformed_prediction.values, target.values)),
    )
    metrics = ROI.range_metrics_from_predictions(
        model,
        records,
        native_prediction,
        transformed_prediction,
        target,
        plan,
        split=split,
    )
    return metrics, transformed_prediction, residual


def run_actual_range_subspace_comparison(
    archive_root: str | Path,
    output_dir: str | Path,
    *,
    target_id: str,
    saved_native_root: str | Path,
    microbatch_size: int = 2,
    provenance_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the explicit saved-native/BP/range-CGLS comparison arm bundle."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"comparison output directory must be fresh and not already exist: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"comparison output parent must already exist: {output.parent}")
    target_id = str(target_id)
    if target_id == "toyota_camry":
        train, validation, source_info = _select_camry_records(archive_root)
    elif target_id == "tophat":
        train, validation, source_info = _select_tophat_records(archive_root)
    else:
        raise ValueError(f"unsupported actual target_id: {target_id}")
    placement = ROI.TOPHAT_PROVISIONAL_PLACEMENT if target_id == "tophat" else None
    model = ROI.NativeComplexDirectionalSHROIModel(target_id, placement=placement)
    ROI.validate_record_set(train, split="train", target_id=target_id)
    ROI.validate_record_set(validation, split="validation", target_id=target_id)
    # The projector/retention gate is completed before any arm does an actual
    # fit or re-renders a saved checkpoint.
    plan = ROI.build_range_subspace_plan(model, tuple(train) + tuple(validation), require_retention=True)
    saved = _load_saved_native_arm(
        saved_native_root,
        model=model,
        target_id=target_id,
        train_records=train,
        validation_records=validation,
    )
    output.mkdir(parents=True, exist_ok=False)
    active_provenance = {
        "mode": "actual_range_subspace_comparison",
        "source_kind": "trusted_native_shard_source_af_records",
        "archive_opened": True,
        "test_payload_opened": False,
        "archive_root": str(Path(archive_root).expanduser().resolve()),
        "target_id": target_id,
        "saved_native_root": str(Path(saved_native_root).expanduser().resolve()),
        "saved_native_source_mode": saved["source_mode"],
        "range_operator_schema": ROI.RANGE_SUBSPACE_SCHEMA,
    }
    if provenance_overrides is not None:
        active_provenance.update(dict(provenance_overrides))
    archive_opened = bool(active_provenance.get("archive_opened", False))
    test_payload_opened = bool(active_provenance.get("test_payload_opened", False))
    active_provenance["selector_contract"] = {
        "train_ids": _canonical_id_lists(train),
        "validation_ids": _canonical_id_lists(validation),
    }

    saved_dir = output / "saved_native_objective_cgls24"
    saved_metrics_train, saved_range_train, saved_residual_train = _supplied_range_metrics(
        model, train, saved["native_train"], plan, split="train"
    )
    saved_metrics_validation, saved_range_validation, saved_residual_validation = _supplied_range_metrics(
        model, validation, saved["native_validation"], plan, split="validation"
    )
    saved_dir.mkdir(parents=True)
    saved_native_residual_train = ROI._range_native_residuals(train, saved["native_train"])
    saved_native_residual_validation = ROI._range_native_residuals(validation, saved["native_validation"])
    _write_comparison_ragged(saved_dir / "predictions_final_train.npz", saved["native_train"], domain="native_complex")
    _write_comparison_ragged(saved_dir / "predictions_final_validation.npz", saved["native_validation"], domain="native_complex")
    _write_comparison_ragged(saved_dir / "residuals_final_train.npz", saved_native_residual_train, domain="native_complex")
    _write_comparison_ragged(saved_dir / "residuals_final_validation.npz", saved_native_residual_validation, domain="native_complex")
    _write_comparison_ragged(saved_dir / "predictions_final.npz", saved["native_train"], domain="native_complex")
    _write_comparison_ragged(saved_dir / "residuals_final.npz", saved_native_residual_train, domain="native_complex")
    _write_comparison_ragged(saved_dir / "range_predictions_final_train.npz", saved_range_train, domain="B_native_train")
    _write_comparison_ragged(saved_dir / "range_predictions_final_validation.npz", saved_range_validation, domain="B_native_validation")
    _write_comparison_ragged(saved_dir / "range_residuals_final_train.npz", saved_residual_train, domain="B_native_train")
    _write_comparison_ragged(saved_dir / "range_residuals_final_validation.npz", saved_residual_validation, domain="B_native_validation")
    _write_json(
        saved_dir / "arm_metadata.json",
        {
            "schema": ROI.RANGE_COMPARISON_SCHEMA + ".arm",
            "arm": "saved_native_objective_cgls24",
            "target_id": target_id,
            "step": ROI.MAX_CGLS_ITERATIONS,
            "domain": "native_complex_and_B_native",
            "source_mode": saved["source_mode"],
            "saved_native_leaf": saved["saved_native_leaf"],
            "sidecar_reuse": bool(saved["sidecar_reuse"]),
            "checkpoint_forward_fallback": bool(saved["checkpoint_forward_fallback"]),
            "support": {"shape": list(model.support.shape), "point_count": model.point_count},
            "placement": ROI.placement_contract(model.placement, target_id=target_id),
            "source_af": {"representation": ROI.SOURCE_AF_REPRESENTATION, "formula": ROI.SOURCE_AF_FORMULA},
            "global_complex_gain": [1.0, 0.0],
            "test_payload_opened": test_payload_opened,
            "run_provenance": active_provenance,
        },
    )

    bp = ROI.range_isotropic_bp(model, train, validation, plan)
    bp_artifacts = _save_comparison_coefficients(
        output / "range_focused_isotropic_bp",
        model=model,
        train_records=train,
        validation_records=validation,
        plan=plan,
        coefficients=bp["coefficients"],
        step=0,
        arm_name="range_focused_isotropic_bp",
        run_provenance=active_provenance,
    )
    bp_metrics_train, bp_range_train, bp_residual_train = _supplied_range_metrics(
        model, train, bp_artifacts["native_train"], plan, split="train"
    )
    bp_metrics_validation, bp_range_validation, bp_residual_validation = _supplied_range_metrics(
        model, validation, bp_artifacts["native_validation"], plan, split="validation"
    )
    _write_comparison_ragged(output / "range_focused_isotropic_bp" / "range_residuals_final_train.npz", bp_residual_train, domain="B_native_train")
    _write_comparison_ragged(output / "range_focused_isotropic_bp" / "range_residuals_final_validation.npz", bp_residual_validation, domain="B_native_validation")

    cgls_dir = output / "range_focused_degree1_cgls24"
    cgls_result = ROI.fit_range_focused_cgls24(
        model,
        train,
        validation,
        plan,
        output_dir=cgls_dir,
        run_provenance=active_provenance,
    )
    cgls_state = ROI.load_checkpoint(cgls_dir / "checkpoint_final.npz")
    if cgls_state["step"] != int(cgls_result.selected_final_step):
        raise AssertionError("range CGLS reload step differs from selected terminal step")
    cgls_native_train = _load_ragged_sidecar(cgls_dir / "predictions_final.npz", train)
    cgls_native_validation = _load_ragged_sidecar(cgls_dir / "predictions_final_validation.npz", validation)
    cgls_metrics_train, cgls_range_train, cgls_residual_train = _supplied_range_metrics(
        model, train, cgls_native_train, plan, split="train"
    )
    cgls_metrics_validation, cgls_range_validation, cgls_residual_validation = _supplied_range_metrics(
        model, validation, cgls_native_validation, plan, split="validation"
    )
    _write_comparison_ragged(cgls_dir / "predictions_final.npz", cgls_native_train, domain="native_complex")
    _write_comparison_ragged(cgls_dir / "predictions_final_validation.npz", cgls_native_validation, domain="native_complex")
    _write_comparison_ragged(cgls_dir / "residuals_final.npz", ROI._range_native_residuals(train, cgls_native_train), domain="native_complex")
    _write_comparison_ragged(cgls_dir / "residuals_final_validation.npz", ROI._range_native_residuals(validation, cgls_native_validation), domain="native_complex")
    _write_comparison_ragged(cgls_dir / "range_residuals_final_train.npz", cgls_residual_train, domain="B_native_train")
    _write_comparison_ragged(cgls_dir / "range_residuals_final_validation.npz", cgls_residual_validation, domain="B_native_validation")
    cgls_provenance = {
        **active_provenance,
        "solver": cgls_result.solver,
        "max_iterations": int(cgls_result.max_iterations),
        "executed_iterations": int(cgls_result.executed_iterations),
        "selected_final_step": int(cgls_result.selected_final_step),
        "termination_reason": cgls_result.termination_reason,
        "diagnostic_steps": list(cgls_result.diagnostic_steps),
    }
    active_provenance.update(cgls_provenance)
    _write_json(
        cgls_dir / "arm_metadata.json",
        {
            "schema": ROI.RANGE_COMPARISON_SCHEMA + ".arm",
            "arm": "range_focused_degree1_cgls24",
            "target_id": target_id,
            "step": int(cgls_result.selected_final_step),
            "domain": "native_complex_and_B_native",
            "solver": cgls_result.solver,
            "support": {"shape": list(model.support.shape), "point_count": model.point_count},
            "placement": ROI.placement_contract(model.placement, target_id=target_id),
            "source_af": {"representation": ROI.SOURCE_AF_REPRESENTATION, "formula": ROI.SOURCE_AF_FORMULA},
            "global_complex_gain": [1.0, 0.0],
            "test_payload_opened": test_payload_opened,
            "run_provenance": cgls_provenance,
            "selected_final_reloaded_before_artifacts": True,
        },
    )

    cgls_breakdown = cgls_result.termination_reason == "nonpositive_search_denominator_breakdown"
    ledger = ROI.range_comparison_resource_ledger(
        model,
        source_info["train_frequency_counts"],
        source_info["validation_frequency_counts"],
        plan,
        saved_native_reuse=bool(saved["sidecar_reuse"]),
        saved_native_fallback=bool(saved["checkpoint_forward_fallback"]),
        actual_iterations=int(cgls_result.executed_iterations),
        diagnostic_steps=cgls_result.diagnostic_steps,
        cgls_breakdown=cgls_breakdown,
    )
    ledger["source_selection_contract"] = {
        "train_record_count": int(source_info["train_record_count"]),
        "train_frequency_sample_count": int(source_info["train_frequency_sample_count"]),
        "validation_record_count": int(source_info["validation_record_count"]),
        "validation_frequency_sample_count": int(source_info["validation_frequency_sample_count"]),
        "train_selector": source_info["train_selector"],
        "validation_selector": source_info["validation_selector"],
        "train_panel_contract": source_info.get("train_panel_contract"),
        "validation_panel_contract": source_info.get("validation_panel_contract"),
    }
    programmatic_success = not cgls_breakdown
    report = {
        "schema": ROI.RANGE_COMPARISON_SCHEMA,
        "mode": "actual_range_subspace_comparison",
        "status": (
            "RANGE_CGLS_NONPOSITIVE_SEARCH_DENOMINATOR_BREAKDOWN_FORENSIC_ONLY"
            if cgls_breakdown
            else "COMPLETED_EXPLICIT_RANGE_SUBSPACE_THREE_ARM_COMPARISON"
        ),
        "programmatic_success": programmatic_success,
        "forensic_only": cgls_breakdown,
        "target_id": target_id,
        "archive_opened": archive_opened,
        "test_payload_opened": test_payload_opened,
        "pace_action": False,
        "manager_touched": False,
        "deployment": "none",
        "run_provenance": active_provenance,
        "source": source_info,
        "placement": ROI.placement_contract(model.placement, target_id=target_id),
        "support": {
            "point_count": model.point_count,
            "shape": list(model.support.shape),
            "cell_size_m": model.support.cell_size_m,
            "readout_label": ROI.READOUT_LABEL,
        },
        "source_af": {
            "representation": ROI.SOURCE_AF_REPRESENTATION,
            "formula": ROI.SOURCE_AF_FORMULA,
            "raw_payload_treatment": "native raw r0/response retained and converted exactly once after metadata/header/role/identity gates",
        },
        "global_complex_gain": {
            "value": [1.0, 0.0],
            "fixed": True,
            "per_record_gains": False,
            "calibration_or_autofocus_retune": False,
        },
        "range_operator": plan.metadata(),
        "N_T": {
            "train": int(plan.normalization(train)),
            "validation": int(plan.normalization(validation)),
            "definition": "sum M over the split after rank(E)=M; native K is reported separately",
        },
        "arms": {
            "saved_native_objective_cgls24": {
                "source_mode": saved["source_mode"],
                "saved_native_leaf": saved["saved_native_leaf"],
                "sidecar_reuse": saved["sidecar_reuse"],
                "checkpoint_forward_fallback": saved["checkpoint_forward_fallback"],
                "compatibility_rejection_reasons": saved["compatibility_rejection_reasons"],
                "additional_direct_term_forecast": saved["additional_direct_term_forecast"],
                "native_artifacts": [
                    "saved_native_objective_cgls24/predictions_final_train.npz",
                    "saved_native_objective_cgls24/predictions_final_validation.npz",
                    "saved_native_objective_cgls24/residuals_final_train.npz",
                    "saved_native_objective_cgls24/residuals_final_validation.npz",
                    "saved_native_objective_cgls24/arm_metadata.json",
                ],
                "train": saved_metrics_train,
                "validation": saved_metrics_validation,
                "range_artifacts": [
                    "saved_native_objective_cgls24/range_predictions_final_train.npz",
                    "saved_native_objective_cgls24/range_predictions_final_validation.npz",
                    "saved_native_objective_cgls24/range_residuals_final_train.npz",
                    "saved_native_objective_cgls24/range_residuals_final_validation.npz",
                ],
            },
            "range_focused_isotropic_bp": {
                **{key: value for key, value in bp.items() if key != "coefficients" and key != "unscaled_isotropic_coefficients"},
                "train": bp_metrics_train,
                "validation": bp_metrics_validation,
                "checkpoint": bp_artifacts["checkpoint"],
                "readout": bp_artifacts["readout"],
                "diagnostics": bp_artifacts["diagnostics"],
                "native_artifacts": [
                    "range_focused_isotropic_bp/predictions_final.npz",
                    "range_focused_isotropic_bp/predictions_final_validation.npz",
                    "range_focused_isotropic_bp/residuals_final.npz",
                    "range_focused_isotropic_bp/residuals_final_validation.npz",
                    "range_focused_isotropic_bp/arm_metadata.json",
                ],
                "selected_final_reloaded_before_artifacts": bp_artifacts["selected_final_reloaded_before_artifacts"],
            },
            "range_focused_degree1_cgls24": {
                "train": cgls_metrics_train,
                "validation": cgls_metrics_validation,
                "solver": cgls_result.solver,
                "max_iterations": cgls_result.max_iterations,
                "executed_iterations": cgls_result.executed_iterations,
                "selected_final_step": cgls_result.selected_final_step,
                "termination_reason": cgls_result.termination_reason,
                "diagnostic_steps": list(cgls_result.diagnostic_steps),
                "checkpoint": str(cgls_dir / "checkpoint_final.npz"),
                "readout": str(cgls_dir / "readout_final.npz"),
                "diagnostics": str(cgls_dir / "diagnostics_final.npz"),
                "native_artifacts": [
                    "range_focused_degree1_cgls24/predictions_final.npz",
                    "range_focused_degree1_cgls24/predictions_final_validation.npz",
                    "range_focused_degree1_cgls24/residuals_final.npz",
                    "range_focused_degree1_cgls24/residuals_final_validation.npz",
                    "range_focused_degree1_cgls24/arm_metadata.json",
                ],
                "selected_final_reloaded_before_artifacts": True,
                "forensic_only": cgls_breakdown,
            },
        },
        "resource_ledger": ledger,
        "scope_statement": (
            "The range projector is a geometry-derived complex-linear observable, not exact target isolation, calibration proof, or full manuscript RIFT. "
            "A CGLS denominator breakdown is retained for forensic diagnostics only."
        ),
    }
    _write_json(output / "resource_ledger.json", ledger)
    _write_json(output / "comparison_report.json", report)
    protocol = dict(ROI.protocol_payload())
    protocol.update(
        {
            "mode": "actual_range_subspace_comparison",
            "target_id": target_id,
            "fit_release_status": report["status"],
            "actual_archive_opened": archive_opened,
            "programmatic_success": programmatic_success,
            "forensic_only": cgls_breakdown,
            "scope_statement": report["scope_statement"],
            "source": source_info,
            "placement": report["placement"],
            "source_af_provenance": report["source_af"],
            "global_complex_gain_contract": report["global_complex_gain"],
            "range_subspace_operator": report["range_operator"],
            "comparison_arms": report["arms"],
            "resource_ledger": ledger,
            "actual_run": {
                "mode": "actual_range_subspace_comparison",
                "target_id": target_id,
                "archive_opened": archive_opened,
                "test_payload_opened": test_payload_opened,
                "saved_native_root": str(Path(saved_native_root).expanduser().resolve()),
                "validation_selector_bound": True,
            },
        }
    )
    _write_json(output / "protocol_payload.json", protocol)
    return report


run_actual_comparison = run_actual_range_subspace_comparison


def run_actual(
    archive_root: str | Path,
    output_dir: str | Path,
    *,
    target_id: str,
    updates: int,
    microbatch_size: int,
    solver: str = "gd",
    provenance_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run actual mode; optional provenance overrides are only for injected local tests."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"actual output directory must be fresh and not already exist: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"actual output parent must already exist: {output.parent}")
    target_id = str(target_id)
    solver = str(solver).lower()
    if solver == "cgls" and int(updates) != ROI.MAX_CGLS_ITERATIONS:
        raise ValueError(
            f"actual archive CGLS route requires exactly {ROI.MAX_CGLS_ITERATIONS} updates; "
            "smaller schedules are reserved for solver-unit tests"
        )
    if target_id == "toyota_camry":
        train, validation, source_info = _select_camry_records(archive_root)
    elif target_id == "tophat":
        train, validation, source_info = _select_tophat_records(archive_root)
    else:
        raise ValueError(f"unsupported actual target_id: {target_id}")
    placement = ROI.TOPHAT_PROVISIONAL_PLACEMENT if target_id == "tophat" else None
    model = ROI.NativeComplexDirectionalSHROIModel(target_id, placement=placement)
    ROI.validate_record_set(train, split="train", target_id=target_id)
    ROI.validate_record_set(validation, split="validation", target_id=target_id)
    config = ROI.actual_fit_config(
        model,
        updates=int(updates),
        microbatch_size=int(microbatch_size),
        solver=solver,
    )
    provenance = {
        "mode": "actual_archive",
        "source_kind": "trusted_native_shard_source_af_records",
        "archive_opened": True,
        "test_payload_opened": False,
        "archive_root": str(Path(archive_root).expanduser().resolve()),
        "target_id": target_id,
        "shards": source_info["shards"],
        "source_af_conversion_batches": source_info["source_af_conversion_batches"],
        "source_af_exactly_once_per_selected_record": source_info["source_af_exactly_once_per_selected_record"],
        "panel_wise_image_pooling": bool(source_info.get("panel_wise_image_pooling", False)),
        "solver": solver,
        "max_iterations": int(config.updates),
    }
    if provenance_overrides is not None:
        provenance.update(dict(provenance_overrides))
    archive_opened = bool(provenance.get("archive_opened", False))
    test_payload_opened = bool(provenance.get("test_payload_opened", False))
    result = ROI.fit_full_aperture_schedule(
        model,
        train,
        validation,
        config=config,
        output_dir=output,
        run_provenance=provenance,
    )
    selected_state = ROI.load_checkpoint(output / "checkpoint_final.npz")
    if not np.array_equal(selected_state["coefficients"], result.coefficients_selected):
        raise AssertionError("reloaded selected-final state differs from fit result")
    max_iterations = int(config.updates)
    executed_iterations = int(
        max_iterations if result.executed_iterations is None else result.executed_iterations
    )
    selected_final_step = int(
        executed_iterations if result.selected_final_step is None else result.selected_final_step
    )
    termination_reason = str(result.termination_reason)
    diagnostic_steps = tuple(
        int(value) for value in (result.diagnostic_steps or tuple(config.validation_steps))
    )
    provenance.update(
        {
            "solver": solver,
            "max_iterations": max_iterations,
            "executed_iterations": executed_iterations,
            "selected_final_step": selected_final_step,
            "termination_reason": termination_reason,
            "diagnostic_steps": list(diagnostic_steps),
        }
    )
    ledger = ROI.direct_term_resource_ledger(
        model,
        source_info["train_frequency_counts"],
        source_info["validation_frequency_counts"],
        updates=max_iterations,
        validation_steps=diagnostic_steps,
        solver=solver,
        actual_iterations=executed_iterations if solver == "cgls" else None,
        termination_reason=termination_reason,
    )
    cgls_breakdown = (
        solver == "cgls" and termination_reason == "nonpositive_search_denominator_breakdown"
    )
    if solver == "cgls":
        ledger["run_outcome"] = {
            "programmatic_success": not cgls_breakdown,
            "termination_reason": termination_reason,
            "max_iterations": max_iterations,
            "executed_iterations": executed_iterations,
            "selected_final_step": selected_final_step,
            "forensic_only": cgls_breakdown,
        }
    if int(selected_state["step"]) != selected_final_step:
        raise AssertionError("reloaded selected-final state differs from result.selected_final_step")
    ledger["source_selection_contract"] = {
        "train_record_count": int(source_info["train_record_count"]),
        "train_frequency_sample_count": int(source_info["train_frequency_sample_count"]),
        "validation_record_count": int(source_info["validation_record_count"]),
        "validation_frequency_sample_count": int(source_info["validation_frequency_sample_count"]),
        "train_selector": source_info["train_selector"],
        "validation_selector": source_info["validation_selector"],
        "train_panel_contract": source_info.get("train_panel_contract"),
        "validation_panel_contract": source_info.get("validation_panel_contract"),
        "source_af_conversion_batches": int(source_info["source_af_conversion_batches"]),
        "source_af_converted_record_count": int(source_info["source_af_converted_record_count"]),
        "source_af_exactly_once_per_selected_record": bool(
            source_info["source_af_exactly_once_per_selected_record"]
        ),
        "source_af_selected_rows_provenance": source_info.get("source_af_selected_rows_provenance"),
    }
    figure_paths = ROI.render_fit_figures(model, selected_state["coefficients"], output)
    report = {
        "schema": ROI.SCHEMA,
        "status": (
            "CGLS_NONPOSITIVE_SEARCH_DENOMINATOR_BREAKDOWN_FORENSIC_ONLY"
            if cgls_breakdown
            else "COMPLETED_SCOPED_NUMERICAL_FIT_NO_RECOVERY_REGISTRATION_AF_VALIDATION_OR_PHYSICAL_GEOMETRY_CLAIM"
        ),
        "programmatic_success": not cgls_breakdown,
        "forensic_only": cgls_breakdown,
        "target_id": target_id,
        "solver": solver,
        "max_iterations": max_iterations,
        "executed_iterations": executed_iterations,
        "selected_final_step": selected_final_step,
        "termination_reason": termination_reason,
        "archive_opened": archive_opened,
        "test_payload_opened": test_payload_opened,
        "run_provenance": provenance,
        "pace_action": False,
        "manager_touched": False,
        "deployment": "none",
        "source": source_info,
        "coverage": _coverage(
            train,
            split="train",
            optimizer_update_count=executed_iterations,
            diagnostic_steps=diagnostic_steps,
            solver=solver,
            max_iterations=max_iterations,
            executed_iterations=executed_iterations,
            selected_final_step=selected_final_step,
            termination_reason=termination_reason,
        ),
        "validation_coverage": _coverage(
            validation,
            split="validation",
            optimizer_update_count=0,
            diagnostic_steps=diagnostic_steps,
            solver=solver,
            max_iterations=max_iterations,
            executed_iterations=executed_iterations,
            selected_final_step=selected_final_step,
            termination_reason=termination_reason,
        ),
        "support": {
            "point_count": model.point_count,
            "shape": list(model.support.shape),
            "cell_size_m": model.support.cell_size_m,
            "readout_label": ROI.READOUT_LABEL,
        },
        "placement": ROI.placement_contract(
            model.placement,
            target_id=target_id,
        ),
        "declared_placements": ROI.protocol_payload()["placements"],
        "source_af": {
            "representation": ROI.SOURCE_AF_REPRESENTATION,
            "formula": ROI.SOURCE_AF_FORMULA,
            "raw_payload_treatment": "native raw r0/response retained and converted exactly once after metadata/header/role/identity gates",
            "selected_rows_provenance": source_info.get("source_af_selected_rows_provenance"),
        },
        "global_complex_gain": {
            "value": [1.0, 0.0],
            "fixed": True,
            "per_record_gains": False,
            "calibration_or_autofocus_retune": False,
        },
        "fit_config": {
            "solver": solver,
            "updates": config.updates,
            "control_kind": config.control_kind,
            "initialization": config.initialization,
            "step_size": config.step_size,
            "checkpoint_steps": list(config.checkpoint_steps),
            "validation_steps": list(config.validation_steps),
            "record_microbatch_size_argument": int(microbatch_size),
            "record_microbatch_size_used": False,
            "step_size_used": solver == "gd",
            "cgls_max_iterations": max_iterations if solver == "cgls" else None,
            "cgls_passes_per_iteration": 3 if solver == "cgls" else None,
            "array_memory_chunks": {
                "spatial_chunk_size": model.spatial_chunk_size,
                "frequency_chunk_size": model.frequency_chunk_size,
            },
        },
        "scope_statement": (
            "CGLS stopped at a nonpositive search denominator; the selected terminal state is retained for forensic diagnostics only; "
            "no successful fit, target recovery, survey registration, autofocus validation, or physical geometry is claimed."
            if cgls_breakdown
            else (
                "Completed bounded local numerical fit on the explicitly selected archive TRAIN records; "
                "does not claim target recovery, survey registration, autofocus validation, or physical geometry."
                if archive_opened
                else "Completed bounded local numerical fit with injected synthetic selector records for orchestration testing; "
                "no archive was opened and no target recovery, survey registration, autofocus validation, or physical geometry is claimed."
            )
        ),
        "ledger": ledger,
        "metrics": {
            "train": ROI.residual_metrics(model, train, selected_state["coefficients"], split="train"),
            "validation": ROI.residual_metrics(model, validation, selected_state["coefficients"], split="validation"),
        },
        "selected_model": (
            "checkpoint_final_cgls_terminal_step_" + str(selected_final_step)
            if solver == "cgls"
            else "checkpoint_final_update_" + str(config.updates)
        ),
        "selected_final_reloaded_before_artifacts": True,
        "figures": figure_paths,
        "readout_arrays": ["readout_final.npz", "readout_final_validation.npz"],
        "exterior_returns": "retained in native residual; no crop/subtract/pad",
        "gain_af_calibration_retune": False,
    }
    _write_json(output / "resource_ledger.json", ledger)
    _write_json(output / "actual_fit_report.json", report)
    template_protocol = ROI.protocol_payload()
    protocol = dict(template_protocol)
    protocol["mode"] = "actual_archive"
    protocol["fit_release_status"] = report["status"]
    protocol["actual_archive_opened"] = archive_opened
    protocol["target_id"] = target_id
    protocol["programmatic_success"] = report["programmatic_success"]
    protocol["forensic_only"] = report["forensic_only"]
    protocol["solver"] = solver
    protocol["max_iterations"] = max_iterations
    protocol["executed_iterations"] = executed_iterations
    protocol["selected_final_step"] = selected_final_step
    protocol["termination_reason"] = termination_reason
    protocol["scope_statement"] = report["scope_statement"]
    protocol["actual_fit_config"] = report["fit_config"]
    protocol["source"] = source_info
    protocol["resource_ledger"] = ledger
    protocol["placement"] = report["placement"]
    protocol["source_af_provenance"] = report["source_af"]
    protocol["global_complex_gain_contract"] = report["global_complex_gain"]
    actual_schedule = {
        **dict(template_protocol["schedule"]),
        "deterministic_adjoint_initial_seed": False,
        "initialization": "zero",
        "adjoint_seed_used": False,
        "fixed_checkpoint_and_validation_points": list(config.checkpoint_steps),
        "persistence_points_only": list(config.checkpoint_steps),
        "selected_model": (
            f"cgls_terminal_step_{selected_final_step}_final_for_train_and_validation"
            if solver == "cgls"
            else f"update_{config.updates}_final_for_train_and_validation"
        ),
        "resolved_updates": int(config.updates),
        "solver": solver,
        "max_iterations": max_iterations,
        "executed_iterations": executed_iterations,
        "selected_final_step": selected_final_step,
        "termination_reason": termination_reason,
        "programmatic_success": report["programmatic_success"],
        "forensic_only": report["forensic_only"],
        "record_microbatch_size_argument": int(microbatch_size),
        "record_microbatch_size_used": False,
        "array_memory_chunks": {
            "spatial_chunk_size": model.spatial_chunk_size,
            "frequency_chunk_size": model.frequency_chunk_size,
        },
    }
    if solver == "cgls":
        actual_schedule["persistence_points_only"] = list(diagnostic_steps)
        actual_schedule.update(
            {
                "actual_optimizer_schedule": "matrix-free complex CGLS over all selected TRAIN records; zero initialization; no extra factor two; no denominator floor",
                "cgls_passes_per_iteration": 3,
                "cgls_initial_normal_residual_passes": 2,
                "cgls_diagnostic_steps": list(diagnostic_steps),
            }
        )
    actual_schedule.pop("minimum_actual_optimizer_updates", None)
    protocol["schedule"] = actual_schedule
    protocol["fake_control_template"] = {
        "fit_release_status": template_protocol["fit_release_status"],
        "actual_archive_opened": template_protocol["actual_archive_opened"],
        "test_payload_opened": template_protocol["test_payload_opened"],
        "scope_statement": template_protocol.get(
            "scope_statement", "Local fake-record control only; archive and PACE remain unopened."
        ),
    }
    protocol["actual_run"] = {
        "archive_opened": archive_opened,
        "test_payload_opened": test_payload_opened,
        "target_id": target_id,
        "source_kind": provenance.get("source_kind"),
        "solver": solver,
        "max_iterations": max_iterations,
        "executed_iterations": executed_iterations,
        "selected_final_step": selected_final_step,
        "termination_reason": termination_reason,
        "source": source_info,
        "resource_class": ledger["resource_estimate"]["class"],
        "validation_selector_bound": True,
    }
    _write_json(output / "protocol_payload.json", protocol)
    return report


def run_fake(args: argparse.Namespace) -> dict[str, Any]:
    if str(getattr(args, "solver", "gd")).lower() != "gd":
        raise ValueError("fake mode supports only the existing gd control")
    if args.updates != 12:
        raise ValueError("fake control fixes its exact 12-update schedule at 0/4/8/12")
    if args.target_id == "toyota_camry":
        model, train, validation, planted = ROI.make_fake_camry_control_case(
            train_count=args.train_count,
            validation_count=args.validation_count,
            frequency_count=args.frequency_count,
        )
    else:
        model, train, planted = ROI.make_fake_tophat_control_case(
            records_per_panel=1,
            frequency_count=args.frequency_count,
        )
        validation = None
    result = ROI.fit_fixed_schedule(
        model,
        train,
        validation,
        config=ROI.FitConfig(updates=12),
        output_dir=args.output_dir,
    )
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(args.output_dir / "protocol_payload.json", ROI.protocol_payload())
    return {
        "schema": ROI.SCHEMA,
        "status": "PASS_LOCAL_FAKE_TWO_TARGET_FIXED_SUPPORT_CONTROL",
        "target_id": model.target_id,
        "train_records": len(train),
        "validation_records": 0 if validation is None else len(validation),
        "support_points": model.point_count,
        "actual_optimizer_updates": 12,
        "history_points": len(result.history),
        "checkpoint_paths_written": len(result.checkpoint_paths),
        "archive_opened": False,
        "pace_action": False,
        "manager_touched": False,
        "planted_coefficients_used_only_for_fake_targets": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the separate fake control or actual archive-bound ROI fit.")
    parser.add_argument(
        "--mode",
        choices=("fake", "actual", "actual_range_comparison", "range_comparison", "actual-comparison"),
        required=True,
        help="actual_range_comparison is the explicit saved-native/BP/range-CGLS three-arm mode",
    )
    parser.add_argument("--target-id", choices=("toyota_camry", "tophat"), default="toyota_camry")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="actual mode: archive parent or exact converted_v3_joint8_fullpol/shards directory",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="required and fresh for actual mode")
    parser.add_argument(
        "--saved-native-root",
        type=Path,
        default=None,
        help="range-comparison mode: launch7 target leaf or its parent for saved native final sidecars/checkpoint",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=None,
        help="bounded iterations; defaults to 12 for GD/fake and 24 for CGLS actual mode",
    )
    parser.add_argument(
        "--solver",
        choices=("gd", "cgls"),
        default="gd",
        help="actual solver: existing normalized GD (default) or optional matrix-free complex CGLS (max 24 iterations)",
    )
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=2,
        help="record microbatch argument accepted for compatibility; ignored in actual full-aperture mode; spatial/frequency chunks control array memory",
    )
    parser.add_argument("--train-count", type=int, default=4, help="fake Camry records only")
    parser.add_argument("--validation-count", type=int, default=2, help="fake Camry records only")
    parser.add_argument("--frequency-count", type=int, default=8, help="fake records only")
    args = parser.parse_args(argv)
    if args.updates is None:
        args.updates = (
            ROI.MAX_CGLS_ITERATIONS
            if args.solver == "cgls" or args.mode in {"actual_range_comparison", "range_comparison", "actual-comparison"}
            else 12
        )
    if args.mode == "fake":
        if args.archive_root is not None:
            parser.error("fake mode cannot accept --archive-root")
        report = run_fake(args)
    elif args.mode == "actual":
        if args.archive_root is None or args.output_dir is None:
            parser.error("actual mode requires both --archive-root and a fresh --output-dir")
        report = run_actual(
            args.archive_root,
            args.output_dir,
            target_id=args.target_id,
            updates=args.updates,
            microbatch_size=args.microbatch_size,
            solver=args.solver,
        )
    else:
        if args.archive_root is None or args.output_dir is None or args.saved_native_root is None:
            parser.error(
                "actual_range_comparison mode requires --archive-root, --output-dir, and --saved-native-root"
            )
        if args.updates not in (24,):
            parser.error("actual_range_comparison mode has a fixed 24-iteration range-CGLS arm")
        report = run_actual_range_subspace_comparison(
            args.archive_root,
            args.output_dir,
            target_id=args.target_id,
            saved_native_root=args.saved_native_root,
            microbatch_size=args.microbatch_size,
        )
    print(json.dumps(_json_ready(report), sort_keys=True))
    return 0 if report.get("programmatic_success", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
