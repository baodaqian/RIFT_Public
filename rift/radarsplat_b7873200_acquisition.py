"""Direct, role-limited acquisition record for the RadarSplat B7873200 cache.

The cache needs enough calibrated geometry to re-derive each stored target,
but the development lane must not materialize calibration rows for the sealed
test or unused views. This module therefore persists direct arrays for the
authorized train/validation IDs only. It never stores a response sample.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from rift.radarsplat_b7873200_protocol import (
    ACQUISITION_FILENAME,
    ACQUISITION_SCHEMA,
    B787_3200_NUM_VIEWS,
    atomic_save_npz,
)


_FIELDS = frozenset(
    {
        "schema",
        "view_indices",
        "frequency_hz",
        "viewpoint_positions",
        "tx_pos",
        "rx_pos",
        "scene_center_m",
        "response_shape",
        "response_dtype",
        "metadata_json",
    }
)


def _metadata_text(metadata: Mapping[str, object]) -> str:
    try:
        return json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("RadarSplat B7873200 acquisition metadata must be JSON serializable") from error


def acquisition_payload(
    *,
    view_indices: Sequence[int] | np.ndarray,
    frequency_hz: np.ndarray,
    viewpoint_positions: np.ndarray,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    scene_center_m: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    metadata: Mapping[str, object],
    response_shape: tuple[int, ...],
    response_dtype: str,
) -> dict[str, np.ndarray]:
    """Validate direct authorized-view calibration and return a serializable record."""

    indices = np.asarray(view_indices, dtype=np.int64)
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    viewpoints = np.asarray(viewpoint_positions, dtype=np.float64)
    tx = np.asarray(tx_pos, dtype=np.float64)
    rx = np.asarray(rx_pos, dtype=np.float64)
    center = np.asarray(scene_center_m, dtype=np.float64)
    if indices.ndim != 1 or indices.size < 1 or len(set(int(value) for value in indices.tolist())) != indices.size:
        raise ValueError("RadarSplat B7873200 acquisition needs unique authorized view IDs")
    if np.any(indices < 0) or np.any(indices >= B787_3200_NUM_VIEWS):
        raise ValueError("RadarSplat B7873200 acquisition has out-of-range authorized view IDs")
    if frequency.shape != (600,) or not np.isfinite(frequency).all() or np.any(np.diff(frequency) <= 0.0):
        raise ValueError("RadarSplat B7873200 acquisition needs 600 increasing finite frequencies")
    if viewpoints.shape != (indices.size, 3) or not np.isfinite(viewpoints).all():
        raise ValueError("RadarSplat B7873200 acquisition has invalid authorized viewpoint positions")
    from .antenna_selection import validate_selection
    acquisition = validate_selection(metadata.get('rift_antenna_selection'))
    nt, nr = (acquisition['num_tx'], acquisition['num_rx']) if acquisition else (16, 16)
    if tx.shape != (indices.size, nt, 3) or rx.shape != (indices.size, nr, 3):
        raise ValueError("RadarSplat B7873200 acquisition requires the declared authorized calibrated Tx/Rx poses")
    if not np.isfinite(tx).all() or not np.isfinite(rx).all() or center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("RadarSplat B7873200 acquisition has non-finite calibrated coordinates")
    if tuple(int(value) for value in response_shape) != (B787_3200_NUM_VIEWS, nt, nr, 1, 600):
        raise ValueError("RadarSplat B7873200 acquisition has the wrong complete archive response shape")
    if str(response_dtype) != "complex64":
        raise ValueError("RadarSplat B7873200 acquisition requires complex64 responses")
    return {
        "schema": np.asarray(ACQUISITION_SCHEMA),
        "view_indices": indices,
        "frequency_hz": frequency,
        "viewpoint_positions": viewpoints,
        "tx_pos": tx,
        "rx_pos": rx,
        "scene_center_m": center,
        "response_shape": np.asarray(response_shape, dtype=np.int64),
        "response_dtype": np.asarray("complex64"),
        "metadata_json": np.asarray(_metadata_text(metadata)),
    }


def _normalise_payload(payload: Mapping[str, object]) -> dict[str, np.ndarray]:
    missing = sorted(_FIELDS.difference(payload))
    extra = sorted(set(payload).difference(_FIELDS))
    if missing or extra:
        raise ValueError(f"RadarSplat B7873200 acquisition record fields differ; missing={missing}, extra={extra}")
    metadata = json.loads(str(np.asarray(payload["metadata_json"]).reshape(()).item()))
    return acquisition_payload(
        view_indices=np.asarray(payload["view_indices"]),
        frequency_hz=np.asarray(payload["frequency_hz"]),
        viewpoint_positions=np.asarray(payload["viewpoint_positions"]),
        tx_pos=np.asarray(payload["tx_pos"]),
        rx_pos=np.asarray(payload["rx_pos"]),
        scene_center_m=np.asarray(payload["scene_center_m"]),
        metadata=metadata,
        response_shape=tuple(int(value) for value in np.asarray(payload["response_shape"]).tolist()),
        response_dtype=str(np.asarray(payload["response_dtype"]).reshape(()).item()),
    )


def write_acquisition_record(root: str | Path, **payload: np.ndarray) -> Path:
    """Write or direct-compare one response-free, role-limited record."""

    destination = Path(root) / ACQUISITION_FILENAME
    expected = _normalise_payload(payload)
    if destination.exists():
        observed = load_acquisition_record(root)
        if not acquisition_records_equal(observed, expected):
            raise ValueError("RadarSplat B7873200 acquisition record differs from the authorized calibration")
        return destination
    return atomic_save_npz(destination, **expected)


def load_acquisition_record(
    root: str | Path,
    *,
    expected_view_indices: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Load only exact record fields and validate direct calibration semantics."""

    path = Path(root) / ACQUISITION_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"RadarSplat B7873200 acquisition record is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        observed_fields = set(archive.files)
        if observed_fields != _FIELDS:
            missing = sorted(_FIELDS.difference(observed_fields))
            extra = sorted(observed_fields.difference(_FIELDS))
            raise ValueError(f"RadarSplat B7873200 acquisition record fields differ; missing={missing}, extra={extra}")
        observed = {name: np.asarray(archive[name]) for name in _FIELDS}
    if str(observed["schema"].reshape(()).item()) != ACQUISITION_SCHEMA:
        raise ValueError("RadarSplat B7873200 acquisition record has the wrong schema")
    payload = _normalise_payload(observed)
    if expected_view_indices is not None:
        expected = np.asarray(tuple(int(value) for value in expected_view_indices), dtype=np.int64)
        if not np.array_equal(payload["view_indices"], expected):
            raise ValueError("RadarSplat B7873200 acquisition record exposes the wrong view roles")
    return payload


def validate_acquisition_record(
    root: str | Path,
    *,
    expected_view_indices: Sequence[int] | None = None,
) -> Path:
    """Validate the cached direct calibration record without a raw response read."""

    load_acquisition_record(root, expected_view_indices=expected_view_indices)
    return Path(root) / ACQUISITION_FILENAME


def acquisition_records_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Compare calibration arrays directly; no digest acts as an identity proxy."""

    try:
        if (
            str(np.asarray(left["schema"]).reshape(()).item()) != ACQUISITION_SCHEMA
            or str(np.asarray(right["schema"]).reshape(()).item()) != ACQUISITION_SCHEMA
        ):
            return False
        left_normalized = _normalise_payload(left)
        right_normalized = _normalise_payload(right)
    except (KeyError, TypeError, ValueError):
        return False
    return set(left_normalized) == set(right_normalized) and all(
        left_normalized[name].shape == right_normalized[name].shape
        and np.array_equal(left_normalized[name], right_normalized[name])
        for name in left_normalized
    )


def acquisition_row(record: Mapping[str, np.ndarray], view_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return direct calibrated pose inputs for one authorized stored view."""

    indices = np.asarray(record["view_indices"], dtype=np.int64)
    matches = np.where(indices == int(view_index))[0]
    if matches.size != 1:
        raise ValueError("RadarSplat B7873200 target has no authorized calibration row")
    row = int(matches[0])
    return (
        np.asarray(record["viewpoint_positions"])[row],
        np.asarray(record["tx_pos"])[row],
        np.asarray(record["rx_pos"])[row],
    )


def _fp32_axis_tolerance(values: np.ndarray, *, ulps: float) -> float:
    scale = max(1.0e-3, float(np.max(np.abs(np.asarray(values, dtype=np.float64)))))
    return float(ulps) * abs(float(np.spacing(np.float32(scale))))


def _expected_target_geometry(
    record: Mapping[str, np.ndarray],
    view_index: int,
    grid: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild a target frame directly from authorized calibration arrays.

    Unlike the matched-filter target builder, this allocates no response or
    3-D query volume. It only recreates the pose and one-dimensional axes that
    define the stored native power image.
    """

    import torch

    from rift.power_baseline_dataset import calibrated_sensor_to_world

    viewpoint, tx, rx = acquisition_row(record, view_index)
    # The response-free acquisition record is the direct calibration source.
    # Refuse a recipe whose declared support centre differs before reconstructing
    # any target frame, then use that persisted direct centre rather than a
    # second independently supplied value from the recipe.
    recipe_center = np.asarray(grid["scene_center_m"], dtype=np.float64)
    record_center = np.asarray(record["scene_center_m"], dtype=np.float64)
    if recipe_center.shape != (3,) or record_center.shape != (3,) or not np.array_equal(
        recipe_center, record_center
    ):
        raise ValueError(
            "RadarSplat B7873200 target-grid scene centre disagrees with direct acquisition calibration"
        )
    center = record_center.astype(np.float32, copy=False)
    pose = calibrated_sensor_to_world(
        viewpoint,
        tx,
        rx,
        scene_center=center,
        dtype=torch.float32,
    ).detach().cpu().numpy().astype(np.float32, copy=False)
    n_range = int(grid["n_range"])
    n_azimuth = int(grid["n_azimuth"])
    n_elevation = int(grid["n_elevation"])
    extent = float(grid["scene_extent_m"])
    output_resolution_deg = float(grid["output_azimuth_resolution_deg"])
    elevation_resolution_deg = float(grid["elevation_sampling_resolution_deg"])
    center_tensor = torch.as_tensor(center, dtype=torch.float32)
    viewpoint_tensor = torch.as_tensor(viewpoint, dtype=torch.float32)
    center_range = float(torch.linalg.vector_norm(center_tensor - viewpoint_tensor).detach().cpu())
    near = center_range - extent
    far = center_range + extent
    if near <= 0.0:
        raise ValueError("RadarSplat B7873200 calibrated scene support crosses the sensor origin")
    range_resolution = (far - near) / n_range
    range_axis = (
        near + (torch.arange(n_range, dtype=torch.float32) + 0.5) * range_resolution
    ).cpu().numpy()
    azimuth_center_deg = float(grid["azimuth_center_deg"])
    az_half = 0.5 * n_azimuth * output_resolution_deg
    el_half = 0.5 * n_elevation * elevation_resolution_deg
    az_edges = torch.linspace(
        azimuth_center_deg - az_half,
        azimuth_center_deg + az_half,
        n_azimuth + 1,
        dtype=torch.float32,
    )
    el_edges = torch.linspace(-el_half, el_half, n_elevation + 1, dtype=torch.float32)
    azimuth_axis = torch.deg2rad(0.5 * (az_edges[:-1] + az_edges[1:])).cpu().numpy()
    elevation_axis = torch.deg2rad(0.5 * (el_edges[:-1] + el_edges[1:])).cpu().numpy()
    return pose, range_axis, azimuth_axis, elevation_axis


def validate_target_calibration(
    target: Mapping[str, np.ndarray],
    record: Mapping[str, np.ndarray],
    grid: Mapping[str, object],
) -> None:
    """Reject a structurally valid target whose direct calibrated geometry changed."""

    expected_pose, expected_range, expected_azimuth, expected_elevation = _expected_target_geometry(
        record, int(np.asarray(target["view_index"]).reshape(())), grid
    )
    observed_pose = np.asarray(target["sensor_to_world"], dtype=np.float32)
    pose_tolerance = _fp32_axis_tolerance(expected_pose, ulps=8.0)
    if not np.allclose(observed_pose, expected_pose, rtol=0.0, atol=pose_tolerance):
        raise ValueError("RadarSplat B7873200 stored target pose disagrees with direct acquisition calibration")
    for name, expected in (
        ("range_m", expected_range),
        ("azimuth_rad", expected_azimuth),
        ("elevation_rad", expected_elevation),
    ):
        observed = np.asarray(target[name], dtype=np.float32)
        tolerance = _fp32_axis_tolerance(expected, ulps=2.0)
        if not np.allclose(observed, expected, rtol=0.0, atol=tolerance):
            raise ValueError(
                f"RadarSplat B7873200 stored target {name} disagrees with direct acquisition calibration"
            )
