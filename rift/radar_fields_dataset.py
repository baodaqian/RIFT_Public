"""Radar-only preprocessing for the Radar Fields baseline.

RIFT npz files store coherent swept-frequency responses. Radar Fields consumes
real range-bin intensity, so the only supervision derived here is

    response(f) --IFFT over frequency--> |range response|**2.

No point cloud or auxiliary sensor is loaded or accepted by this module.
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


LIGHT_SPEED = 299792458.0

# A cache created while the sealed protocol is active must prove that its
# normalization scan used only the current train-role source IDs.  This is
# provenance for response access, not an archive or manifest integrity pin.
NORMALIZATION_PROVENANCE_VERSION = 1


@dataclass
class RadarFieldsArrays:
    """Measured pose metadata plus either eager or per-view response access.

    ``response`` remains the legacy eager array when ``load_response=True``.
    Diagnostic callers can instead set it to ``None`` and use
    :meth:`response_view`, which holds only the selected view in memory.  The
    latter matters for the multi-gigabyte RIFT acquisitions: resolving a split
    must not retain the sealed test payload just to inspect metadata.
    """

    response: Optional[np.ndarray]
    viewpoint_positions: np.ndarray
    tx_pos: np.ndarray
    rx_pos: np.ndarray
    metadata: Dict[str, object]
    source_path: Optional[str] = None
    response_shape: Optional[Tuple[int, ...]] = None
    response_dtype: Optional[np.dtype] = None
    allowed_response_view_indices: Optional[FrozenSet[int]] = None
    public_arrays: Optional[dict] = None
    acquisition_identity: Optional[dict] = None

    def _response_shape(self) -> Tuple[int, ...]:
        if self.response_shape is not None:
            return tuple(int(value) for value in self.response_shape)
        if self.response is not None:
            return tuple(int(value) for value in self.response.shape)
        raise RuntimeError("RadarFieldsArrays has neither a response array nor response header")

    @property
    def num_views(self) -> int:
        return int(self._response_shape()[0])

    @property
    def num_tx(self) -> int:
        return int(self._response_shape()[1])

    @property
    def num_rx(self) -> int:
        return int(self._response_shape()[2])

    @property
    def num_freq(self) -> int:
        return int(self._response_shape()[-1])

    @property
    def response_is_materialized(self) -> bool:
        return self.response is not None

    @property
    def response_access_is_restricted(self) -> bool:
        """Whether this lazy handle has an explicit source-view capability."""

        return self.allowed_response_view_indices is not None

    def source_sensor_rotation(self, view_index: int, *, device=None) -> torch.Tensor:
        """Simulator attitude from original Tx=Y/Rx=Z axes, before selection.

        A single selected element does not determine an axis; ordered channel
        subsets must not redefine the physical sensor's boresight or roll.
        """
        if self.acquisition_identity is not None:
            source = self.public_arrays or {}
            if 'source_tx_pos' not in source or 'source_rx_pos' not in source:
                raise ValueError('Selected Radar Fields acquisition requires full source sensor geometry')
            tx, rx = source['source_tx_pos'], source['source_rx_pos']
        else:
            tx, rx = self.tx_pos, self.rx_pos
        axes = []
        for positions in (tx, rx):
            positions = torch.as_tensor(positions[view_index], dtype=torch.float64, device=device)
            if positions.ndim != 2 or positions.shape[-1] != 3 or len(positions) < 2:
                raise ValueError('Source sensor axes require at least two original Tx and Rx elements')
            axis = positions[-1] - positions[0]
            length = axis.norm()
            if not torch.isfinite(axis).all() or length <= 0:
                raise ValueError('Source sensor axes must be finite and nondegenerate')
            axes.append(axis / length)
        y_axis, z_axis = axes
        rotation = torch.stack((torch.linalg.cross(y_axis, z_axis), y_axis, z_axis), -1)
        if not torch.allclose(rotation.T @ rotation, torch.eye(3, dtype=rotation.dtype, device=device),
                              atol=1e-7, rtol=1e-7):
            raise ValueError('Source sensor axes must define an orthonormal frame')
        return rotation

    def _validate_response_indices(self, view_indices: Iterable[int]) -> Tuple[int, ...]:
        """Validate source IDs before a lazy reader opens ``response.npy``."""

        normalized = _normalize_view_indices(view_indices, self.num_views)
        if self.allowed_response_view_indices is not None:
            denied = sorted(set(normalized).difference(self.allowed_response_view_indices))
            if denied:
                raise PermissionError(
                    "response access is restricted to the authorized development roles; "
                    f"denied source-view IDs: {denied}"
                )
        return normalized

    def response_view(self, view_index: int) -> np.ndarray:
        """Return one ``[tx,rx,chirp,freq]`` response without retaining all views."""

        view_index = self._validate_response_indices([view_index])[0]
        if self.public_arrays is not None:
            from .npz_dataset import get_npz_response_view
            return get_npz_response_view(self.public_arrays, view_index)
        if self.response is not None:
            return self.response[view_index]
        if self.source_path is None or self.response_dtype is None:
            raise RuntimeError("lazy response access requires source_path and response_dtype")
        return _read_response_view_from_npz(
            self.source_path,
            view_index,
            self._response_shape(),
            self.response_dtype,
        )

    def iter_response_views(self, view_indices: Iterable[int]):
        """Yield requested views once, in ascending source index, with bounded memory."""

        ordered = sorted(set(self._validate_response_indices(view_indices)))
        if self.public_arrays is not None:
            from .npz_dataset import iter_npz_response_views
            yield from iter_npz_response_views(self.public_arrays, ordered)
            return
        if self.response is not None:
            for index in ordered:
                yield index, self.response[index]
            return
        if self.source_path is None or self.response_dtype is None:
            raise RuntimeError("lazy response access requires source_path and response_dtype")
        yield from _iter_response_views_from_npz(
            self.source_path,
            ordered,
            self._response_shape(),
            self.response_dtype,
        )


def from_collection_arrays(public, contract):
    reader = public['_lazy_response_reader']
    while hasattr(reader, 'reader'):
        reader = reader.reader
    return RadarFieldsArrays(response=None, viewpoint_positions=public['viewpoint_positions'],
        tx_pos=public['tx_pos'], rx_pos=public['rx_pos'], metadata=public['meta'],
        source_path=contract['source_path'], response_shape=tuple(public['response_shape']),
        response_dtype=public['response_dtype'], allowed_response_view_indices=reader.allowed_view_indices,
        public_arrays=public, acquisition_identity=({k: contract[k] for k in
            ('antenna_selection', 'source_geometry_sha256', 'source_response_shape')}
            if contract.get('antenna_selection') else None))


def validate_power_stats_acquisition(stats, acquisition):
    if not isinstance(stats, Mapping) or stats.get('acquisition_identity') != acquisition:
        raise ValueError('Power stats antenna acquisition identity does not match selected channels')


def _normalize_view_indices(indices: Iterable[int], num_views: int) -> Tuple[int, ...]:
    """Reject fractional/out-of-range source IDs without touching responses."""

    normalized = []
    for value in indices:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"view index must be an integer ID, got boolean {value!r}")
        integer = int(value)
        if integer != value:
            raise ValueError(f"view index must be integral, got {value!r}")
        if not 0 <= integer < int(num_views):
            raise IndexError(f"view index {integer} is outside [0,{num_views})")
        normalized.append(integer)
    return tuple(normalized)


def restrict_radar_fields_response_views(
    arrays: RadarFieldsArrays,
    allowed_view_indices: Iterable[int],
) -> RadarFieldsArrays:
    """Return a lazy handle that can materialize only the supplied source IDs.

    A full eager response cube cannot be made sealed retroactively, so this is
    deliberately limited to ``load_radar_fields_npz(..., load_response=False)``.
    Re-restricting an already restricted handle can only narrow its existing
    capability.  The pose/header metadata remains available for all source
    views; only the response payload is protected.
    """

    if arrays.response_is_materialized:
        raise ValueError(
            "role restriction requires load_radar_fields_npz(..., load_response=False)"
        )
    authorized = frozenset(arrays._validate_response_indices(allowed_view_indices))
    return replace(arrays, allowed_response_view_indices=authorized)


def decode_metadata_json(raw: object) -> Dict[str, object]:
    """Decode scalar or singleton-array ``metadata_json`` without repr artifacts."""

    while isinstance(raw, np.ndarray):
        if raw.size != 1:
            raise ValueError(
                "metadata_json must be a scalar or singleton numpy array, "
                f"got shape {raw.shape}"
            )
        raw = raw.reshape(-1)[0]
    if isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, (bytes, bytearray, np.bytes_)):
        raw = bytes(raw).decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError(f"metadata_json must decode to text, got {type(raw).__name__}")
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("metadata_json is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must encode a JSON object")
    return metadata


@dataclass(frozen=True)
class RadarFieldsSealedSplit:
    """Explicit train/validation/test source IDs for a sealed RF recipe.

    This is intentionally small provenance, not an archive integrity gate.  A
    caller binds these IDs against the NPZ response header before any response
    payload is opened, then restricts the lazy reader to train plus validation.
    """

    manifest_path: str
    manifest_name: Optional[str]
    schema_version: int
    response_shape: Tuple[int, ...]
    response_dtype: str
    train_indices: Tuple[int, ...]
    validation_indices: Tuple[int, ...]
    test_indices: Tuple[int, ...]
    unused_indices: Tuple[int, ...]
    test_sealed: bool
    unused_sealed: bool
    complete_partition: bool

    def protocol_contract(self) -> Dict[str, object]:
        """The resume-relevant sealed-development policy, excluding any hash."""

        return {
            "version": 1,
            "role_binding": "explicit_manifest_source_view_ids",
            "manifest_path": self.manifest_path,
            "manifest_name": self.manifest_name,
            "manifest_schema_version": self.schema_version,
            "response_header_shape": list(self.response_shape),
            "response_header_dtype": self.response_dtype,
            "test_sealed": self.test_sealed,
            "unused_sealed": self.unused_sealed,
            "complete_partition": self.complete_partition,
            "unused_role_ids": list(self.unused_indices),
            "authorized_response_roles": ["train", "val"],
            "test_response_materialized": False,
            "unused_response_materialized": False,
        }


def _manifest_indices(split: Mapping[str, object], key: str) -> Tuple[int, ...]:
    if key not in split:
        raise ValueError(f"sealed split manifest lacks {key!r}")
    raw = split[key]
    if isinstance(raw, (str, bytes)):
        raise ValueError(f"sealed split manifest {key!r} must be an integer-ID sequence")
    try:
        values = tuple(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sealed split manifest {key!r} must be an integer-ID sequence") from exc
    return values


def _validate_manifest_role(
    split: Mapping[str, object],
    key: str,
    num_views: int,
    *,
    expected_count: Optional[int] = None,
    required: bool = True,
) -> Tuple[int, ...]:
    if key not in split:
        if required:
            raise ValueError(f"sealed split manifest lacks {key!r}")
        return ()
    indices = _normalize_view_indices(_manifest_indices(split, key), num_views)
    if len(set(indices)) != len(indices):
        raise ValueError(f"sealed split manifest {key!r} contains duplicate source-view IDs")
    if expected_count is not None and len(indices) != int(expected_count):
        raise ValueError(
            f"sealed split manifest {key!r} has {len(indices)} IDs, expected {expected_count}"
        )
    return indices


def _manifest_count(split: Mapping[str, object], key: str, actual: int) -> None:
    if key not in split:
        return
    try:
        declared = int(split[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sealed split manifest {key} must be an integer") from exc
    if declared != int(actual):
        raise ValueError(
            f"sealed split manifest {key}={split[key]!r} disagrees with {actual} explicit IDs"
        )


def _normalize_manifest_response_shape(raw: object) -> Tuple[int, ...]:
    """Return a JSON manifest shape without accepting lossy dimensions."""

    if isinstance(raw, (str, bytes)):
        raise ValueError("sealed split manifest dataset response_shape must be a sequence")
    try:
        values = tuple(raw)
    except TypeError as exc:
        raise ValueError("sealed split manifest dataset response_shape must be a sequence") from exc
    if not values:
        raise ValueError("sealed split manifest dataset response_shape must not be empty")
    normalized = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("sealed split manifest dataset response_shape dimensions must be integers")
        integer = int(value)
        if integer <= 0:
            raise ValueError(
                "sealed split manifest dataset response_shape dimensions must be positive integers"
            )
        normalized.append(integer)
    return tuple(normalized)


def _normalize_manifest_response_dtype(raw: object) -> np.dtype:
    """Decode a declared header dtype without making it an identity pin."""

    if not isinstance(raw, str) or not raw:
        raise ValueError("sealed split manifest dataset response_dtype must be nonempty text")
    try:
        return np.dtype(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sealed split manifest dataset response_dtype must be a NumPy dtype string"
        ) from exc


def _validate_manifest_dataset_header(
    manifest: Mapping[str, object],
    num_views: int,
    response_shape: Sequence[int],
    response_dtype: object,
) -> None:
    """Bind explicit roles to the complete NPZ response header before reads.

    A manifest's historical path hints or digest-like notes remain ordinary
    provenance.  The concrete source binding is the header shape/dtype plus
    explicit source-view roles; both are available without materializing any
    response payload.
    """

    actual_shape = _normalize_manifest_response_shape(response_shape)
    if actual_shape[0] != int(num_views):
        raise ValueError(
            "sealed split manifest caller view count disagrees with the NPZ response header"
        )
    try:
        actual_dtype = np.dtype(response_dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError("sealed split manifest caller response dtype is invalid") from exc
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("sealed split manifest dataset must be an object")

    # Sealed access needs an unambiguous header binding.  Unlike historical
    # nonsealed recipes, omitting either field is not safe to treat as an
    # equivalent acquisition merely because its view count happens to match.
    if "response_shape" not in dataset:
        raise ValueError("sealed split manifest dataset lacks response_shape")
    declared_shape = _normalize_manifest_response_shape(dataset["response_shape"])
    if manifest.get('antenna_selection') is not None:
        from .antenna_selection import validate_selection
        acquisition = validate_selection(manifest['antenna_selection'])
        if declared_shape != (10000, 16, 16, 1, 600):
            raise ValueError('Selected collection must retain the original source header')
        declared_shape = (10000, acquisition['num_tx'], acquisition['num_rx'], 1, 600)
    if declared_shape != actual_shape:
        raise ValueError(
            "sealed split manifest dataset response_shape disagrees with the NPZ response header: "
            f"manifest={list(declared_shape)}, header={list(actual_shape)}"
        )
    if "response_dtype" not in dataset:
        raise ValueError("sealed split manifest dataset lacks response_dtype")
    declared_dtype = _normalize_manifest_response_dtype(dataset["response_dtype"])
    if declared_dtype != actual_dtype:
        raise ValueError(
            "sealed split manifest dataset response_dtype disagrees with the NPZ response header: "
            f"manifest={declared_dtype}, header={actual_dtype}"
        )

    declared = []
    for key in ("num_views", "viewpoint_count", "view_count"):
        if key in dataset:
            declared.append((key, int(dataset[key])))
    declared.append(("response_shape[0]", declared_shape[0]))
    for label, count in declared:
        if count != int(num_views):
            raise ValueError(
                f"sealed split manifest dataset {label}={count} disagrees with NPZ header "
                f"view count {num_views}"
            )


def load_radar_fields_sealed_split_manifest(
    path: str,
    num_views: int,
    *,
    response_shape: Sequence[int],
    response_dtype: object,
    expected_num_train: Optional[int] = None,
    expected_num_val: Optional[int] = None,
    expected_num_test: Optional[int] = None,
) -> RadarFieldsSealedSplit:
    """Load explicit sealed roles without opening an NPZ response payload.

    The generic schema is the project-wide ``dataset`` plus ``split`` object
    convention.  The split must contain ordered ``train_indices``,
    ``validation_indices``, ``test_indices``, and ``unused_indices`` arrays
    forming a complete partition.  We validate
    source-ID range, cardinality, disjointness, sealing policy, and the full
    response header before a payload read; no manifest digest or archive hash
    is introduced.
    """

    manifest_path = str(Path(path).resolve())
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise ValueError("sealed split manifest must encode a JSON object")
    if "schema_version" not in manifest:
        raise ValueError("sealed split manifest lacks schema_version")
    try:
        schema_version = int(manifest["schema_version"])
    except (TypeError, ValueError) as exc:
        raise ValueError("sealed split manifest schema_version must be an integer") from exc
    if schema_version <= 0:
        raise ValueError("sealed split manifest schema_version must be positive")
    name = manifest.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError("sealed split manifest name must be text when present")
    split = manifest.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("sealed split manifest lacks a split object")
    if split.get("test_sealed") is not True:
        raise ValueError("sealed split manifest must set split.test_sealed=true")
    if split.get("unused_sealed") is not True:
        raise ValueError("sealed split manifest must set split.unused_sealed=true")
    if split.get("complete_partition") is not True:
        raise ValueError("sealed split manifest must set split.complete_partition=true")

    actual_response_shape = _normalize_manifest_response_shape(response_shape)
    try:
        actual_response_dtype = np.dtype(response_dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError("sealed split manifest caller response dtype is invalid") from exc
    _validate_manifest_dataset_header(
        manifest,
        num_views,
        response_shape=actual_response_shape,
        response_dtype=actual_response_dtype,
    )
    train = _validate_manifest_role(
        split, "train_indices", num_views, expected_count=expected_num_train
    )
    validation = _validate_manifest_role(
        split, "validation_indices", num_views, expected_count=expected_num_val
    )
    test = _validate_manifest_role(
        split, "test_indices", num_views, expected_count=expected_num_test
    )
    unused = _validate_manifest_role(
        split, "unused_indices", num_views
    )
    complete = train + validation + test + unused
    if len(complete) != int(num_views) or set(complete) != set(range(int(num_views))):
        raise ValueError(
            "complete sealed split manifest must partition every NPZ source-view ID exactly once"
        )

    all_roles = train + validation + test + unused
    if len(set(all_roles)) != len(all_roles):
        raise ValueError("sealed split manifest roles overlap")
    _manifest_count(split, "num_train", len(train))
    _manifest_count(split, "num_validation", len(validation))
    _manifest_count(split, "num_test", len(test))
    _manifest_count(split, "num_unused", len(unused))
    return RadarFieldsSealedSplit(
        manifest_path=manifest_path,
        manifest_name=name,
        schema_version=schema_version,
        response_shape=actual_response_shape,
        response_dtype=str(actual_response_dtype),
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        unused_indices=unused,
        test_sealed=True,
        unused_sealed=True,
        complete_partition=True,
    )


def _response_header(path: str) -> Tuple[Tuple[int, ...], np.dtype, bool]:
    """Read the ``response.npy`` header inside an NPZ without loading its payload."""

    with zipfile.ZipFile(path) as archive:
        try:
            member = archive.getinfo("response.npy")
        except KeyError as exc:
            raise ValueError(f"{path} is missing required key: response") from exc
        with archive.open(member) as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version in ((2, 0), (3, 0)):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"unsupported response.npy version {version}")
    return tuple(int(value) for value in shape), np.dtype(dtype), bool(fortran_order)


def _discard_bytes(handle, count: int) -> None:
    """Consume a compressed NPZ stream with bounded temporary memory."""

    buffer = bytearray(min(1024 * 1024, max(1, count)))
    remaining = int(count)
    while remaining:
        wanted = min(len(buffer), remaining)
        read = handle.readinto(memoryview(buffer)[:wanted])
        if not read:
            raise EOFError(f"response payload ended with {remaining} bytes still required")
        remaining -= int(read)


def _read_exact_bytes(handle, count: int) -> bytes:
    pieces = []
    remaining = int(count)
    while remaining:
        piece = handle.read(min(1024 * 1024, remaining))
        if not piece:
            raise EOFError(f"response payload ended with {remaining} bytes still required")
        pieces.append(piece)
        remaining -= len(piece)
    return b"".join(pieces)


def _read_response_view_from_npz(
    path: str,
    view_index: int,
    shape: Tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    """Stream one C-order response view from a zip member, retaining only it."""

    if len(shape) != 5:
        raise ValueError(f"response must have shape [view,tx,rx,chirp,freq], got {shape}")
    with zipfile.ZipFile(path) as archive:
        with archive.open("response.npy") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                streamed_shape, fortran_order, streamed_dtype = np.lib.format.read_array_header_1_0(handle)
            elif version in ((2, 0), (3, 0)):
                streamed_shape, fortran_order, streamed_dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"unsupported response.npy version {version}")
            if tuple(int(value) for value in streamed_shape) != tuple(shape) or np.dtype(streamed_dtype) != dtype:
                raise RuntimeError("response.npy header changed while the dataset was being read")
            if fortran_order:
                raise ValueError("lazy response access supports only C-order response.npy arrays")
            bytes_per_view = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
            _discard_bytes(handle, int(view_index) * bytes_per_view)
            raw = _read_exact_bytes(handle, bytes_per_view)
    return np.frombuffer(raw, dtype=dtype).reshape(shape[1:]).copy()


def _iter_response_views_from_npz(
    path: str,
    view_indices: Sequence[int],
    shape: Tuple[int, ...],
    dtype: np.dtype,
):
    """Stream sorted unique response views in one pass through the NPZ member."""

    if len(shape) != 5:
        raise ValueError(f"response must have shape [view,tx,rx,chirp,freq], got {shape}")
    ordered = sorted(set(int(value) for value in view_indices))
    if not ordered:
        return
    with zipfile.ZipFile(path) as archive:
        with archive.open("response.npy") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                streamed_shape, fortran_order, streamed_dtype = np.lib.format.read_array_header_1_0(handle)
            elif version in ((2, 0), (3, 0)):
                streamed_shape, fortran_order, streamed_dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"unsupported response.npy version {version}")
            if tuple(int(value) for value in streamed_shape) != tuple(shape) or np.dtype(streamed_dtype) != dtype:
                raise RuntimeError("response.npy header changed while the dataset was being read")
            if fortran_order:
                raise ValueError("lazy response access supports only C-order response.npy arrays")
            bytes_per_view = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
            target_position = 0
            for current_view in range(shape[0]):
                selected = target_position < len(ordered) and current_view == ordered[target_position]
                if selected:
                    raw = _read_exact_bytes(handle, bytes_per_view)
                    yield current_view, np.frombuffer(raw, dtype=dtype).reshape(shape[1:]).copy()
                    target_position += 1
                    if target_position == len(ordered):
                        return
                else:
                    _discard_bytes(handle, bytes_per_view)


def load_radar_fields_npz(path: str, *, load_response: bool = True) -> RadarFieldsArrays:
    """Load dataset metadata, with optional legacy eager response materialization.

    ``load_response=False`` resolves response shape and all role-defining pose
    metadata before accessing any response payload.  It is intended for
    diagnostics, where a selected authorized view is streamed on demand.
    """

    path = os.fspath(path)
    response_shape, response_dtype, response_fortran = _response_header(path)
    required = ("metadata_json", "viewpoint_positions", "tx_pos", "rx_pos")
    # Preserve the previous loader's object-array compatibility.  The decoded
    # metadata still has to become a JSON object, so this does not introduce a
    # second metadata representation; it only keeps older NPZ writers usable.
    with np.load(path, allow_pickle=True) as data:
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path} is missing required keys: {missing}")
        metadata = decode_metadata_json(data["metadata_json"])
        viewpoint_positions = np.asarray(data["viewpoint_positions"])
        tx_pos = np.asarray(data["tx_pos"])
        rx_pos = np.asarray(data["rx_pos"])
        response = np.asarray(data["response"]) if load_response else None

    if len(response_shape) != 5:
        raise ValueError(
            "response must have shape [view,tx,rx,chirp,freq], "
            f"got {response_shape}"
        )
    if response_fortran and not load_response:
        raise ValueError("lazy response access supports only C-order response.npy arrays")
    if not np.issubdtype(response_dtype, np.complexfloating):
        raise ValueError("response must be complex-valued")
    if response_shape[0] != viewpoint_positions.shape[0]:
        raise ValueError("response and viewpoint_positions view counts disagree")
    if tx_pos.shape != (response_shape[0], response_shape[1], 3):
        raise ValueError("tx_pos must have shape [view,tx,3] matching response")
    if rx_pos.shape != (response_shape[0], response_shape[2], 3):
        raise ValueError("rx_pos must have shape [view,rx,3] matching response")
    return RadarFieldsArrays(
        response=response,
        viewpoint_positions=viewpoint_positions,
        tx_pos=tx_pos,
        rx_pos=rx_pos,
        metadata=metadata,
        source_path=str(Path(path).resolve()),
        response_shape=response_shape,
        response_dtype=response_dtype,
    )


def dataset_provenance(arrays: RadarFieldsArrays) -> Dict[str, object]:
    """JSON-safe source identity for diagnostics and checkpoint provenance."""

    source = Path(arrays.source_path).resolve() if arrays.source_path else None
    return {
        "dataset_path": str(source) if source is not None else None,
        "dataset_file_size_bytes": int(source.stat().st_size) if source is not None and source.exists() else None,
        "response_shape": list(arrays._response_shape()),
        "response_dtype": str(arrays.response_dtype) if arrays.response_dtype is not None else (
            str(arrays.response.dtype) if arrays.response is not None else None
        ),
        "viewpoint_count": int(arrays.num_views),
        "tx_count": int(arrays.num_tx),
        "rx_count": int(arrays.num_rx),
        "frequency_count": int(arrays.num_freq),
        "metadata": dict(arrays.metadata),
        "response_payload_materialized": bool(arrays.response_is_materialized),
        "response_access_restricted": bool(arrays.response_access_is_restricted),
        "authorized_response_view_count": (
            len(arrays.allowed_response_view_indices)
            if arrays.allowed_response_view_indices is not None
            else None
        ),
    }


def build_frequency_grid(metadata: Dict[str, object]) -> np.ndarray:
    center = float(metadata["radar_fc_hz"])
    bandwidth = float(metadata["radar_bandwidth_hz"])
    count = int(metadata["num_adc_samples"])
    return (center - bandwidth / 2.0) + np.arange(count, dtype=np.float64) * (bandwidth / count)


def range_bin_size(metadata: Dict[str, object]) -> float:
    """One-way range spacing after IFFT of the swept-frequency response."""

    return LIGHT_SPEED / (2.0 * float(metadata["radar_bandwidth_hz"]))


def range_bin_centers(metadata: Dict[str, object], device=None, dtype=torch.float32) -> torch.Tensor:
    count = int(metadata["num_adc_samples"])
    return torch.arange(count, device=device, dtype=dtype) * range_bin_size(metadata)


def split_view_indices(
    num_views: int,
    num_train: int,
    num_val: int,
    num_test: int,
    seed: int,
    val_from_tail: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wanted = num_train + num_val + num_test
    if wanted > num_views:
        raise ValueError(f"requested {wanted} views from a {num_views}-view dataset")
    permutation = np.random.default_rng(seed).permutation(num_views)
    if val_from_tail:
        train = permutation[:num_train]
        val = permutation[num_views - num_val :] if num_val else permutation[:0]
        test_stop = num_views - num_val
        test = permutation[test_stop - num_test : test_stop] if num_test else permutation[:0]
    else:
        train = permutation[:num_train]
        val = permutation[num_train : num_train + num_val]
        test = permutation[num_train + num_val : wanted]
    return train, val, test


def response_view_to_range_power(
    response_view: np.ndarray,
    pair_indices: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return ``[pairs,range_bins]`` power after coherent chirp averaging."""

    if response_view.ndim != 4:
        raise ValueError("response_view must have shape [tx,rx,chirp,freq]")
    cube = response_view.mean(axis=2)  # [tx,rx,freq], static chirps are repeats
    flat = cube.reshape(-1, cube.shape[-1])
    if pair_indices is not None:
        flat = flat[np.asarray(pair_indices, dtype=np.int64)]
    signal = torch.as_tensor(flat, dtype=torch.complex64, device=device)
    range_response = torch.fft.ifft(signal, dim=-1)
    return range_response.abs().square()


def estimate_power_peak(
    arrays: RadarFieldsArrays,
    view_indices: Iterable[int],
    max_views: int = 0,
) -> float:
    """Dataset-global range-power maximum used for fixed dB normalization."""

    indices = normalization_scan_indices(view_indices, max_views=max_views)
    peak = 0.0
    for count, (_view, response_view) in enumerate(arrays.iter_response_views(indices), start=1):
        cube = response_view.mean(axis=2).reshape(-1, arrays.num_freq)
        power = np.abs(np.fft.ifft(cube, axis=-1)) ** 2
        peak = max(peak, float(power.max()))
        if count % 100 == 0:
            print(f"Radar Fields normalization scan: {count}/{len(indices)} views, peak={peak:.6e}")
    if not np.isfinite(peak) or peak <= 0:
        raise RuntimeError(f"invalid range-power peak {peak}")
    return peak


def normalization_scan_indices(
    view_indices: Iterable[int],
    max_views: int = 0,
) -> list[int]:
    """Return the exact train-role IDs contributing to normalization stats."""

    indices = [int(value) for value in view_indices]
    if max_views > 0 and len(indices) > max_views:
        pick = np.linspace(0, len(indices) - 1, max_views).round().astype(int)
        indices = [indices[index] for index in pick]
    return indices


def normalize_power_db(
    power: torch.Tensor,
    peak_power: float,
    dynamic_range_db: float = 60.0,
) -> torch.Tensor:
    """Map power to the release's normalized FFT-image intensity domain."""

    if peak_power <= 0 or dynamic_range_db <= 0:
        raise ValueError("peak_power and dynamic_range_db must be positive")
    relative = power / float(peak_power)
    db = 10.0 * torch.log10(relative.clamp_min(10.0 ** (-dynamic_range_db / 10.0)))
    return ((db + dynamic_range_db) / dynamic_range_db).clamp(0.0, 1.0)


def scene_range_mask(
    ranges: torch.Tensor,
    viewpoint_position: torch.Tensor,
    extent: float,
    margin: float = 0.05,
) -> torch.Tensor:
    """Bins that can intersect ``[-extent,extent]^3`` from this viewpoint."""

    center_range = torch.linalg.vector_norm(viewpoint_position)
    radius = math_sqrt3() * float(extent) + float(margin)
    return (ranges >= center_range - radius) & (ranges <= center_range + radius)


def math_sqrt3() -> float:
    return 1.7320508075688772


def _strict_cached_view_indices(
    raw: object,
    *,
    num_views: int,
) -> Optional[list[int]]:
    """Decode cache IDs only when every value is an exact source-view ID."""

    if isinstance(raw, (str, bytes)):
        return None
    try:
        values = tuple(raw)
    except TypeError:
        return None
    try:
        normalized = list(_normalize_view_indices(values, num_views))
    except (IndexError, TypeError, ValueError):
        return None
    if len(set(normalized)) != len(normalized):
        return None
    return normalized


def _sealed_stats_cache_matches_train_role(
    stats: Mapping[str, object],
    *,
    requested_train_indices: Sequence[int],
    requested_scan_indices: Sequence[int],
    num_views: int,
) -> bool:
    """Whether a cache proves its exact normalization scan was train-only.

    The redundant top-level fields are retained for older reporting code, but
    a sealed run trusts them only when the versioned nested provenance agrees
    with the current train role and deterministic scan selection exactly.
    """

    provenance = stats.get("normalization_provenance")
    if not isinstance(provenance, Mapping):
        return False
    try:
        version = int(provenance.get("version"))
    except (TypeError, ValueError):
        return False
    if version != NORMALIZATION_PROVENANCE_VERSION:
        return False
    if provenance.get("normalization_scan_role") != "train":
        return False

    expected_train = list(requested_train_indices)
    expected_scan = list(requested_scan_indices)
    fields = (
        (stats.get("train_view_indices"), expected_train),
        (stats.get("normalization_scan_view_indices"), expected_scan),
        (provenance.get("train_view_indices"), expected_train),
        (provenance.get("normalization_scan_view_indices"), expected_scan),
    )
    for raw_indices, expected_indices in fields:
        parsed = _strict_cached_view_indices(raw_indices, num_views=num_views)
        if parsed != expected_indices:
            return False
    try:
        stored_train_count = int(stats.get("train_view_count"))
    except (TypeError, ValueError):
        return False
    return stored_train_count == len(expected_train)


def _stats_result(
    stats: Mapping[str, object],
    *,
    dynamic_range_db: float,
    train_view_indices: Optional[Sequence[int]],
    normalization_scan_view_indices: Optional[Sequence[int]],
    train_view_ids_verified: bool,
    normalization_provenance_verified: bool,
    normalization_stats_cache_reused: bool,
) -> Dict[str, object]:
    """Return the stable public stats payload plus cache-verification state."""

    return {
        **({"dataset_identity": dict(stats["dataset_identity"])} if "dataset_identity" in stats else {}),
        **({"acquisition_identity": stats["acquisition_identity"]} if "acquisition_identity" in stats else {}),
        "peak_power": float(stats["peak_power"]),
        "dynamic_range_db": float(dynamic_range_db),
        "normalization_domain": str(
            stats.get("normalization_domain", "normalized_dB_range_power")
        ),
        "train_view_indices": (
            list(train_view_indices) if train_view_indices is not None else None
        ),
        "train_view_count": int(
            stats.get(
                "train_view_count",
                len(train_view_indices) if train_view_indices is not None else 0,
            )
        ),
        "normalization_scan_view_indices": (
            list(normalization_scan_view_indices)
            if normalization_scan_view_indices is not None
            else None
        ),
        "normalization_provenance": stats.get("normalization_provenance"),
        "train_view_ids_verified": bool(train_view_ids_verified),
        "normalization_provenance_verified": bool(normalization_provenance_verified),
        "normalization_stats_cache_reused": bool(normalization_stats_cache_reused),
    }


def validate_power_stats_identity(stats: Mapping[str, object], expected_identity: Mapping[str, object]) -> None:
    """Never reuse another object's normalization, even when its role IDs match."""
    from .rift_dataset import object_identity

    if (not isinstance(expected_identity, Mapping)
            or expected_identity != object_identity(expected_identity.get("object_id"))):
        raise ValueError("Power stats require a canonical RIFT dataset object identity")
    if not isinstance(stats, Mapping) or stats.get("dataset_identity") != expected_identity:
        raise ValueError("Power stats dataset_identity is missing or does not match the selected object")


def load_or_create_stats(
    stats_path: str,
    arrays: RadarFieldsArrays,
    train_indices: Sequence[int],
    dynamic_range_db: float,
    max_views: int = 0,
    *,
    sealed_protocol: bool = False,
    dataset_identity: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Load or calibrate fixed range-power normalization statistics.

    Ordinary legacy recipes retain their historic cache compatibility.  An
    opt-in sealed recipe instead accepts a cached peak only if the cache's
    versioned provenance proves that the exact deterministic scan came from
    its current authorized train IDs.  An unsafe existing cache fails closed
    without reading responses or overwriting a user-owned artifact; a fresh
    sealed run creates its cache through the role-restricted lazy reader.
    Collection callers additionally bind dataset_identity: absent/wrong-object
    caches are rejected rather than relabeled or recalibrated.
    """

    if dataset_identity is not None:
        from .rift_dataset import metadata_object_id, object_identity, validate_metadata

        if not sealed_protocol:
            raise ValueError("RIFT dataset normalization requires the sealed protocol")
        validate_metadata(arrays.metadata)
        validate_power_stats_identity({"dataset_identity": dataset_identity},
                                      object_identity(metadata_object_id(arrays.metadata)))
    requested_train_indices = list(_normalize_view_indices(train_indices, arrays.num_views))
    requested_scan_indices = normalization_scan_indices(requested_train_indices, max_views=max_views)
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as handle:
            stats = json.load(handle)
        validate_power_stats_acquisition(stats, getattr(arrays, "acquisition_identity", None))
        if dataset_identity is not None:
            validate_power_stats_identity(stats, dataset_identity)

        if sealed_protocol:
            if not isinstance(stats, Mapping):
                stats = {}
            cache_matches = _sealed_stats_cache_matches_train_role(
                stats,
                requested_train_indices=requested_train_indices,
                requested_scan_indices=requested_scan_indices,
                num_views=arrays.num_views,
            )
            try:
                cache_matches = cache_matches and (
                    float(stats["dynamic_range_db"]) == float(dynamic_range_db)
                )
                # Ensure a cached scalar cannot become an implicit NaN/zero
                # normalization just because its role provenance was sound.
                cache_matches = cache_matches and (
                    np.isfinite(float(stats["peak_power"])) and float(stats["peak_power"]) > 0
                )
            except (KeyError, TypeError, ValueError):
                cache_matches = False
            if cache_matches:
                return _stats_result(
                    stats,
                    dynamic_range_db=dynamic_range_db,
                    train_view_indices=requested_train_indices,
                    normalization_scan_view_indices=requested_scan_indices,
                    train_view_ids_verified=True,
                    normalization_provenance_verified=True,
                    normalization_stats_cache_reused=True,
                )
            raise ValueError(
                "sealed normalization cache lacks exact train-only scan provenance; "
                "refusing to reuse, stream responses, or overwrite the existing cache"
            )
        else:
            if float(stats["dynamic_range_db"]) != float(dynamic_range_db):
                raise ValueError(
                    f"cached dynamic_range_db={stats['dynamic_range_db']} disagrees with requested "
                    f"{dynamic_range_db}"
                )
            stored_indices = stats.get("train_view_indices")
            if stored_indices is not None:
                stored_indices = [int(value) for value in stored_indices]
                if stored_indices != requested_train_indices:
                    raise ValueError(
                        "cached power stats were calibrated on different train view IDs; "
                        "refusing to reuse normalization across roles"
                    )
            return _stats_result(
                stats,
                dynamic_range_db=dynamic_range_db,
                train_view_indices=stored_indices,
                normalization_scan_view_indices=stats.get("normalization_scan_view_indices"),
                train_view_ids_verified=stored_indices is not None,
                normalization_provenance_verified=False,
                normalization_stats_cache_reused=True,
            )

    peak = estimate_power_peak(arrays, requested_train_indices, max_views=max_views)
    stats = {
        **({"acquisition_identity": arrays.acquisition_identity} if getattr(arrays, "acquisition_identity", None) else {}),
        **({"dataset_identity": dict(dataset_identity)} if dataset_identity is not None else {}),
        "peak_power": peak,
        "dynamic_range_db": float(dynamic_range_db),
        "normalization_domain": "normalized_dB_range_power",
        "train_view_indices": requested_train_indices,
        "train_view_count": len(requested_train_indices),
        "normalization_scan_view_indices": requested_scan_indices,
        "normalization_provenance": {
            "version": NORMALIZATION_PROVENANCE_VERSION,
            "normalization_scan_role": "train",
            "train_view_indices": requested_train_indices,
            "normalization_scan_view_indices": requested_scan_indices,
        },
    }
    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
    temporary = stats_path + f".tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, stats_path)
    return _stats_result(
        stats,
        dynamic_range_db=dynamic_range_db,
        train_view_indices=requested_train_indices,
        normalization_scan_view_indices=requested_scan_indices,
        train_view_ids_verified=True,
        normalization_provenance_verified=bool(sealed_protocol),
        normalization_stats_cache_reused=False,
    )
