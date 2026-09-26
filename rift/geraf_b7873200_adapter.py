"""Restricted raw-response adapter for the corrected GeRaF B7873200 lane.

The historical :mod:`rift.power_baseline_dataset` loader intentionally exposes
a memory-mapped ``response`` member because its old, unsealed recipes need
arbitrary view access.  This module is deliberately separate: it exposes the
same B787 geometry and metadata surface needed by GeRaF while making raw
response access possible only through :meth:`SealedB787PowerArrays.response_view`
for the manifest-authorized train and validation IDs.

There is no archive hash here.  The factory first asks the established sealed
protocol to bind the manifest against the response header, then opens the
stored ``response.npy`` member for selected-row reads.  A compressed member is
rejected rather than streamed, because streaming a late authorized row would
consume preceding reserved-response bytes.
"""

from __future__ import annotations

import copy
import json
import os
import weakref
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class B787ResponseHeader:
    """Non-indexable response metadata exposed by a sealed B787 adapter."""

    shape: Tuple[int, ...]
    dtype: np.dtype


def _normalize_view_index(value: object, num_views: int) -> int:
    """Reject coercions before the raw response member is indexed."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"view index must be an integer ID, got boolean {value!r}")
    try:
        index = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"view index must be an integer ID, got {value!r}") from exc
    if index != value:
        raise ValueError(f"view index must be integral, got {value!r}")
    if not 0 <= index < int(num_views):
        raise IndexError(f"view index {index} is outside [0,{int(num_views)})")
    return index


def _authorized_train_validation_ids(identity: Mapping[str, object]) -> FrozenSet[int]:
    """Extract and re-check the only raw-response capability from an identity."""

    roles = identity.get("role_ids")
    if not isinstance(roles, Mapping):
        raise ValueError("sealed B787 identity lacks role_ids")
    shape = identity.get("response_shape")
    if not isinstance(shape, (list, tuple)) or not shape:
        raise ValueError("sealed B787 identity lacks a response shape")
    try:
        num_views = int(shape[0])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sealed B787 identity has an invalid response shape") from exc
    if num_views <= 0:
        raise ValueError("sealed B787 identity has an invalid view count")

    normalized_roles: Dict[str, tuple[int, ...]] = {}
    for role in ("train", "validation", "reserved_test", "unused"):
        values = roles.get(role)
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"sealed B787 identity role {role!r} is not an ordered ID sequence")
        normalized_roles[role] = tuple(
            _normalize_view_index(value, num_views) for value in values
        )
    all_role_ids = [
        value
        for values in normalized_roles.values()
        for value in values
    ]
    if len(set(all_role_ids)) != len(all_role_ids) or set(all_role_ids) != set(range(num_views)):
        raise ValueError("sealed B787 identity roles must be a complete disjoint view partition")
    selected = normalized_roles["train"] + normalized_roles["validation"]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("sealed B787 train/validation response roles are empty or overlap")

    access = identity.get("response_access")
    expected_access = {
        "train_materialized": True,
        "validation_materialized": True,
        "reserved_test_materialized": False,
        "unused_materialized": False,
    }
    if access != expected_access:
        raise ValueError("sealed B787 identity would expose a reserved response role")
    return frozenset(selected)


def _preflight_b7873200_identity(
    npz_path: str | os.PathLike[str], role_manifest_path: str | os.PathLike[str]
) -> Dict[str, object]:
    """Bind the manifest/header before this module touches ``response.npy``."""

    # This import is intentionally delayed.  The data-free adapter validator
    # substitutes this one narrow preflight seam and therefore needs neither
    # PyTorch nor the ordinary RIFT trainer import graph.
    from rift.geraf_b7873200_protocol import load_b7873200_sealed_protocol_identity

    return load_b7873200_sealed_protocol_identity(npz_path, role_manifest_path)


def _stored_response_memmap(path: str):
    """Import the historical stored-member mapper only after sealed preflight."""

    # ``power_baseline_dataset`` imports torch for the historical renderer.
    # Keeping this import at the final opening step lets data-free adapter
    # checks exercise the capability boundary without importing torch.
    from rift.power_baseline_dataset import _stored_member_memmap

    return _stored_member_memmap(path, "response.npy")


def _require_stored_response_member(path: str) -> None:
    """Reject compressed archives before a selected response row can be read."""

    with zipfile.ZipFile(path, "r") as archive:
        try:
            response_info = archive.getinfo("response.npy")
        except KeyError as exc:
            raise ValueError(f"{path} is missing response.npy") from exc
        if response_info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                "sealed B787 response access requires a ZIP_STORED response.npy member; "
                "refusing a streaming fallback that could consume reserved rows"
            )


def _load_b787_geometry_metadata(path: str) -> tuple[Dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    """Load precisely the historical B787 non-response fields after preflight."""

    with np.load(path, allow_pickle=True) as archive:
        required = {"response", "viewpoint_positions", "tx_pos", "rx_pos", "metadata_json"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays {missing}")
        try:
            metadata = json.loads(archive["metadata_json"].item())
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path} has invalid metadata_json") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"{path} metadata_json must decode to an object")
        viewpoint_positions = np.asarray(archive["viewpoint_positions"], dtype=np.float64)
        tx_pos = np.asarray(archive["tx_pos"], dtype=np.float64)
        rx_pos = np.asarray(archive["rx_pos"], dtype=np.float64)
    return metadata, viewpoint_positions, tx_pos, rx_pos


def _identity_response_header(identity: Mapping[str, object]) -> B787ResponseHeader:
    """Normalize the preflight response header without opening its payload."""

    raw_shape = identity.get("response_shape")
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 5:
        raise ValueError("sealed B787 identity has an invalid response shape")
    shape = tuple(_normalize_view_index(value, 1 << 62) for value in raw_shape)
    if any(value <= 0 for value in shape):
        raise ValueError("sealed B787 response dimensions must be positive")
    raw_dtype = identity.get("response_dtype")
    if not isinstance(raw_dtype, str) or not raw_dtype:
        raise ValueError("sealed B787 identity has an invalid response dtype")
    try:
        dtype = np.dtype(raw_dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError("sealed B787 identity has an invalid response dtype") from exc
    if not np.issubdtype(dtype, np.complexfloating):
        raise ValueError("sealed B787 response dtype must be complex")
    return B787ResponseHeader(shape=shape, dtype=dtype)


_RESPONSE_READERS: weakref.WeakKeyDictionary[object, object] = weakref.WeakKeyDictionary()


@dataclass(frozen=True, eq=False)
class SealedB787PowerArrays:
    """B787PowerArrays-compatible geometry with a sealed response accessor.

    ``response`` intentionally provides only ``shape`` and ``dtype``.  It is
    not an ndarray or memmap, so legacy ``arrays.response[index]`` code cannot
    bypass :meth:`response_view`.  The stored-member reader is held only in a
    module-private weak registry, rather than on this returned object, so an
    ordinary caller cannot index a mapper attribute around the role guard.
    """

    path: str
    response: B787ResponseHeader
    viewpoint_positions: np.ndarray
    tx_pos: np.ndarray
    rx_pos: np.ndarray
    metadata: Dict[str, object]
    sealed_protocol_identity: Dict[str, object]
    _allowed_response_view_indices: FrozenSet[int] = field(repr=False)

    @property
    def num_views(self) -> int:
        return int(self.response.shape[0])

    @property
    def num_tx(self) -> int:
        return int(self.response.shape[1])

    @property
    def num_rx(self) -> int:
        return int(self.response.shape[2])

    @property
    def num_chirps(self) -> int:
        return int(self.response.shape[3])

    @property
    def num_freq(self) -> int:
        return int(self.response.shape[4])

    @property
    def allowed_response_view_indices(self) -> FrozenSet[int]:
        """The explicit train/validation capability, for provenance checks."""

        return self._allowed_response_view_indices

    @property
    def response_payload_materialized(self) -> bool:
        """The adapter maps selected rows but never materializes the full payload."""

        return False

    @property
    def response_access_is_restricted(self) -> bool:
        return True

    def response_view(self, index: int, chirp_average: bool = True) -> np.ndarray:
        """Return one authorized view as ``[Tx,Rx,F]`` or ``[Tx,Rx,C,F]``."""

        view_index = _normalize_view_index(index, self.num_views)
        if view_index not in self._allowed_response_view_indices:
            raise PermissionError(
                "B787 response access is restricted to the manifest train/validation roles; "
                f"denied source-view ID: {view_index}"
            )
        mapper = _RESPONSE_READERS.get(self)
        if mapper is None:
            raise RuntimeError("sealed B787 response reader is no longer available")
        view = (mapper.response_view(view_index) if hasattr(mapper, 'response_view')
                else np.asarray(mapper[view_index]))
        expected_shape = (self.num_tx, self.num_rx, self.num_chirps, self.num_freq)
        if view.shape != expected_shape or np.dtype(view.dtype) != self.response.dtype:
            raise RuntimeError("stored B787 response member changed after sealed-header preflight")
        return view.mean(axis=2) if chirp_average else view


def load_b7873200_sealed_power_arrays(
    npz_path: str | os.PathLike[str], role_manifest_path: str | os.PathLike[str]
) -> tuple[SealedB787PowerArrays, Dict[str, object]]:
    """Build a manifest-bound B787 response adapter for a new GeRaF cache.

    The returned identity is the exact policy used to construct the adapter.
    It is suitable for a single versioned target-cache record after that cache
    is complete; callers must not reinterpret a historical cache/checkpoint as
    this sealed protocol.
    """

    # This must remain the first action which could inspect the archive.  The
    # protocol checks metadata/header and ordered manifest roles before this
    # adapter maps the response member.
    identity = _preflight_b7873200_identity(npz_path, role_manifest_path)
    if not isinstance(identity, Mapping):
        raise ValueError("B787 sealed protocol preflight did not return an identity mapping")
    identity = copy.deepcopy(dict(identity))
    if identity.get('antenna_selection'):
        from .rift_dataset import load_object_contract, collection_contract
        public, contract = load_object_contract(npz_path, role_manifest_path)
        if collection_contract(contract) != identity:
            raise ValueError('Selected acquisition changed during adapter preflight')
        arrays = SealedB787PowerArrays(path=os.path.abspath(os.fspath(npz_path)),
            response=_identity_response_header(identity), viewpoint_positions=public['viewpoint_positions'],
            tx_pos=public['tx_pos'], rx_pos=public['rx_pos'], metadata=public['meta'],
            sealed_protocol_identity=identity, _allowed_response_view_indices=_authorized_train_validation_ids(identity))
        _RESPONSE_READERS[arrays] = public['_lazy_response_reader']
        return arrays, copy.deepcopy(identity)
    header = _identity_response_header(identity)
    allowed = _authorized_train_validation_ids(identity)
    path = os.path.abspath(os.fspath(npz_path))

    # No response member access occurs in this metadata/geometry pass.
    metadata, viewpoint_positions, tx_pos, rx_pos = _load_b787_geometry_metadata(path)
    expected_shape = (
        int(viewpoint_positions.shape[0]),
        int(tx_pos.shape[1]) if tx_pos.ndim >= 2 else -1,
        int(rx_pos.shape[1]) if rx_pos.ndim >= 2 else -1,
        int(metadata.get("num_chirps_cpi", -1)),
        int(metadata.get("num_adc_samples", -1)),
    )
    if header.shape != expected_shape:
        raise ValueError(f"response shape {header.shape} != metadata/geometry {expected_shape}")
    if tx_pos.shape != (header.shape[0], header.shape[1], 3):
        raise ValueError("tx_pos must have shape [view,tx,3] matching response")
    if rx_pos.shape != (header.shape[0], header.shape[2], 3):
        raise ValueError("rx_pos must have shape [view,rx,3] matching response")

    # The stored-member check intentionally follows protocol preflight.  It
    # rejects compressed archives rather than accepting a sequential reader
    # which would consume prior sealed response bytes on a late view request.
    _require_stored_response_member(path)
    mapper = _stored_response_memmap(path)
    if tuple(int(value) for value in mapper.shape) != header.shape or np.dtype(mapper.dtype) != header.dtype:
        raise RuntimeError("stored B787 response header changed after sealed protocol preflight")

    arrays = SealedB787PowerArrays(
        path=path,
        response=header,
        viewpoint_positions=viewpoint_positions,
        tx_pos=tx_pos,
        rx_pos=rx_pos,
        metadata=metadata,
        sealed_protocol_identity=identity,
        _allowed_response_view_indices=allowed,
    )
    _RESPONSE_READERS[arrays] = mapper
    return arrays, copy.deepcopy(identity)
