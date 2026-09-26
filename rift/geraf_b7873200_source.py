"""One narrow data-source facade for the corrected sealed GeRaF B787 lane.

Historical GeRaF loaders remain intentionally untouched because they expose
arbitrary response rows and bind their caches to legacy digest fields.  This
facade instead returns the restricted adapter used by the new B7873200 target
preparer.  It supplies geometry/metadata plus a train-or-validation-only
``response_view`` capability; it never exposes raw reserved-test or unused
responses.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np

from rift.geraf_b7873200_adapter import (
    B787ResponseHeader,
    SealedB787PowerArrays,
    _identity_response_header,
    _load_b787_geometry_metadata,
    load_b7873200_sealed_power_arrays,
)
from rift.geraf_b7873200_protocol import (
    B787_3200_NUM_TEST,
    B787_3200_NUM_TRAIN,
    B787_3200_NUM_UNUSED,
    B787_3200_NUM_VALIDATION,
    B787_3200_NUM_VIEWS,
    load_b7873200_sealed_protocol_identity,
)


@dataclass(frozen=True)
class B7873200MetadataArrays:
    """B787 geometry/header surface with no raw response capability at all."""

    path: str
    response: B787ResponseHeader
    viewpoint_positions: np.ndarray
    tx_pos: np.ndarray
    rx_pos: np.ndarray
    metadata: Dict[str, object]

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
    def response_payload_materialized(self) -> bool:
        return False

    def response_view(self, _index: int, chirp_average: bool = True) -> np.ndarray:
        del chirp_average
        raise PermissionError(
            "this B7873200 metadata-only source has no raw response capability; "
            "target preparation must use the sealed development adapter"
        )


@dataclass(frozen=True)
class B7873200DevelopmentSource:
    """Sealed B787 geometry and the only permitted response accessor."""

    arrays: SealedB787PowerArrays | B7873200MetadataArrays
    identity: Dict[str, object]

    def response_view(self, index: int) -> np.ndarray:
        """Return one chirp-averaged train/validation row through the guard."""

        return self.arrays.response_view(index, chirp_average=True)


def load_b7873200_development_source(
    npz_path: str | os.PathLike[str], role_manifest_path: str | os.PathLike[str]
) -> B7873200DevelopmentSource:
    """Preflight the canonical archive then return its restricted adapter.

    ``load_b7873200_sealed_power_arrays`` performs the ordering-critical
    manifest/header preflight before opening the stored response member.  The
    checks below are redundant semantic guards for the versioned GeRaF route,
    not file content checks: they prevent a future caller from repurposing this
    facade for a different acquisition or split.
    """

    arrays, identity = load_b7873200_sealed_power_arrays(npz_path, role_manifest_path)
    _validate_source_semantics(arrays, identity)
    return B7873200DevelopmentSource(arrays=arrays, identity=copy.deepcopy(identity))


def _validate_source_semantics(
    arrays: SealedB787PowerArrays | B7873200MetadataArrays, identity: Dict[str, object]
) -> None:
    """Validate only explicit acquisition and role semantics for this lane."""

    expected_shape = (B787_3200_NUM_VIEWS, 16, 16, 1, 600)
    if tuple(int(value) for value in arrays.response.shape) != expected_shape:
        raise ValueError(
            "B7873200 GeRaF source requires response header "
            f"{expected_shape}, got {arrays.response.shape}"
        )
    if arrays.response.dtype != np.dtype(np.complex64):
        raise ValueError("B7873200 GeRaF source requires complex64 radar responses")
    roles = identity.get("role_ids")
    if not isinstance(roles, dict):
        raise ValueError("B7873200 GeRaF source identity lacks role IDs")
    expected_counts = {
        "train": B787_3200_NUM_TRAIN,
        "validation": B787_3200_NUM_VALIDATION,
        "reserved_test": B787_3200_NUM_TEST,
        "unused": B787_3200_NUM_UNUSED,
    }
    for role, count in expected_counts.items():
        values = roles.get(role)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"B7873200 GeRaF source has invalid {role!r} role length")
    if isinstance(arrays, SealedB787PowerArrays):
        authorized = frozenset(int(value) for value in roles["train"] + roles["validation"])
        if arrays.allowed_response_view_indices != authorized:
            raise ValueError("B7873200 GeRaF adapter does not expose exactly train plus validation")


def load_b7873200_metadata_source(
    npz_path: str | os.PathLike[str], role_manifest_path: str | os.PathLike[str]
) -> B7873200DevelopmentSource:
    """Return sealed B787 geometry/header without mapping any response member.

    The trainer needs only cached native-MF targets plus calibrated geometry, so
    it uses this stricter source.  Manifest/header validation remains first;
    afterward it loads only the non-response NPZ members needed to recheck the
    frozen target geometry.
    """

    identity = load_b7873200_sealed_protocol_identity(npz_path, role_manifest_path)
    path = os.path.abspath(os.fspath(npz_path))
    metadata, viewpoint_positions, tx_pos, rx_pos = _load_b787_geometry_metadata(path)
    arrays = B7873200MetadataArrays(
        path=path,
        response=_identity_response_header(identity),
        viewpoint_positions=viewpoint_positions,
        tx_pos=tx_pos,
        rx_pos=rx_pos,
        metadata=metadata,
    )
    _validate_source_semantics(arrays, identity)
    return B7873200DevelopmentSource(arrays=arrays, identity=copy.deepcopy(identity))
