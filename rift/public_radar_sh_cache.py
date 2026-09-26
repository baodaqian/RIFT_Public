"""Immutable directional real-SH basis caches for PublicRadar datasets.

The learned scene coefficients are optimizer state and must never be cached.
This module caches only the dataset-derived basis values
``Y_lm(theta_view, phi_view)``.  Rows are indexed by the canonical NPZ view
index, before train/validation/test selection or epoch permutation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from rift.spherical_harmonics import real_sh_basis


SCHEMA = "rift.public_radar_directional_sh_cache_v1"
MAX_DEGREE = 6
MAX_BASIS = (MAX_DEGREE + 1) ** 2
BASIS_FILENAME = "real_sh_degree6_f32.npy"
THETA_FILENAME = "theta_rad_f32.npy"
PHI_FILENAME = "phi_rad_f32.npy"
MANIFEST_FILENAME = "manifest.json"
POSITIONS_FILENAME = "viewpoint_positions.npy"
BASIS_ORDER = "degree_major_then_m_negative_to_positive"
ANGLE_SOURCE = "viewpoint_positions_float64_then_dataset_float32"
THETA_CONVENTION = "acos(z / norm(position))"
PHI_CONVENTION = "atan2(y, x)"
VIEW_INDEX_SPACE = "canonical_npz_row_before_role_selection"


def canonical_theta_phi(viewpoint_positions):
    """Reproduce ``PecSphereNPZDataset``'s exact direction convention.

    Geometry is evaluated in float64, then each angle is rounded to float32
    because the dataset exposes float32 direction tensors to scene models.
    Stored PublicRadar azimuth/elevation columns are intentionally not used.
    """

    positions = np.asarray(viewpoint_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError("viewpoint_positions must have nonempty shape [views, 3]")
    if not np.isfinite(positions).all():
        raise ValueError("viewpoint_positions contains a non-finite value")
    radii = np.linalg.norm(positions, axis=1)
    if not np.isfinite(radii).all() or np.any(radii <= 0.0):
        raise ValueError("viewpoint_positions contains a zero or invalid radius")
    theta = np.arccos(np.clip(positions[:, 2] / radii, -1.0, 1.0)).astype(
        np.float32
    )
    phi = np.arctan2(positions[:, 1], positions[:, 0]).astype(np.float32)
    return np.ascontiguousarray(theta), np.ascontiguousarray(phi)


@torch.no_grad()
def basis_from_angles(theta, phi, max_degree=MAX_DEGREE):
    """Return row-major float32 basis values with shape ``[views, H]``."""

    max_degree = int(max_degree)
    if not 0 <= max_degree <= MAX_DEGREE:
        raise ValueError(f"cache basis degree must be in [0, {MAX_DEGREE}]")
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    if (
        theta.dtype != np.float32
        or phi.dtype != np.float32
        or theta.ndim != 1
        or phi.ndim != 1
        or theta.shape != phi.shape
        or theta.size == 0
    ):
        raise ValueError("theta and phi must be matching nonempty float32 vectors")
    theta_tensor = torch.from_numpy(np.ascontiguousarray(theta))
    phi_tensor = torch.from_numpy(np.ascontiguousarray(phi))
    basis = real_sh_basis(theta_tensor, phi_tensor, max_degree)
    expected = ((max_degree + 1) ** 2, theta.size)
    if basis.shape != expected or basis.dtype != torch.float32:
        raise RuntimeError("real-SH implementation returned an unexpected cache tensor")
    values = basis.transpose(0, 1).contiguous().cpu().numpy()
    if not np.isfinite(values).all():
        raise RuntimeError("computed real-SH cache contains a non-finite value")
    return values


def _manifest(dataset_name, view_count, viewpoint_dtype):
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "dataset_name": str(dataset_name),
        "view_count": int(view_count),
        "max_degree": MAX_DEGREE,
        "basis_count": MAX_BASIS,
        "basis_dtype": "float32",
        "basis_shape": [int(view_count), MAX_BASIS],
        "basis_order": BASIS_ORDER,
        "angle_dtype": "float32",
        "angle_source": ANGLE_SOURCE,
        "theta_convention": THETA_CONVENTION,
        "phi_convention": PHI_CONVENTION,
        "view_index_space": VIEW_INDEX_SPACE,
        "viewpoint_positions_source_dtype": str(np.dtype(viewpoint_dtype)),
        "basis_file": BASIS_FILENAME,
        "theta_file": THETA_FILENAME,
        "phi_file": PHI_FILENAME,
        "viewpoint_positions_file": POSITIONS_FILENAME,
    }


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def materialize_cache(cache_dir, viewpoint_positions, dataset_name):
    """Create an immutable cache directory, or validate an existing one.

    Existing artifacts are never overwritten.  A temporary sibling directory
    is fully validated before its atomic rename to ``cache_dir``.
    """

    cache_dir = Path(cache_dir).resolve()
    positions = np.asarray(viewpoint_positions)
    theta, phi = canonical_theta_phi(positions)
    if cache_dir.exists():
        cache = PublicRadarSHBasisCache(
            cache_dir, positions, expected_dataset_name=dataset_name
        )
        return dict(cache.manifest)

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{cache_dir.name}.tmp.", dir=cache_dir.parent)
    )
    try:
        basis = basis_from_angles(theta, phi, MAX_DEGREE)
        np.save(temporary / BASIS_FILENAME, basis, allow_pickle=False)
        np.save(temporary / THETA_FILENAME, theta, allow_pickle=False)
        np.save(temporary / PHI_FILENAME, phi, allow_pickle=False)
        np.save(temporary / POSITIONS_FILENAME, positions, allow_pickle=False)
        manifest = _manifest(dataset_name, positions.shape[0], positions.dtype)
        _write_json(temporary / MANIFEST_FILENAME, manifest)
        PublicRadarSHBasisCache(
            temporary, positions, expected_dataset_name=dataset_name
        )
        os.replace(temporary, cache_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


class PublicRadarSHBasisCache:
    """Read-only, memory-mapped degree-6 directional basis sidecar."""

    def __init__(self, cache_dir, viewpoint_positions, expected_dataset_name):
        self.root = Path(cache_dir).resolve()
        manifest_path = self.root / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing SH cache manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_name = str(expected_dataset_name)
        positions_expected = np.asarray(viewpoint_positions)
        required = {
            "schema": SCHEMA,
            "schema_version": 1,
            "dataset_name": expected_name,
            "max_degree": MAX_DEGREE,
            "basis_count": MAX_BASIS,
            "basis_dtype": "float32",
            "basis_order": BASIS_ORDER,
            "angle_dtype": "float32",
            "angle_source": ANGLE_SOURCE,
            "theta_convention": THETA_CONVENTION,
            "phi_convention": PHI_CONVENTION,
            "view_index_space": VIEW_INDEX_SPACE,
            "viewpoint_positions_source_dtype": str(positions_expected.dtype),
            "basis_file": BASIS_FILENAME,
            "theta_file": THETA_FILENAME,
            "phi_file": PHI_FILENAME,
            "viewpoint_positions_file": POSITIONS_FILENAME,
        }
        for key, expected in required.items():
            if self.manifest.get(key) != expected:
                raise ValueError(
                    f"SH cache manifest {key} changed: "
                    f"{self.manifest.get(key)!r} != {expected!r}"
                )

        theta_expected, phi_expected = canonical_theta_phi(viewpoint_positions)
        self.view_count = int(theta_expected.size)
        if int(self.manifest.get("view_count", -1)) != self.view_count:
            raise ValueError("SH cache view count disagrees with the dataset")
        if self.manifest.get("basis_shape") != [self.view_count, MAX_BASIS]:
            raise ValueError("SH cache manifest basis shape is invalid")

        self._theta = np.load(
            self.root / THETA_FILENAME, mmap_mode="r", allow_pickle=False
        )
        self._phi = np.load(
            self.root / PHI_FILENAME, mmap_mode="r", allow_pickle=False
        )
        self._basis = np.load(
            self.root / BASIS_FILENAME, mmap_mode="r", allow_pickle=False
        )
        self._positions = np.load(
            self.root / POSITIONS_FILENAME, mmap_mode="r", allow_pickle=False
        )
        if (
            self._positions.shape != positions_expected.shape
            or self._positions.dtype != positions_expected.dtype
            or not np.array_equal(self._positions, positions_expected)
        ):
            raise ValueError("SH cache viewpoint positions disagree with the dataset")
        if (
            self._theta.dtype != np.float32
            or self._phi.dtype != np.float32
            or self._theta.shape != (self.view_count,)
            or self._phi.shape != (self.view_count,)
            or not np.array_equal(self._theta, theta_expected)
            or not np.array_equal(self._phi, phi_expected)
        ):
            raise ValueError("SH cache angles disagree with canonical viewpoint positions")
        if self._basis.dtype != np.float32 or self._basis.shape != (
            self.view_count,
            MAX_BASIS,
        ):
            raise ValueError("SH cache basis array has an invalid dtype or shape")
        if not np.isfinite(self._basis).all():
            raise ValueError("SH cache basis contains a non-finite value")

        expected_basis = basis_from_angles(theta_expected, phi_expected, MAX_DEGREE)
        if not np.array_equal(np.asarray(self._basis), expected_basis):
            raise ValueError(
                "SH cache basis order or values disagree with canonical full-basis evaluation"
            )

    def contract(self):
        """Return checkpoint-safe cache provenance without mutable arrays."""

        return {
            "schema": self.manifest["schema"],
            "schema_version": self.manifest["schema_version"],
            "dataset_name": self.manifest["dataset_name"],
            "view_count": self.view_count,
            "max_degree": MAX_DEGREE,
            "basis_count": MAX_BASIS,
            "basis_dtype": "float32",
            "basis_order": self.manifest["basis_order"],
            "angle_source": self.manifest["angle_source"],
            "theta_convention": self.manifest["theta_convention"],
            "phi_convention": self.manifest["phi_convention"],
            "view_index_space": self.manifest["view_index_space"],
            "viewpoint_positions_source_dtype": self.manifest[
                "viewpoint_positions_source_dtype"
            ],
            "cache_dir": str(self.root),
        }

    def basis_rows(self, view_indices, degree, device, dtype=torch.float32):
        """Load selected rows and return ``[H, B]`` on the requested device."""

        degree = int(degree)
        if not 0 <= degree <= MAX_DEGREE:
            raise ValueError(f"requested SH degree must be in [0, {MAX_DEGREE}]")
        indices = np.asarray(view_indices)
        if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError("view_indices must be a nonempty integer vector")
        indices = indices.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= self.view_count):
            raise IndexError("SH cache view index is outside the canonical dataset")
        basis_count = (degree + 1) ** 2
        rows = np.array(
            self._basis[indices, :basis_count], dtype=np.float32, copy=True, order="C"
        )
        return torch.from_numpy(rows).to(device=device, dtype=dtype).transpose(0, 1).contiguous()


class IndexedPublicRadarDataset:
    """Attach canonical IDs after proving every item/ID direction pairing."""

    def __init__(self, base_dataset, canonical_indices, viewpoint_positions):
        self.base_dataset = base_dataset
        indices = np.asarray(canonical_indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("canonical_indices must be a one-dimensional integer array")
        self.canonical_indices = indices.astype(np.int64, copy=True)
        if len(base_dataset) != self.canonical_indices.size:
            raise ValueError("dataset length disagrees with canonical_indices")
        theta, phi = canonical_theta_phi(viewpoint_positions)
        if np.any(self.canonical_indices < 0) or np.any(
            self.canonical_indices >= theta.size
        ):
            raise IndexError("canonical_indices contains an out-of-range view id")
        self._canonical_theta = theta
        self._canonical_phi = phi
        self.canonical_indices.setflags(write=False)
        for local_index in range(len(self.base_dataset)):
            self._validated_item(local_index)

    def __len__(self):
        return len(self.base_dataset)

    @staticmethod
    def _direction_bits(value, field_name):
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.shape != (1,)
        ):
            raise ValueError(
                f"PublicRadar {field_name} must be a length-one CPU float32 tensor"
            )
        return int(value.detach().numpy().view(np.uint32)[0])

    def _validated_item(self, index):
        item = self.base_dataset[index]
        if not isinstance(item, tuple) or len(item) != 7:
            raise ValueError("PublicRadar base item must be a seven-tuple")
        view_id = int(self.canonical_indices[index])
        expected_phi_bits = int(
            np.asarray([self._canonical_phi[view_id]], dtype=np.float32).view(np.uint32)[0]
        )
        expected_theta_bits = int(
            np.asarray([self._canonical_theta[view_id]], dtype=np.float32).view(np.uint32)[0]
        )
        if (
            self._direction_bits(item[1], "dphi") != expected_phi_bits
            or self._direction_bits(item[2], "dtheta") != expected_theta_bits
        ):
            raise ValueError(
                "PublicRadar item direction disagrees with its canonical global view id "
                f"at local index {index} (global id {view_id})"
            )
        return item, view_id

    def __getitem__(self, index):
        item, view_id = self._validated_item(index)
        return (view_id,) + item[1:]


def item_view_indices(items):
    """Extract strict canonical view ids from indexed PublicRadar items."""

    values = []
    for item in items:
        if not isinstance(item, tuple) or len(item) != 7:
            raise ValueError("PublicRadar cached item must be a seven-tuple")
        value = item[0]
        if isinstance(value, torch.Tensor):
            if value.ndim != 0 or value.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise ValueError("cached item view index tensor must be scalar integer")
            value = int(value.item())
        elif isinstance(value, (int, np.integer)):
            value = int(value)
        else:
            raise ValueError("cached item is missing its canonical view index")
        values.append(value)
    if not values:
        raise ValueError("cached view batch must not be empty")
    return np.asarray(values, dtype=np.int64)
