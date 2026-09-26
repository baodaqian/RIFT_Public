"""Shared B787 data and sampling contracts for matched-filter baselines.

RIFT trains on coherent complex frequency responses.  GeRaF v1 instead
trains on the paper's matched-filter magnitude ``|MF|`` (called ``P`` or
"power" in its prose), while RadarSplat trains on a native 2-D squared-power
image.  This module contains only the pieces that must be identical on both
sides of a comparison:

* random-access reads of the uncompressed ``response.npy`` member in a RIFT
  NPZ without materialising the multi-gigabyte array;
* the historical seed-42 fixed-tail split used by the B787 experiments;
* a deterministic GeRaF lensless 3-D sampling grid and a distinct local
  polar RadarSplat target grid, both expressed in the calibrated per-view
  array frame; and
* a small on-disk target-cache contract.

No STL, mesh, point cloud, backprojection, or other geometry truth is read.
The only scene-space centre used below is the acquisition coordinate origin
recorded in the radar metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


LIGHT_SPEED = 299_792_458.0
CACHE_VERSION = 1


def _npy_header(handle) -> Tuple[Tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
    elif version == (3, 0) and hasattr(np.lib.format, "read_array_header_3_0"):
        shape, fortran, dtype = np.lib.format.read_array_header_3_0(handle)
    else:
        raise ValueError(f"unsupported NPY header version {version}")
    return tuple(int(value) for value in shape), bool(fortran), np.dtype(dtype)


def _stored_member_memmap(npz_path: str | os.PathLike[str], member_name: str) -> np.memmap:
    """Memory-map one ``ZIP_STORED`` NPY member in an NPZ archive.

    ``numpy.load(..., mmap_mode='r')`` does not actually memory-map arrays
    inside an NPZ.  RIFT datasets deliberately store ``response.npy`` without
    compression, so its payload is a contiguous byte range in the outer file
    and can be mapped after parsing the ZIP-local and NPY headers.
    """

    path = os.fspath(npz_path)
    with zipfile.ZipFile(path, "r") as archive:
        try:
            info = archive.getinfo(member_name)
        except KeyError as exc:
            raise ValueError(f"{path} is missing {member_name}") from exc
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"{member_name} is compressed; random-access baseline training requires "
                "the project NPZ contract produced by numpy.savez (not savez_compressed)"
            )

    with open(path, "rb") as handle:
        handle.seek(info.header_offset)
        local = handle.read(30)
        if len(local) != 30:
            raise ValueError(f"truncated ZIP local header for {member_name}")
        fields = struct.unpack("<IHHHHHIIIHH", local)
        signature, compression = fields[0], fields[3]
        name_length, extra_length = fields[-2], fields[-1]
        if signature != 0x04034B50 or compression != zipfile.ZIP_STORED:
            raise ValueError(f"invalid ZIP local header for {member_name}")
        payload_start = info.header_offset + 30 + name_length + extra_length
        handle.seek(payload_start)
        shape, fortran, dtype = _npy_header(handle)
        if fortran:
            raise ValueError(f"Fortran-ordered {member_name} is unsupported")
        array_start = handle.tell()

    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    npy_header_bytes = array_start - payload_start
    if expected + npy_header_bytes != info.file_size:
        raise ValueError(
            f"{member_name} byte count disagrees with its NPY header: "
            f"member={info.file_size}, header={npy_header_bytes}, data={expected}"
        )
    return np.memmap(path, dtype=dtype, mode="r", offset=array_start, shape=shape, order="C")


@dataclass
class B787PowerArrays:
    """Geometry, metadata, and lazy raw response access for one B787 NPZ."""

    path: str
    response: np.memmap
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

    def response_view(self, index: int, chirp_average: bool = True) -> np.ndarray:
        """Return one view as ``[Tx,Rx,F]`` (or ``[Tx,Rx,C,F]``)."""

        view = np.asarray(self.response[int(index)])
        return view.mean(axis=2) if chirp_average else view


def load_b787_power_arrays(path: str | os.PathLike[str]) -> B787PowerArrays:
    path = os.path.abspath(os.fspath(path))
    with np.load(path, allow_pickle=True) as archive:
        required = {"response", "viewpoint_positions", "tx_pos", "rx_pos", "metadata_json"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays {missing}")
        metadata = json.loads(archive["metadata_json"].item())
        viewpoint_positions = np.asarray(archive["viewpoint_positions"], dtype=np.float64)
        tx_pos = np.asarray(archive["tx_pos"], dtype=np.float64)
        rx_pos = np.asarray(archive["rx_pos"], dtype=np.float64)
    response = _stored_member_memmap(path, "response.npy")
    expected_shape = (
        viewpoint_positions.shape[0],
        tx_pos.shape[1],
        rx_pos.shape[1],
        int(metadata["num_chirps_cpi"]),
        int(metadata["num_adc_samples"]),
    )
    if response.shape != expected_shape:
        raise ValueError(f"response shape {response.shape} != metadata/geometry {expected_shape}")
    if not np.issubdtype(response.dtype, np.complexfloating):
        raise ValueError(f"response must be complex, got {response.dtype}")
    return B787PowerArrays(
        path=path,
        response=response,
        viewpoint_positions=viewpoint_positions,
        tx_pos=tx_pos,
        rx_pos=rx_pos,
        metadata=metadata,
    )


def frequency_grid_hz(metadata: Mapping[str, object]) -> np.ndarray:
    center = float(metadata["radar_fc_hz"])
    bandwidth = float(metadata["radar_bandwidth_hz"])
    count = int(metadata["num_adc_samples"])
    return (center - bandwidth / 2.0) + np.arange(count, dtype=np.float64) * bandwidth / count


def fixed_tail_split(
    num_views: int,
    num_train: int,
    num_val: int,
    num_test: int = 0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The seed-fixed permutation-tail split used by the B787 baselines."""

    wanted = int(num_train) + int(num_val) + int(num_test)
    if min(num_train, num_val, num_test) < 0 or wanted > num_views:
        raise ValueError(f"invalid split {num_train}/{num_val}/{num_test} for {num_views} views")
    permutation = np.random.default_rng(seed).permutation(num_views)
    val = permutation[num_views - num_val :] if num_val else permutation[:0]
    test_stop = num_views - num_val
    test = permutation[test_stop - num_test : test_stop] if num_test else permutation[:0]
    train = permutation[:num_train]
    return train.astype(np.int64), val.astype(np.int64), test.astype(np.int64)


def _unit(vector: torch.Tensor, name: str) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or float(norm) < 1.0e-12:
        raise ValueError(f"cannot construct lensless frame: degenerate {name}")
    return vector / norm


def _fallback_tangent(primary: torch.Tensor) -> torch.Tensor:
    basis = torch.eye(3, dtype=primary.dtype, device=primary.device)
    candidate = basis[torch.argmin(primary.abs())]
    return _unit(candidate - torch.dot(candidate, primary) * primary, "fallback tangent")


@dataclass(frozen=True)
class LenslessGrid:
    """Paper-shaped primary-ray samples for one calibrated radar viewpoint."""

    points: torch.Tensor
    shape: Tuple[int, int, int]
    ray_origins: torch.Tensor
    primary_direction: torch.Tensor
    azimuth_axis: torch.Tensor
    elevation_axis: torch.Tensor
    depth_m: torch.Tensor
    azimuth_offsets_m: torch.Tensor
    elevation_offsets_m: torch.Tensor
    scene_center: torch.Tensor

    @property
    def flat_points(self) -> torch.Tensor:
        return self.points.reshape(-1, 3)


@dataclass(frozen=True)
class RadarSplatTargetGrid:
    """Local polar azimuth/range observable for the B787 sensor adapter.

    RadarSplat natively predicts a 2-D polar radar image.  The B787 array is
    pointed at a compact target rather than spinning through 360 degrees, so
    its reportable adapter uses a *cropped polar sensor grid* around boresight.
    Elevation is sampled only to perform the documented projection that the
    native RadarSplat renderer leaves unresolved.
    """

    points: torch.Tensor
    shape: Tuple[int, int, int]
    sensor_to_world: torch.Tensor
    range_m: torch.Tensor
    azimuth_rad: torch.Tensor
    elevation_rad: torch.Tensor
    scene_center: torch.Tensor

    @property
    def flat_points(self) -> torch.Tensor:
        return self.points.reshape(-1, 3)

    @property
    def range_resolution_m(self) -> float:
        if self.range_m.numel() < 2:
            raise ValueError("at least two range bins are required to infer resolution")
        return float((self.range_m[1] - self.range_m[0]).detach().cpu())

    @property
    def azimuth_resolution_deg(self) -> float:
        if self.azimuth_rad.numel() < 2:
            raise ValueError("at least two azimuth bins are required to infer resolution")
        return float(torch.rad2deg(self.azimuth_rad[1] - self.azimuth_rad[0]).detach().cpu())

    @property
    def azimuth_start_deg(self) -> float:
        half_bin = 0.5 * (self.azimuth_rad[1] - self.azimuth_rad[0])
        return float(torch.rad2deg(self.azimuth_rad[0] - half_bin).detach().cpu())

    @property
    def azimuth_span_deg(self) -> float:
        return self.azimuth_resolution_deg * int(self.azimuth_rad.numel())


def calibrated_sensor_to_world(
    viewpoint_position,
    tx_positions,
    rx_positions,
    *,
    scene_center=(0.0, 0.0, 0.0),
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Recover the B787 sensor frame from calibrated Tx/Rx coordinates.

    Sensor ``+x`` is boresight toward the acquisition origin, ``+y`` follows
    the measured Tx line (azimuth), and ``+z`` follows the measured Rx line
    (elevation).  The returned matrix maps sensor coordinates into world
    coordinates and is directly consumable by :mod:`rift.radarsplat`.
    """

    vp = torch.as_tensor(viewpoint_position, dtype=dtype, device=device).reshape(3)
    tx = torch.as_tensor(tx_positions, dtype=dtype, device=device).reshape(-1, 3)
    rx = torch.as_tensor(rx_positions, dtype=dtype, device=device).reshape(-1, 3)
    center = torch.as_tensor(scene_center, dtype=dtype, device=device).reshape(3)
    primary = _unit(center - vp, "primary direction")

    tx_span = tx[-1] - tx[0] if len(tx) > 1 else torch.zeros_like(primary)
    tx_tangent = tx_span - torch.dot(tx_span, primary) * primary
    azimuth = (
        _fallback_tangent(primary)
        if float(torch.linalg.vector_norm(tx_tangent)) < 1.0e-12
        else _unit(tx_tangent, "Tx tangent")
    )

    rx_span = rx[-1] - rx[0] if len(rx) > 1 else torch.zeros_like(primary)
    rx_tangent = rx_span - torch.dot(rx_span, primary) * primary
    rx_tangent = rx_tangent - torch.dot(rx_tangent, azimuth) * azimuth
    if float(torch.linalg.vector_norm(rx_tangent)) < 1.0e-12:
        elevation = _unit(torch.linalg.cross(primary, azimuth), "elevation tangent")
    else:
        elevation = _unit(rx_tangent, "Rx tangent")
    if torch.dot(torch.linalg.cross(azimuth, elevation), primary) < 0:
        elevation = -elevation

    pose = torch.eye(4, dtype=dtype, device=device)
    pose[:3, :3] = torch.stack((primary, azimuth, elevation), dim=1)
    pose[:3, 3] = vp
    return pose


def build_radarsplat_target_grid(
    viewpoint_position,
    tx_positions,
    rx_positions,
    *,
    scene_center=(0.0, 0.0, 0.0),
    scene_extent_m: float = 0.15,
    n_azimuth: int = 32,
    n_elevation: int = 32,
    n_range: int = 32,
    azimuth_half_angle_deg: Optional[float] = None,
    elevation_half_angle_deg: Optional[float] = None,
    output_azimuth_resolution_deg: float = 0.9,
    elevation_sampling_resolution_deg: float = 0.9,
    device=None,
    dtype=torch.float32,
) -> RadarSplatTargetGrid:
    """Build the matched-filter query grid for native RadarSplat supervision.

    Unlike :func:`build_lensless_grid`, this is a genuine polar grid: each
    azimuth/elevation cell is a ray from the physical viewpoint.  The default
    angular crop is ``n_azimuth * 0.9 deg`` by default, retaining the pinned
    release's output azimuth sampling while cropping its otherwise mostly
    empty 360-degree scan.  Elevation uses the same documented sensor-adapter
    convention by default.  The cached 2-D target is the sum of its squared
    MF power over elevation, yielding exactly ``[azimuth, range]``.
    """

    if min(n_azimuth, n_elevation, n_range) < 2:
        raise ValueError("RadarSplat target dimensions must each be at least two")
    if scene_extent_m <= 0:
        raise ValueError("scene_extent_m must be positive")
    if output_azimuth_resolution_deg <= 0 or elevation_sampling_resolution_deg <= 0:
        raise ValueError("polar angular resolutions must be positive")
    pose = calibrated_sensor_to_world(
        viewpoint_position,
        tx_positions,
        rx_positions,
        scene_center=scene_center,
        device=device,
        dtype=dtype,
    )
    vp = pose[:3, 3]
    center = torch.as_tensor(scene_center, dtype=dtype, device=device).reshape(3)
    center_range = torch.linalg.vector_norm(center - vp)
    near = float(center_range.detach().cpu()) - float(scene_extent_m)
    far = float(center_range.detach().cpu()) + float(scene_extent_m)
    if near <= 0:
        raise ValueError("scene ROI reaches or crosses the radar origin")

    az_default_half = 0.5 * int(n_azimuth) * float(output_azimuth_resolution_deg)
    el_default_half = 0.5 * int(n_elevation) * float(elevation_sampling_resolution_deg)
    az_half = az_default_half if azimuth_half_angle_deg is None else float(azimuth_half_angle_deg)
    el_half = el_default_half if elevation_half_angle_deg is None else float(elevation_half_angle_deg)
    if not (0.0 < az_half <= 180.0 and 0.0 < el_half < 90.0):
        raise ValueError("invalid RadarSplat azimuth/elevation half-angle")

    range_resolution = (far - near) / int(n_range)
    ranges = near + (
        torch.arange(n_range, dtype=dtype, device=device) + 0.5
    ) * range_resolution
    az_edges = torch.linspace(-az_half, az_half, n_azimuth + 1, dtype=dtype, device=device)
    el_edges = torch.linspace(-el_half, el_half, n_elevation + 1, dtype=dtype, device=device)
    azimuth = torch.deg2rad(0.5 * (az_edges[:-1] + az_edges[1:]))
    elevation = torch.deg2rad(0.5 * (el_edges[:-1] + el_edges[1:]))

    el_grid, az_grid, range_grid = torch.meshgrid(
        elevation, azimuth, ranges, indexing="ij"
    )
    cos_el = torch.cos(el_grid)
    directions_sensor = torch.stack(
        (
            cos_el * torch.cos(az_grid),
            cos_el * torch.sin(az_grid),
            torch.sin(el_grid),
        ),
        dim=-1,
    )
    directions_world = directions_sensor @ pose[:3, :3].transpose(0, 1)
    points = vp + range_grid[..., None] * directions_world
    return RadarSplatTargetGrid(
        points=points,
        shape=(n_elevation, n_azimuth, n_range),
        sensor_to_world=pose,
        range_m=ranges,
        azimuth_rad=azimuth,
        elevation_rad=elevation,
        scene_center=center,
    )


def build_lensless_grid(
    viewpoint_position,
    tx_positions,
    rx_positions,
    *,
    scene_center=(0.0, 0.0, 0.0),
    scene_extent_m: float = 0.15,
    n_azimuth: int = 32,
    n_elevation: int = 32,
    n_depth: int = 32,
    aperture_scale: float = 1.0,
    device=None,
    dtype=torch.float32,
) -> LenslessGrid:
    """Construct ``N_ray=N_azimuth*N_elevation`` parallel primary rays.

    The two tangent axes are recovered from the *measured* Tx/Rx coordinates,
    not from an assumed array convention.  Ray origins span the calibrated
    virtual MIMO aperture ``(tx + rx)/2``.  Samples cover the scene centre
    plus/minus ``scene_extent_m`` along the primary direction.  The defaults
    reproduce GeRaF's 1024 rays x 32 temporal/depth samples.
    """

    if min(n_azimuth, n_elevation, n_depth) <= 0:
        raise ValueError("lensless grid dimensions must be positive")
    if scene_extent_m <= 0 or aperture_scale <= 0:
        raise ValueError("scene_extent_m and aperture_scale must be positive")

    vp = torch.as_tensor(viewpoint_position, dtype=dtype, device=device).reshape(3)
    tx = torch.as_tensor(tx_positions, dtype=dtype, device=device).reshape(-1, 3)
    rx = torch.as_tensor(rx_positions, dtype=dtype, device=device).reshape(-1, 3)
    center = torch.as_tensor(scene_center, dtype=dtype, device=device).reshape(3)
    primary = _unit(center - vp, "primary direction")

    tx_span = tx[-1] - tx[0] if len(tx) > 1 else torch.zeros_like(primary)
    tx_tangent = tx_span - torch.dot(tx_span, primary) * primary
    azimuth = _fallback_tangent(primary) if float(torch.linalg.vector_norm(tx_tangent)) < 1e-12 else _unit(tx_tangent, "Tx tangent")

    rx_span = rx[-1] - rx[0] if len(rx) > 1 else torch.zeros_like(primary)
    rx_tangent = rx_span - torch.dot(rx_span, primary) * primary
    rx_tangent = rx_tangent - torch.dot(rx_tangent, azimuth) * azimuth
    if float(torch.linalg.vector_norm(rx_tangent)) < 1e-12:
        elevation = _unit(torch.linalg.cross(primary, azimuth), "elevation tangent")
    else:
        elevation = _unit(rx_tangent, "Rx tangent")
    # Make the frame right-handed and deterministic without changing which
    # measured array line defines each named axis.
    if torch.dot(torch.linalg.cross(azimuth, elevation), primary) < 0:
        elevation = -elevation

    virtual = 0.5 * (tx[:, None, :] + rx[None, :, :])
    virtual_offset = virtual.reshape(-1, 3) - vp
    az_projection = virtual_offset @ azimuth
    el_projection = virtual_offset @ elevation
    az_lo, az_hi = aperture_scale * az_projection.min(), aperture_scale * az_projection.max()
    el_lo, el_hi = aperture_scale * el_projection.min(), aperture_scale * el_projection.max()
    # Degenerate one-dimensional arrays still need a finite ray bank.  Use the
    # scene ROI only for the missing axis; this is recorded by the cache spec.
    if float((az_hi - az_lo).abs()) < 1.0e-9:
        az_lo, az_hi = az_projection.new_tensor(-scene_extent_m), az_projection.new_tensor(scene_extent_m)
    if float((el_hi - el_lo).abs()) < 1.0e-9:
        el_lo, el_hi = el_projection.new_tensor(-scene_extent_m), el_projection.new_tensor(scene_extent_m)

    az_offsets = torch.linspace(az_lo, az_hi, n_azimuth, dtype=dtype, device=device)
    el_offsets = torch.linspace(el_lo, el_hi, n_elevation, dtype=dtype, device=device)
    centre_range = torch.linalg.vector_norm(center - vp)
    depth = torch.linspace(
        centre_range - scene_extent_m,
        centre_range + scene_extent_m,
        n_depth,
        dtype=dtype,
        device=device,
    )
    el_grid, az_grid = torch.meshgrid(el_offsets, az_offsets, indexing="ij")
    origins = vp + az_grid[..., None] * azimuth + el_grid[..., None] * elevation
    points = origins[..., None, :] + depth[None, None, :, None] * primary
    return LenslessGrid(
        points=points,
        shape=(n_elevation, n_azimuth, n_depth),
        ray_origins=origins,
        primary_direction=primary,
        azimuth_axis=azimuth,
        elevation_axis=elevation,
        depth_m=depth,
        azimuth_offsets_m=az_offsets,
        elevation_offsets_m=el_offsets,
        scene_center=center,
    )


def file_sha256(path: str | os.PathLike[str], chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def split_manifest(
    arrays: B787PowerArrays,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
    *,
    seed: int,
    dataset_sha256: Optional[str] = None,
) -> Dict[str, object]:
    manifest = {
        "version": CACHE_VERSION,
        "strategy": "pcg64_seeded_fixed_tail",
        "seed": int(seed),
        "dataset_path": os.path.abspath(arrays.path),
        "dataset_size_bytes": int(os.path.getsize(arrays.path)),
        "dataset_sha256": dataset_sha256,
        "response_shape": list(arrays.response.shape),
        "response_dtype": str(arrays.response.dtype),
        "train_indices": [int(value) for value in train_indices],
        "validation_indices": [int(value) for value in val_indices],
        "test_indices": [int(value) for value in test_indices],
    }
    roles = [manifest["train_indices"], manifest["validation_indices"], manifest["test_indices"]]
    flat = [value for role in roles for value in role]
    if len(flat) != len(set(flat)):
        raise ValueError("split roles overlap")
    return manifest


def canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)


def target_cache_path(cache_root: str | os.PathLike[str], view_index: int) -> Path:
    return Path(cache_root) / "views" / f"view_{int(view_index):06d}.npz"


def atomic_save_target(path: str | os.PathLike[str], **arrays) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # numpy appends .npz when the filename does not already end in it, so the
    # temporary name deliberately retains that suffix.
    temporary = destination.with_name(destination.stem + f".tmp.{os.getpid()}.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, destination)


def load_target_view(cache_root: str | os.PathLike[str], view_index: int) -> Dict[str, np.ndarray]:
    path = target_cache_path(cache_root, view_index)
    if not path.exists():
        raise FileNotFoundError(f"missing prepared power target {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def iter_split_indices(manifest: Mapping[str, object], roles: Iterable[str]) -> Iterable[int]:
    key_for_role = {"train": "train_indices", "validation": "validation_indices", "test": "test_indices"}
    for role in roles:
        try:
            key = key_for_role[role]
        except KeyError as exc:
            raise ValueError(f"unknown split role {role!r}") from exc
        yield from (int(value) for value in manifest[key])
