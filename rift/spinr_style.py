"""Disclosed SpINR-style neural baseline primitives.

This module is deliberately separate from the legacy :mod:`rift.model` and
``PecSphereNPZDataset`` paths.  The former predicts a complex RIFT field; the
latter converts the archive to float32 magnitude/phase tensors.  The
SpINR-style B787 adapter instead needs a signed-real field and the raw complex
FMCW observations, with float64 geometry and frequency coordinates.

Nothing here changes an existing RIFT recipe.  Consumers must opt in through
``train_spinr_style.py`` after the normal sealed-NPZ manifest preflight has
completed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from rift.npz_dataset import (
    build_freqs,
    iter_npz_response_views,
    restrict_npz_response_views,
)


SPINR_STYLE_METHOD_ID = "spinr_style_inr"
SPINR_STYLE_RECIPE_ID = "b78710k_spinr_style_inr_pm_v1"
SPINR_STYLE_SUPPORT_M = 0.15
SPINR_STYLE_ENCODING_BANDS = 6
SPINR_STYLE_INPUT_FEATURES = 39
SPINR_STYLE_HIDDEN_LAYERS = 6
SPINR_STYLE_HIDDEN_WIDTH = 840
SPINR_STYLE_PARAMETER_COUNT = 3_566_641
SPINR_STYLE_PHASE_SIGN = -1.0
SPINR_STYLE_RANGE_MODEL = "product"
SPINR_STYLE_ACQUISITION_SCHEMA = "spinr_style_b787_acquisition_v1"
SPINR_STYLE_GAUSS2_RECIPE_ID = "b78710k_spinr_style_inr_pm_gauss2_v2"
SPINR_STYLE_GAUSS3_RECIPE_ID = "b78710k_spinr_style_inr_pm_gauss3_v1"
SPINR_STYLE_GAUSS2_BASE_GRID_SIZE = 48
SPINR_STYLE_GAUSS2_NODES_PER_CELL = 2
SPINR_STYLE_GAUSS3_NODES_PER_CELL = 3
SPINR_STYLE_GAUSS5_NODES_PER_CELL = 5


def spinr_style_parameter_count(
    input_features: int = SPINR_STYLE_INPUT_FEATURES,
    hidden_layers: int = SPINR_STYLE_HIDDEN_LAYERS,
    hidden_width: int = SPINR_STYLE_HIDDEN_WIDTH,
) -> int:
    """Return the scalar count of the fixed scalar-real MLP.

    ``hidden_layers`` counts ``Linear + ReLU`` hidden blocks.  Every linear
    layer has a bias and the final scalar head is linear.
    """

    if input_features <= 0 or hidden_layers <= 0 or hidden_width <= 0:
        raise ValueError("network dimensions must be positive")
    return (
        (int(input_features) + 1) * int(hidden_width)
        + (int(hidden_layers) - 1) * (int(hidden_width) + 1) * int(hidden_width)
        + (int(hidden_width) + 1)
    )


def encode_spinr_style_coordinates(
    coordinates_m: torch.Tensor,
    *,
    support_m: float = SPINR_STYLE_SUPPORT_M,
    bands: int = SPINR_STYLE_ENCODING_BANDS,
) -> torch.Tensor:
    """Encode world coordinates as raw normalized XYZ plus fixed Fourier bands.

    The output is exactly 39 features for the frozen B787 recipe.  Positions
    outside the reference cube are rejected rather than silently clipped:
    clipping would invent a different physical support.
    """

    coords = torch.as_tensor(coordinates_m)
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError("coordinates must have shape [points, 3]")
    if not math.isfinite(float(support_m)) or float(support_m) <= 0:
        raise ValueError("support_m must be finite and positive")
    if int(bands) <= 0:
        raise ValueError("bands must be positive")
    normalized = coords / float(support_m)
    if not torch.isfinite(normalized).all():
        raise ValueError("coordinates must be finite")
    # A tiny tolerance avoids rejecting the exact floating-point cube boundary.
    if bool((normalized.abs() > 1.0 + 1e-6).any()):
        raise ValueError("coordinates lie outside the declared SpINR-style support cube")
    pieces = [normalized]
    for band in range(int(bands)):
        phase = math.pi * float(2 ** band) * normalized
        pieces.extend((torch.sin(phase), torch.cos(phase)))
    encoded = torch.cat(pieces, dim=-1)
    expected = 3 + 6 * int(bands)
    if encoded.shape[-1] != expected:
        raise AssertionError("unexpected SpINR-style encoding width")
    return encoded


class SpinrStyleINR(nn.Module):
    """Six-hidden-layer signed-real Fourier-feature field used by this baseline."""

    def __init__(
        self,
        *,
        support_m: float = SPINR_STYLE_SUPPORT_M,
        bands: int = SPINR_STYLE_ENCODING_BANDS,
        hidden_layers: int = SPINR_STYLE_HIDDEN_LAYERS,
        hidden_width: int = SPINR_STYLE_HIDDEN_WIDTH,
    ) -> None:
        super().__init__()
        self.support_m = float(support_m)
        self.bands = int(bands)
        self.hidden_layers = int(hidden_layers)
        self.hidden_width = int(hidden_width)
        self.input_features = 3 + 6 * self.bands
        if self.input_features != SPINR_STYLE_INPUT_FEATURES:
            raise ValueError("the selected independent SpINR-style recipe fixes 39 input features")
        if self.hidden_layers != SPINR_STYLE_HIDDEN_LAYERS:
            raise ValueError("the selected independent SpINR-style recipe fixes six hidden layers")
        if self.hidden_width != SPINR_STYLE_HIDDEN_WIDTH:
            raise ValueError("the selected independent SpINR-style recipe fixes width 840")

        layers: list[nn.Linear] = []
        in_features = self.input_features
        for _ in range(self.hidden_layers):
            layers.append(nn.Linear(in_features, self.hidden_width, bias=True))
            in_features = self.hidden_width
        self.hidden = nn.ModuleList(layers)
        self.head = nn.Linear(self.hidden_width, 1, bias=True)
        self.reset_parameters()
        if self.trainable_parameter_count() != SPINR_STYLE_PARAMETER_COUNT:
            raise AssertionError("SpINR-style parameter count drifted from the selected recipe")

    def reset_parameters(self) -> None:
        for layer in self.hidden:
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        # A small nonzero head gives every preceding layer a gradient on the
        # first data-fit update; an all-zero head would starve them.
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01 / math.sqrt(self.hidden_width))
        nn.init.zeros_(self.head.bias)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, coordinates_m: torch.Tensor) -> torch.Tensor:
        # The physical quadrature grid deliberately remains float64, while the
        # neural field is deliberately FP32.  Convert only the neural input;
        # geometry, phase, integration weights, and the renderer never pass
        # through this path and stay in their prescribed FP64/complex128 form.
        parameter = self.hidden[0].weight
        if torch.is_tensor(coordinates_m):
            # ``Tensor.to`` preserves a possible coordinate autograd path;
            # focused physics tests use that property even though production
            # quadrature locations are fixed buffers.
            coordinates_m = coordinates_m.to(device=parameter.device, dtype=parameter.dtype)
        else:
            coordinates_m = torch.as_tensor(
                coordinates_m, device=parameter.device, dtype=parameter.dtype)
        values = encode_spinr_style_coordinates(
            coordinates_m, support_m=self.support_m, bands=self.bands)
        for layer in self.hidden:
            values = torch.relu(layer(values))
        # Signed real reflectivity; deliberately no complex head or learned gain.
        return self.head(values).squeeze(-1)


def midpoint_grid(
    grid_size: int,
    *,
    support_m: float = SPINR_STYLE_SUPPORT_M,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, float]:
    """Return midpoint quadrature nodes and one constant cell volume."""

    grid_size = int(grid_size)
    if grid_size <= 1:
        raise ValueError("grid_size must be greater than one")
    if not math.isfinite(float(support_m)) or float(support_m) <= 0:
        raise ValueError("support_m must be finite and positive")
    pitch = 2.0 * float(support_m) / grid_size
    axis = (-float(support_m) + (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) * pitch)
    points = torch.cartesian_prod(axis, axis, axis)
    return points, float(pitch ** 3)


def gauss_legendre_cell_grid(
    grid_size: int,
    *,
    nodes_per_cell: int = SPINR_STYLE_GAUSS2_NODES_PER_CELL,
    support_m: float = SPINR_STYLE_SUPPORT_M,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounded tensor Gauss–Legendre cell nodes and physical weights.

    The parent support is still divided into ``grid_size**3`` cells.  The
    nodes are an integration rule inside each cell, not a new trainable voxel
    grid.  A two-node rule has uniform weights and is the opt-in SpINR repair;
    the three-node rule is retained as an independent frozen reference.
    """

    from rift.spinr_quadrature import gauss_legendre_cell_grid_arrays

    points_np, weights_np = gauss_legendre_cell_grid_arrays(
        grid_size,
        nodes_per_cell=nodes_per_cell,
        support_m=support_m,
    )
    points = torch.as_tensor(points_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
    if points.ndim != 2 or points.shape[-1] != 3 or weights.shape != (points.shape[0],):
        raise AssertionError("cell quadrature returned incompatible point/weight shapes")
    return points, weights


def metadata_frequency_grid(meta: Mapping[str, object], *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Construct the archive's exact uniform frequency coordinates in float64."""

    required = ("radar_fc_hz", "radar_bandwidth_hz", "num_adc_samples")
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"dataset metadata is missing {missing}")
    fc = float(meta["radar_fc_hz"])
    bandwidth = float(meta["radar_bandwidth_hz"])
    samples = int(meta["num_adc_samples"])
    if not (math.isfinite(fc) and math.isfinite(bandwidth) and bandwidth > 0 and samples >= 2):
        raise ValueError("invalid FMCW frequency metadata")
    # Use the affine form instead of endpoint linspace: the NPZ contract is
    # f_n = fc - BW/2 + n*BW/N, so the final sample is below fc+BW/2.
    indices = torch.arange(samples, dtype=dtype)
    return (fc - bandwidth / 2.0) + indices * (bandwidth / samples)


def _canonical_b787_roles(sealed_contract: Mapping[str, object]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate and return the development and sealed role IDs without response access."""

    roles = sealed_contract.get("role_ids")
    access = sealed_contract.get("response_access")
    if not isinstance(roles, Mapping) or not isinstance(access, Mapping):
        raise ValueError("SpINR-style acquisition requires a complete sealed NPZ contract")
    if access.get("reserved_test_materialized") or access.get("unused_materialized"):
        raise ValueError("SpINR-style acquisition requires sealed test and unused response access")
    train = tuple(int(item) for item in roles.get("train", ()))
    validation = tuple(int(item) for item in roles.get("validation", ()))
    reserved_test = tuple(int(item) for item in roles.get("reserved_test", ()))
    unused = tuple(int(item) for item in roles.get("unused", ()))
    from rift.rift_dataset import collection_contract
    collection = collection_contract(sealed_contract)
    if collection is None and (len(train), len(validation), len(reserved_test), len(unused)) != (3200, 1000, 1000, 4800):
        raise ValueError(
            "SpINR-style B787 requires canonical 3200/1000/1000/4800 sealed roles")
    authorized = train + validation
    if (len(set(authorized)) != len(authorized)
            or set(authorized).intersection(reserved_test)
            or set(authorized).intersection(unused)):
        raise ValueError("SpINR-style B787 roles overlap")
    return authorized, train


def _canonical_b787_metadata(meta: Mapping[str, object]) -> dict[str, object]:
    """Validate the fixed acquisition fields and return portable primitives."""

    from rift.rift_dataset import metadata_object_id
    source_object = metadata_object_id(meta)
    if str(meta.get("experiment", "")).lower() != "sphere10k":
        raise ValueError("SpINR-style adapter is restricted to B787 sphere10k")
    try:
        fc_hz = float(meta["radar_fc_hz"])
        bandwidth_hz = float(meta["radar_bandwidth_hz"])
        num_adc_samples = int(meta["num_adc_samples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("SpINR-style adapter requires complete B787 acquisition metadata") from exc
    if not (
        math.isclose(fc_hz, 10.0e9, rel_tol=0.0, abs_tol=1.0)
        and math.isclose(bandwidth_hz, 3.0e9, rel_tol=0.0, abs_tol=1.0)
        and num_adc_samples == 600
    ):
        raise ValueError("SpINR-style adapter requires the B787 10 GHz / 3 GHz / 600-bin acquisition")
    return {
        "target_type": str(meta["target_type"]).lower(),
        **({"target_id": source_object} if source_object != "b787" else {}),
        "experiment": "sphere10k",
        "radar_fc_hz": fc_hz,
        "radar_bandwidth_hz": bandwidth_hz,
        "num_adc_samples": num_adc_samples,
    }


def _selected_response_shape(contract):
    from .rift_dataset import collection_contract
    if contract.get('antenna_selection'):
        return tuple(collection_contract(contract)['response_shape'])
    return (10000, 16, 16, 1, 600)


def build_spinr_style_acquisition_identity(
    arrays: Mapping[str, object],
    sealed_contract: Mapping[str, object],
) -> dict[str, object]:
    """Record B787 physics inputs before any response payload is materialized.

    The record is ordinary checkpoint metadata, not a source hash or a lock.
    It deliberately excludes path provenance so an equivalent archive/manifest
    relocation remains a valid continuation.  Only authorized development
    poses are retained; reserved-test and unused response rows remain
    inaccessible and uninvolved.
    """

    if arrays.get("response") is not None:
        raise ValueError("SpINR-style acquisition identity must be built before response materialization")
    response_shape = tuple(int(value) for value in sealed_contract.get("response_shape", ()))
    if response_shape != _selected_response_shape(sealed_contract) or sealed_contract.get("response_dtype") != "complex64":
        raise ValueError("SpINR-style B787 requires [10000,16,16,1,600] complex64 observations")
    authorized, _train = _canonical_b787_roles(sealed_contract)
    meta = arrays.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError("SpINR-style acquisition identity requires decoded metadata")
    metadata = _canonical_b787_metadata(meta)
    frequency_hz = metadata_frequency_grid(metadata, dtype=torch.float64).cpu().contiguous()
    # Do not silently promote lower-precision pose arrays.  The acquisition
    # record is intended to bind the actual FP64 physics inputs, not a rounded
    # reconstruction of some other archive convention.
    raw_rx_pos = np.asarray(arrays.get("rx_pos"))
    raw_tx_pos = np.asarray(arrays.get("tx_pos"))
    if raw_rx_pos.dtype != np.dtype(np.float64) or raw_tx_pos.dtype != np.dtype(np.float64):
        raise ValueError("SpINR-style B787 requires source Tx/Rx pose arrays stored as float64")
    rx_pos = np.ascontiguousarray(raw_rx_pos, dtype=np.float64)
    tx_pos = np.ascontiguousarray(raw_tx_pos, dtype=np.float64)
    if rx_pos.shape != (10000, response_shape[2], 3) or tx_pos.shape != (10000, response_shape[1], 3):
        raise ValueError("SpINR-style B787 requires [view,16,3] Tx/Rx pose arrays")
    if not (np.isfinite(rx_pos).all() and np.isfinite(tx_pos).all()):
        raise ValueError("SpINR-style B787 requires finite Tx/Rx pose arrays")
    indices = np.asarray(authorized, dtype=np.int64)
    rx_authorized = torch.from_numpy(np.ascontiguousarray(rx_pos[indices], dtype=np.float64)).clone()
    tx_authorized = torch.from_numpy(np.ascontiguousarray(tx_pos[indices], dtype=np.float64)).clone()
    return {
        "schema": SPINR_STYLE_ACQUISITION_SCHEMA,
        "metadata": metadata,
        "frequency_hz": frequency_hz,
        "authorized_view_ids": [int(item) for item in authorized],
        "rx_pos_m": rx_authorized,
        "tx_pos_m": tx_authorized,
    }


def validate_spinr_style_acquisition_identity(
    saved: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    """Reject a continuation whose declared B787 physics inputs changed."""

    if not isinstance(saved, Mapping):
        raise ValueError("SpINR-style resume requires an acquisition identity record")
    if saved.get("schema") != SPINR_STYLE_ACQUISITION_SCHEMA:
        raise ValueError("SpINR-style resume has an incompatible acquisition identity schema")
    if saved.get("metadata") != expected.get("metadata"):
        raise ValueError("SpINR-style resume would change B787 acquisition metadata")
    if saved.get("authorized_view_ids") != expected.get("authorized_view_ids"):
        raise ValueError("SpINR-style resume would change authorized B787 view IDs")
    for key in ("frequency_hz", "rx_pos_m", "tx_pos_m"):
        left = saved.get(key)
        right = expected.get(key)
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            raise ValueError(f"SpINR-style acquisition identity is missing tensor {key}")
        if left.dtype != torch.float64 or right.dtype != torch.float64 or left.shape != right.shape:
            raise ValueError(f"SpINR-style acquisition identity has incompatible {key} layout")
        if not (torch.isfinite(left).all() and torch.equal(left.cpu(), right.cpu())):
            raise ValueError(f"SpINR-style resume would change B787 {key}")


@dataclass(frozen=True)
class RawComplexView:
    """One authorized raw FMCW view in physical coordinates.

    ``response`` remains complex64 in host memory to keep the 4,200 authorized
    views manageable.  :meth:`SealedRawComplexViews.tensor_view` promotes it
    to complex128 before the physics renderer.
    """

    source_id: int
    response: np.ndarray  # [Tx, Rx, frequency], complex64
    rx_pos_m: np.ndarray  # [Rx, 3], float64
    tx_pos_m: np.ndarray  # [Tx, 3], float64


def npz_tx_rx_frequency_to_renderer(response: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert an NPZ raw response from ``[Tx,Rx,F]`` to ``[F,Rx,Tx]``.

    This intentionally has no B787-shaped assertion so the axis convention can
    be tested with an asymmetric Tx/Rx sentinel.  The B787 adapter separately
    verifies its required 16 x 16 x 600 input contract.
    """

    tensor = torch.as_tensor(response)
    if tensor.ndim != 3:
        raise ValueError("raw response must have shape [Tx, Rx, frequency]")
    return tensor.permute(2, 1, 0).contiguous()


def _validate_and_average_b787_raw_response(raw_response: object, expected_shape=(16, 16, 1, 600)) -> np.ndarray:
    """Validate one source view and average only its redundant chirp axis."""

    response = np.asarray(raw_response)
    if response.shape != expected_shape or response.dtype != np.dtype(np.complex64):
        raise ValueError("raw response disagrees with the B787 SpINR-style contract")
    # Only chirps are redundant static repeats.  Do not average any antenna or
    # frequency axis; preserving those phases is part of the all-pair/full-bin
    # objective.
    return np.asarray(response.mean(axis=2), dtype=np.complex64).copy()


class SealedRawComplexViews:
    """Raw-complex materialization capability limited to train plus validation.

    Instances can only be built from a manifest contract already validated by
    ``train._load_sealed_npz_protocol_contract``.  They never retain or expose
    reserved-test or unused response rows.
    """

    def __init__(self, arrays: Mapping[str, object], sealed_contract: Mapping[str, object]) -> None:
        roles = sealed_contract.get("role_ids")
        access = sealed_contract.get("response_access")
        if not isinstance(roles, Mapping) or not isinstance(access, Mapping):
            raise ValueError("raw-complex adapter requires a complete sealed NPZ contract")
        if access.get("reserved_test_materialized") or access.get("unused_materialized"):
            raise ValueError("raw-complex adapter refuses a contract that materializes sealed roles")
        train = tuple(int(item) for item in roles.get("train", ()))
        validation = tuple(int(item) for item in roles.get("validation", ()))
        reserved_test = tuple(int(item) for item in roles.get("reserved_test", ()))
        unused = tuple(int(item) for item in roles.get("unused", ()))
        _canonical_b787_roles(sealed_contract)
        authorized = train + validation
        if set(authorized).intersection(reserved_test) or set(authorized).intersection(unused):
            raise ValueError("raw-complex adapter received overlapping authorized and sealed roles")
        self._roles = {"train": train, "validation": validation}
        self._sealed_ids = frozenset(reserved_test + unused)
        self._contract = dict(sealed_contract)
        meta = arrays.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError("raw-complex adapter requires decoded metadata")
        self.meta = dict(meta)
        self.frequencies_hz = metadata_frequency_grid(self.meta).cpu().numpy()
        expected_shape = tuple(int(value) for value in sealed_contract.get("response_shape", ()))
        if len(expected_shape) != 5:
            raise ValueError("raw-complex adapter requires a five-dimensional response contract")
        if expected_shape != _selected_response_shape(sealed_contract):
            raise ValueError("SpINR-style B787 adapter requires [view,16,16,1,600] responses")
        _canonical_b787_metadata(self.meta)
        if str(self.meta.get("experiment", "")).lower() != "sphere10k":
            raise ValueError("SpINR-style adapter is restricted to B787 sphere10k")
        try:
            fc_hz = float(self.meta["radar_fc_hz"])
            bandwidth_hz = float(self.meta["radar_bandwidth_hz"])
            num_adc_samples = int(self.meta["num_adc_samples"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("SpINR-style adapter requires complete B787 acquisition metadata") from exc
        if not (
            math.isclose(fc_hz, 10.0e9, rel_tol=0.0, abs_tol=1.0)
            and math.isclose(bandwidth_hz, 3.0e9, rel_tol=0.0, abs_tol=1.0)
            and num_adc_samples == 600
        ):
            raise ValueError("SpINR-style adapter requires the B787 10 GHz / 3 GHz / 600-bin acquisition")

        restricted = restrict_npz_response_views(dict(arrays), authorized)
        response_by_id: dict[int, np.ndarray] = {}
        train_set = frozenset(train)
        training_energy_sum = 0.0
        training_energy_count = 0
        raw_rx_pos = np.asarray(arrays["rx_pos"])
        raw_tx_pos = np.asarray(arrays["tx_pos"])
        if raw_rx_pos.dtype != np.dtype(np.float64) or raw_tx_pos.dtype != np.dtype(np.float64):
            raise ValueError("raw-complex adapter requires source Tx/Rx pose arrays stored as float64")
        rx_pos = np.ascontiguousarray(raw_rx_pos, dtype=np.float64)
        tx_pos = np.ascontiguousarray(raw_tx_pos, dtype=np.float64)
        for source_id, raw_response in iter_npz_response_views(restricted, authorized):
            source_id = int(source_id)
            if source_id not in authorized:
                raise AssertionError("restricted raw reader yielded an unauthorized view")
            chirp_averaged = _validate_and_average_b787_raw_response(raw_response, expected_shape[1:])
            # Compute the normalization while the restricted reader streams
            # its authorized rows.  The adapter intentionally retains the
            # chirp-averaged train/validation cache for later epoch access,
            # but never revisits it to call this quantity "streamed".
            if source_id in train_set:
                energy_view = chirp_averaged.astype(np.complex128, copy=False)
                training_energy_sum += float(np.vdot(energy_view.reshape(-1), energy_view.reshape(-1)).real)
                training_energy_count += int(energy_view.size)
            response_by_id[source_id] = chirp_averaged
        if set(response_by_id) != set(authorized):
            raise RuntimeError("raw-complex adapter did not materialize exactly its authorized roles")
        if not (np.isfinite(rx_pos).all() and np.isfinite(tx_pos).all()):
            raise ValueError("raw-complex adapter requires finite antenna coordinates")
        self._views = {
            source_id: RawComplexView(
                source_id=source_id,
                response=response_by_id[source_id],
                rx_pos_m=np.asarray(rx_pos[source_id], dtype=np.float64).copy(),
                tx_pos_m=np.asarray(tx_pos[source_id], dtype=np.float64).copy(),
            )
            for source_id in authorized
        }
        if training_energy_count <= 0 or not math.isfinite(training_energy_sum) or training_energy_sum <= 0:
            raise ValueError("training raw-complex energy must be finite and positive")
        self._training_mean_raw_power = training_energy_sum / training_energy_count
        self._materialized_response_bytes = sum(item.response.nbytes for item in self._views.values())

    @property
    def sealed_contract(self) -> Mapping[str, object]:
        return dict(self._contract)

    def role_ids(self, role: str) -> tuple[int, ...]:
        if role not in self._roles:
            raise ValueError("role must be 'train' or 'validation'; sealed roles are never exposed")
        return self._roles[role]

    def view(self, source_id: int) -> RawComplexView:
        source_id = int(source_id)
        if source_id in self._sealed_ids:
            raise PermissionError("reserved-test and unused response access is forbidden")
        try:
            return self._views[source_id]
        except KeyError as exc:
            raise PermissionError("response access is restricted to train plus validation roles") from exc

    def tensor_view(
        self,
        source_id: int,
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``S[frequency,Rx,Tx]``, Rx and Tx positions in float64 precision."""

        item = self.view(source_id)
        response = torch.as_tensor(item.response, device=device, dtype=torch.complex128)
        # NPZ order is [Tx, Rx, frequency]; range renderer order is [frequency, Rx, Tx].
        response = npz_tx_rx_frequency_to_renderer(response)
        rx_pos = torch.as_tensor(item.rx_pos_m, device=device, dtype=torch.float64)
        tx_pos = torch.as_tensor(item.tx_pos_m, device=device, dtype=torch.float64)
        return response, rx_pos, tx_pos

    def raw_training_mean_power(self) -> float:
        """Return streamed mean raw |S|² over exactly the 3,200 train roles."""

        return float(self._training_mean_raw_power)

    @property
    def materialized_response_bytes(self) -> int:
        """Raw chirp-averaged host-cache bytes, excluding Python/container overhead."""

        return int(self._materialized_response_bytes)


def build_sealed_raw_complex_views(
    arrays: Mapping[str, object],
    sealed_contract: Mapping[str, object],
) -> SealedRawComplexViews:
    """Construct the raw-complex adapter after sealed header/role preflight."""

    return SealedRawComplexViews(arrays, sealed_contract)


def scale_field_to_renderer_weights(
    field: torch.Tensor,
    *,
    cell_volume_m3: float | torch.Tensor,
    initial_output_scale: float,
) -> torch.Tensor:
    """Map real quadrature field values to the existing range-renderer weights.

    ``range_forward_operator`` includes ``(4π)^-2``.  Multiplying by
    ``(4π)^2`` here makes its product-spreading kernel exactly
    ``volume*sigma/(R_T R_R)``.  The volume and fixed initial scale appear
    exactly once in this conversion.
    """

    if not math.isfinite(float(initial_output_scale)) or float(initial_output_scale) <= 0:
        raise ValueError("initial_output_scale must be finite and positive")
    if torch.is_complex(field):
        raise ValueError("SpINR-style field must be signed real")
    volume = torch.as_tensor(cell_volume_m3, device=field.device, dtype=torch.float64)
    if volume.ndim == 0:
        if not torch.isfinite(volume).all() or bool((volume <= 0).any()):
            raise ValueError("cell_volume_m3 must be finite and positive")
    elif volume.shape == field.shape:
        if not torch.isfinite(volume).all() or bool((volume <= 0).any()):
            raise ValueError("cell_volume_m3 entries must be finite and positive")
    else:
        raise ValueError("cell_volume_m3 must be a scalar or one volume per field point")
    scale = ((4.0 * math.pi) ** 2) * float(initial_output_scale)
    return field.to(dtype=torch.float64) * volume * scale


def frequency_to_range_bins(response: torch.Tensor) -> torch.Tensor:
    """Apply the frozen all-bin forward-normalized FFT on the frequency axis.

    The final three axes are always ``[frequency, Rx, Tx]``.  Optional leading
    axes are logical view-batch axes and are preserved rather than flattened.
    """

    if response.ndim < 3:
        raise ValueError("response must end with [frequency, Rx, Tx]")
    return torch.fft.fft(response, dim=-3, norm="forward")


def range_to_frequency_cotangent(range_cotangent: torch.Tensor) -> torch.Tensor:
    """Adjoint of :func:`frequency_to_range_bins` under PyTorch's VJP convention."""

    if range_cotangent.ndim < 3:
        raise ValueError("range_cotangent must end with [frequency, Rx, Tx]")
    # fft(..., norm='forward') is F/N.  Its Hermitian adjoint is F^H/N,
    # implemented by the default/backward-normalized ifft.
    return torch.fft.ifft(range_cotangent, dim=-3, norm="backward")


def spinr_style_objective(
    predicted_frequency: torch.Tensor,
    observed_frequency: torch.Tensor,
    *,
    training_mean_raw_power: float,
    range_bin_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Magnitude-plus-half-complex loss, optionally on geometry-selected bins.

    Masking retains sum-over-bins scaling: excluded bins contribute zero to the
    original denominator, rather than reweighting pairs by their bin counts.
    Omitting the mask preserves the historical objective exactly.
    """

    if predicted_frequency.shape != observed_frequency.shape:
        raise ValueError("predicted and observed frequency responses must have the same shape")
    if predicted_frequency.ndim < 3:
        raise ValueError("responses must end with [frequency, Rx, Tx]")
    if not math.isfinite(float(training_mean_raw_power)) or float(training_mean_raw_power) <= 0:
        raise ValueError("training_mean_raw_power must be finite and positive")
    predicted = frequency_to_range_bins(predicted_frequency)
    observed = frequency_to_range_bins(observed_frequency)
    if range_bin_mask is not None:
        if (range_bin_mask.shape != predicted.shape or range_bin_mask.dtype != torch.bool
                or range_bin_mask.device != predicted.device or not range_bin_mask.any()):
            raise ValueError("range_bin_mask must be a nonempty matching boolean tensor")
        predicted = torch.where(range_bin_mask, predicted, 0)
        observed = torch.where(range_bin_mask, observed, 0)
    magnitude_term = (predicted.abs() - observed.abs()).square().mean()
    complex_term = (predicted - observed).abs().square().mean()
    return (predicted.shape[-3] / float(training_mean_raw_power)) * (magnitude_term + 0.5 * complex_term)
