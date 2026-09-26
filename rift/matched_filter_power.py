"""Shared differentiable matched-filter-power contract.

This module turns a coherent swept-frequency response into a spatial
matched-filter image.  It is the common observable used by the GeRaF v1
baseline, the model-free matched-filter anchor, and coherent RIFT rerenders.

For a response whose propagation convention is

    S(tx, rx, f) ~ exp(phase_sign * 1j * 2*pi*f*R(tx, rx, x)/c),

the complex matched-filter amplitude at a query point is

    A(x) = sum_tx,rx,f S(tx, rx, f)
           * exp(-phase_sign * 1j * 2*pi*f*R(tx, rx, x)/c),

where ``R = ||x - tx|| + ||x - rx||`` is the exact bistatic path length.
The reported *power* is explicitly ``|A|**2``.  This distinction matters:
GeRaF calls the norm in its Eq. 3/A.5 "power", while this contract retains
both complex amplitude and squared power so no method silently compares
amplitude against power.

The default ``phase_sign=-1`` and ``range_model='none'`` implement GeRaF's
phase-only matched filter for the RIFT FMCW ``npz`` convention.  The optional
``product`` and ``sum2`` range models instead form the conjugate of RIFT's
forward-model geometric factor.  They do *not* compensate free-space loss.

Two backends are available:

``range_nufft``
    Production path for a complete, uniform frequency grid.  It reuses the
    validated differentiable ``rift.range_operator.range_adjoint_operator``
    and removes that operator's unavoidable ``(4*pi)^-2`` constant when the
    requested matched-filter policy omits it.  It never forms a frequency
    kernel per query point.  Like the underlying range operator, this is a
    controlled Gaussian-gridding approximation, not the direct sum.

``direct``
    Exact, chunked reference that also accepts non-uniform frequencies.  It
    never materializes the full ``[points, tx, rx, frequency]`` kernel; its
    largest phase block is ``[point_chunk, pair_chunk, freq_chunk]``.

All tensor operations remain in PyTorch and preserve gradients with respect
to the response and geometry.  Normalization is deliberately a separate,
frozen object fitted from training targets only.

Tx, Rx, and frequency samples are summed coherently *within* each view.
Leading view/batch dimensions remain independent: this module never averages
or coherently combines held-out viewpoints behind the caller's back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


LIGHT_SPEED_M_S = 299792458.0
_FOUR_PI_SQUARED = (4.0 * math.pi) ** 2
_VALID_LAYOUTS = {"freq_rx_tx", "tx_rx_freq"}
_VALID_RANGE_MODELS = {"none", "product", "sum2"}
_VALID_BACKENDS = {"direct", "range_nufft"}


def _real_dtype_for_complex(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    raise TypeError("response must have dtype torch.complex64 or torch.complex128")


def _complex_dtype_for_real(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float32:
        return torch.complex64
    if dtype == torch.float64:
        return torch.complex128
    raise ValueError("compute_dtype must be torch.float32 or torch.float64")


def _canonical_response(
    response: torch.Tensor,
    response_layout: str,
) -> Tuple[torch.Tensor, Tuple[int, ...], int, int, int]:
    """Return response as ``[*batch, tx, rx, frequency]``."""

    response = torch.as_tensor(response)
    if not torch.is_complex(response):
        raise TypeError("response must be complex-valued")
    if response.ndim < 3:
        raise ValueError("response must have at least three dimensions")
    if response_layout not in _VALID_LAYOUTS:
        raise ValueError(
            f"response_layout must be one of {sorted(_VALID_LAYOUTS)}, "
            f"got {response_layout!r}"
        )

    batch_shape = tuple(int(v) for v in response.shape[:-3])
    if response_layout == "freq_rx_tx":
        num_freq, num_rx, num_tx = (int(v) for v in response.shape[-3:])
        leading = list(range(response.ndim - 3))
        response = response.permute(*leading, response.ndim - 1,
                                    response.ndim - 2, response.ndim - 3)
    else:
        num_tx, num_rx, num_freq = (int(v) for v in response.shape[-3:])

    if min(num_tx, num_rx, num_freq) <= 0:
        raise ValueError("tx, rx, and frequency dimensions must all be non-empty")
    return response, batch_shape, num_tx, num_rx, num_freq


def _broadcast_positions(
    positions: torch.Tensor,
    batch_shape: Tuple[int, ...],
    count: int,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.as_tensor(positions, device=device, dtype=dtype)
    if positions.ndim < 2 or tuple(positions.shape[-2:]) != (count, 3):
        raise ValueError(
            f"{name} must end in [{count},3], got {tuple(positions.shape)}"
        )
    wanted = batch_shape + (count, 3)
    try:
        return torch.broadcast_to(positions, wanted)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} leading dimensions {tuple(positions.shape[:-2])} do not "
            f"broadcast to response batch shape {batch_shape}"
        ) from exc


def _broadcast_mask(mask: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(mask, dtype=torch.bool, device=values.device)
    if mask.ndim > values.ndim:
        raise ValueError(
            f"ROI mask has {mask.ndim} dimensions but values have {values.ndim}"
        )
    shaped = mask.reshape((1,) * (values.ndim - mask.ndim) + tuple(mask.shape))
    try:
        return torch.broadcast_to(shaped, values.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"ROI mask shape {tuple(mask.shape)} does not broadcast to "
            f"values shape {tuple(values.shape)}"
        ) from exc


@dataclass(frozen=True)
class MatchedFilterGrid:
    """Query points plus the metadata needed to interpret a flat MF image.

    ``points`` follows the exact flattening order described by ``shape``.
    ``roi_mask`` may have ``shape`` or be flat.  Axis metadata is descriptive;
    it never changes point ordering or silently constructs a different grid.
    """

    points: torch.Tensor
    shape: Tuple[int, ...]
    axis_names: Tuple[str, ...] = ()
    spacing_m: Optional[Tuple[float, ...]] = None
    bounds_m: Optional[Tuple[Tuple[float, float], ...]] = None
    roi_mask: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        points = torch.as_tensor(self.points)
        shape = tuple(int(v) for v in self.shape)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f"grid points must have shape [P,3], got {tuple(points.shape)}")
        if not shape or any(v <= 0 for v in shape):
            raise ValueError("grid shape must contain positive dimensions")
        if math.prod(shape) != int(points.shape[0]):
            raise ValueError(
                f"grid shape {shape} contains {math.prod(shape)} cells, but points "
                f"contains {int(points.shape[0])} rows"
            )

        axis_names = self.axis_names or tuple(f"axis_{i}" for i in range(len(shape)))
        if len(axis_names) != len(shape):
            raise ValueError("axis_names must have one entry per grid dimension")
        if self.spacing_m is not None and len(self.spacing_m) != len(shape):
            raise ValueError("spacing_m must have one entry per grid dimension")
        if self.bounds_m is not None and len(self.bounds_m) != len(shape):
            raise ValueError("bounds_m must have one entry per grid dimension")
        if self.roi_mask is not None:
            mask_shape = tuple(int(v) for v in torch.as_tensor(self.roi_mask).shape)
            valid_shapes = {shape, (math.prod(shape),)}
            if mask_shape not in valid_shapes:
                raise ValueError(
                    "roi_mask shape must equal the grid shape or be flat with one "
                    f"value per point; got {mask_shape}, expected {shape} or "
                    f"({math.prod(shape)},)"
                )

        object.__setattr__(self, "points", points)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "axis_names", tuple(str(v) for v in axis_names))
        if self.spacing_m is not None:
            object.__setattr__(self, "spacing_m", tuple(float(v) for v in self.spacing_m))
        if self.bounds_m is not None:
            bounds = tuple((float(lo), float(hi)) for lo, hi in self.bounds_m)
            object.__setattr__(self, "bounds_m", bounds)
        if self.roi_mask is not None:
            object.__setattr__(self, "roi_mask", torch.as_tensor(self.roi_mask, dtype=torch.bool))

    @classmethod
    def from_axes(
        cls,
        x_m: torch.Tensor,
        y_m: torch.Tensor,
        z_m: torch.Tensor,
        *,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> "MatchedFilterGrid":
        """Construct an ``ij``-ordered Cartesian grid from three 1D axes."""

        axes = [torch.as_tensor(axis) for axis in (x_m, y_m, z_m)]
        if any(axis.ndim != 1 or axis.numel() == 0 for axis in axes):
            raise ValueError("x_m, y_m, and z_m must be non-empty 1D tensors")
        device = axes[0].device
        dtype = axes[0].dtype
        axes = [axis.to(device=device, dtype=dtype) for axis in axes]
        meshes = torch.meshgrid(*axes, indexing="ij")
        points = torch.stack(meshes, dim=-1).reshape(-1, 3)

        def spacing(axis: torch.Tensor) -> float:
            if axis.numel() < 2:
                return 0.0
            differences = axis[1:] - axis[:-1]
            if not torch.allclose(differences, differences[:1]):
                return float("nan")
            return float(differences[0].detach().cpu())

        return cls(
            points=points,
            shape=tuple(int(axis.numel()) for axis in axes),
            axis_names=("x", "y", "z"),
            spacing_m=tuple(spacing(axis) for axis in axes),
            bounds_m=tuple(
                (float(axis[0].detach().cpu()), float(axis[-1].detach().cpu()))
                for axis in axes
            ),
            roi_mask=roi_mask,
        )

    def reshape(self, values: torch.Tensor) -> torch.Tensor:
        """Reshape a ``[...,P]`` tensor into ``[...,*shape]``."""

        if values.shape[-1] != self.points.shape[0]:
            raise ValueError(
                f"values last dimension is {values.shape[-1]}, expected "
                f"{self.points.shape[0]} grid points"
            )
        return values.reshape(tuple(values.shape[:-1]) + self.shape)

    def flat_roi_mask(self, *, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if self.roi_mask is None:
            return None
        return self.roi_mask.reshape(-1).to(device=device)

    def metadata_dict(self) -> Dict[str, object]:
        """Small JSON-safe metadata record (the possibly large mask is summarized)."""

        return {
            "shape": list(self.shape),
            "axis_names": list(self.axis_names),
            "spacing_m": None if self.spacing_m is None else list(self.spacing_m),
            "bounds_m": None if self.bounds_m is None else [list(v) for v in self.bounds_m],
            "num_points": int(self.points.shape[0]),
            "roi_points": None if self.roi_mask is None else int(self.roi_mask.sum().item()),
            "flattening": "C-order over the supplied point rows / ij axes",
        }


@dataclass(frozen=True)
class PowerNormalization:
    """Frozen training-target normalization for matched-filter power."""

    mode: str
    peak_power: float
    dynamic_range_db: float = 60.0
    clip: bool = False
    fitted_split: str = "train"
    fitted_values: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"linear_peak", "db_peak"}:
            raise ValueError("normalization mode must be 'linear_peak' or 'db_peak'")
        if not math.isfinite(self.peak_power) or self.peak_power <= 0.0:
            raise ValueError("peak_power must be positive and finite")
        if self.dynamic_range_db <= 0.0:
            raise ValueError("dynamic_range_db must be positive")
        if self.fitted_split != "train":
            raise ValueError("matched-filter normalization may only be fitted on split='train'")

    def apply(self, power: torch.Tensor) -> torch.Tensor:
        """Apply frozen statistics without detaching prediction gradients."""

        if torch.is_complex(power):
            raise TypeError("PowerNormalization.apply expects real squared power, not amplitude")
        peak = power.new_tensor(self.peak_power)
        if self.mode == "linear_peak":
            normalized = power / peak
        else:
            floor = 10.0 ** (-self.dynamic_range_db / 10.0)
            relative = (power / peak).clamp_min(floor)
            normalized = (10.0 * torch.log10(relative) + self.dynamic_range_db) / self.dynamic_range_db
        if self.clip:
            normalized = normalized.clamp(0.0, 1.0)
        return normalized

    def as_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "peak_power": self.peak_power,
            "dynamic_range_db": self.dynamic_range_db,
            "clip": self.clip,
            "fitted_split": self.fitted_split,
            "fitted_values": self.fitted_values,
        }

    @classmethod
    def from_dict(cls, values: Dict[str, object]) -> "PowerNormalization":
        return cls(
            mode=str(values["mode"]),
            peak_power=float(values["peak_power"]),
            dynamic_range_db=float(values.get("dynamic_range_db", 60.0)),
            clip=bool(values.get("clip", False)),
            fitted_split=str(values.get("fitted_split", "train")),
            fitted_values=int(values.get("fitted_values", 0)),
        )


@dataclass(frozen=True)
class MatchedFilterResult:
    """Complex amplitude and unambiguously squared matched-filter power."""

    complex_amplitude: torch.Tensor
    power: torch.Tensor
    grid: Optional[MatchedFilterGrid] = None
    normalized_power: Optional[torch.Tensor] = None

    def amplitude_image(self) -> torch.Tensor:
        if self.grid is None:
            return self.complex_amplitude
        return self.grid.reshape(self.complex_amplitude)

    def power_image(self, normalized: bool = False) -> torch.Tensor:
        values = self.normalized_power if normalized else self.power
        if values is None:
            raise ValueError("no normalizer was supplied when this result was computed")
        if self.grid is None:
            return values
        return self.grid.reshape(values)


def fit_power_normalization(
    training_target_power: torch.Tensor,
    *,
    roi_mask: Optional[torch.Tensor] = None,
    mode: str = "linear_peak",
    dynamic_range_db: float = 60.0,
    clip: bool = False,
    split: str = "train",
) -> PowerNormalization:
    """Fit a single global peak using *training target* images only.

    The explicit ``split`` guard prevents a validation/test call site from
    fitting by accident.  It cannot infer data provenance, so callers must
    still pass only the frozen training manifest selected by the experiment.
    """

    if split != "train":
        raise ValueError("normalization fitting is forbidden outside split='train'")
    values = torch.as_tensor(training_target_power).detach()
    if torch.is_complex(values):
        raise TypeError("fit_power_normalization expects real squared power")
    if roi_mask is not None:
        values = values[_broadcast_mask(roi_mask, values)]
    else:
        values = values.reshape(-1)
    finite = torch.isfinite(values)
    values = values[finite]
    if values.numel() == 0:
        raise ValueError("training power contains no finite ROI values")
    if bool((values < 0).any()):
        raise ValueError("squared power must be non-negative")
    peak = float(values.max().cpu())
    return PowerNormalization(
        mode=mode,
        peak_power=peak,
        dynamic_range_db=float(dynamic_range_db),
        clip=bool(clip),
        fitted_split="train",
        fitted_values=int(values.numel()),
    )


def _range_gain(
    r_tx: torch.Tensor,
    r_rx: torch.Tensor,
    r_sum: torch.Tensor,
    range_model: str,
    include_four_pi: bool,
    eps: float,
) -> torch.Tensor:
    if range_model == "none":
        gain = torch.ones_like(r_sum)
    elif range_model == "product":
        gain = 1.0 / (r_tx * r_rx + eps)
    elif range_model == "sum2":
        gain = 1.0 / (r_sum.square() + eps)
    else:
        raise ValueError(f"range_model must be one of {sorted(_VALID_RANGE_MODELS)}")
    if include_four_pi:
        gain = gain / _FOUR_PI_SQUARED
    return gain


def _matched_filter_direct_single(
    response_tx_rx_freq: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    frequencies_hz: torch.Tensor,
    query_points: torch.Tensor,
    *,
    phase_sign: float,
    range_model: str,
    include_four_pi: bool,
    point_chunk: int,
    pair_chunk: int,
    freq_chunk: int,
    wave_speed_m_s: float,
    eps: float,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    complex_dtype = _complex_dtype_for_real(compute_dtype)
    signal = response_tx_rx_freq.to(dtype=complex_dtype)
    tx_positions = tx_positions.to(dtype=compute_dtype)
    rx_positions = rx_positions.to(dtype=compute_dtype)
    frequencies_hz = frequencies_hz.to(dtype=compute_dtype)
    query_points = query_points.to(dtype=compute_dtype)

    num_tx, num_rx, num_freq = signal.shape
    tx_all = torch.arange(num_tx, device=signal.device).repeat_interleave(num_rx)
    rx_all = torch.arange(num_rx, device=signal.device).repeat(num_tx)
    angular_scale = (-float(phase_sign) * 2.0 * torch.pi) / float(wave_speed_m_s)
    outputs = []

    for point_start in range(0, query_points.shape[0], point_chunk):
        points = query_points[point_start: point_start + point_chunk]
        r_tx_all = torch.linalg.vector_norm(
            points[:, None, :] - tx_positions[None, :, :], dim=-1
        ).clamp_min(eps)
        r_rx_all = torch.linalg.vector_norm(
            points[:, None, :] - rx_positions[None, :, :], dim=-1
        ).clamp_min(eps)
        point_amplitude = None

        for pair_start in range(0, tx_all.numel(), pair_chunk):
            tx_idx = tx_all[pair_start: pair_start + pair_chunk]
            rx_idx = rx_all[pair_start: pair_start + pair_chunk]
            r_tx = r_tx_all[:, tx_idx]
            r_rx = r_rx_all[:, rx_idx]
            r_sum = r_tx + r_rx
            gain = _range_gain(
                r_tx, r_rx, r_sum, range_model, include_four_pi, eps
            ).to(complex_dtype)
            pair_amplitude = None

            for freq_start in range(0, num_freq, freq_chunk):
                freq_slice = slice(freq_start, min(freq_start + freq_chunk, num_freq))
                frequencies = frequencies_hz[freq_slice]
                phase = angular_scale * r_sum[..., None] * frequencies[None, None, :]
                kernel = gain[..., None] * torch.exp(1j * phase)
                pair_signal = signal[tx_idx, rx_idx, freq_slice]
                contribution = torch.einsum("pqf,qf->p", kernel, pair_signal)
                pair_amplitude = contribution if pair_amplitude is None else pair_amplitude + contribution

            point_amplitude = pair_amplitude if point_amplitude is None else point_amplitude + pair_amplitude
        outputs.append(point_amplitude)

    return torch.cat(outputs, dim=0)


def _matched_filter_nufft_single(
    response_tx_rx_freq: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    frequencies_hz: torch.Tensor,
    query_points: torch.Tensor,
    *,
    phase_sign: float,
    range_model: str,
    include_four_pi: bool,
    point_chunk: int,
    pair_chunk: int,
    eps: float,
    compute_dtype: torch.dtype,
    nufft_oversample: int,
    nufft_kernel_width: int,
) -> torch.Tensor:
    """Uniform-grid accelerator backed by the validated RIFT adjoint."""

    from rift.range_operator import range_adjoint_operator

    frequencies_hz = frequencies_hz.to(device=query_points.device)
    kvector = (2.0 * torch.pi * frequencies_hz) / LIGHT_SPEED_M_S
    signal_freq_rx_tx = response_tx_rx_freq.permute(2, 1, 0).contiguous()
    amplitude = range_adjoint_operator(
        frequencies_hz,
        kvector,
        rx_positions,
        tx_positions,
        query_points,
        signal_freq_rx_tx,
        phase_sign=phase_sign,
        eps=eps,
        oversample=nufft_oversample,
        kernel_width=nufft_kernel_width,
        pair_chunk=pair_chunk,
        point_chunk=point_chunk,
        compute_dtype=compute_dtype,
        range_model=range_model,
    )
    # range_operator always contains g_const=(4*pi)^-2, including its
    # range_model='none'.  GeRaF Eq. 3 omits that constant.  Correct exactly
    # once here; the other range_model choices keep the same policy.
    if not include_four_pi:
        amplitude = amplitude * _FOUR_PI_SQUARED
    return amplitude


def matched_filter_complex(
    response: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    frequencies_hz: torch.Tensor,
    query_points_or_grid: Union[torch.Tensor, MatchedFilterGrid],
    *,
    phase_sign: float = -1.0,
    response_layout: str = "freq_rx_tx",
    range_model: str = "none",
    include_four_pi: bool = False,
    backend: str = "range_nufft",
    point_chunk: Optional[int] = None,
    pair_chunk: int = 64,
    freq_chunk: int = 64,
    compute_dtype: Optional[torch.dtype] = None,
    nufft_oversample: int = 2,
    nufft_kernel_width: int = 20,
    wave_speed_m_s: float = LIGHT_SPEED_M_S,
    eps: float = 1.0e-9,
) -> torch.Tensor:
    """Compute coherent matched-filter amplitude at exact bistatic points.

    Parameters
    ----------
    response:
        ``[...,F,Rx,Tx]`` for ``freq_rx_tx`` or ``[...,Tx,Rx,F]`` for
        ``tx_rx_freq``.  Chirps must be coherently averaged before this call.
    tx_positions, rx_positions:
        ``[...,Tx,3]`` / ``[...,Rx,3]`` in metres.  Leading dimensions may
        broadcast over the response batch (usually viewpoints).
    frequencies_hz:
        One complete frequency grid ``[F]``.  ``range_nufft`` requires it to
        be uniform; ``direct`` also accepts non-uniform grids.
    query_points_or_grid:
        Common ``[P,3]`` points in metres, or :class:`MatchedFilterGrid`.
    phase_sign:
        Sign in the *received* propagation response.  The MF automatically
        applies the conjugate sign.  RIFT FMCW npz files use ``-1``.
    range_model:
        ``none`` is GeRaF Eq. 3/A.5 (phase only). ``product`` multiplies the
        conjugate kernel by ``1/(R_tx R_rx)``; ``sum2`` by ``1/R_sum^2``.
        Neither option is inverse-range compensation.
    include_four_pi:
        Additionally multiply the kernel by ``(4*pi)^-2``.  Set true only to
        reproduce the exact RIFT forward-operator adjoint convention.
    backend:
        ``range_nufft`` is the practical uniform-frequency implementation;
        ``direct`` is the exact chunked reference.

    Returns
    -------
    torch.Tensor
        Complex amplitude with shape ``[*response_batch,P]``.  Use the grid's
        ``reshape`` method to recover image/volume axes.
    """

    if phase_sign not in (-1, -1.0, 1, 1.0):
        raise ValueError("phase_sign must be -1 or +1")
    if range_model not in _VALID_RANGE_MODELS:
        raise ValueError(f"range_model must be one of {sorted(_VALID_RANGE_MODELS)}")
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_VALID_BACKENDS)}")
    if pair_chunk <= 0 or freq_chunk <= 0:
        raise ValueError("pair_chunk and freq_chunk must be positive")
    if wave_speed_m_s <= 0.0:
        raise ValueError("wave_speed_m_s must be positive")
    if backend == "range_nufft" and float(wave_speed_m_s) != LIGHT_SPEED_M_S:
        raise ValueError(
            "range_nufft uses rift.config.cc; choose backend='direct' for a "
            "non-default propagation speed"
        )

    canonical, batch_shape, num_tx, num_rx, num_freq = _canonical_response(
        response, response_layout
    )
    device = canonical.device
    input_real_dtype = _real_dtype_for_complex(canonical.dtype)
    if compute_dtype is None:
        compute_dtype = input_real_dtype
    _complex_dtype_for_real(compute_dtype)  # validates

    frequencies_hz = torch.as_tensor(frequencies_hz, device=device, dtype=compute_dtype)
    if frequencies_hz.ndim != 1 or frequencies_hz.shape[0] != num_freq:
        raise ValueError(
            f"frequencies_hz must have shape [{num_freq}], got "
            f"{tuple(frequencies_hz.shape)}"
        )
    if not bool(torch.isfinite(frequencies_hz).all()) or not bool((frequencies_hz > 0).all()):
        raise ValueError("frequencies_hz must be finite and positive")

    grid = query_points_or_grid if isinstance(query_points_or_grid, MatchedFilterGrid) else None
    query_points = grid.points if grid is not None else torch.as_tensor(query_points_or_grid)
    query_points = query_points.to(device=device, dtype=compute_dtype)
    if query_points.ndim != 2 or query_points.shape[-1] != 3 or query_points.shape[0] == 0:
        raise ValueError("query points must have non-empty shape [P,3]")

    tx_positions = _broadcast_positions(
        tx_positions, batch_shape, num_tx, "tx_positions",
        device=device, dtype=compute_dtype,
    )
    rx_positions = _broadcast_positions(
        rx_positions, batch_shape, num_rx, "rx_positions",
        device=device, dtype=compute_dtype,
    )

    if point_chunk is None:
        point_chunk = 32768 if backend == "range_nufft" else 1024
    if point_chunk <= 0:
        raise ValueError("point_chunk must be positive")

    flat_batch = math.prod(batch_shape) if batch_shape else 1
    canonical = canonical.reshape(flat_batch, num_tx, num_rx, num_freq)
    tx_positions = tx_positions.reshape(flat_batch, num_tx, 3)
    rx_positions = rx_positions.reshape(flat_batch, num_rx, 3)
    outputs = []
    for batch_index in range(flat_batch):
        arguments = (
            canonical[batch_index], tx_positions[batch_index],
            rx_positions[batch_index], frequencies_hz, query_points,
        )
        if backend == "range_nufft":
            amplitude = _matched_filter_nufft_single(
                *arguments,
                phase_sign=float(phase_sign),
                range_model=range_model,
                include_four_pi=include_four_pi,
                point_chunk=point_chunk,
                pair_chunk=pair_chunk,
                eps=eps,
                compute_dtype=compute_dtype,
                nufft_oversample=nufft_oversample,
                nufft_kernel_width=nufft_kernel_width,
            )
        else:
            amplitude = _matched_filter_direct_single(
                *arguments,
                phase_sign=float(phase_sign),
                range_model=range_model,
                include_four_pi=include_four_pi,
                point_chunk=point_chunk,
                pair_chunk=pair_chunk,
                freq_chunk=freq_chunk,
                wave_speed_m_s=wave_speed_m_s,
                eps=eps,
                compute_dtype=compute_dtype,
            )
        outputs.append(amplitude)

    stacked = torch.stack(outputs, dim=0)
    return stacked.reshape(batch_shape + (int(query_points.shape[0]),))


def matched_filter_power(
    response: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    frequencies_hz: torch.Tensor,
    query_points_or_grid: Union[torch.Tensor, MatchedFilterGrid],
    *,
    normalizer: Optional[PowerNormalization] = None,
    **matched_filter_kwargs,
) -> torch.Tensor:
    """Return squared MF power, optionally transformed by frozen statistics."""

    amplitude = matched_filter_complex(
        response,
        tx_positions,
        rx_positions,
        frequencies_hz,
        query_points_or_grid,
        **matched_filter_kwargs,
    )
    power = amplitude.real.square() + amplitude.imag.square()
    return power if normalizer is None else normalizer.apply(power)


def matched_filter(
    response: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    frequencies_hz: torch.Tensor,
    query_points_or_grid: Union[torch.Tensor, MatchedFilterGrid],
    *,
    normalizer: Optional[PowerNormalization] = None,
    **matched_filter_kwargs,
) -> MatchedFilterResult:
    """Return complex amplitude, squared power, and optional normalized power."""

    amplitude = matched_filter_complex(
        response,
        tx_positions,
        rx_positions,
        frequencies_hz,
        query_points_or_grid,
        **matched_filter_kwargs,
    )
    power = amplitude.real.square() + amplitude.imag.square()
    normalized = None if normalizer is None else normalizer.apply(power)
    grid = query_points_or_grid if isinstance(query_points_or_grid, MatchedFilterGrid) else None
    return MatchedFilterResult(amplitude, power, grid=grid, normalized_power=normalized)


def ssim_2d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    roi_mask: Optional[torch.Tensor] = None,
    window_size: int = 11,
    sigma: float = 1.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """Differentiable single-channel 2D structural similarity.

    Leading dimensions are treated as independent images.  A ROI mask selects
    output-window centres; callers should crop a rectangular ROI first when
    they do not want neighbourhoods to include values just outside the ROI.
    """

    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have the same [...,H,W] shape")
    if torch.is_complex(prediction) or torch.is_complex(target):
        raise TypeError("SSIM is defined here for real power images only")
    if data_range <= 0.0 or window_size <= 0 or sigma <= 0.0:
        raise ValueError("data_range, window_size, and sigma must be positive")
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be 'mean' or 'none'")

    height, width = int(prediction.shape[-2]), int(prediction.shape[-1])
    actual_window = min(int(window_size), height, width)
    if actual_window % 2 == 0:
        actual_window -= 1
    actual_window = max(actual_window, 1)
    dtype = torch.promote_types(prediction.dtype, target.dtype)
    x = prediction.to(dtype=dtype).reshape(-1, 1, height, width)
    y = target.to(dtype=dtype).reshape(-1, 1, height, width)

    coordinates = torch.arange(actual_window, dtype=dtype, device=x.device)
    coordinates = coordinates - (actual_window - 1.0) / 2.0
    gaussian = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    gaussian = gaussian / gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :]).reshape(
        1, 1, actual_window, actual_window
    )
    padding = actual_window // 2

    def local_mean(values: torch.Tensor) -> torch.Tensor:
        if padding:
            values = F.pad(values, (padding, padding, padding, padding), mode="reflect")
        return F.conv2d(values, kernel)

    mu_x = local_mean(x)
    mu_y = local_mean(y)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y
    var_x = (local_mean(x.square()) - mu_x2).clamp_min(0.0)
    var_y = (local_mean(y.square()) - mu_y2).clamp_min(0.0)
    cov_xy = local_mean(x * y) - mu_xy
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * cov_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (var_x + var_y + c2)
    )

    if roi_mask is None:
        per_image = ssim_map.mean(dim=(-2, -1)).reshape(prediction.shape[:-2])
    else:
        mask = _broadcast_mask(roi_mask, prediction).reshape(-1, 1, height, width)
        count = mask.sum(dim=(-2, -1)).clamp_min(1)
        per_image = (ssim_map * mask.to(ssim_map.dtype)).sum(dim=(-2, -1)) / count
        per_image = per_image.reshape(prediction.shape[:-2])
    return per_image.mean() if reduction == "mean" else per_image


def power_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    roi_mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
    ssim_shape: Optional[Tuple[int, int]] = None,
    ssim_window_size: int = 11,
) -> Dict[str, torch.Tensor]:
    """Compute common metrics from per-image predictions, not scalar summaries.

    Relative MSE is a global ratio of sums, matching the existing RIFT signal
    reports. RMSE and PSNR use the global per-pixel MSE.  To request SSIM for
    flat images ``[...,P]``, pass their explicit ``(H,W)`` shape; a 3D volume
    must first be projected or sliced according to a frozen protocol.
    """

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if torch.is_complex(prediction) or torch.is_complex(target):
        raise TypeError("power_metrics expects real squared power images")
    if data_range <= 0.0:
        raise ValueError("data_range must be positive")
    prediction, target = torch.broadcast_tensors(prediction, target)
    difference_squared = (prediction - target).square()
    target_squared = target.square()
    if roi_mask is None:
        error_sum = difference_squared.sum()
        target_sum = target_squared.sum()
        count = difference_squared.new_tensor(difference_squared.numel())
    else:
        mask = _broadcast_mask(roi_mask, difference_squared)
        mask_values = mask.to(difference_squared.dtype)
        count = mask_values.sum()
        if not bool(count.detach() > 0):
            raise ValueError("ROI mask selects no values")
        error_sum = (difference_squared * mask_values).sum()
        target_sum = (target_squared * mask_values).sum()

    tiny = torch.finfo(difference_squared.dtype).tiny
    mse = error_sum / count
    relative_mse = error_sum / target_sum.clamp_min(tiny)
    rmse = torch.sqrt(mse)
    psnr = torch.where(
        mse > 0,
        10.0 * torch.log10(mse.new_tensor(float(data_range) ** 2) / mse),
        mse.new_tensor(float("inf")),
    )
    result = {
        "mse": mse,
        "relative_mse": relative_mse,
        "rmse": rmse,
        "psnr_db": psnr,
    }

    if ssim_shape is not None:
        shape = tuple(int(v) for v in ssim_shape)
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError("ssim_shape must be a positive (H,W) pair")
        if prediction.shape[-1] != math.prod(shape):
            raise ValueError(
                f"prediction last dimension {prediction.shape[-1]} does not match "
                f"ssim_shape product {math.prod(shape)}"
            )
        image_shape = tuple(prediction.shape[:-1]) + shape
        pred_images = prediction.reshape(image_shape)
        target_images = target.reshape(image_shape)
        ssim_mask = None
        if roi_mask is not None:
            full_mask = _broadcast_mask(roi_mask, prediction)
            ssim_mask = full_mask.reshape(image_shape)
        result["ssim"] = ssim_2d(
            pred_images,
            target_images,
            data_range=data_range,
            roi_mask=ssim_mask,
            window_size=ssim_window_size,
        )
    return result


__all__ = [
    "LIGHT_SPEED_M_S",
    "MatchedFilterGrid",
    "MatchedFilterResult",
    "PowerNormalization",
    "fit_power_normalization",
    "matched_filter",
    "matched_filter_complex",
    "matched_filter_power",
    "power_metrics",
    "ssim_2d",
]
