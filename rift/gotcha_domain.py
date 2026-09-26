"""Frozen full-domain support and range-window mask for GOTCHA.

The released GOTCHA scene is the ``z=0`` square ``[-50, 50]^2``.  A 50 m
disc is valid for every view by the reverse triangle inequality.  Points in
the square outside that disc still use the exact per-view, one-way virtual
range window; the square must not be replaced by a view-independent disc.
"""

from __future__ import annotations

import math

import torch


SPEED_OF_LIGHT_M_S = 299_792_458.0
GOTCHA_DOMAIN_SCHEMA = "rift.gotcha_full_domain.v1"
GOTCHA_XY_BOUNDS_M = (-50.0, 50.0)
GOTCHA_Z_REFERENCE_M = 0.0
GOTCHA_SCENE_CENTER_M = (0.0, 0.0, 0.0)
GOTCHA_CORE_RADIUS_M = 50.0


def _finite_scalar(name, value):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.is_complex():
            raise ValueError(f"{name} must be a real scalar")
        value = value.detach().item()
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _geometry_tensor(value, name, *, device=None, dtype=None):
    try:
        tensor = torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must be real numeric coordinates") from exc
    if tensor.is_complex() or tensor.dtype == torch.bool:
        raise TypeError(f"{name} must be real numeric coordinates")
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains a non-finite value")
    return tensor


def _validated_center(scene_center_m, *, device=None, dtype=None):
    center = _geometry_tensor(
        scene_center_m,
        "scene_center_m",
        device=device,
        dtype=dtype,
    )
    if center.shape != (3,):
        raise ValueError("scene_center_m must have shape [3]")
    expected = torch.tensor(
        GOTCHA_SCENE_CENTER_M,
        device=center.device,
        dtype=center.dtype,
    )
    if not torch.equal(center, expected):
        raise ValueError("GOTCHA scene_center_m must be exactly (0, 0, 0)")
    return center


def validate_gotcha_full_domain(
    *,
    reference_range_m,
    frequency_step_hz,
    scene_center_m=GOTCHA_SCENE_CENTER_M,
):
    """Validate acquisition values against the frozen full-domain contract.

    Returns the one-way unambiguous range ``c / (2 df)`` in metres.  Requiring
    ``Rref - 50 > 0`` and ``Rref + 50 < U`` proves that the entire 50 m core
    lies strictly inside the open range window for every viewpoint.
    """

    reference_range_m = _finite_scalar("reference_range_m", reference_range_m)
    frequency_step_hz = _finite_scalar("frequency_step_hz", frequency_step_hz)
    _validated_center(scene_center_m)
    if reference_range_m <= 0.0:
        raise ValueError("reference_range_m must be positive")
    if frequency_step_hz <= 0.0:
        raise ValueError("frequency_step_hz must be positive")

    unambiguous_range_m = SPEED_OF_LIGHT_M_S / (2.0 * frequency_step_hz)
    lower_core_edge_m = reference_range_m - GOTCHA_CORE_RADIUS_M
    upper_core_edge_m = reference_range_m + GOTCHA_CORE_RADIUS_M
    if not lower_core_edge_m > 0.0:
        raise ValueError(
            "GOTCHA core reaches the lower open unambiguous-range boundary"
        )
    if not upper_core_edge_m < unambiguous_range_m:
        raise ValueError(
            "GOTCHA core reaches the upper open unambiguous-range boundary"
        )
    return unambiguous_range_m


def gotcha_full_domain_metadata(
    *,
    reference_range_m,
    frequency_step_hz,
    scene_center_m=GOTCHA_SCENE_CENTER_M,
):
    """Return JSON-compatible metadata for a validated GOTCHA domain."""

    reference_range_m = _finite_scalar("reference_range_m", reference_range_m)
    frequency_step_hz = _finite_scalar("frequency_step_hz", frequency_step_hz)
    unambiguous_range_m = validate_gotcha_full_domain(
        reference_range_m=reference_range_m,
        frequency_step_hz=frequency_step_hz,
        scene_center_m=scene_center_m,
    )
    return {
        "schema": GOTCHA_DOMAIN_SCHEMA,
        "coordinate_units": "m",
        "x_bounds_m": list(GOTCHA_XY_BOUNDS_M),
        "y_bounds_m": list(GOTCHA_XY_BOUNDS_M),
        "z_reference_m": GOTCHA_Z_REFERENCE_M,
        "scene_center_m": list(GOTCHA_SCENE_CENTER_M),
        "core_radius_m": GOTCHA_CORE_RADIUS_M,
        "reference_range_m": reference_range_m,
        "frequency_step_hz": frequency_step_hz,
        "unambiguous_one_way_range_m": unambiguous_range_m,
        "per_view_range_condition": (
            "0 < reference_range_m + ||x-p|| - ||p-c|| "
            "< unambiguous_one_way_range_m"
        ),
    }


def validate_gotcha_frequency_grid(
    frequencies_hz,
    *,
    nominal_spacing_hz,
    unambiguous_range_m=None,
):
    """Validate GOTCHA frequencies under float32-source quantization.

    Rounding one exact frequency to float32 incurs at most half the larger
    adjacent float32 ULP.  A stored frequency difference therefore has a
    jitter bound equal to the sum of the two endpoint rounding bounds.  The
    allowed step jitter is consequently derived from frequency magnitude,
    rather than from an X-band-specific constant.
    """

    try:
        frequencies = torch.as_tensor(frequencies_hz)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError("frequencies_hz must be real numeric values") from exc
    if frequencies.is_complex() or frequencies.dtype == torch.bool:
        raise TypeError("frequencies_hz must be real numeric values")
    if frequencies.ndim != 1:
        raise ValueError("frequencies_hz must be one-dimensional")
    if frequencies.numel() < 2:
        raise ValueError("frequencies_hz must contain at least two values")
    frequencies = frequencies.to(dtype=torch.float64)
    if not bool(torch.isfinite(frequencies).all().item()):
        raise ValueError("frequencies_hz contains a non-finite value")

    steps = torch.diff(frequencies)
    if not bool((steps > 0.0).all().item()):
        raise ValueError("frequencies_hz must be strictly increasing")
    nominal_spacing_hz = _finite_scalar(
        "nominal_spacing_hz", nominal_spacing_hz
    )
    if nominal_spacing_hz <= 0.0:
        raise ValueError("nominal_spacing_hz must be positive")

    source_frequencies = frequencies.to(dtype=torch.float32)
    if not bool(torch.isfinite(source_frequencies).all().item()):
        raise ValueError(
            "frequencies_hz cannot be represented by a finite float32 source"
        )
    if not torch.equal(source_frequencies.to(dtype=torch.float64), frequencies):
        raise ValueError(
            "frequencies_hz does not exactly round-trip through its declared "
            "float32 source representation"
        )
    toward_positive = torch.nextafter(
        source_frequencies,
        torch.full_like(source_frequencies, math.inf),
    )
    toward_negative = torch.nextafter(
        source_frequencies,
        torch.full_like(source_frequencies, -math.inf),
    )
    upper_ulp = (toward_positive - source_frequencies).abs().to(torch.float64)
    lower_ulp = (source_frequencies - toward_negative).abs().to(torch.float64)
    endpoint_rounding_bound = 0.5 * torch.maximum(upper_ulp, lower_ulp)
    step_tolerances = endpoint_rounding_bound[:-1] + endpoint_rounding_bound[1:]
    if not bool(torch.isfinite(step_tolerances).all().item()):
        raise ValueError(
            "frequencies_hz is too close to the float32 limit for a finite jitter bound"
        )

    deviations = (steps - nominal_spacing_hz).abs()
    violations = deviations > step_tolerances
    if bool(violations.any().item()):
        index = int(torch.nonzero(violations)[0, 0].item())
        raise ValueError(
            "frequency step jitter exceeds the float32 source-quantization bound "
            f"at step {index}"
        )

    endpoint_spacing_hz = float(
        ((frequencies[-1] - frequencies[0]) / (frequencies.numel() - 1)).item()
    )
    endpoint_affine = torch.linspace(
        float(frequencies[0].item()),
        float(frequencies[-1].item()),
        int(frequencies.numel()),
        dtype=torch.float64,
        device=frequencies.device,
    )
    endpoint_affine_residual_hz = float(
        (frequencies - endpoint_affine).abs().max().item()
    )
    endpoint_affine_residual_fraction = (
        endpoint_affine_residual_hz / abs(endpoint_spacing_hz)
    )
    # The active range operator reconstructs this exact endpoint linspace and
    # rejects a raw grid at or above a 1e-2 residual/spacing ratio.
    endpoint_affine_tolerance_fraction = 1.0e-2
    if not endpoint_affine_residual_fraction < endpoint_affine_tolerance_fraction:
        raise ValueError(
            "frequencies_hz is incompatible with the range operator endpoint "
            "linspace contract"
        )

    nominal_unambiguous_range_m = SPEED_OF_LIGHT_M_S / (
        2.0 * nominal_spacing_hz
    )
    if unambiguous_range_m is not None:
        metadata_range_m = _finite_scalar(
            "unambiguous_range_m", unambiguous_range_m
        )
        if metadata_range_m <= 0.0:
            raise ValueError("unambiguous_range_m must be positive")
        comparison_tolerance_m = 8.0 * max(
            math.ulp(nominal_unambiguous_range_m),
            math.ulp(metadata_range_m),
        )
        if (
            abs(metadata_range_m - nominal_unambiguous_range_m)
            > comparison_tolerance_m
        ):
            raise ValueError(
                "unambiguous_range_m disagrees with c / (2 * nominal_spacing_hz)"
            )

    min_spacing_hz = float(steps.min().item())
    max_spacing_hz = float(steps.max().item())
    conservative_spacing_hz = max(nominal_spacing_hz, max_spacing_hz)
    return {
        "frequency_count": int(frequencies.numel()),
        "nominal_spacing_hz": nominal_spacing_hz,
        "min_spacing_hz": min_spacing_hz,
        "max_spacing_hz": max_spacing_hz,
        "max_abs_spacing_deviation_hz": float(deviations.max().item()),
        "float32_source_step_tolerance_hz": float(step_tolerances.max().item()),
        "endpoint_affine_spacing_hz": endpoint_spacing_hz,
        "max_abs_endpoint_affine_residual_hz": endpoint_affine_residual_hz,
        "endpoint_affine_residual_fraction": endpoint_affine_residual_fraction,
        "endpoint_affine_tolerance_fraction": endpoint_affine_tolerance_fraction,
        "conservative_spacing_hz": conservative_spacing_hz,
        "nominal_unambiguous_range_m": nominal_unambiguous_range_m,
        "conservative_unambiguous_range_m": SPEED_OF_LIGHT_M_S
        / (2.0 * conservative_spacing_hz),
    }


def validate_gotcha_position_chunk(position_chunk):
    """Validate and return a tensor of planar full-domain points ``[N, 3]``."""

    positions = _geometry_tensor(position_chunk, "position_chunk")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("position_chunk must have shape [N, 3]")
    if positions.numel() == 0:
        return positions

    xy = positions[:, :2]
    lo, hi = GOTCHA_XY_BOUNDS_M
    if bool(((xy < lo) | (xy > hi)).any().item()):
        raise ValueError("position_chunk lies outside GOTCHA x,y bounds [-50, 50] m")
    if bool((positions[:, 2] != GOTCHA_Z_REFERENCE_M).any().item()):
        raise ValueError("position_chunk must lie exactly on the GOTCHA z=0 plane")
    return positions


def gotcha_nonwrapping_chunk_mask(
    position_chunk,
    viewpoint_positions,
    *,
    reference_range_m,
    frequency_step_hz,
    scene_center_m=GOTCHA_SCENE_CENTER_M,
):
    """Return the exact nonwrapping mask for one or more GOTCHA views.

    ``position_chunk`` has shape ``[N, 3]``.  A single viewpoint ``[3]``
    returns ``[N]``; batched viewpoints ``[V, 3]`` return ``[V, N]``.  The
    interval is deliberately open at both boundaries.
    """

    unambiguous_range_m = validate_gotcha_full_domain(
        reference_range_m=reference_range_m,
        frequency_step_hz=frequency_step_hz,
        scene_center_m=scene_center_m,
    )
    reference_range_m = _finite_scalar("reference_range_m", reference_range_m)
    positions = validate_gotcha_position_chunk(position_chunk)
    # Platform ranges are about 10 km while the admissible-window margin can
    # be sub-metre.  Evaluate the subtraction in float64 even when learned
    # scene parameters use float32; the boolean mask itself is nondifferentiable.
    geometry_positions = positions.to(dtype=torch.float64)
    center = _validated_center(
        scene_center_m,
        device=geometry_positions.device,
        dtype=geometry_positions.dtype,
    )
    viewpoints = _geometry_tensor(
        viewpoint_positions,
        "viewpoint_positions",
        device=geometry_positions.device,
        dtype=geometry_positions.dtype,
    )
    single_view = viewpoints.ndim == 1
    if single_view:
        if viewpoints.shape != (3,):
            raise ValueError("viewpoint_positions must have shape [3] or [V, 3]")
        viewpoints = viewpoints.unsqueeze(0)
    elif viewpoints.ndim != 2 or viewpoints.shape[1] != 3:
        raise ValueError("viewpoint_positions must have shape [3] or [V, 3]")
    if viewpoints.shape[0] == 0:
        raise ValueError("viewpoint_positions must contain at least one view")

    point_ranges = torch.linalg.vector_norm(
        geometry_positions.unsqueeze(0) - viewpoints.unsqueeze(1),
        dim=-1,
    )
    center_ranges = torch.linalg.vector_norm(viewpoints - center, dim=-1)
    virtual_ranges = (
        reference_range_m + point_ranges - center_ranges.unsqueeze(1)
    )
    mask = (virtual_ranges > 0.0) & (virtual_ranges < unambiguous_range_m)
    return mask[0] if single_view else mask


def validate_gotcha_planar_square_views(
    viewpoint_positions,
    *,
    reference_range_m,
    frequency_step_hz,
    scene_center_m=GOTCHA_SCENE_CENTER_M,
):
    """Prove the complete frozen square is nonwrapping for every view.

    For a fixed viewpoint, the minimum distance to the axis-aligned square is
    attained by coordinate clamping and the maximum is attained at a corner.
    Checking those exact extrema avoids materializing the full dense plane.
    The returned margins are positive or the function fails closed.
    """

    unambiguous_range_m = validate_gotcha_full_domain(
        reference_range_m=reference_range_m,
        frequency_step_hz=frequency_step_hz,
        scene_center_m=scene_center_m,
    )
    reference_range_m = _finite_scalar("reference_range_m", reference_range_m)
    viewpoints = _geometry_tensor(
        viewpoint_positions,
        "viewpoint_positions",
        dtype=torch.float64,
    )
    if viewpoints.ndim != 2 or viewpoints.shape[1] != 3:
        raise ValueError("viewpoint_positions must have shape [V, 3]")
    if viewpoints.shape[0] == 0:
        raise ValueError("viewpoint_positions must contain at least one view")
    center = _validated_center(
        scene_center_m,
        device=viewpoints.device,
        dtype=viewpoints.dtype,
    )
    lo, hi = GOTCHA_XY_BOUNDS_M
    nearest = torch.stack(
        (
            viewpoints[:, 0].clamp(lo, hi),
            viewpoints[:, 1].clamp(lo, hi),
            torch.full_like(viewpoints[:, 2], GOTCHA_Z_REFERENCE_M),
        ),
        dim=-1,
    )
    corners = torch.tensor(
        (
            (lo, lo, GOTCHA_Z_REFERENCE_M),
            (lo, hi, GOTCHA_Z_REFERENCE_M),
            (hi, lo, GOTCHA_Z_REFERENCE_M),
            (hi, hi, GOTCHA_Z_REFERENCE_M),
        ),
        dtype=viewpoints.dtype,
        device=viewpoints.device,
    )
    center_ranges = torch.linalg.vector_norm(viewpoints - center, dim=-1)
    minimum_point_ranges = torch.linalg.vector_norm(viewpoints - nearest, dim=-1)
    maximum_point_ranges = torch.linalg.vector_norm(
        viewpoints[:, None, :] - corners[None, :, :], dim=-1
    ).amax(dim=1)
    minimum_virtual = reference_range_m + minimum_point_ranges - center_ranges
    maximum_virtual = reference_range_m + maximum_point_ranges - center_ranges
    lower_margin = float(minimum_virtual.amin().item())
    upper_margin = float((unambiguous_range_m - maximum_virtual).amin().item())
    if not lower_margin > 0.0 or not upper_margin > 0.0:
        raise ValueError(
            "GOTCHA planar square is not wholly inside the per-view "
            "unambiguous range window"
        )
    return {
        "view_count": int(viewpoints.shape[0]),
        "minimum_lower_window_margin_m": lower_margin,
        "minimum_upper_window_margin_m": upper_margin,
        "minimum_virtual_range_m": float(minimum_virtual.amin().item()),
        "maximum_virtual_range_m": float(maximum_virtual.amax().item()),
        "unambiguous_one_way_range_m": float(unambiguous_range_m),
        "mask_realization": "identity_after_exact_planar_extrema_preflight",
    }


__all__ = [
    "GOTCHA_CORE_RADIUS_M",
    "GOTCHA_DOMAIN_SCHEMA",
    "GOTCHA_SCENE_CENTER_M",
    "GOTCHA_XY_BOUNDS_M",
    "GOTCHA_Z_REFERENCE_M",
    "SPEED_OF_LIGHT_M_S",
    "gotcha_full_domain_metadata",
    "gotcha_nonwrapping_chunk_mask",
    "validate_gotcha_frequency_grid",
    "validate_gotcha_full_domain",
    "validate_gotcha_planar_square_views",
    "validate_gotcha_position_chunk",
]
