"""Shared propagation geometry for coherent monostatic radar datasets.

The historical RIFT datasets contain absolute bistatic Tx/Rx coordinates.
CVDomes and GOTCHA do not share that acquisition contract:

* CVDomes is a far-field monostatic angular phase history.
* GOTCHA is monostatic circular SAR with measured platform positions, but its
  delivered phase history is referenced to the scene centre.

Both public datasets are therefore represented in a *virtual reference-range*
gauge.  A constant, positive ``reference_range_m`` places the scene centre in
the unambiguous stepped-frequency range window.  This is only a phase gauge;
it does not change the scene geometry.  The per-point two-way path is

``monostatic_far_field_reference``
    ``2 * (R_ref - dot(u, x - scene_center))``

``monostatic_near_field_reference``
    ``2 * (||x-p|| - ||scene_center-p|| + R_ref)``

The legacy ``bistatic_near_field_absolute`` path remains exactly
``||x-tx|| + ||x-rx||``.  Keeping these formulas in one module prevents the
forward operator, matched filter, Radar Fields adapter, and GeRaF tracer from
quietly using different phase conventions.
"""

from __future__ import annotations

from typing import Sequence

import torch


BISTATIC_NEAR_FIELD_ABSOLUTE = "bistatic_near_field_absolute"
MONOSTATIC_NEAR_FIELD_REFERENCE = "monostatic_near_field_reference"
MONOSTATIC_FAR_FIELD_REFERENCE = "monostatic_far_field_reference"
VALID_PROPAGATION_MODELS = frozenset(
    {
        BISTATIC_NEAR_FIELD_ABSOLUTE,
        MONOSTATIC_NEAR_FIELD_REFERENCE,
        MONOSTATIC_FAR_FIELD_REFERENCE,
    }
)


def _scene_center_tensor(
    scene_center_m: Sequence[float] | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    center = torch.as_tensor(scene_center_m, device=device, dtype=dtype)
    if center.shape != (3,) or not bool(torch.isfinite(center).all()):
        raise ValueError("scene_center_m must contain three finite coordinates")
    return center


def validate_monostatic_geometry(
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    atol_m: float = 1.0e-5,
) -> torch.Tensor:
    """Return the monostatic platform position after strict co-location checks."""

    if tx_positions.ndim != 2 or tx_positions.shape[-1] != 3:
        raise ValueError("tx_positions must have shape [Tx,3]")
    if rx_positions.ndim != 2 or rx_positions.shape[-1] != 3:
        raise ValueError("rx_positions must have shape [Rx,3]")
    if tx_positions.shape[0] != 1 or rx_positions.shape[0] != 1:
        raise ValueError(
            "public coherent-radar monostatic adapters require exactly one Tx and one Rx"
        )
    tx = tx_positions[0]
    rx = rx_positions[0]
    if not bool(torch.isfinite(tx).all()) or not bool(torch.isfinite(rx).all()):
        raise ValueError("monostatic platform coordinates must be finite")
    if not bool(torch.allclose(tx, rx, rtol=0.0, atol=float(atol_m))):
        raise ValueError("monostatic Tx and Rx phase centres are not co-located")
    return 0.5 * (tx + rx)


def point_pair_path_lengths(
    points: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_indices: torch.Tensor,
    rx_indices: torch.Tensor,
    *,
    propagation_model: str = BISTATIC_NEAR_FIELD_ABSOLUTE,
    reference_range_m: float | None = None,
    scene_center_m: Sequence[float] | torch.Tensor = (0.0, 0.0, 0.0),
    eps: float = 1.0e-9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Tx leg, Rx leg, and two-way path for ``[point,pair]``.

    The first two outputs are virtual one-way legs for the reference-range
    models.  That lets existing geometric-gain code remain well-defined, even
    though external-dataset commands deliberately use ``range_model=none``.
    """

    if propagation_model not in VALID_PROPAGATION_MODELS:
        raise ValueError(
            f"unknown propagation_model {propagation_model!r}; expected one of "
            f"{sorted(VALID_PROPAGATION_MODELS)}"
        )
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape [N,3]")
    if propagation_model == BISTATIC_NEAR_FIELD_ABSOLUTE:
        r_tx = torch.linalg.vector_norm(
            points[:, None, :] - tx_positions[None, :, :], dim=-1
        ).clamp_min(eps)
        r_rx = torch.linalg.vector_norm(
            points[:, None, :] - rx_positions[None, :, :], dim=-1
        ).clamp_min(eps)
        r_tx_pair = r_tx[:, tx_indices]
        r_rx_pair = r_rx[:, rx_indices]
        return r_tx_pair, r_rx_pair, r_tx_pair + r_rx_pair

    if reference_range_m is None or not torch.isfinite(
        torch.as_tensor(reference_range_m)
    ) or float(reference_range_m) <= 0.0:
        raise ValueError("reference_range_m must be finite and positive")
    platform = validate_monostatic_geometry(tx_positions, rx_positions)
    center = _scene_center_tensor(
        scene_center_m, device=points.device, dtype=points.dtype
    )
    if propagation_model == MONOSTATIC_NEAR_FIELD_REFERENCE:
        centre_range = torch.linalg.vector_norm(platform - center).clamp_min(eps)
        one_way = (
            torch.linalg.vector_norm(points - platform[None, :], dim=-1)
            - centre_range
            + float(reference_range_m)
        )
    else:
        look = platform - center
        look = look / torch.linalg.vector_norm(look).clamp_min(eps)
        one_way = float(reference_range_m) - (points - center[None, :]) @ look
    if not bool(torch.isfinite(one_way).all()) or bool((one_way <= 0.0).any()):
        raise ValueError(
            "reference range is too small for the requested scene points"
        )
    # The public adapters have exactly one pair, but preserve the [N,P] shape.
    pair_count = int(tx_indices.numel())
    one_way_pair = one_way[:, None].expand(-1, pair_count)
    return one_way_pair, one_way_pair, 2.0 * one_way_pair


def one_way_range_coordinates(
    points: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    pair_indices: torch.Tensor,
    *,
    propagation_model: str = BISTATIC_NEAR_FIELD_ABSOLUTE,
    reference_range_m: float | None = None,
    scene_center_m: Sequence[float] | torch.Tensor = (0.0, 0.0, 0.0),
    eps: float = 1.0e-9,
) -> torch.Tensor:
    """Return one-way Radar-Fields bin coordinates with shape ``[pair,N]``."""

    num_rx = int(rx_positions.shape[0])
    tx_indices = torch.div(pair_indices.long(), num_rx, rounding_mode="floor")
    rx_indices = torch.remainder(pair_indices.long(), num_rx)
    r_tx, r_rx, _ = point_pair_path_lengths(
        points,
        tx_positions,
        rx_positions,
        tx_indices,
        rx_indices,
        propagation_model=propagation_model,
        reference_range_m=reference_range_m,
        scene_center_m=scene_center_m,
        eps=eps,
    )
    return (0.5 * (r_tx + r_rx)).transpose(0, 1).contiguous()


__all__ = [
    "BISTATIC_NEAR_FIELD_ABSOLUTE",
    "MONOSTATIC_NEAR_FIELD_REFERENCE",
    "MONOSTATIC_FAR_FIELD_REFERENCE",
    "VALID_PROPAGATION_MODELS",
    "one_way_range_coordinates",
    "point_pair_path_lengths",
    "validate_monostatic_geometry",
]
