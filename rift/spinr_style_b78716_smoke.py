"""Bounded B787 data adapter for the SpINR-style 16-view engineering smoke.

The production SpINR-style adapter intentionally keeps all 3,200 training and
1,000 validation responses available for its full development recipe.  This
module is a separate, opt-in capability for the approved small-fit gate: it
streams all parent-training responses only to calculate the frozen scalar
normalizer, while retaining just the selected 16 fit views, 16 validation
views, and the first 32 parent-training initialization views.  It never gives
callers a route to parent test, unused, or unretained development responses.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import torch

from rift.npz_dataset import iter_npz_response_views, restrict_npz_response_views
from rift.spinr_style import (
    RawComplexView,
    _canonical_b787_metadata,
    _canonical_b787_roles,
    _validate_and_average_b787_raw_response,
    metadata_frequency_grid,
    npz_tx_rx_frequency_to_renderer,
)


SMOKE_FIT_VIEW_COUNT = 16
SMOKE_VALIDATION_VIEW_COUNT = 16
SMOKE_INITIALIZATION_VIEW_COUNT = 32


@dataclass(frozen=True)
class B78716SmokeWorklists:
    """Frozen parent-role worklists for one bounded B787 SpINR-style gate."""

    fit_training_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]
    normalization_training_ids: tuple[int, ...]
    initialization_training_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "fit_training_ids": [int(item) for item in self.fit_training_ids],
            "validation_ids": [int(item) for item in self.validation_ids],
            "normalization_training_ids": [int(item) for item in self.normalization_training_ids],
            "initialization_training_ids": [int(item) for item in self.initialization_training_ids],
        }


def bounded_b78716_smoke_worklists(sealed_contract: Mapping[str, object]) -> B78716SmokeWorklists:
    """Derive the approved ordered parent prefixes without response access."""

    _authorized, parent_train = _canonical_b787_roles(sealed_contract)
    roles = sealed_contract.get("role_ids")
    if not isinstance(roles, Mapping):
        raise ValueError("bounded SpINR-style smoke requires canonical role IDs")
    parent_validation = tuple(int(item) for item in roles.get("validation", ()))
    if len(parent_train) != 3200 or len(parent_validation) != 1000:
        raise ValueError("bounded SpINR-style smoke requires the canonical parent train/validation roles")
    worklists = B78716SmokeWorklists(
        fit_training_ids=parent_train[:SMOKE_FIT_VIEW_COUNT],
        validation_ids=parent_validation[:SMOKE_VALIDATION_VIEW_COUNT],
        normalization_training_ids=parent_train,
        initialization_training_ids=parent_train[:SMOKE_INITIALIZATION_VIEW_COUNT],
    )
    if (
        len(worklists.fit_training_ids) != SMOKE_FIT_VIEW_COUNT
        or len(worklists.validation_ids) != SMOKE_VALIDATION_VIEW_COUNT
        or len(worklists.normalization_training_ids) != 3200
        or len(worklists.initialization_training_ids) != SMOKE_INITIALIZATION_VIEW_COUNT
    ):
        raise AssertionError("canonical B787 parent roles no longer support the frozen bounded worklists")
    if worklists.fit_training_ids != worklists.initialization_training_ids[:SMOKE_FIT_VIEW_COUNT]:
        raise AssertionError("bounded fit rows must remain the ordered parent-training prefix")
    return worklists


class BoundedSealedRawComplexViews:
    """Capability-limited raw complex views for the approved B787 small fit."""

    def __init__(self, arrays: Mapping[str, object], sealed_contract: Mapping[str, object]) -> None:
        if arrays.get("response") is not None:
            raise ValueError("bounded SpINR-style adapter requires a lazy pre-payload archive reader")
        worklists = bounded_b78716_smoke_worklists(sealed_contract)
        metadata = arrays.get("meta")
        if not isinstance(metadata, Mapping):
            raise ValueError("bounded SpINR-style adapter requires decoded acquisition metadata")
        self.meta = _canonical_b787_metadata(metadata)
        self.frequencies_hz = metadata_frequency_grid(self.meta).cpu().numpy()
        expected_shape = tuple(int(value) for value in sealed_contract.get("response_shape", ()))
        if expected_shape != (10000, 16, 16, 1, 600):
            raise ValueError("bounded SpINR-style adapter requires [10000,16,16,1,600] responses")
        if sealed_contract.get("response_dtype") != "complex64":
            raise ValueError("bounded SpINR-style adapter requires complex64 raw responses")

        raw_rx_pos = np.asarray(arrays.get("rx_pos"))
        raw_tx_pos = np.asarray(arrays.get("tx_pos"))
        if (
            raw_rx_pos.dtype != np.dtype(np.float64)
            or raw_tx_pos.dtype != np.dtype(np.float64)
            or raw_rx_pos.shape != (10000, 16, 3)
            or raw_tx_pos.shape != (10000, 16, 3)
        ):
            raise ValueError("bounded SpINR-style adapter requires source [view,16,3] float64 poses")
        if not (np.isfinite(raw_rx_pos).all() and np.isfinite(raw_tx_pos).all()):
            raise ValueError("bounded SpINR-style adapter requires finite source poses")

        # The authorized lazy reader can decompress only the parent-training
        # rows required for the streamed normalizer and the selected validation
        # rows.  Parent test/unused IDs never become reader-authorized.
        stream_ids = tuple(dict.fromkeys(
            worklists.normalization_training_ids + worklists.validation_ids))
        retained_ids = frozenset(
            worklists.fit_training_ids
            + worklists.validation_ids
            + worklists.initialization_training_ids
        )
        restricted = restrict_npz_response_views(dict(arrays), stream_ids)
        training_ids = frozenset(worklists.normalization_training_ids)
        response_by_id: dict[int, np.ndarray] = {}
        training_energy_sum = 0.0
        training_energy_count = 0
        for source_id, raw_response in iter_npz_response_views(restricted, stream_ids):
            source_id = int(source_id)
            if source_id not in stream_ids:
                raise AssertionError("bounded lazy reader yielded an unauthorized response")
            chirp_averaged = _validate_and_average_b787_raw_response(raw_response)
            if source_id in training_ids:
                energy_view = chirp_averaged.astype(np.complex128, copy=False)
                training_energy_sum += float(np.vdot(energy_view.reshape(-1), energy_view.reshape(-1)).real)
                training_energy_count += int(energy_view.size)
            if source_id in retained_ids:
                response_by_id[source_id] = chirp_averaged
        expected_retained = set(retained_ids)
        if set(response_by_id) != expected_retained:
            raise RuntimeError("bounded SpINR-style adapter did not retain exactly its frozen worklists")
        expected_energy_count = 3200 * 16 * 16 * 600
        if training_energy_count != expected_energy_count:
            raise RuntimeError(
                "bounded SpINR-style adapter did not stream exactly all 3,200 parent-training response rows")
        if not math.isfinite(training_energy_sum) or training_energy_sum <= 0:
            raise ValueError("bounded SpINR-style streamed training energy must be finite and positive")

        self._worklists = worklists
        self._sealed_contract = dict(sealed_contract)
        self._retained_ids = retained_ids
        self._views = {
            source_id: RawComplexView(
                source_id=source_id,
                response=response_by_id[source_id],
                rx_pos_m=np.asarray(raw_rx_pos[source_id], dtype=np.float64).copy(),
                tx_pos_m=np.asarray(raw_tx_pos[source_id], dtype=np.float64).copy(),
            )
            for source_id in sorted(retained_ids)
        }
        self._training_mean_raw_power = training_energy_sum / training_energy_count
        self._materialized_response_bytes = sum(view.response.nbytes for view in self._views.values())

    @property
    def sealed_contract(self) -> Mapping[str, object]:
        return dict(self._sealed_contract)

    @property
    def worklists(self) -> B78716SmokeWorklists:
        return self._worklists

    @property
    def materialized_response_bytes(self) -> int:
        return int(self._materialized_response_bytes)

    def role_ids(self, role: str) -> tuple[int, ...]:
        if role == "train":
            return self._worklists.fit_training_ids
        if role == "validation":
            return self._worklists.validation_ids
        raise ValueError("bounded SpINR-style smoke exposes only selected train and validation roles")

    def initialization_ids(self) -> tuple[int, ...]:
        return self._worklists.initialization_training_ids

    def raw_training_mean_power(self) -> float:
        return float(self._training_mean_raw_power)

    def view(self, source_id: int) -> RawComplexView:
        """Expose only the selected fit/validation rows to generic callers."""

        source_id = int(source_id)
        selected_ids = frozenset(
            self._worklists.fit_training_ids + self._worklists.validation_ids
        )
        if source_id not in selected_ids:
            raise PermissionError(
                "bounded SpINR-style generic access is limited to selected fit and validation responses")
        return self._views[source_id]

    def initialization_view(self, source_id: int) -> RawComplexView:
        """Expose retained parent train[:32] only to fixed-scale initialization."""

        source_id = int(source_id)
        if source_id not in self._worklists.initialization_training_ids:
            raise PermissionError("bounded SpINR-style initialization access is limited to parent train[:32]")
        return self._views[source_id]

    @staticmethod
    def _tensor_from_view(
        item: RawComplexView,
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        response = torch.as_tensor(item.response, device=device, dtype=torch.complex128)
        response = npz_tx_rx_frequency_to_renderer(response)
        rx_pos = torch.as_tensor(item.rx_pos_m, device=device, dtype=torch.float64)
        tx_pos = torch.as_tensor(item.tx_pos_m, device=device, dtype=torch.float64)
        return response, rx_pos, tx_pos

    def tensor_view(
        self,
        source_id: int,
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one retained response as [frequency,Rx,Tx] complex128 data."""

        return self._tensor_from_view(self.view(source_id), device=device)

    def initialization_tensor_view(
        self,
        source_id: int,
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return an initialization-only parent train[:32] response."""

        return self._tensor_from_view(self.initialization_view(source_id), device=device)


def build_bounded_sealed_raw_complex_views(
    arrays: Mapping[str, object],
    sealed_contract: Mapping[str, object],
) -> BoundedSealedRawComplexViews:
    """Create the frozen 16-fit/16-validation B787 small-fit adapter."""

    return BoundedSealedRawComplexViews(arrays, sealed_contract)
