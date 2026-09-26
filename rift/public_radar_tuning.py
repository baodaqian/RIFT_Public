"""Pure helpers for the isolated CVDomes Camry tuning pilots.

This module intentionally contains no Torch or cluster dependencies.  The
Slurm entrypoint imports these helpers, while local tests can exercise the
selection, ordering, update-budget, and closed-form-gain contracts without a
PACE environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPOCH_STRIDE = 0x9E3779B1


def _as_1d_int(values, name):
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def validate_embedded_partition(
    n_view,
    train_indices,
    validation_indices,
    test_indices,
    group_ids,
):
    """Validate complete, disjoint roles and acquisition-group isolation."""
    roles = {
        "train": _as_1d_int(train_indices, "train_indices"),
        "validation": _as_1d_int(validation_indices, "validation_indices"),
        "test": _as_1d_int(test_indices, "test_indices"),
    }
    combined = np.concatenate(tuple(roles.values()))
    if combined.size != int(n_view):
        raise ValueError(
            f"embedded roles contain {combined.size} entries for {n_view} views"
        )
    if np.any(combined < 0) or np.any(combined >= int(n_view)):
        raise ValueError("embedded role index is outside the dataset")
    if np.unique(combined).size != int(n_view):
        raise ValueError("embedded roles are not a disjoint complete partition")

    group_ids = _as_1d_int(group_ids, "group_ids")
    if group_ids.size != int(n_view):
        raise ValueError("group_ids length disagrees with n_view")
    owner = {}
    for role, indices in roles.items():
        for group in np.unique(group_ids[indices]):
            previous = owner.setdefault(int(group), role)
            if previous != role:
                raise ValueError(
                    f"acquisition group {int(group)} leaks across {previous}/{role}"
                )
    return {
        "n_view": int(n_view),
        "train": int(roles["train"].size),
        "validation": int(roles["validation"].size),
        "test": int(roles["test"].size),
        "groups_by_role": {
            role: int(np.unique(group_ids[indices]).size)
            for role, indices in roles.items()
        },
    }


def interleaved_group_split(group_ids, azimuth_deg, seed=42):
    """Build the sealed public-radar interpolation split.

    Acquisition groups are ordered once around azimuth.  In every ten-group
    block, slot 0 is validation, slot 5 is test, and the other eight slots are
    training.  Membership is therefore deterministic and uniformly
    interleaved over the full orbit.  ``seed`` only scrambles the order of the
    whole groups within each role; it never changes membership or separates a
    group's views.
    """
    groups = _as_1d_int(group_ids, "group_ids")
    azimuth = np.asarray(azimuth_deg, dtype=np.float64)
    if azimuth.ndim != 1 or azimuth.size != groups.size:
        raise ValueError("azimuth_deg must be one-dimensional and match group_ids")
    if not np.isfinite(azimuth).all():
        raise ValueError("azimuth_deg must be finite")

    unique_groups = np.unique(groups)
    if unique_groups.size == 0 or unique_groups.size % 10 != 0:
        raise ValueError("sealed interleaved split requires a nonzero multiple of 10 groups")

    group_angles = []
    for group in unique_groups:
        radians = np.deg2rad(azimuth[groups == group])
        mean_vector = np.exp(1j * radians).mean()
        if abs(mean_vector) <= 1.0e-12:
            raise ValueError(f"acquisition group {int(group)} has undefined mean azimuth")
        angle = float(np.degrees(np.angle(mean_vector)) % 360.0)
        group_angles.append(angle)
    group_angles = np.asarray(group_angles, dtype=np.float64)
    order = np.lexsort((unique_groups, group_angles))
    angular_groups = unique_groups[order]
    angular_degrees = group_angles[order]

    slots = np.arange(angular_groups.size, dtype=np.int64) % 10
    group_roles = {
        "train": angular_groups[(slots != 0) & (slots != 5)],
        "validation": angular_groups[slots == 0],
        "test": angular_groups[slots == 5],
    }
    role_indices = {}
    for role_index, (role, role_groups) in enumerate(group_roles.items()):
        rng = np.random.default_rng(int(seed) + _EPOCH_STRIDE * role_index)
        shuffled_groups = rng.permutation(role_groups)
        pieces = [np.flatnonzero(groups == group) for group in shuffled_groups]
        role_indices[role] = np.concatenate(pieces).astype(np.int64, copy=False)

    partition = validate_embedded_partition(
        groups.size,
        role_indices["train"],
        role_indices["validation"],
        role_indices["test"],
        groups,
    )
    partition.update(
        {
            "strategy": "angular_group_interleaved_8train_1validation_1test_v2",
            "seed": int(seed),
            "ordered_group_ids": angular_groups.tolist(),
            "ordered_group_azimuth_deg": angular_degrees.tolist(),
            "groups": {role: values.tolist() for role, values in group_roles.items()},
        }
    )
    return role_indices, partition


def select_balanced_direction_fps(
    train_indices,
    positions,
    group_ids,
    count,
):
    """Select a deterministic, angularly broad BP subset in group rounds.

    Farthest-point sampling is restricted to the least-used acquisition
    groups.  Consequently every available training group is used once before
    any is used twice, and so on.  This supports BP counts larger than the
    number of groups without allowing a narrow sector to dominate.
    """
    train = np.sort(_as_1d_int(train_indices, "train_indices"))
    count = int(count)
    if count <= 0 or count > train.size:
        raise ValueError("count must be positive and no larger than the train role")
    positions = np.asarray(positions, dtype=np.float64)
    groups = _as_1d_int(group_ids, "group_ids")
    if positions.ndim != 2 or positions.shape != (groups.size, 3):
        raise ValueError("positions must have shape [len(group_ids), 3]")
    if np.any(train < 0) or np.any(train >= groups.size):
        raise ValueError("train_indices contains an out-of-range index")

    directions = unit_directions(positions[train])
    candidate_groups, group_codes = np.unique(groups[train], return_inverse=True)
    group_counts = np.zeros(candidate_groups.size, dtype=np.int64)
    chosen = np.zeros(train.size, dtype=bool)
    # For each candidate, max cosine over selected directions is the cosine
    # of its *nearest* selected direction.  FPS minimizes that value.
    nearest_cosine = np.full(train.size, -np.inf, dtype=np.float64)
    selected_local = []

    for _ in range(count):
        least_used = int(group_counts.min())
        eligible = (~chosen) & (group_counts[group_codes] == least_used)
        if not eligible.any():
            raise RuntimeError("balanced FPS exhausted an acquisition-group round")
        eligible_local = np.flatnonzero(eligible)
        if not selected_local:
            selected = int(eligible_local[0])
        else:
            scores = nearest_cosine[eligible_local]
            best = float(scores.min())
            tied = eligible_local[np.isclose(scores, best, rtol=0.0, atol=1.0e-12)]
            selected = int(tied[np.argmin(train[tied])])
        selected_local.append(selected)
        chosen[selected] = True
        group_counts[group_codes[selected]] += 1
        cosine = directions @ directions[selected]
        nearest_cosine = np.maximum(nearest_cosine, cosine)

    selected_indices = train[np.asarray(selected_local, dtype=np.int64)]
    if np.unique(selected_indices).size != count:
        raise AssertionError("balanced FPS repeated a viewpoint")
    selected_group_counts = np.unique(groups[selected_indices], return_counts=True)[1]
    if int(selected_group_counts.max() - selected_group_counts.min()) > 1:
        raise AssertionError("balanced FPS group counts differ by more than one")
    return selected_indices


def unit_directions(positions):
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [n, 3]")
    norms = np.linalg.norm(positions, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("positions must be finite and nonzero")
    return positions / norms


def _elevation_levels(elevations, indices, tolerance=1.0e-3):
    values = np.asarray(elevations, dtype=np.float64)[indices]
    if not np.isfinite(values).all():
        raise ValueError("elevations must be finite")
    ordered = np.sort(values)
    levels = []
    for value in ordered:
        if not levels or abs(value - levels[-1]) > tolerance:
            levels.append(float(value))
        else:
            levels[-1] = 0.5 * (levels[-1] + float(value))
    return np.asarray(levels, dtype=np.float64)


def _quota_by_level(levels, count):
    if levels.size == 0:
        raise ValueError("no elevation levels are available")
    base, remainder = divmod(int(count), int(levels.size))
    return {
        float(level): base + (1 if index < remainder else 0)
        for index, level in enumerate(levels)
    }


def select_group_stratified_fps(
    train_indices,
    positions,
    group_ids,
    elevations,
    count=100,
    elevation_tolerance=1.0e-3,
):
    """Select angularly broad views with unique groups and elevation quotas.

    The selection is deterministic.  Elevation levels take turns; at each turn
    the unused-group candidate farthest from the already selected directions is
    chosen, with global dataset index as the tie-breaker.
    """
    train_indices = _as_1d_int(train_indices, "train_indices")
    if count <= 0 or count > train_indices.size:
        raise ValueError("count must be positive and no larger than the train role")
    positions = np.asarray(positions, dtype=np.float64)
    groups = _as_1d_int(group_ids, "group_ids")
    elevations = np.asarray(elevations, dtype=np.float64)
    if positions.shape[0] != groups.size or groups.size != elevations.size:
        raise ValueError("positions/group_ids/elevations length mismatch")

    directions = unit_directions(positions)
    levels = _elevation_levels(elevations, train_indices, elevation_tolerance)
    quotas = _quota_by_level(levels, count)
    candidates = {}
    for level in levels:
        mask = np.abs(elevations[train_indices] - level) <= elevation_tolerance
        level_candidates = np.sort(train_indices[mask])
        if level_candidates.size < quotas[float(level)]:
            raise ValueError(f"elevation {level:g} cannot satisfy its quota")
        candidates[float(level)] = level_candidates

    selected = []
    selected_counts = {float(level): 0 for level in levels}
    used_groups = set()
    while len(selected) < int(count):
        made_progress = False
        for level in levels:
            key = float(level)
            if selected_counts[key] >= quotas[key]:
                continue
            available = np.asarray(
                [
                    int(index)
                    for index in candidates[key]
                    if int(groups[index]) not in used_groups
                ],
                dtype=np.int64,
            )
            if available.size == 0:
                raise ValueError(
                    f"elevation {level:g} exhausted unique acquisition groups"
                )
            if not selected:
                chosen = int(available[0])
            else:
                cosine = directions[available] @ directions[np.asarray(selected)].T
                nearest_angle = np.arccos(np.clip(cosine, -1.0, 1.0)).min(axis=1)
                best = float(nearest_angle.max())
                tied = available[np.isclose(nearest_angle, best, rtol=0.0, atol=1.0e-12)]
                chosen = int(tied.min())
            selected.append(chosen)
            used_groups.add(int(groups[chosen]))
            selected_counts[key] += 1
            made_progress = True
            if len(selected) == int(count):
                break
        if not made_progress:
            raise RuntimeError("stratified FPS made no progress")

    selected_array = np.asarray(selected, dtype=np.int64)
    if np.unique(groups[selected_array]).size != selected_array.size:
        raise AssertionError("stratified FPS repeated an acquisition group")
    return selected_array


def angular_coverage_hole_deg(reference_positions, selected_positions):
    """Largest nearest-selected angular distance over reference directions."""
    reference = unit_directions(reference_positions)
    selected = unit_directions(selected_positions)
    nearest_cosine = (reference @ selected.T).max(axis=1)
    return float(np.degrees(np.arccos(np.clip(nearest_cosine, -1.0, 1.0))).max())


def epoch_group_order(group_ids, seed, epoch):
    """Epoch-keyed shuffle that keeps each acquisition group contiguous."""
    groups = _as_1d_int(group_ids, "group_ids")
    derived_seed = (int(seed) + _EPOCH_STRIDE * int(epoch)) % (2**63 - 1)
    rng = np.random.default_rng(derived_seed)
    unique_groups = np.unique(groups)
    shuffled_groups = rng.permutation(unique_groups)
    pieces = []
    for group in shuffled_groups:
        members = np.flatnonzero(groups == group)
        pieces.append(rng.permutation(members))
    order = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
    if order.size != groups.size or np.unique(order).size != groups.size:
        raise AssertionError("epoch group shuffle did not return a permutation")
    return order


def accumulation_windows(order, update_count):
    """Partition all views into exactly ``update_count`` near-equal windows."""
    order = _as_1d_int(order, "order")
    update_count = int(update_count)
    if update_count <= 0 or update_count > order.size:
        raise ValueError("update_count must be in [1, number of views]")
    base, remainder = divmod(int(order.size), update_count)
    sizes = np.full(update_count, base, dtype=np.int64)
    sizes[:remainder] += 1
    boundaries = np.concatenate(([0], np.cumsum(sizes)))
    windows = [order[boundaries[i] : boundaries[i + 1]] for i in range(update_count)]
    if not np.array_equal(np.concatenate(windows), order):
        raise AssertionError("accumulation windows changed the view order")
    return windows


@dataclass(frozen=True)
class ComplexAccumulators:
    """Sufficient statistics for fitting/scoring one global complex gain."""

    cross: complex
    predicted_power: float
    measured_power: float
    sample_count: int

    def validate(self):
        values = (
            self.cross.real,
            self.cross.imag,
            self.predicted_power,
            self.measured_power,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("gain accumulators must be finite")
        if self.predicted_power < 0 or self.measured_power <= 0:
            raise ValueError("gain accumulators have invalid powers")
        if int(self.sample_count) <= 0:
            raise ValueError("sample_count must be positive")
        return self


def closed_form_gain(accumulators):
    accumulators.validate()
    if accumulators.predicted_power <= 0:
        return 0.0 + 0.0j
    return complex(accumulators.cross / accumulators.predicted_power)


def score_gain(accumulators, gain):
    accumulators.validate()
    gain = complex(gain)
    if not np.isfinite(gain.real) or not np.isfinite(gain.imag):
        raise ValueError("gain must be finite")
    a = accumulators.cross
    b = float(accumulators.predicted_power)
    c = float(accumulators.measured_power)
    squared_error = abs(gain) ** 2 * b - 2.0 * np.real(np.conjugate(gain) * a) + c
    # Roundoff can make a mathematically zero residual very slightly negative.
    squared_error = max(float(squared_error), 0.0)
    correlation = abs(a) / np.sqrt(b * c) if b > 0 else 0.0
    return {
        "gain_real": float(gain.real),
        "gain_imag": float(gain.imag),
        "gain_magnitude": float(abs(gain)),
        "gain_phase_rad": float(np.angle(gain)),
        "relative_mse": float(squared_error / c),
        "relative_l2": float(np.sqrt(squared_error / c)),
        "predicted_to_measured_power": float(abs(gain) ** 2 * b / c),
        "coherent_correlation": float(correlation),
        "sample_count": int(accumulators.sample_count),
    }
