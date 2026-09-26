#!/usr/bin/env python3
"""CPU regression gate for AdaptivePointSHScene split/prune slot semantics.

This deliberately exercises optimizer state, not only scene values. The legacy
public split must retire its parent and allocate eight fresh child slots, while
the explicit v2 in-place route keeps the live heir’s coefficient moments.
Recentered positions and every recycled/new row must carry no stale state.
"""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.sparse_scene import AdaptivePointSHScene


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def check_equal(actual, expected, message):
    if not torch.equal(actual, expected):
        raise AssertionError(f"{message}\nactual={actual}\nexpected={expected}")


def check_close(actual, expected, message, atol=1e-6, rtol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{message}\nactual={actual}\nexpected={expected}")


def make_scene(n_active, capacity, max_degree=1, init_degree=1, half=1.0):
    anchors = torch.stack([
        torch.tensor([float(i), 0.25 * i, -0.125 * i])
        for i in range(n_active)
    ])
    return AdaptivePointSHScene(
        anchors, half, torch.device("cpu"), max_degree=max_degree,
        init_degree=init_degree, init_scale=0.0, learn_positions=True,
        capacity=capacity,
    )


def make_optimizer(scene):
    return torch.optim.AdamW([
        {"params": [scene.w_re, scene.w_im], "lr": 1e-3, "weight_decay": 0.0},
        {"params": [scene.delta_raw], "lr": 1e-3, "weight_decay": 0.0},
    ])


def seed_dense_adam_state(scene, optimizer):
    """Populate every row, including spare rows, with distinguishable state."""
    for parameter_index, parameter in enumerate((scene.w_re, scene.w_im, scene.delta_raw), start=1):
        values = torch.arange(1, parameter.numel() + 1, dtype=parameter.dtype).reshape_as(parameter)
        parameter.grad = values * (parameter_index * 1e-3)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def moment_row(optimizer, parameter, key, row):
    return optimizer.state[parameter][key][row].detach().clone()


def check_zero_moments(optimizer, parameter, indices, message):
    for key in ("exp_avg", "exp_avg_sq"):
        state = optimizer.state[parameter][key][indices]
        check_equal(state, torch.zeros_like(state), f"{message}: {key}")


def test_world_position_gradient_accumulation():
    scene = make_scene(1, 2, max_degree=0, init_degree=0, half=2.0)
    raw = math.atanh(0.5)
    with torch.no_grad():
        scene.delta_raw[0] = torch.tensor([raw, 0.0, -raw])

    # jacobian = h*(1-tanh(delta)^2) = [1.5, 2.0, 1.5]. Choose
    # dL/dx=[3,4,0], hence dL/ddelta=[4.5,8,0].
    scene.w_re.grad = torch.zeros_like(scene.w_re)
    scene.w_im.grad = torch.zeros_like(scene.w_im)
    scene.delta_raw.grad = torch.zeros_like(scene.delta_raw)
    scene.delta_raw.grad[0] = torch.tensor([4.5, 8.0, 0.0])
    scene.delta_raw.grad[1] = 1e6  # inactive row must never enter the statistic
    scene.accumulate_grad_stats()

    check_close(scene.pos_grad_accum[0], torch.tensor(5.0),
                "pos_grad_accum must be physical ||dL/dx||")
    check_close(scene.pos_raw_grad_accum[0], torch.tensor(math.hypot(4.5, 8.0)),
                "pos_raw_grad_accum must retain ||dL/ddelta_raw|| as a diagnostic")
    check_close(scene.pos_world_grad_sum[0], torch.tensor([3.0, 4.0, 0.0]),
                "world-gradient vector sum must undo the raw-offset chain rule")
    check_close(scene.pos_world_grad_sq_accum[0], torch.tensor([9.0, 16.0, 0.0]),
                "world-gradient diagonal second moment must be accumulated")
    check_equal(scene.pos_grad_accum[1], torch.tensor(0.0),
                "inactive position gradients must be masked")
    check_equal(scene.pos_raw_grad_accum[1], torch.tensor(0.0),
                "inactive raw-position gradients must be masked")
    check(int(scene.grad_accum_count) == 1, "gradient-window count must advance once")


def test_legacy_eight_child_slot_layout_and_optimizer_state():
    """The public default keeps PACE's retired-parent/eight-child layout."""
    scene = make_scene(2, 10)
    optimizer = make_optimizer(scene)
    seed_dense_adam_state(scene, optimizer)

    with torch.no_grad():
        scene.w_re.zero_()
        scene.w_im.zero_()
        scene.delta_raw.zero_()
        scene.w_re[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scene.w_im[0] = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        scene.w_re[1] = torch.tensor([5.0, 6.0, 7.0, 8.0])
        scene.w_im[1] = torch.tensor([0.5, 1.5, 2.5, 3.5])
        scene.delta_raw[1] = torch.tensor([0.2, -0.1, 0.3])
        # Position statistic selects slot 1 rather than the coefficient-rich
        # slot 0, so the exact historical source/destination slots are clear.
        scene.grad_accum[:2] = torch.tensor([100.0, 1.0])
        scene.pos_grad_accum[:2] = torch.tensor([1.0, 10.0])
        scene.grad_accum_count.fill_(1)

    theta = torch.tensor([[0.7]])
    phi = torch.tensor([[-0.4]])
    positions_before, weights_before = scene.active_scatterers(theta, phi)
    shape_before = tuple(scene.w_re.shape)
    parent_anchor = scene.anchors[1].detach().clone()
    parent_half = scene.cell_half[1, 0].detach().clone()
    parent_position = scene.positions()[1].detach().clone()
    parent_w_re = scene.w_re[1].detach().clone()
    parent_w_im = scene.w_im[1].detach().clone()
    untouched_state = moment_row(optimizer, scene.w_re, "exp_avg", 0)

    n_split, n_active, report = scene.split(
        criterion="position_world", mode="count", count=1,
        max_active=9, optimizer=optimizer, return_report=True,
    )

    children = torch.arange(2, 10)
    octant_bits = (parent_position >= parent_anchor).long()
    heir_local = int((octant_bits[0] * 4 + octant_bits[1] * 2 + octant_bits[2]).item())
    heir_idx = int(children[heir_local].item())
    non_heir = children[children != heir_idx]
    signs = torch.tensor(
        [[sx, sy, sz] for sx in (-1.0, 1.0)
         for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=parent_anchor.dtype,
    )
    expected_anchors = parent_anchor[None, :] + 0.5 * parent_half * signs
    expected_anchors[heir_local] = parent_position

    check(n_split == 1 and n_active == 9,
          "legacy split must retire one parent and activate exactly eight children")
    check(tuple(scene.w_re.shape) == shape_before, "fixed-capacity tensor shape must not change")
    check(bool(scene.active_mask[0]) and not bool(scene.active_mask[1]),
          "legacy layout must retire the selected parent slot")
    check(bool(scene.active_mask[children].all()),
          "legacy layout must activate the first eight inactive child slots")
    check_equal(scene.anchors[children], expected_anchors,
                "legacy children must occupy the historical octant slot order, with a recentered heir")
    check_equal(scene.cell_half[children], torch.ones_like(scene.cell_half[children]) * (0.5 * parent_half),
                "legacy child cells must use half the parent pitch")
    check_equal(scene.level[children], torch.ones_like(scene.level[children]),
                "legacy children must increment the parent refinement level")
    check_equal(scene.w_re[heir_idx], parent_w_re,
                "the legacy heir child must inherit the parent real SH coefficients")
    check_equal(scene.w_im[heir_idx], parent_w_im,
                "the legacy heir child must inherit the parent imaginary SH coefficients")
    check_equal(scene.w_re[non_heir], torch.zeros_like(scene.w_re[non_heir]),
                "the seven non-heir legacy children must start at zero real SH")
    check_equal(scene.w_im[non_heir], torch.zeros_like(scene.w_im[non_heir]),
                "the seven non-heir legacy children must start at zero imaginary SH")
    check_equal(scene.anchors[1], torch.zeros(3), "retired legacy parent anchor must be tombstoned")
    check_equal(scene.cell_half[1], torch.zeros(1), "retired legacy parent cell must be tombstoned")
    check_equal(scene.delta_raw[1], torch.zeros(3), "retired legacy parent offset must be tombstoned")
    check_equal(scene.w_re[1], torch.zeros_like(scene.w_re[1]),
                "retired legacy parent real SH must be tombstoned")
    check_equal(scene.w_im[1], torch.zeros_like(scene.w_im[1]),
                "retired legacy parent imaginary SH must be tombstoned")

    fresh_rows = torch.cat([torch.tensor([1]), children])
    for parameter in (scene.w_re, scene.w_im, scene.delta_raw):
        check_zero_moments(optimizer, parameter, fresh_rows,
                           "retired/new legacy rows must not retain Adam moments")
    check_equal(moment_row(optimizer, scene.w_re, "exp_avg", 0), untouched_state,
                "unselected optimizer state must remain untouched")

    positions_after, weights_after = scene.active_scatterers(theta, phi)
    check_equal(positions_after[0], positions_before[0],
                "unselected legacy scatterer position must remain identical")
    check_close(weights_after[0], weights_before[0],
                "unselected legacy scatterer weight must remain identical", atol=0, rtol=0)
    check_equal(positions_after[1 + heir_local], parent_position,
                "legacy heir must preserve the parent rendered position")
    check_close(weights_after[1 + heir_local], weights_before[1],
                "legacy heir must preserve the parent rendered weight", atol=0, rtol=0)
    child_output_idx = 1 + torch.arange(8)
    non_heir_output = child_output_idx[child_output_idx != 1 + heir_local]
    check_equal(weights_after[non_heir_output],
                torch.zeros_like(weights_after[non_heir_output]),
                "legacy non-heir children must be render-zero")
    check("legacy_densify" in report and "free=1" in report and "active=1" in report,
          "legacy split report must expose the eight-slot and active-count budgets")

    # Seven free slots used to be enough for the accidental in-place default;
    # the historical public route must now refuse that allocation cleanly.
    no_space = make_scene(1, 8)
    with torch.no_grad():
        no_space.pos_grad_accum[0] = 1.0
        no_space.grad_accum_count.fill_(1)
    check(no_space.split(criterion="position_world", mode="count", count=1) == (0, 1),
          "legacy split must require eight free child slots")

    public_shape = make_scene(1, 9)
    with torch.no_grad():
        public_shape.pos_grad_accum[0] = 1.0
        public_shape.grad_accum_count.fill_(1)
    result = public_shape.split(criterion="position_world", mode="count", count=1)
    check(isinstance(result, tuple) and len(result) == 2 and result == (1, 8),
          "legacy public split must retain its two-value return shape and +7 active count")


def test_legacy_default_parent_order_and_budget():
    """Default coefficient/relmax keeps old slot ordering until capacity binds."""
    ample = make_scene(2, 18)
    with torch.no_grad():
        ample.w_re.zero_()
        ample.w_im.zero_()
        ample.w_re[0, 0] = 11.0
        ample.w_re[1, 0] = 22.0
        # Both clear the default 10%-of-max cutoff, but parent 1 scores more.
        ample.grad_accum[:2] = torch.tensor([2.0, 10.0])
        ample.grad_accum_count.fill_(1)
    parent0_anchor = ample.anchors[0].detach().clone()
    parent1_anchor = ample.anchors[1].detach().clone()
    result = ample.split()
    check(result == (2, 16), "ample legacy capacity must split both selected parents")
    # Historical split_mask.nonzero() assigned the first eight free child rows
    # to parent 0 and the second eight to parent 1, not score-ranked order.
    check_equal(ample.anchors[2], parent0_anchor + torch.tensor([-0.5, -0.5, -0.5]),
                "ample legacy first child block must belong to ascending parent 0")
    check_equal(ample.anchors[10], parent1_anchor + torch.tensor([-0.5, -0.5, -0.5]),
                "ample legacy second child block must belong to ascending parent 1")
    check_equal(ample.w_re[9, 0], torch.tensor(11.0),
                "ample legacy first heir must remain in parent-0 child block")
    check_equal(ample.w_re[17, 0], torch.tensor(22.0),
                "ample legacy second heir must remain in parent-1 child block")

    limited = make_scene(2, 10)
    with torch.no_grad():
        limited.w_re.zero_()
        limited.w_im.zero_()
        limited.w_re[0, 0] = 11.0
        limited.w_re[1, 0] = 22.0
        limited.grad_accum[:2] = torch.tensor([2.0, 10.0])
        limited.grad_accum_count.fill_(1)
    limited_parent1_anchor = limited.anchors[1].detach().clone()
    result = limited.split()
    check(result == (1, 9), "one eight-child free-slot budget must select one legacy parent")
    check(bool(limited.active_mask[0]) and not bool(limited.active_mask[1]),
          "budget-limited legacy default must select the higher-score parent")
    check_equal(limited.anchors[2], limited_parent1_anchor + torch.tensor([-0.5, -0.5, -0.5]),
                "budget-limited child block must belong to the higher-score parent")
    check_equal(limited.w_re[9, 0], torch.tensor(22.0),
                "budget-limited legacy heir must inherit the higher-score parent")

    active_capped = make_scene(2, 18)
    with torch.no_grad():
        active_capped.w_re.zero_()
        active_capped.w_im.zero_()
        active_capped.w_re[0, 0] = 11.0
        active_capped.w_re[1, 0] = 22.0
        active_capped.grad_accum[:2] = torch.tensor([2.0, 10.0])
        active_capped.grad_accum_count.fill_(1)
    result = active_capped.split(max_active=9)
    check(result == (1, 9), "active cap must limit the legacy route to one +7 split")
    check(bool(active_capped.active_mask[0]) and not bool(active_capped.active_mask[1]),
          "active-cap truncation must select the higher-score legacy parent")
    check_equal(active_capped.w_re[9, 0], torch.tensor(22.0),
                "active-cap-limited legacy heir must inherit the higher-score parent")


def test_in_place_heir_and_optimizer_state():
    scene = make_scene(2, 9)
    optimizer = make_optimizer(scene)
    seed_dense_adam_state(scene, optimizer)

    with torch.no_grad():
        scene.w_re.zero_()
        scene.w_im.zero_()
        scene.delta_raw.zero_()
        scene.w_re[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scene.w_im[0] = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        scene.w_re[1] = torch.tensor([5.0, 6.0, 7.0, 8.0])
        scene.w_im[1] = torch.tensor([0.5, 1.5, 2.5, 3.5])
        scene.delta_raw[1] = torch.tensor([0.2, -0.1, 0.3])
        # Coefficient statistic prefers slot 0; position statistic prefers 1.
        scene.grad_accum[:2] = torch.tensor([100.0, 1.0])
        scene.pos_grad_accum[:2] = torch.tensor([1.0, 10.0])
        scene.grad_accum_count.fill_(1)

    theta = torch.tensor([[0.7]])
    phi = torch.tensor([[-0.4]])
    positions_before, weights_before = scene.active_scatterers(theta, phi)
    shape_before = tuple(scene.w_re.shape)
    parent_w_re = scene.w_re[1].detach().clone()
    parent_w_im = scene.w_im[1].detach().clone()
    parent_position = scene.positions()[1].detach().clone()
    parent_coeff_state = {
        (name, key): moment_row(optimizer, parameter, key, 1)
        for name, parameter in (("re", scene.w_re), ("im", scene.w_im))
        for key in ("exp_avg", "exp_avg_sq")
    }
    untouched_state = moment_row(optimizer, scene.w_re, "exp_avg", 0)

    n_split, n_active, report = scene.split(
        criterion="position_world", mode="count", count=1,
        max_active=9, optimizer=optimizer, return_report=True,
        in_place_heir=True,
    )

    check(n_split == 1 and n_active == 9, "one parent must add exactly seven active siblings")
    check(tuple(scene.w_re.shape) == shape_before, "fixed-capacity tensor shape must not change")
    check(bool(scene.active_mask[1]), "the selected parent slot must remain active as the heir")
    check(int(scene.level[1]) == 1 and int(scene.level[0]) == 0,
          "position criterion must select slot 1, not coefficient-favoured slot 0")
    check_equal(scene.positions()[1], parent_position, "heir position must be exactly preserved")
    check_equal(scene.w_re[1], parent_w_re, "heir real SH coefficients must remain in place")
    check_equal(scene.w_im[1], parent_w_im, "heir imaginary SH coefficients must remain in place")
    check_equal(scene.delta_raw[1], torch.zeros(3), "heir raw offset must be recentered")
    check_close(scene.cell_half[1], torch.tensor([0.5]), "heir movement cell must halve")

    siblings = torch.arange(2, 9)
    check(bool(scene.active_mask[siblings].all()), "seven spare slots must become active siblings")
    check_equal(scene.w_re[siblings], torch.zeros_like(scene.w_re[siblings]),
                "new sibling real coefficients must be zero")
    check_equal(scene.w_im[siblings], torch.zeros_like(scene.w_im[siblings]),
                "new sibling imaginary coefficients must be zero")
    check_equal(scene.delta_raw[siblings], torch.zeros_like(scene.delta_raw[siblings]),
                "new sibling offsets must be zero")

    for name, parameter in (("re", scene.w_re), ("im", scene.w_im)):
        for key in ("exp_avg", "exp_avg_sq"):
            check_equal(moment_row(optimizer, parameter, key, 1), parent_coeff_state[(name, key)],
                        f"heir {name} coefficient {key} must be preserved")
        check_zero_moments(optimizer, parameter, siblings,
                           f"new sibling {name} coefficient state must be cleared")
    check_zero_moments(optimizer, scene.delta_raw, torch.cat([torch.tensor([1]), siblings]),
                       "recentered heir and sibling position state must be cleared")
    check_equal(moment_row(optimizer, scene.w_re, "exp_avg", 0), untouched_state,
                "unselected coefficient state must remain untouched")

    positions_after, weights_after = scene.active_scatterers(theta, phi)
    check_equal(positions_after[:2], positions_before,
                "the two pre-split nonzero scatterer positions must remain identical")
    check_close(weights_after[:2], weights_before,
                "the two pre-split nonzero angular weights must remain identical", atol=0, rtol=0)
    check_equal(weights_after[2:], torch.zeros_like(weights_after[2:]),
                "seven activated siblings must be render-zero")
    check(scene.active_parameter_count() == 99,
          "nine degree-1 complex-SH points with positions must expose 99 active scalars")
    check("eligible 2" in report and "selected 1" in report,
          "split report must expose selection and budget counts")
    check(int(scene.grad_accum_count) == 0 and not bool(scene.grad_accum.any())
          and not bool(scene.pos_raw_grad_accum.any()) and not bool(scene.pos_grad_accum.any())
          and not bool(scene.pos_world_grad_sum.any()) and not bool(scene.pos_world_grad_sq_accum.any()),
          "split must reset the accumulation window")


def configured_split(scores, *, mode, threshold=0.1, count=0,
                     capacity=32, max_active=None, levels=None,
                     criterion="position_world", in_place_heir=True):
    scene = make_scene(4, capacity)
    with torch.no_grad():
        scene.pos_grad_accum[:4] = torch.tensor(scores, dtype=torch.float32)
        scene.pos_raw_grad_accum[:4] = torch.tensor(scores, dtype=torch.float32)
        scene.grad_accum[:4] = torch.tensor(list(reversed(scores)), dtype=torch.float32)
        scene.grad_accum_count.fill_(1)
        if levels is not None:
            scene.level[:4] = torch.tensor(levels)
    result = scene.split(
        threshold_fraction=threshold, criterion=criterion, mode=mode,
        count=count, max_active=max_active, return_report=True,
        in_place_heir=in_place_heir,
    )
    return scene, result


def test_deterministic_selection_and_budgets():
    quantile_scene, (n_split, _, _) = configured_split(
        [5.0, 5.0, 4.0, 3.0], mode="quantile", threshold=0.5, capacity=18)
    check(n_split == 2, "top-half quantile must select exactly ceil(0.5*4)=2")
    check_equal(quantile_scene.level[:4], torch.tensor([1, 1, 0, 0]),
                "equal-score ties must resolve by ascending slot index")

    count_scene, (n_split, n_active, report) = configured_split(
        [9.0, 8.0, 7.0, 6.0], mode="count", count=3,
        capacity=32, max_active=11)
    check(n_split == 1 and n_active == 11,
          "active budget 11 from four parents must permit only one +7 split")
    check(int(count_scene.level[0]) == 1 and int(count_scene.level[1:].sum()) == 7,
          "highest-score parent must win the active-budget cap")
    check("requested 3" in report and "selected 1" in report,
          "report must distinguish requested from resource-limited selection")

    no_space_scene, (n_split, n_active, _) = configured_split(
        [4.0, 3.0, 2.0, 1.0], mode="count", count=2, capacity=10)
    check(n_split == 0 and n_active == 4,
          "fewer than seven free slots must make splitting a no-op")

    level_scene, (n_split, _, _) = configured_split(
        [100.0, 9.0, 8.0, 7.0], mode="count", count=1,
        capacity=11, levels=[2, 0, 0, 0])
    check(n_split == 1 and int(level_scene.level[0]) == 2 and int(level_scene.level[1]) == 1,
          "max-level parents must be excluded before ranking")

    coeff_scene, (n_split, _, _) = configured_split(
        [1.0, 10.0, 2.0, 3.0], mode="count", count=1,
        capacity=11, criterion="coefficient")
    check(n_split == 1 and int(coeff_scene.level[2]) == 1,
          "legacy coefficient criterion must remain available and explicit")

    raw_scene, (n_split, _, _) = configured_split(
        [1.0, 10.0, 2.0, 3.0], mode="count", count=1,
        capacity=11, criterion="position_raw")
    check(n_split == 1 and int(raw_scene.level[1]) == 1,
          "raw-position diagnostic criterion must select its own accumulator")

    snr_scene = make_scene(4, 11)
    with torch.no_grad():
        # Point 0 is bright but has a cancelling physical gradient. Point 1 is
        # dimmer but directionally consistent, so the selector chooses point 1.
        snr_scene.w_re.zero_()
        snr_scene.w_im.zero_()
        snr_scene.w_re[0, 0] = 10.0
        snr_scene.w_re[1, 0] = 1.0
        snr_scene.pos_world_grad_sum[0] = torch.tensor([0.0, 0.0, 0.0])
        snr_scene.pos_world_grad_sq_accum[0] = torch.tensor([2.0, 0.0, 0.0])
        snr_scene.pos_world_grad_sum[1] = torch.tensor([2.0, 0.0, 0.0])
        snr_scene.pos_world_grad_sq_accum[1] = torch.tensor([2.0, 0.0, 0.0])
        snr_scene.grad_accum_count.fill_(2)
    n_split, _, report = snr_scene.split(
        criterion="position_world_snr", mode="count", count=1,
        return_report=True, in_place_heir=True,
    )
    check(n_split == 1 and int(snr_scene.level[1]) == 1,
          "second-moment world selector must favor a consistent nonzero gradient")
    check("position_world_snr/count" in report,
          "normalized position statistic must be explicit in the split report")


def test_random_selector_and_recenter_control():
    scene = make_scene(4, 32)
    optimizer = make_optimizer(scene)
    seed_dense_adam_state(scene, optimizer)
    with torch.no_grad():
        scene.delta_raw[:4] = torch.tensor([
            [0.1, -0.2, 0.3], [-0.3, 0.2, 0.1],
            [0.4, 0.1, -0.2], [-0.1, -0.4, 0.2],
        ])
    theta = torch.tensor([[0.2]])
    phi = torch.tensor([[-0.7]])
    positions_before, weights_before = scene.active_scatterers(theta, phi)
    active_before = scene.active_mask.clone()
    level_before = scene.level.clone()
    half_before = scene.cell_half.clone()
    anchors_before = scene.anchors.clone()
    delta_before = scene.delta_raw.clone()

    n_selected, n_active, report = scene.split(
        criterion="random", mode="count", count=2, random_seed=12345,
        densify=False, optimizer=optimizer, return_report=True,
    )
    positions_after, weights_after = scene.active_scatterers(theta, phi)
    check(n_selected == 2 and n_active == 4,
          "recenter-only must select requested parents without adding capacity")
    check_equal(scene.active_mask, active_before,
                "recenter-only active mask must not change")
    check_equal(scene.level, level_before, "recenter-only levels must not change")
    check_equal(scene.cell_half, half_before,
                "recenter-only cell widths must not change")
    check_close(positions_after, positions_before,
                "recenter-only world positions must be render-preserving", atol=0, rtol=0)
    check_close(weights_after, weights_before,
                "recenter-only weights must be render-preserving", atol=0, rtol=0)
    changed = (scene.anchors[:4] != anchors_before[:4]).any(dim=1)
    check(int(changed.sum()) == 2, "exactly two selected parents must be recentered")
    check_equal(scene.delta_raw[:4][changed], torch.zeros_like(scene.delta_raw[:4][changed]),
                "selected recenter-control offsets must reset")
    check_equal(scene.delta_raw[:4][~changed], delta_before[:4][~changed],
                "unselected offsets must remain unchanged")
    check("recenter_only random/count" in report,
          "report must label the reset-only control explicitly")

    # The event counter is checkpointed: resumed and uninterrupted copies must
    # draw the identical selector on the next event.
    state_after_event = scene.state_dict()
    uninterrupted = make_scene(4, 32)
    uninterrupted.load_state_dict(state_after_event)
    resumed = make_scene(4, 32)
    resumed.load_state_dict(state_after_event)
    out_a = uninterrupted.split(
        criterion="random", mode="count", count=2, random_seed=12345,
        densify=False, return_report=True,
    )
    out_b = resumed.split(
        criterion="random", mode="count", count=2, random_seed=12345,
        densify=False, return_report=True,
    )
    check(out_a == out_b, "resumed random selector must report the same event result")
    check_mapping_equal(uninterrupted.state_dict(), resumed.state_dict(),
                        "resumed random selector")


def test_prune_tombstone_and_recycle():
    scene = make_scene(2, 9, max_degree=0, init_degree=0)
    optimizer = make_optimizer(scene)
    seed_dense_adam_state(scene, optimizer)
    with torch.no_grad():
        scene.w_re.zero_()
        scene.w_im.zero_()
        scene.w_re[0, 0] = 1.0
        scene.w_re[1, 0] = 0.01
        scene.delta_raw[1] = 0.4
    parent_state = moment_row(optimizer, scene.w_re, "exp_avg", 0)

    n_active, _, _ = scene.prune(
        threshold_fraction=0.5, criterion="energy", mode="relmax",
        optimizer=optimizer,
    )
    check(n_active == 1 and not bool(scene.active_mask[1]),
          "small-energy point must be pruned")
    check_equal(scene.anchors[1], torch.zeros(3), "pruned anchor must be tombstoned")
    check_equal(scene.cell_half[1], torch.zeros(1), "pruned half-width must be tombstoned")
    check_equal(scene.delta_raw[1], torch.zeros(3), "pruned offset must be tombstoned")
    for parameter in (scene.w_re, scene.w_im, scene.delta_raw):
        check_zero_moments(optimizer, parameter, torch.tensor([1]),
                           "pruned optimizer row must be cleared")
    check_equal(moment_row(optimizer, scene.w_re, "exp_avg", 0), parent_state,
                "surviving coefficient state must not be touched by prune")

    with torch.no_grad():
        scene.pos_grad_accum[0] = 1.0
        scene.grad_accum_count.fill_(1)
    n_split, n_active = scene.split(
        criterion="position_world", mode="count", count=1, optimizer=optimizer)
    check(n_split == 1 and n_active == 8, "pruned slots must be reusable as eight legacy children")
    check(bool(scene.active_mask[1]), "lowest recycled slot should become a legacy child deterministically")
    for parameter in (scene.w_re, scene.w_im, scene.delta_raw):
        check_zero_moments(optimizer, parameter, torch.arange(1, 9),
                           "recycled legacy children must start without stale row moments")


def check_mapping_equal(left, right, message):
    check(left.keys() == right.keys(), f"{message}: key mismatch")
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, dict):
            check_mapping_equal(a, b, f"{message}.{key}")
        elif isinstance(a, list):
            check(len(a) == len(b), f"{message}.{key}: list length mismatch")
            for i, (aa, bb) in enumerate(zip(a, b)):
                if isinstance(aa, dict):
                    check_mapping_equal(aa, bb, f"{message}.{key}[{i}]")
                else:
                    check(aa == bb, f"{message}.{key}[{i}] mismatch")
        elif torch.is_tensor(a):
            check_equal(a, b, f"{message}.{key}")
        else:
            check(a == b, f"{message}.{key} mismatch: {a!r} != {b!r}")


def test_checkpoint_round_trip():
    scene = make_scene(1, 9, max_degree=0, init_degree=0)
    optimizer = make_optimizer(scene)
    seed_dense_adam_state(scene, optimizer)
    with torch.no_grad():
        scene.pos_grad_accum[0] = 2.0
        scene.grad_accum_count.fill_(1)
    scene.split(criterion="position_world", mode="count", count=1, optimizer=optimizer)

    buffer = io.BytesIO()
    torch.save({"model": scene.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, map_location="cpu", weights_only=False)

    resumed = make_scene(1, 9, max_degree=0, init_degree=0)
    resumed_optimizer = make_optimizer(resumed)
    resumed.load_state_dict(payload["model"])
    resumed_optimizer.load_state_dict(payload["optimizer"])

    check_mapping_equal(scene.state_dict(), resumed.state_dict(), "model round-trip")
    check_mapping_equal(optimizer.state_dict(), resumed_optimizer.state_dict(), "optimizer round-trip")

    # Identical next gradients must produce identical next parameters and state.
    for left, right in zip(
            (scene.w_re, scene.w_im, scene.delta_raw),
            (resumed.w_re, resumed.w_im, resumed.delta_raw)):
        gradient = torch.linspace(-0.25, 0.5, left.numel()).reshape_as(left)
        left.grad = gradient.clone()
        right.grad = gradient.clone()
    optimizer.step()
    resumed_optimizer.step()
    check_mapping_equal(scene.state_dict(), resumed.state_dict(), "post-resume model")
    check_mapping_equal(optimizer.state_dict(), resumed_optimizer.state_dict(), "post-resume optimizer")

    # Known additive buffers load as zero from older checkpoints, while strict
    # loading continues to reject every unrelated mismatch.
    legacy_state = dict(scene.state_dict())
    legacy_state.pop("pos_raw_grad_accum")
    legacy_state.pop("pos_world_grad_sum")
    legacy_state.pop("pos_world_grad_sq_accum")
    legacy_state.pop("split_event_count")
    legacy = make_scene(1, 9, max_degree=0, init_degree=0)
    legacy.load_state_dict(legacy_state, strict=True)
    check_equal(legacy.pos_raw_grad_accum, torch.zeros_like(legacy.pos_raw_grad_accum),
                "legacy checkpoint must initialize the raw-gradient buffer")
    check_equal(legacy.pos_world_grad_sum, torch.zeros_like(legacy.pos_world_grad_sum),
                "legacy checkpoint must initialize the world-gradient sum")
    check_equal(legacy.pos_world_grad_sq_accum,
                torch.zeros_like(legacy.pos_world_grad_sq_accum),
                "legacy checkpoint must initialize the world-gradient second moment")
    check(int(legacy.split_event_count) == 0,
          "legacy checkpoint must initialize the random-selector event counter")


def main():
    torch.manual_seed(7)
    tests = [
        test_world_position_gradient_accumulation,
        test_legacy_eight_child_slot_layout_and_optimizer_state,
        test_legacy_default_parent_order_and_budget,
        test_in_place_heir_and_optimizer_state,
        test_deterministic_selection_and_budgets,
        test_random_selector_and_recenter_control,
        test_prune_tombstone_and_recycle,
        test_checkpoint_round_trip,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS validate_adaptive_split ({len(tests)} blocks)")


if __name__ == "__main__":
    main()
