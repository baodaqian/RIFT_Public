"""CPU regression gates for opt-in adaptive-capacity v2.

Run in a PyTorch environment (the PACE allocated validation is the
authoritative hardware check)::

    PYTHONPATH=. python scripts/validate_adaptive_capacity_v2.py

These are deliberately tiny, data-free gates. They cover invariants which
must survive before a coherent-renderer pilot is allowed to use the new
controller: zero-coefficient densification, immutable scene support,
finite-only evidence, compact SH evaluation, and exact checkpoint recovery.
They do *not* claim a radar-data result.
"""
from __future__ import annotations

import math
import random
import tempfile

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset

from rift.sparse_scene import AdaptivePointSHScene
from train import (
    adaptive_training_contract,
    adaptive_refinement_config,
    capture_rng_state,
    load_tensor_checkpoint,
    load_run_checkpoint,
    optimizer_scheduler_requested_recipe,
    regularization_loss,
    restore_rng_state,
    save_run_checkpoint,
    select_freq_indices,
    set_seed,
    validate_adaptive_resume_config,
)


DEV = "cpu"
torch.manual_seed(17)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def check_close(actual, expected, message, atol=1e-7, rtol=1e-6):
    if actual.dtype == torch.bool or not (actual.is_floating_point() or actual.is_complex()):
        if not torch.equal(actual, expected):
            raise AssertionError(f"{message}; non-floating tensors differ")
        return
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        delta = (actual - expected).detach().abs().max().item()
        raise AssertionError(f"{message}; maximum absolute difference={delta:.3e}")


def expect_raises(exc_type, fn, text):
    try:
        fn()
    except exc_type as exc:
        check(text in str(exc), f"expected {text!r} in {exc!r}")
    else:
        raise AssertionError(f"expected {exc_type.__name__}: {text}")


def make_scene(
    *, capacity=8, max_degree=2, init_degree=0,
    enforce_support_bounds=False, compact_sh_eval=False,
):
    scene = AdaptivePointSHScene.from_regular_grid(
        1,
        1.0,
        DEV,
        max_degree=max_degree,
        init_degree=init_degree,
        capacity=capacity,
        enforce_support_bounds=enforce_support_bounds,
        compact_sh_eval=compact_sh_eval,
    )
    with torch.no_grad():
        scene.w_re[0, 0] = 0.7
        scene.w_im[0, 0] = -0.2
    return scene


def directions():
    # Shapes mirror train.py after its batch squeeze path.
    return (
        torch.tensor([[0.73]], dtype=torch.float32),
        torch.tensor([[1.12]], dtype=torch.float32),
    )


def force_position_split(scene, *, max_level=2, birth_event=None):
    """Densify active row zero through the public lifecycle operation."""
    with torch.no_grad():
        scene.pos_grad_accum.zero_()
        scene.pos_grad_accum[0] = 1.0
        scene.grad_accum_count.fill_(1)
    n_split, n_active, _ = scene.split(
        criterion="position_world",
        mode="count",
        count=1,
        max_level=max_level,
        return_report=True,
        birth_event=birth_event,
        in_place_heir=True,
    )
    check((n_split, n_active) == (1, 8), "one parent must create seven zero siblings")


def continuation_step(scene, optimizer):
    """One synthetic train/probe/refine event after a checkpoint boundary.

    The step deliberately consumes every persisted RNG (Python, NumPy,
    PyTorch global, and train.py's frequency-subset generator), updates both
    coefficients and coordinates, then makes an actual adaptive decision.
    It is not a radar surrogate; it is a deterministic continuation contract.
    """
    py_draw = random.random()
    np_draw = float(np.random.rand())
    torch_draw = torch.rand(2)
    frequencies = select_freq_indices(17, 5, DEV)
    theta = torch.tensor([[0.20 + 0.35 * py_draw]], dtype=torch.float32)
    phi = torch.tensor([[0.40 + 0.20 * np_draw]], dtype=torch.float32)
    scale = 0.01 + 0.001 * (float(frequencies.float().mean()) + float(torch_draw.sum()))

    optimizer.zero_grad(set_to_none=True)
    positions, weights = scene.active_scatterers(theta, phi)
    data_loss = (
        scale * weights.real.sum()
        - 0.5 * scale * weights.imag.sum()
        + 0.17 * (positions[:, 0] - 0.25).square().sum()
    )
    data_loss.backward()
    data_delta = scene.delta_raw.grad.detach().clone()

    # The zero-next-band probe is separate from the optimizer objective, just
    # as in train.py. It supplies the angular evidence only.
    _, probe_weights = scene.active_scatterers(theta, phi, probe_next_band=True)
    probe_loss = 0.73 * probe_weights.real.sum() + 0.31 * probe_weights.imag.sum()
    next_re, next_im = torch.autograd.grad(probe_loss, (scene.w_re, scene.w_im))
    scene.accumulate_refinement_data_stats(data_delta, next_re, next_im)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    snapshot = scene.refinement_snapshot(
        max_level=1,
        min_spatial_exposure=1,
        min_angular_exposure=1,
        spatial_floor=0.0,
        angular_floor=0.0,
        cooldown_events=0,
        child_maturity_events=0,
    )
    selected = (
        scene._select_refinement_indices(snapshot["spatial_score"], snapshot["spatial_eligible"], 1.0),
        scene._select_refinement_indices(snapshot["angular_score"], snapshot["angular_eligible"], 1.0),
    )
    action = scene.apply_refinement_snapshot(
        snapshot,
        spatial_fraction=1.0,
        angular_fraction=1.0,
        max_level=1,
        optimizer=optimizer,
        max_active=8,
    )
    return {
        "draws": (py_draw, np_draw, torch_draw, frequencies),
        "selected": tuple(index.detach().clone() for index in selected),
        "action": action[:3],
        "loss": data_loss.detach().clone(),
    }


def check_optimizer_equal(actual_scene, actual_optimizer, expected_scene, expected_optimizer, message):
    """Exact comparison for the rowwise Adam state relevant to topology."""
    for actual_param, expected_param, label in zip(
        (actual_scene.w_re, actual_scene.w_im, actual_scene.delta_raw),
        (expected_scene.w_re, expected_scene.w_im, expected_scene.delta_raw),
        ("w_re", "w_im", "delta_raw"),
    ):
        actual_state = actual_optimizer.state[actual_param]
        expected_state = expected_optimizer.state[expected_param]
        check(set(actual_state) == set(expected_state), f"{message}: {label} state keys differ")
        for key in actual_state:
            a, b = actual_state[key], expected_state[key]
            if torch.is_tensor(a):
                check_close(a, b, f"{message}: {label}.{key}", atol=0.0, rtol=0.0)
            else:
                check(a == b, f"{message}: {label}.{key} differs")


class ContractDataset(Dataset):
    """Tiny role-identifiable dataset; no tensor items are needed by this gate."""

    def __init__(self, file_paths):
        self.file_paths = list(file_paths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        return torch.tensor(index)


def make_training_contract(optimizer, scheduler, **overrides):
    """Build the complete v2 scientific contract used by resume tests."""
    train_loader = DataLoader(ContractDataset(["synthetic/train_00.csv", "synthetic/train_01.csv"]), batch_size=1)
    val_loader = DataLoader(ContractDataset(["synthetic/val_00.csv"]), batch_size=1)
    settings = {
        "train_loader": train_loader,
        "validation_loader": val_loader,
        "num_freq_selected": 5,
        "loss_mode": "complex", "w1": 1.0, "w2": 2.0,
        "l1_weight": 0.23, "sh_smooth_weight": 0.19,
        "regularizer_normalization": "fixed_initial",
        "regularizer_reference_active_count": 1,
        "phase_sign": 1.0, "forward_operator_name": "brute",
        "compute_dtype": torch.float64, "data_format": "csv",
        "op_kwargs": {"range_model": "sum2"},
        "arr_dist": 0.03, "spacing": 0.004, "num_rx": 2, "num_tx": 3,
        "prune_every": 0, "prune_threshold": 0.01, "prune_criterion": "energy",
        "prune_start_epoch": 0, "prune_mode": "relmax", "prune_target_active": 0,
        "prune_end_epoch": 0, "prune_min_active": 0,
        "grow_every": 0, "grow_threshold": 0.1, "grow_threshold_mode": "relmax",
        "grow_criterion": "grad", "grow_tail_ratio": 0.05,
        "split_every": 0, "split_threshold": 0.1, "split_max_level": 2,
        "step_every": 1, "clip_grad_norm": 0.0, "checkpoint_metric": "train",
        "mag_weight": 0.0, "mag_warmup_epochs": 0,
        "view_weight_alpha": 0.0, "view_weight_max_ratio": 0.0,
        "scene_repr": "point_sh", "gain": None, "occlusion": None,
        "val_cap_axis": None, "num_epochs": 15,
        "optimizer": optimizer, "scheduler": scheduler,
    }
    settings.update(overrides)
    return adaptive_training_contract(**settings)


print("--- v2 probe is render preserving at zero next band")
scene = make_scene()
dtheta, dphi = directions()
_, normal = scene.active_scatterers(dtheta, dphi)
_, probe = scene.active_scatterers(dtheta, dphi, probe_next_band=True)
check(torch.equal(normal, probe), "zero next-band probe changed the coherent prediction")
probe.real.sum().backward()
next_band = scene.basis_degree == 1
check(scene.w_re.grad[0, next_band].abs().sum() > 0, "probe did not expose next-band gradient")
scene.zero_grad(set_to_none=True)


print("--- data-only physical statistic and joint snapshot action")
scene = make_scene()
optimizer = torch.optim.AdamW([
    {"params": [scene.w_re, scene.w_im], "lr": 1e-2},
    {"params": [scene.delta_raw], "lr": 1e-2},
])
# Materialize optimizer rows whose survival/reset matters below.
(scene.w_re.square().sum() + scene.w_im.square().sum() + scene.delta_raw.square().sum()).backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
with torch.no_grad():
    scene.w_re[0, 0] = 0.7
    scene.w_im[0, 0] = -0.2
before_pos, before_weight = scene.active_scatterers(dtheta, dphi)
before_dc_avg = optimizer.state[scene.w_re]["exp_avg"][0, 0].clone()

data_delta = torch.tensor([[0.25, -0.50, 0.75]] + [[0.0, 0.0, 0.0]] * 7)
next_re = torch.zeros_like(scene.w_re)
next_im = torch.zeros_like(scene.w_im)
next_re[0, next_band] = 1.0
scene.accumulate_refinement_data_stats(data_delta, next_re, next_im)
snapshot = scene.refinement_snapshot(
    max_level=1,
    min_spatial_exposure=1,
    min_angular_exposure=1,
    spatial_floor=0.0,
    angular_floor=0.0,
    cooldown_events=0,
)
check(int(snapshot["spatial_eligible"].sum()) == 1, "spatial evidence did not select parent")
check(int(snapshot["angular_eligible"].sum()) == 1, "angular evidence did not select parent")
n_split, n_grown, active_after, report = scene.apply_refinement_snapshot(
    snapshot,
    spatial_fraction=1.0,
    angular_fraction=1.0,
    max_level=1,
    optimizer=optimizer,
)
print("  " + report)
check((n_split, n_grown, active_after) == (1, 1, 8), "joint action changed requested topology")
after_pos, after_weight = scene.active_scatterers(dtheta, dphi)
check(torch.equal(before_pos, after_pos[:1]), "heir changed scatterer location")
check_close(before_weight, after_weight[:1], "heir changed coherent prediction", atol=1e-7, rtol=0.0)
check(after_weight[1:].abs().sum() == 0, "zero siblings changed coherent prediction")
check(scene.order[0].item() == 1 and scene.level[0].item() == 1, "joint action did not unlock/split")
check(torch.equal(optimizer.state[scene.w_re]["exp_avg"][0, 0], before_dc_avg), "heir lost DC Adam state")
check(optimizer.state[scene.delta_raw]["exp_avg"][0].abs().sum() == 0, "recentered position kept Adam state")
check(scene.w_re[1:].abs().sum() == 0 and scene.w_im[1:].abs().sum() == 0, "new siblings are not zero")


print("--- fixed-initial regularizer is invariant to zero siblings")
scene = make_scene(max_degree=1, init_degree=1)
with torch.no_grad():
    # Exercise both the group-L1 and l(l+1) SH terms.
    scene.w_re[0, :] = torch.tensor([0.7, -0.3, 0.2, 0.1])
    scene.w_im[0, :] = torch.tensor([-0.2, 0.4, -0.5, 0.6])


def fixed_prior_and_grad():
    scene.zero_grad(set_to_none=True)
    loss, terms = regularization_loss(
        scene,
        l1_weight=0.23,
        sh_smooth_weight=0.19,
        normalization="fixed_initial",
        reference_active_count=1,
        return_terms=True,
    )
    check(loss is not None and set(terms) == {"l1", "sh_degree"}, "expected both regularizer terms")
    loss.backward()
    return loss.detach().clone(), scene.w_re.grad[0].detach().clone(), scene.w_im.grad[0].detach().clone()


before_prior, before_re_grad, before_im_grad = fixed_prior_and_grad()
force_position_split(scene, max_level=1)
after_prior, after_re_grad, after_im_grad = fixed_prior_and_grad()
# fixed_initial explicitly removes the zero-vector epsilon contribution, so
# both the objective and retained-heir derivatives must be bit-identical.
check_close(after_prior, before_prior, "zero siblings diluted fixed-initial prior", atol=0.0, rtol=0.0)
check_close(after_re_grad, before_re_grad, "zero siblings changed fixed-initial real gradient", atol=0.0, rtol=0.0)
check_close(after_im_grad, before_im_grad, "zero siblings changed fixed-initial imaginary gradient", atol=0.0, rtol=0.0)


print("--- immutable support bounds contain cells and later motion")
scene = make_scene(enforce_support_bounds=True)
with torch.no_grad():
    # Original one-cell support is [-1, 1]^3. Make the live location close
    # to its +x face, then split; the child cell must not grow past that face.
    scene.delta_raw[0, 0] = torch.atanh(torch.tensor(0.9))
before_pos = scene.positions()[0].detach().clone()
force_position_split(scene, max_level=1)
after_pos = scene.positions()[0].detach().clone()
check_close(after_pos, before_pos, "support refinement moved immediate heir", atol=1e-7, rtol=0.0)
active = scene.active_mask
cell_min = scene.anchors[active] - scene.cell_half[active]
cell_max = scene.anchors[active] + scene.cell_half[active]
check(bool((cell_min >= scene.support_min - 1e-6).all()), "active child cell escaped lower immutable support")
check(bool((cell_max <= scene.support_max + 1e-6).all()), "active child cell escaped upper immutable support")
# A near-boundary heir may need a small radius, but its seven interior
# siblings must retain their own usable radius rather than inheriting it.
check(bool((scene.cell_half[1:, 0] > 0).all()), "boundary-near split collapsed an interior sibling")
check(scene.cell_half[1:, 0].max().item() > scene.cell_half[0, 0].item(),
      "siblings reused the heir boundary radius")
scene.zero_grad(set_to_none=True)
scene.positions()[1:, 0].sum().backward()
check(scene.delta_raw.grad[1:].abs().sum() > 0,
      "usable boundary-near siblings lost their position gradients")
with torch.no_grad():
    # An optimizer step can subsequently drive raw offsets toward the former
    # exterior; rendered locations must remain in the frozen original domain.
    scene.delta_raw[0, 0] = 12.0
    scene.delta_raw[1, 0] = -12.0
positions = scene.positions()[scene.active_mask]
check(bool((positions >= scene.support_min - 1e-6).all()), "post-split motion escaped lower support")
check(bool((positions <= scene.support_max + 1e-6).all()), "post-split motion escaped upper support")

# At a float32-saturated face the retained heir legitimately has zero radius;
# independently-contained siblings must still be alive and movable.
scene = make_scene(enforce_support_bounds=True)
with torch.no_grad():
    scene.delta_raw[0, 0] = 12.0
force_position_split(scene, max_level=1)
check(bool((scene.cell_half[1:, 0] > 0).all()), "face-saturated heir collapsed sibling cells")
scene.zero_grad(set_to_none=True)
scene.positions()[1:, 0].sum().backward()
check(scene.delta_raw.grad[1:].abs().sum() > 0,
      "face-saturated split left no movable sibling")


print("--- non-finite/masked evidence cannot poison a refinement window")
scene = make_scene()
data_delta = torch.zeros_like(scene.delta_raw)
data_delta[0] = torch.tensor([1.0, -2.0, 3.0])
data_delta[1] = torch.tensor([float("nan"), float("inf"), -float("inf")])  # inactive
next_re = torch.zeros_like(scene.w_re)
next_im = torch.zeros_like(scene.w_im)
next_re[0, next_band] = 1.0
next_re[1, next_band] = float("nan")  # inactive and masked from the controller
next_im[1, next_band] = float("inf")
scene.accumulate_refinement_data_stats(data_delta, next_re, next_im)
for name in ("refine_spatial_sum", "refine_spatial_exposure", "refine_angular_sum", "refine_angular_exposure"):
    check(bool(torch.isfinite(getattr(scene, name)).all()), f"{name} was poisoned by inactive NaN/Inf")
check(scene.refine_spatial_exposure[0].item() == 1 and scene.refine_spatial_exposure[1].item() == 0,
      "inactive data-gradient row gained spatial exposure")
check(scene.refine_angular_exposure[0].item() == 1 and scene.refine_angular_exposure[1].item() == 0,
      "inactive next-band row gained angular exposure")
snapshot = scene.refinement_snapshot(1, 1, 1)
check(bool(torch.isfinite(snapshot["spatial_score"]).all()), "spatial score became non-finite")
check(bool(torch.isfinite(snapshot["angular_score"]).all()), "angular score became non-finite")

scene = make_scene()
data_delta = torch.zeros_like(scene.delta_raw)
next_re = torch.zeros_like(scene.w_re)
next_im = torch.zeros_like(scene.w_im)
next_re[0, next_band] = float("nan")  # active target band: exclude the bad observation
scene.accumulate_refinement_data_stats(data_delta, next_re, next_im)
check(scene.refine_angular_exposure[0].item() == 0, "non-finite active probe was treated as evidence")
check(bool(torch.isfinite(scene.refine_angular_sum).all()), "active NaN probe poisoned score accumulator")
check(not scene.refinement_snapshot(1, 1, 1)["angular_eligible"][0], "bad probe became an unlock candidate")

scene = make_scene()
data_delta = torch.zeros_like(scene.delta_raw)
data_delta[0] = 1e20  # finite inputs whose float32 score norm overflows
next_re = torch.zeros_like(scene.w_re)
next_im = torch.zeros_like(scene.w_im)
next_re[0, next_band] = 1e20
next_im[0, next_band] = 1e20
scene.accumulate_refinement_data_stats(data_delta, next_re, next_im)
for name in ("refine_spatial_sum", "refine_spatial_exposure", "refine_angular_sum", "refine_angular_exposure"):
    check(bool(torch.isfinite(getattr(scene, name)).all()), f"{name} was poisoned by finite overflow")
check(scene.refine_spatial_exposure[0].item() == 0,
      "overflowing spatial score was counted as refinement evidence")
check(scene.refine_angular_exposure[0].item() == 0,
      "overflowing angular score was counted as refinement evidence")


print("--- max-active ceiling rejects an already-overfull topology")
scene = AdaptivePointSHScene(
    torch.tensor([[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]),
    0.25,
    DEV,
    max_degree=1,
    capacity=16,
)
snapshot = scene.refinement_snapshot(1, 1, 1)
expect_raises(
    ValueError,
    lambda: scene.apply_refinement_snapshot(
        snapshot, spatial_fraction=0.0, angular_fraction=0.0, max_level=1, max_active=1),
    "below the current active count",
)


print("--- child maturity blocks an immediate cascade, then permits fresh evidence")
scene = make_scene(capacity=64)
scene.accumulate_refinement_data_stats(torch.ones_like(scene.delta_raw))
snapshot = scene.refinement_snapshot(2, 1, 99, child_maturity_events=1)
n_split, _, _, _ = scene.apply_refinement_snapshot(
    snapshot, spatial_fraction=1.0, angular_fraction=0.0, max_level=2,
)
check(n_split == 1, "setup event failed to create children")
children = torch.arange(1, 8)
scene.accumulate_refinement_data_stats(torch.ones_like(scene.delta_raw))
snapshot = scene.refinement_snapshot(2, 1, 99, child_maturity_events=1)
check(not bool(snapshot["spatial_eligible"][children].any()), "new children split without a completed maturity interval")
# Consume the blocked event without changing topology. It advances the birth
# clock but resets the evidence window, exactly as a real scheduled event does.
scene.apply_refinement_snapshot(snapshot, spatial_fraction=0.0, angular_fraction=0.0, max_level=2)
scene.accumulate_refinement_data_stats(torch.ones_like(scene.delta_raw))
snapshot = scene.refinement_snapshot(2, 1, 99, child_maturity_events=1)
check(bool(snapshot["spatial_eligible"][children].all()), "mature children with fresh evidence stayed blocked")


print("--- compact SH evaluation matches legacy output, gradients, and probe")
legacy = make_scene(max_degree=3, init_degree=1, compact_sh_eval=False)
with torch.no_grad():
    torch.manual_seed(81)
    legacy.w_re.copy_(torch.randn_like(legacy.w_re))
    legacy.w_im.copy_(torch.randn_like(legacy.w_im))
    # All active points have order one, but deliberately retain nonzero locked
    # bands so the mask, not accidental zeros, establishes equivalence.
    legacy.order[0] = 1
compact = make_scene(max_degree=3, init_degree=1, compact_sh_eval=True)
compact_state = legacy.state_dict()
compact_state["compact_sh_eval_enabled"] = torch.tensor(True)
compact.load_state_dict(compact_state)
check(compact._compact_eval_degree == 1, "compact cap did not refresh on checkpoint load")


def render_loss(model, *, probe_next_band):
    model.zero_grad(set_to_none=True)
    pos, weights = model.active_scatterers(dtheta, dphi, probe_next_band=probe_next_band)
    loss = weights.real.square().sum() + weights.imag.square().sum() + 0.07 * pos.square().sum()
    loss.backward()
    return (
        pos.detach(),
        weights.detach(),
        model.w_re.grad.detach().clone(),
        model.w_im.grad.detach().clone(),
        model.delta_raw.grad.detach().clone(),
    )


for probe_next_band in (False, True):
    old = render_loss(legacy, probe_next_band=probe_next_band)
    new = render_loss(compact, probe_next_band=probe_next_band)
    labels = ("positions", "weights", "real SH gradients", "imaginary SH gradients", "position gradients")
    for label, actual, expected in zip(labels, new, old):
        check_close(actual, expected, f"compact {label} differs from legacy (probe={probe_next_band})")

compact.unlock_next_bands(torch.tensor([0]))
check(compact._compact_eval_degree == 2, "compact cap did not refresh after SH unlock")
with torch.no_grad():
    compact.order[0] = 3
compact.refresh_compact_sh_eval_cap()
check(compact._compact_eval_degree == 3, "explicit compact cap refresh missed externally edited order")


print("--- checkpoint/RNG/config recovery is trajectory-safe")
expected_config = adaptive_refinement_config(
    enabled=True,
    refine_every=2,
    probe_every=3,
    min_spatial_exposure=4,
    min_angular_exposure=5,
    spatial_fraction=0.25,
    angular_fraction=0.5,
    spatial_floor=1e-5,
    angular_floor=2e-5,
    cooldown_events=1,
    child_maturity_events=1,
    max_active=8,
    split_max_level=2,
    regularizer_normalization="fixed_initial",
    regularizer_reference_active_count=1,
)
config_scene = make_scene(enforce_support_bounds=True, compact_sh_eval=True)
config_optimizer = torch.optim.AdamW(config_scene.parameters(), lr=1e-3)
config_scheduler = CosineAnnealingWarmRestarts(config_optimizer, T_0=2)
expected_training_contract = make_training_contract(config_optimizer, config_scheduler)
config_checkpoint = {
    "adaptive_capacity_v2": True,
    "adaptive_refinement": expected_config,
    "adaptive_training_contract": expected_training_contract,
    "model_state_dict": config_scene.state_dict(),
    "rng_state": capture_rng_state(),
}
validate_adaptive_resume_config(
    config_checkpoint, expected_config,
    expected_training_contract=expected_training_contract)
incomplete_scene_state = dict(config_scene.state_dict())
del incomplete_scene_state["support_min"]
expect_raises(
    ValueError,
    lambda: validate_adaptive_resume_config(
        {
            "adaptive_capacity_v2": True,
            "adaptive_refinement": expected_config,
            "adaptive_training_contract": expected_training_contract,
            "model_state_dict": incomplete_scene_state,
            "rng_state": capture_rng_state(),
        },
        expected_config,
        expected_training_contract=expected_training_contract,
    ),
    "support/compact/maturity scene state",
)
bad_config = dict(expected_config)
bad_config["probe_every_views"] += 1
expect_raises(
    ValueError,
    lambda: validate_adaptive_resume_config(
        config_checkpoint, bad_config,
        expected_training_contract=expected_training_contract),
    "trajectory-defining setting",
)
expect_raises(
    ValueError,
    lambda: validate_adaptive_resume_config(
        {"adaptive_capacity_v2": False}, expected_config,
        expected_training_contract=expected_training_contract),
    "disagree",
)
alternate_optimizer = torch.optim.AdamW(config_scene.parameters(), lr=2e-3)
alternate_scheduler = CosineAnnealingWarmRestarts(alternate_optimizer, T_0=3)
for label, changed_contract in (
    ("L1", make_training_contract(config_optimizer, config_scheduler, l1_weight=0.24)),
    ("prune cadence", make_training_contract(config_optimizer, config_scheduler, prune_every=1)),
    ("effective default prune end", make_training_contract(config_optimizer, config_scheduler, num_epochs=16)),
    ("phase sign", make_training_contract(config_optimizer, config_scheduler, phase_sign=-1.0)),
    ("train split", make_training_contract(
        config_optimizer, config_scheduler,
        train_loader=DataLoader(ContractDataset(["synthetic/train_changed.csv"]), batch_size=1))),
    ("optimizer LR", make_training_contract(alternate_optimizer, alternate_scheduler)),
    ("scheduler T0", make_training_contract(
        config_optimizer, CosineAnnealingWarmRestarts(config_optimizer, T_0=3))),
):
    expect_raises(
        ValueError,
        lambda changed_contract=changed_contract: validate_adaptive_resume_config(
            config_checkpoint, expected_config,
            expected_training_contract=changed_contract),
        "loss, physics, observation split",
    )
disabled_scene_state = dict(config_scene.state_dict())
disabled_scene_state["support_bounds_enabled"] = torch.tensor(False)
expect_raises(
    ValueError,
    lambda: validate_adaptive_resume_config(
        {
            **config_checkpoint,
            "model_state_dict": disabled_scene_state,
        },
        expected_config,
        expected_training_contract=expected_training_contract,
    ),
    "disables immutable support",
)
legacy_rng_checkpoint = {
    **config_checkpoint,
    "rng_state": {"version": 1, "numpy": ("unsafe",)},
}
expect_raises(
    ValueError,
    lambda: validate_adaptive_resume_config(
        legacy_rng_checkpoint, expected_config,
        expected_training_contract=expected_training_contract),
    "safe RNG payload version 2",
)
expect_raises(ValueError, lambda: restore_rng_state({}, require_complete=True), "complete RNG payload")


print("--- normalized scene-scale recovery keeps the requested recipe strict")
# A fresh --normalize-scene-scale run starts its scene group at base_lr * a,
# whereas an unchanged --resume deliberately starts from base_lr and lets the
# checkpoint restore that effective LR.  The strict contract must compare the
# same user-facing base LR in both cases, then still reject a changed base LR
# or a changed normalization flag.
base_scene_lr = 1e-3
gauge_scale = 3.25
effective_scene_lr = base_scene_lr * gauge_scale
gauge_scene = make_scene(enforce_support_bounds=True, compact_sh_eval=True)
gauge_optimizer = torch.optim.AdamW(gauge_scene.parameters(), lr=effective_scene_lr)
gauge_scheduler = CosineAnnealingWarmRestarts(gauge_optimizer, T_0=2)
gauge_recipe = optimizer_scheduler_requested_recipe(
    gauge_optimizer,
    gauge_scheduler,
    requested_scene_lr=base_scene_lr,
    normalize_scene_scale=True,
)
gauge_contract = make_training_contract(
    gauge_optimizer,
    gauge_scheduler,
    optimizer_requested_recipe=gauge_recipe,
)
resume_scene = make_scene(enforce_support_bounds=True, compact_sh_eval=True)
resume_optimizer = torch.optim.AdamW(resume_scene.parameters(), lr=base_scene_lr)
resume_scheduler = CosineAnnealingWarmRestarts(resume_optimizer, T_0=2)
resume_recipe = optimizer_scheduler_requested_recipe(
    resume_optimizer,
    resume_scheduler,
    requested_scene_lr=base_scene_lr,
    normalize_scene_scale=True,
)
resume_contract = make_training_contract(
    resume_optimizer,
    resume_scheduler,
    optimizer_requested_recipe=resume_recipe,
)
check(gauge_contract == resume_contract,
      "unchanged normalized resume did not canonicalize its pre-gauge scene LR")
gauge_checkpoint = {
    "adaptive_capacity_v2": True,
    "adaptive_refinement": expected_config,
    "adaptive_training_contract": gauge_contract,
    "model_state_dict": gauge_scene.state_dict(),
    "rng_state": capture_rng_state(),
}
validate_adaptive_resume_config(
    gauge_checkpoint,
    expected_config,
    expected_training_contract=resume_contract,
)
changed_lr_recipe = optimizer_scheduler_requested_recipe(
    resume_optimizer,
    resume_scheduler,
    requested_scene_lr=2 * base_scene_lr,
    normalize_scene_scale=True,
)
expect_raises(
    ValueError,
    lambda: validate_adaptive_resume_config(
        gauge_checkpoint,
        expected_config,
        expected_training_contract=make_training_contract(
            resume_optimizer,
            resume_scheduler,
            optimizer_requested_recipe=changed_lr_recipe,
        ),
    ),
    "loss, physics, observation split",
)
ungauged_recipe = optimizer_scheduler_requested_recipe(
    resume_optimizer,
    resume_scheduler,
    requested_scene_lr=base_scene_lr,
    normalize_scene_scale=False,
)
expect_raises(
    ValueError,
    lambda: validate_adaptive_resume_config(
        gauge_checkpoint,
        expected_config,
        expected_training_contract=make_training_contract(
            resume_optimizer,
            resume_scheduler,
            optimizer_requested_recipe=ungauged_recipe,
        ),
    ),
    "loss, physics, observation split",
)
with tempfile.TemporaryDirectory() as tmp:
    gauge_checkpoint_path = f"{tmp}/adaptive_v2_gauge.pth.tar"
    save_run_checkpoint(
        gauge_checkpoint_path,
        {
            "epoch": 2,
            "loss": 0.125,
            "adaptive_capacity_v2": True,
            "adaptive_refinement": expected_config,
            "adaptive_training_contract": gauge_contract,
        },
        gauge_scene,
        gauge_optimizer,
        gauge_scheduler,
    )
    recovered_scene = make_scene(enforce_support_bounds=True, compact_sh_eval=True)
    recovered_optimizer = torch.optim.AdamW(recovered_scene.parameters(), lr=base_scene_lr)
    recovered_scheduler = CosineAnnealingWarmRestarts(recovered_optimizer, T_0=2)
    epoch, loss = load_run_checkpoint(
        gauge_checkpoint_path,
        recovered_scene,
        recovered_optimizer,
        recovered_scheduler,
        gain=None,
        device=DEV,
        expected_adaptive_config=expected_config,
        expected_training_contract=resume_contract,
        require_rng_state=True,
    )
    check((epoch, loss) == (2, 0.125), "normalized checkpoint epoch/loss did not round-trip")
    check(math.isclose(recovered_optimizer.param_groups[0]["lr"], effective_scene_lr,
                       rel_tol=0.0, abs_tol=0.0),
          "resume did not restore the normalized effective scene LR")
    check(math.isclose(recovered_scheduler.base_lrs[0], effective_scene_lr,
                       rel_tol=0.0, abs_tol=0.0),
          "resume did not restore the normalized scheduler base LR")


set_seed(20260904)
scene = make_scene(enforce_support_bounds=True, compact_sh_eval=True)
optimizer = torch.optim.AdamW(scene.parameters(), lr=1e-3)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2)
# Give the optimizer a nontrivial row state to make checkpoint restoration
# more than a model-only load.
(scene.w_re.square().sum() + scene.w_im.square().sum()).backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
with tempfile.TemporaryDirectory() as tmp:
    checkpoint_path = f"{tmp}/adaptive_v2.pth.tar"
    save_run_checkpoint(
        checkpoint_path,
        {
            "epoch": 3,
            "loss": 0.25,
            "adaptive_capacity_v2": True,
            "adaptive_refinement": expected_config,
            "adaptive_training_contract": expected_training_contract,
        },
        scene,
        optimizer,
        scheduler,
    )
    # A v2 resume must not silently replace the adaptive trajectory with fresh
    # Adam state. Test the strict rejection separately before the valid load.
    checkpoint_payload = load_tensor_checkpoint(checkpoint_path, map_location=DEV)
    check(checkpoint_payload["rng_state"]["version"] == 2,
          "checkpoint did not use the safe RNG schema")
    checkpoint_payload["optimizer_state_dict"] = None
    missing_optimizer_path = f"{tmp}/adaptive_v2_missing_optimizer.pth.tar"
    torch.save(checkpoint_payload, missing_optimizer_path)
    missing_scene = make_scene(enforce_support_bounds=True, compact_sh_eval=True)
    missing_optimizer = torch.optim.AdamW(missing_scene.parameters(), lr=1e-3)
    missing_scheduler = CosineAnnealingWarmRestarts(missing_optimizer, T_0=2)
    expect_raises(
        ValueError,
        lambda: load_run_checkpoint(
            missing_optimizer_path,
            missing_scene,
            missing_optimizer,
            missing_scheduler,
            gain=None,
            device=DEV,
            expected_adaptive_config=expected_config,
            expected_training_contract=expected_training_contract,
            require_rng_state=True,
        ),
        "requires optimizer state",
    )
    # Constructing the intentionally-invalid fixture above allocates a fresh
    # scene and consumes global Torch draws.  It must not move the reference
    # continuation away from the checkpoint boundary being tested.  Rewind
    # explicitly before taking the uninterrupted branch; the valid load below
    # performs the same rewind for the recovered branch.
    restore_rng_state(checkpoint_payload["rng_state"], require_complete=True)
    # Continue the original process first. Loading the restoration below must
    # rewind all four RNGs to this same recovery boundary before it performs
    # the very same synthetic train/probe/refine event.
    uninterrupted = continuation_step(scene, optimizer)

    restored = make_scene(enforce_support_bounds=True, compact_sh_eval=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = CosineAnnealingWarmRestarts(restored_optimizer, T_0=2)
    epoch, loss = load_run_checkpoint(
        checkpoint_path,
        restored,
        restored_optimizer,
        restored_scheduler,
        gain=None,
        device=DEV,
        expected_adaptive_config=expected_config,
        expected_training_contract=expected_training_contract,
        require_rng_state=True,
    )
    check((epoch, loss) == (3, 0.25), "checkpoint epoch/loss did not round-trip")
    for key, value in checkpoint_payload["model_state_dict"].items():
        check_close(restored.state_dict()[key], value, f"checkpoint model key {key} did not round-trip", atol=0.0, rtol=0.0)
    state = restored_optimizer.state[restored.w_re]
    check("exp_avg" in state and state["exp_avg"].abs().sum() > 0, "checkpoint Adam state did not restore")
    resumed = continuation_step(restored, restored_optimizer)
    check(math.isclose(resumed["draws"][0], uninterrupted["draws"][0], rel_tol=0.0, abs_tol=0.0),
          "Python RNG did not reproduce the post-checkpoint continuation")
    check(math.isclose(resumed["draws"][1], uninterrupted["draws"][1], rel_tol=0.0, abs_tol=0.0),
          "NumPy RNG did not reproduce the post-checkpoint continuation")
    check_close(resumed["draws"][2], uninterrupted["draws"][2], "PyTorch CPU RNG continuation diverged", atol=0.0, rtol=0.0)
    check_close(resumed["draws"][3], uninterrupted["draws"][3], "frequency-subset continuation diverged", atol=0.0, rtol=0.0)
    check_close(resumed["loss"], uninterrupted["loss"], "post-checkpoint data loss diverged", atol=0.0, rtol=0.0)
    check(resumed["action"] == uninterrupted["action"], "post-checkpoint topology action diverged")
    for actual_index, expected_index in zip(resumed["selected"], uninterrupted["selected"]):
        check_close(actual_index, expected_index, "post-checkpoint refinement selection diverged", atol=0.0, rtol=0.0)
    for key, value in scene.state_dict().items():
        check_close(restored.state_dict()[key], value, f"post-checkpoint model key {key} diverged", atol=0.0, rtol=0.0)
    check_optimizer_equal(restored, restored_optimizer, scene, optimizer, "post-checkpoint Adam state diverged")


print("ALL ADAPTIVE-CAPACITY V2 GATES PASS")
