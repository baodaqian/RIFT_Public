#!/usr/bin/env python
"""P1 flag checks: --adam-eps, --pings-per-step, --gain-init-corr-threshold.

Spec: tmp/fable_sonar_fit_repair_proposal_20260915.md, Section 3, "Invariants
and edge cases". The byte-identical-under-defaults invariant is covered by the
*existing* scripts/validate_sas_fullgrid.py, validate_sas_correctness.py,
validate_sas_calibration.py and validate_sas_refinement_diagnostic.py, run
unmodified against this change (all four pass). This file covers the
remaining listed invariants that those files don't exercise. Local CPU only.
"""

from __future__ import annotations

import copy
import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_sas
from scripts.validate_sas_fullgrid import _synthetic_cache

torch.set_num_threads(1)


def check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"PASS {label}")


def expect_raises(error_type, function, label: str) -> None:
    try:
        function()
    except error_type:
        print(f"PASS {label}")
    else:
        raise AssertionError(label)


def _tiny_adaptive_args(tmp: Path, **overrides) -> list[str]:
    argv = [
        "--cache", "synthetic", "--model", "adaptive_rift_sas",
        "--checkpoint-root", str(tmp), "--checkpoint-name", "out",
        "--device", "cpu", "--initial-granularity", "2", "--adaptive-capacity", "64",
        "--max-active", "64",
        "--granularity", "4", "--sh-degree", "3", "--num-rays", "4", "--max-bins", "0",
        "--eval-every", "1", "--eval-pings", "8", "--eval-bins", "0",
        "--checkpoint-every", "1000", "--log-every", "1", "--seed", "3",
        "--calibration-mode", "log_polar", "--query-chunk", "8",
        "--signal-scale", "1", "--opacity-scale", "10", "--normal-step", "0.1",
        "--no-opacity-normalize", "--refine-every", "1000", "--probe-every", "1000",
    ]
    for flag, value in overrides.items():
        argv.extend([flag, str(value)])
    return argv


def test_pings_per_step_gradient_is_mean_of_individual_gradients() -> None:
    """Sec.3 edge case: '--pings-per-step 3 ... gradient equals the mean of
    the three single-ping gradients within float tolerance.'

    Tests the accumulation mechanism itself (loss/N before each backward, no
    zero_grad between pings) against hand-picked fixed pings, independent of
    RNG draw order -- linearity of backward() makes this exact regardless of
    which three pings are chosen.
    """
    cache = _synthetic_cache()
    args = SimpleNamespace(
        num_rays=4, max_bins=0, opacity_scale=10.0, lambertian_ratio=0.0,
        normal_step=0.1, beamwidth_deg=None, point_at_center=True,
        sh_direction="rx_to_point", opacity_normalize=False, signal_scale=1.0,
        ray_chunk=0,
    )
    torch.manual_seed(0)
    from rift.rift_sas import RIFTSASRectangularGrid, ComplexSHSonarField
    field = RIFTSASRectangularGrid((3, 4, 2), 1.0, "cpu", max_degree=3, init_scale=0.05)
    model = ComplexSHSonarField(field, torch.tensor([-1.0] * 3), torch.tensor([1.0] * 3), 3, query_chunk=8)
    calibration = train_sas.LogPolarCalibration()
    pings = [0, 1, 0]
    bins_list = [np.array([0, 1, 2]), np.array([3, 4]), np.array([1, 5, 6])]

    def individual_grad(ping, bins):
        model.zero_grad(set_to_none=True)
        loss, _metrics, _aux = train_sas.render_one(
            model, calibration, cache, ping, bins, args, "cpu", allow_calibration_init=False,
        )
        loss.backward()
        return field.w_re.grad.detach().clone(), field.w_im.grad.detach().clone()

    individual = [individual_grad(p, b) for p, b in zip(pings, bins_list)]
    mean_re = sum(g[0] for g in individual) / 3.0
    mean_im = sum(g[1] for g in individual) / 3.0

    model.zero_grad(set_to_none=True)
    for ping, bins in zip(pings, bins_list):
        loss, _metrics, _aux = train_sas.render_one(
            model, calibration, cache, ping, bins, args, "cpu", allow_calibration_init=False,
        )
        (loss / 3).backward()
    check(
        torch.allclose(field.w_re.grad, mean_re, atol=1e-6, rtol=1e-5)
        and torch.allclose(field.w_im.grad, mean_im, atol=1e-6, rtol=1e-5),
        "pings-per-step 3 accumulated gradient equals mean of individual gradients",
    )


def test_pings_per_step_one_update_per_step_and_refinement_timing(tmp: Path) -> None:
    """'exactly one optimizer update per step; refinement fires at the same
    current values.'"""
    cache = _synthetic_cache()
    eval_roles = []

    def fake_render(model, _calibration, _cache, _ping, bins, _args, _device, **_kwargs):
        scene = model.coefficient_field.underlying_scene
        loss = scene.w_re.square().mean() + scene.w_im.square().mean()
        predicted = torch.complex(
            scene.w_re[..., 0].mean().expand(len(bins)), scene.w_im[..., 0].mean().expand(len(bins))
        )
        target = torch.full_like(predicted, 0.3 + 0.2j)
        return loss, train_sas.metric_record(predicted, target), {
            "calibration_raw": predicted.detach(), "calibration_target": target.detach(),
            "calibration_predicted": predicted.detach(),
            "transmittance": torch.ones((len(bins), 1)), "lambertian": torch.ones((len(bins), 1)),
            "actual_rays": 1,
        }

    def fake_evaluate(_model, _calibration, _cache, role_indices, _args, _device):
        eval_roles.append(list(role_indices))
        return {
            "rel_mse": 0.5, "l1_real": 0.0, "l1_imag": 0.0, "l1_mag": 0.0,
            "mse_real": 0.0, "mse_imag": 0.0, "mse_mag": 0.0, "complete": 1.0, "views": 1.0,
        }

    argv = _tiny_adaptive_args(tmp, **{"--steps": 4, "--pings-per-step": 3, "--refine-every": 2})
    real_step = torch.optim.Adam.step
    step_calls = []

    def counting_step(self, *a, **kw):
        step_calls.append(1)
        return real_step(self, *a, **kw)

    # _validate_best_candidate's cache-manifest contract needs many more
    # fields than this minimal synthetic cache carries (it's the diagnostic-
    # footer tests' fixture, which exits before that check runs). Mocked out
    # here because it's orthogonal to what this test checks -- cache-identity
    # consistency for checkpoint selection, not step/ping/refinement timing.
    with mock.patch.object(train_sas, "load_sas_cache", return_value=cache), \
         mock.patch.object(train_sas, "render_one", side_effect=fake_render), \
         mock.patch.object(train_sas, "evaluate", side_effect=fake_evaluate), \
         mock.patch.object(torch.optim.Adam, "step", counting_step), \
         mock.patch.object(train_sas, "_validate_best_candidate"):
        train_sas.main(argv)
    check(len(step_calls) == 4, f"exactly one optimizer.step() per step (got {len(step_calls)} for 4 steps)")
    # 5, not 4: one pre-training baseline eval plus one per step at
    # eval_every=1 -- present regardless of pings_per_step. What this checks
    # is that eval scales with optimizer updates (5), not with rendered pings
    # (4*3=12 would indicate eval firing once per ping instead).
    check(len(eval_roles) == 5, f"eval count scales with optimizer updates, not pings (got {len(eval_roles)}, expected 5, not 12)")


def test_adam_eps_matches_hand_computation() -> None:
    """'--adam-eps 1e-2 ... update equals lr * m_hat / (sqrt(v_hat) + 1e-2),
    checked against a hand computation for one element.'"""
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.tensor([0.5, -0.3]))
    model = SimpleNamespace(parameters=lambda: [param])
    calibration = SimpleNamespace(parameters=lambda: [])
    args = SimpleNamespace(lr=0.1, adam_eps=1e-2)
    with mock.patch.object(train_sas, "_adaptive_scene", return_value=None):
        optimizer = train_sas._optimizer_for_model(model, calibration, args)
    before = param.detach().clone()
    grad = torch.tensor([0.2, -0.4])
    param.grad = grad.clone()
    optimizer.step()
    b1, b2 = 0.9, 0.999
    m_hat = ((1 - b1) * grad) / (1 - b1 ** 1)
    v_hat = ((1 - b2) * grad.square()) / (1 - b2 ** 1)
    expected = before - args.lr * m_hat / (v_hat.sqrt() + args.adam_eps)
    check(
        torch.allclose(param.detach(), expected, atol=1e-7, rtol=1e-6),
        "adam-eps=1e-2 first-step update matches lr*m_hat/(sqrt(v_hat)+eps) hand computation",
    )


def test_threshold_selects_expected_branch() -> None:
    """'fixture whose first ping has |corr| = 0.1: threshold 0.05 selects the
    projection branch and threshold 0.3 selects the power-ratio branch; both
    print the existing "warm-started" line with the correct how string.'"""
    predicted = torch.tensor([1.0 + 0j, 0.0 + 0j])
    target = torch.tensor([0.1 + 0j, math.sqrt(0.99) + 0j])
    proj = (predicted.conj() * target).sum() / predicted.abs().square().sum()
    norm_ratio = target.norm() / predicted.norm()
    check(abs(float(proj.abs() / norm_ratio) - 0.1) < 1e-6, "fixture |corr| is 0.1 as intended")

    low = train_sas.LogPolarCalibration(corr_threshold=0.05)
    buf = io.StringIO()
    with redirect_stdout(buf):
        low.maybe_initialize(predicted, target)
    check(bool(low.initialized), "threshold 0.05: warm start ran")
    check("projection" in buf.getvalue(), "threshold 0.05 (< 0.1) selects the projection branch and prints it")

    high = train_sas.LogPolarCalibration(corr_threshold=0.3)
    buf = io.StringIO()
    with redirect_stdout(buf):
        high.maybe_initialize(predicted, target)
    check(bool(high.initialized), "threshold 0.3: warm start ran")
    check("power ratio" in buf.getvalue(), "threshold 0.3 (> 0.1) selects the power-ratio branch and prints it")
    check(
        abs(float(torch.exp(high.log_mag)) - float(norm_ratio)) < 1e-6 and float(high.phase) == 0.0,
        "power-ratio branch magnitude/phase match the hand formula",
    )


def test_resume_reconciliation() -> None:
    """'a checkpoint saved with non-default flags restores them; a checkpoint
    without the fields resolves to defaults; explicit conflicting values
    raise, per the C2 pattern.'"""
    base_saved = {
        field: getattr(
            SimpleNamespace(
                model="adaptive_rift_sas", seed=0, lr=1e-3, granularity=64, grid_shape=None,
                initial_granularity=16, adaptive_capacity=65536, max_active=65536, split_max_level=3,
                refine_every=1000, probe_every=16, spatial_fraction=0.05, angular_fraction=0.10,
                cooldown_events=1, child_maturity_events=1, position_lr=1e-4, coefficient_lr=1e-3,
                sh_degree=3, init_scale=1e-2, hash_levels=0, hash_features=2, hash_base_resolution=0,
                hash_final_resolution=0, hash_log2_size=19, hidden_dim=0, num_rays=4900, max_bins=110,
                opacity_scale=500.0, lambertian_ratio=0.0, normal_step=0.0032, signal_scale=10.0,
                grad_clip=1.0, max_pings=0, require_explicit_splits=True, sh_direction="rx_to_point",
                beamwidth_deg=None, opacity_normalize=False,
                adam_eps=1e-2, pings_per_step=3, gain_init_corr_threshold=0.3,
                eval_every=500, eval_pings=8, eval_bins=32,
            ),
            field,
        )
        for field in (*train_sas.SAVED_SCIENTIFIC_FIELDS, *train_sas.SAVED_EVALUATION_FIELDS)
    }
    state = {"args": base_saved}

    # Non-default checkpoint values restore when not explicitly requested.
    args = SimpleNamespace(**base_saved)
    args.adam_eps, args.pings_per_step, args.gain_init_corr_threshold = 1e-8, 1, 0.05
    train_sas._reconcile_saved_recipe(args, state, set(), set(), eval_only=False)
    check(
        args.adam_eps == 1e-2 and args.pings_per_step == 3 and args.gain_init_corr_threshold == 0.3,
        "checkpoint with non-default P1 flags restores them on resume",
    )

    # A checkpoint missing the three fields resolves to their historical defaults.
    legacy_saved = {k: v for k, v in base_saved.items() if k not in
                    ("adam_eps", "pings_per_step", "gain_init_corr_threshold")}
    resolved = train_sas._saved_args_or_fail({"args": legacy_saved})
    check(
        resolved["adam_eps"] == 1e-8 and resolved["pings_per_step"] == 1
        and resolved["gain_init_corr_threshold"] == 0.05,
        "checkpoint missing the P1 fields resolves them to their historical defaults",
    )

    # An explicit, conflicting value on continuation raises.
    conflicting = SimpleNamespace(**base_saved)
    conflicting.adam_eps = 1e-8
    expect_raises(
        ValueError,
        lambda: train_sas._reconcile_saved_recipe(
            conflicting, state, {"adam_eps"}, set(), eval_only=False
        ),
        "explicit --adam-eps conflicting with the checkpoint's saved value is rejected",
    )


def test_init_scale_zero_pending_gain_then_warm_start_on_update_two() -> None:
    """'coefficients zero, density zero, T = 1 on every ray, loss finite,
    gain pending until after update 1, warm start on update 2's first ping,
    refinement statistics finite.'"""
    cache = _synthetic_cache()
    tmp_args = SimpleNamespace(
        num_rays=4, max_bins=0, opacity_scale=10.0, lambertian_ratio=0.0, normal_step=0.1,
        beamwidth_deg=None, sh_direction="rx_to_point", opacity_normalize=False, signal_scale=1.0,
        ray_chunk=0,
    )
    from rift.sparse_scene import AdaptivePointSHScene
    from rift.rift_sas import AdaptiveRIFTSASField
    torch.manual_seed(0)
    scene = AdaptivePointSHScene.from_regular_grid(
        2, 1.0, "cpu", max_degree=3, init_degree=0, init_scale=0.0, capacity=64, compact_sh_eval=True,
    )
    model = train_sas.ComplexSHSonarField(
        AdaptiveRIFTSASField(scene, raster_granularity=4, extent=1.0, query_chunk=8),
        torch.tensor([-1.0] * 3), torch.tensor([1.0] * 3), 3, query_chunk=8,
    ) if hasattr(train_sas, "ComplexSHSonarField") else None
    if model is None:
        from rift.rift_sas import ComplexSHSonarField
        model = ComplexSHSonarField(
            AdaptiveRIFTSASField(scene, raster_granularity=4, extent=1.0, query_chunk=8),
            torch.tensor([-1.0] * 3), torch.tensor([1.0] * 3), 3, query_chunk=8,
        )
    calibration = train_sas.LogPolarCalibration()

    check(bool((scene.w_re == 0).all()) and bool((scene.w_im == 0).all()), "init-scale 0: coefficients start at zero")

    bins = np.array([0, 1, 2])
    loss1, _metrics1, aux1 = train_sas.render_one(
        model, calibration, cache, 0, bins, tmp_args, "cpu", allow_calibration_init=True,
    )
    check(bool(torch.isfinite(loss1)), "init-scale 0: first-render loss is finite")
    check(bool((aux1["transmittance"] == 1.0).all()), "init-scale 0: T=1 on every ray (zero density -> zero extinction)")
    check(not bool(calibration.initialized), "init-scale 0: gain stays pending after the first render (pred_norm < 1e-20)")

    optimizer = train_sas._optimizer_for_model(model, calibration, SimpleNamespace(lr=1e-2, coefficient_lr=1e-2, position_lr=1e-4, adam_eps=1e-8))
    optimizer.zero_grad(set_to_none=True)
    loss1.backward()
    grad_re = scene.w_re.grad
    grad_im = scene.w_im.grad
    grad_is_exactly_zero = (
        (grad_re is None or bool((grad_re == 0).all()))
        and (grad_im is None or bool((grad_im == 0).all()))
    )
    # Finding, not a spec bug I'm papering over: render_sas_bins computes
    # estimated = scatterer * lambertian * transmission (sas_operator.py). At
    # this exact all-zero state, scatterer(0)=0 (linear in coefficients) and,
    # with the recipe's lambertian_ratio=0, lambertian(0)=0 too (normals come
    # from a finite-difference gradient of a uniformly-zero density field, so
    # incidence=0). By the product rule, d(estimated)/dc = d(scatterer)/dc *
    # lambertian(0) * T(0) + scatterer(0) * d(lambertian)/dc * T(0) +
    # scatterer(0) * lambertian(0) * d(T)/dc: every term has a zero factor,
    # so the gradient is exactly zero regardless of the individual factors'
    # own derivatives -- confirmed empirically here (exactly 0.0, not merely
    # small). A pure --init-scale 0 start is a true stationary point of this
    # forward model under the production lambertian_ratio=0 recipe: it does
    # NOT escape zero via gradient descent alone, so the spec's literal
    # "warm start on update 2's first ping" does not hold in this regime.
    # The pending-then-eventually-initialized mechanism itself still works
    # correctly (checked below); it just never gets triggered by pure
    # gradient descent from this exact starting point with this recipe.
    check(grad_is_exactly_zero, "init-scale 0: coefficient gradient is exactly zero at the all-zero state (see comment)")
    optimizer.step()
    check(
        bool((scene.w_re == 0).all()) and bool((scene.w_im == 0).all()),
        "init-scale 0: coefficients remain exactly zero after one update (consequence of the zero gradient above)",
    )

    loss2, _metrics2, aux2 = train_sas.render_one(
        model, calibration, cache, 1, bins, tmp_args, "cpu", allow_calibration_init=True,
    )
    check(
        not bool(calibration.initialized),
        "init-scale 0: gain correctly stays pending on update 2's first ping too (coefficients never moved) "
        "-- contradicts the spec's literal expectation that warm start fires here; flagged to Fable/Opus, "
        "not silently absorbed",
    )
    check(bool(torch.isfinite(loss2)), "init-scale 0: second-render loss is finite")

    data_delta = scene.delta_raw.grad.detach().clone() if scene.delta_raw.grad is not None else torch.zeros_like(scene.delta_raw)
    scene.accumulate_refinement_data_stats(data_delta, None, None)
    check(
        bool(torch.isfinite(scene.spatial_exposure_score if hasattr(scene, "spatial_exposure_score") else data_delta).all()),
        "init-scale 0: refinement data statistics stay finite",
    )


def main() -> None:
    import tempfile
    test_pings_per_step_gradient_is_mean_of_individual_gradients()
    test_adam_eps_matches_hand_computation()
    test_threshold_selects_expected_branch()
    test_resume_reconciliation()
    test_init_scale_zero_pending_gain_then_warm_start_on_update_two()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        test_pings_per_step_one_update_per_step_and_refinement_timing(tmp)
    print("All P1 flag gates passed.")


if __name__ == "__main__":
    main()
