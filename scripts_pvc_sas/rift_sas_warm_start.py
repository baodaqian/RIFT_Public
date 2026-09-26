#!/usr/bin/env python
"""CGLS warm start of adaptive RIFT-SAS as a step-0 trainer checkpoint (docs/RIFT_SAS_Train.md, A17/B13/A20).

Usage: rift_sas_warm_start.py [options] -- <the exact train_sas_pvc.py argv of the run to be started>

The trainer arguments are resolved exactly as ``train_sas_pvc.main`` resolves a
fresh run (PVC parser twin, ``--profile full`` defaults, explicit splits,
log-polar calibration), and the model is built by the unchanged
``train_sas.build_model``. Its initialization is then replaced:

1. **Fit.** The degree-0 coefficients of the dense adaptive field (same anchors,
   ``--initial-granularity`` points on the ``--granularity`` raster, positions
   frozen) are fitted by CGLS through the shared shell renderer linearized as in
   ``rift_sas_capacity_check.py`` (``lambertian_ratio 1``, ``opacity_scale 0``,
   normals-free, production rays/beam/SH direction/signal scale). This is an
   initializer only (B13). Fit set: TRAIN pings at every ``--fit-stride``-th
   azimuth, excluding the 64 pre-registered TRAIN read-out pings; VAL and TEST
   are never fitted. Per iteration the held-out TRAIN64/VAL64 rel-MSE (linearized
   operator, g = 1) is printed, full band and 12.5-27.5 kHz.
2. **Scale.** The fitted field is multiplied by s, chosen by bisection so that
   the minimum ``T_all_min`` over ``--scale-pings`` TRAIN pings under the
   production renderer (the run's own ``lambertian_ratio``/``opacity_scale``)
   equals ``--target-t-min`` (B13: >= 0.8). The Lambertian factor is scale-free;
   only transmittance depends on s.
3. **Gain.** The log-polar calibration is set to the best global complex gain
   fitted on ``--gain-pings`` TRAIN pings of the fit set under the production
   renderer, and marked initialized (the trainer's warm start then leaves it).
4. **Learning rate.** ``--coefficient-lr`` is set to ``--lr-fraction`` x the median
   |DC| of the scaled field, rounded to one significant digit (B13); the value to
   pass on the resume command line is printed (``CONTINUE_COEFFICIENT_LR``) and
   stored, because the trainer refuses a mismatched explicit recipe on resume.
5. **Checkpoint.** A step-0 checkpoint is written with the trainer's own
   ``checkpoint(...)`` (PVC twin): fresh optimizer state, ``default_rng(seed)``,
   empty history, best = inf, to ``<checkpoint-root>/<checkpoint-name>/checkpoint_latest.pt``
   (refused if one exists), plus ``warm_start.json`` provenance. The run then
   starts with the ordinary ``--resume`` of that file; no trainer code changes.
Step-0 production-renderer diagnostics (``T_all_min``, ``lambert_positive``,
raw correlation) are printed for the scale pings.

Chained CGLS legs (A29). After every iteration the CGLS state (x, residual,
direction, gamma, iteration, history, fit set) is saved atomically to
``<run>/cgls_state.pt``, so a wall-time kill loses at most the running iteration.
A later leg writes a new step-0 run (steps 2-5 as above) from a continued fit:
  * ``--continue-state <cgls_state.pt>``: exact continuation of that CGLS; the
    saved residual is checked against b - A x before the first iteration;
  * ``--restart-from <step-0 checkpoint>``: for a warm start that predates the
    state file (W1). x0 is the checkpoint's DC field divided by the scale in
    ``warm_start.json`` beside it. The residual is recomputed and CGLS restarts
    (direction = A^H r), which loses the old conjugate direction once. The fit at
    x0 must reproduce the source's last recorded fit within ``--restart-tolerance``.
Either way the fit set, and the trainer fields the linearized operator depends on,
must equal the source's, or the leg refuses. Iterations are numbered on from the
source. ``--iterations`` counts this leg's iterations. The time budget covers the
restart pass too, and a new iteration is started only if the previous one's
duration still fits in the budget.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_sas as _sas  # noqa: E402
from rift.sas_dataset import load_sas_cache  # noqa: E402
from rift_pvc_sas import training as twins  # noqa: E402
from scripts_pvc_sas.rift_sas_capacity_check import LinearShellOperator, dot, spectra  # noqa: E402
from scripts_pvc_sas.rift_sas_references import band_masks  # noqa: E402

IN_BAND = "12.5..27.5"
# trainer fields the linearized operator and the fit set depend on; a chained leg must match its source on all
FIT_FIELDS = ("cache", "granularity", "initial_granularity", "num_rays", "beamwidth_deg", "sh_direction",
              "signal_scale", "seed")
FIT_OPTIONS = ("fit_stride", "max_fit_pings", "readout_pings")


def resolve_trainer_args(raw_argv):
    """Mirror ``train_sas_pvc.main`` for a fresh run (no checkpoint)."""
    explicit = _sas._explicit_cli_fields(raw_argv)
    args = _sas.parse_args(raw_argv)
    _sas._apply_profile(args, explicit)
    if args.opacity_normalize is None:
        args.opacity_normalize = False
    if args.model != "adaptive_rift_sas":
        raise ValueError("the warm start is defined for --model adaptive_rift_sas")
    if args.eval_only or args.resume:
        raise ValueError("pass the argv of a fresh run (no --resume/--eval-only)")
    args.require_explicit_splits = True
    args.calibration_mode = _sas.resolve_calibration_mode(args.model, args.calibration_mode, None)
    return args


def rel(target, prediction, masks, band):
    y, p = target, prediction
    if band != "full":
        y, p = y[:, masks[band]], p[:, masks[band]]
    return float(np.sum(np.abs(p - y) ** 2) / np.sum(np.abs(y) ** 2))


@torch.no_grad()
def production_pass(model, calibration, cache, pings, args, device):
    """Raw predictions, targets and diagnostics under the run's own renderer (calibration untouched)."""
    bins = np.arange(cache.num_bins)
    raws, targets, t_min, lambert = [], [], [], []
    for ping in pings:
        _loss, _metrics, aux = _sas.render_one(model, calibration, cache, int(ping), bins, args, device)
        raws.append(aux["calibration_raw"].to(torch.complex128).cpu().numpy())
        targets.append(aux["calibration_target"].to(torch.complex128).cpu().numpy())
        t_min.append(float(aux["transmittance"].min()))
        lambert.append(float((aux["lambertian"] > 0).float().mean()))
    return np.asarray(raws), np.asarray(targets), np.asarray(t_min), np.asarray(lambert)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: rift_sas_warm_start.py [options] -- <train_sas_pvc.py argv>")
    split = argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-stride", type=int, default=8)
    parser.add_argument("--max-fit-pings", type=int, default=0, help="test only: evenly thin the fit set")
    parser.add_argument("--iterations", type=int, default=20, help="CGLS iterations in this leg")
    parser.add_argument("--time-budget", type=float, default=3300.0,
                        help="seconds for the CGLS stage, restart pass included; no iteration is started that the previous one's duration says would overrun")
    parser.add_argument("--continue-state", default=None, help="cgls_state.pt of an earlier leg: continue that CGLS exactly")
    parser.add_argument("--restart-from", default=None,
                        help="step-0 checkpoint of an earlier warm start (warm_start.json beside it): restart CGLS from its field")
    parser.add_argument("--restart-tolerance", type=float, default=1e-3,
                        help="max |fit rel-MSE at the source field - the source's last recorded fit| (restart), or relative residual drift (continue)")
    parser.add_argument("--target-t-min", type=float, default=0.8)
    parser.add_argument("--scale-pings", type=int, default=8)
    parser.add_argument("--gain-pings", type=int, default=32)
    parser.add_argument("--lr-fraction", type=float, default=0.01)
    parser.add_argument("--readout-pings", type=int, default=64)
    options = parser.parse_args(argv[:split])
    raw = argv[split + 1:]
    if options.continue_state and options.restart_from:
        raise SystemExit("--continue-state and --restart-from are exclusive")

    twins.check_backend()
    twins.install()
    args = resolve_trainer_args(raw)
    output = Path(args.checkpoint_root) / args.checkpoint_name
    target_path = output / "checkpoint_latest.pt"
    if target_path.exists():
        raise FileExistsError(f"refusing to overwrite {target_path}")
    for field, expected in (("signal_scale", 10.0), ("beamwidth_deg", 30.0), ("sh_direction", "rx_to_point")):
        if getattr(args, field) != expected:
            raise ValueError(f"the linearized operator assumes {field}={expected!r}, the run has {getattr(args, field)!r}")
    _sas.seed_all(args.seed)
    device = torch.device(args.device)
    cache = load_sas_cache(args.cache)
    model = _sas.build_model(args, cache, device)
    calibration = _sas.build_calibration(args.calibration_mode, device, args.gain_init_corr_threshold)
    scene = model.coefficient_field.scene
    points = int(args.initial_granularity)
    k_init = points ** 3

    # 1. linearized degree-0 operator on the same anchors
    op = LinearShellOperator(cache, device, (int(args.granularity),) * 3, 0, int(args.num_rays), True, points)
    op_scene = op.full_field.coefficient_field.scene
    if not torch.equal(op_scene.anchors[:k_init].cpu(), scene.anchors[:k_init].cpu()):
        raise AssertionError("operator and model anchors differ")
    if int(scene.active_mask.sum()) != k_init:
        raise AssertionError("the model must start with exactly the regular-grid points active")

    ring_size = int(cache.manifest["ring_size"])
    readout_train = set(_sas.select_eval_indices(cache.train_indices, options.readout_pings).tolist())
    train = cache.train_indices
    fit = np.asarray([p for p in train if (p % ring_size) % options.fit_stride == 0 and int(p) not in readout_train])
    if options.max_fit_pings > 0 and fit.size > options.max_fit_pings:
        fit = fit[np.unique(np.linspace(0, fit.size - 1, options.max_fit_pings).round().astype(int))]
    held_train = _sas.select_eval_indices(cache.train_indices, options.readout_pings)
    held_val = _sas.select_eval_indices(cache.validation_indices, options.readout_pings)
    test_rows = set(cache.test_indices.tolist())
    if test_rows.intersection(fit.tolist()) or set(cache.validation_indices.tolist()).intersection(fit.tolist()):
        raise AssertionError("the fit set must be TRAIN only")
    if test_rows.intersection(held_train.tolist()) or test_rows.intersection(held_val.tolist()):
        raise AssertionError("a reserved-test row was selected")
    masks = band_masks(cache.num_bins, float(cache.manifest["sample_rate_hz"]))
    print(f"warm start: fit {fit.size} TRAIN pings (stride {options.fit_stride}, TRAIN{options.readout_pings} excluded), "
          f"K={2 * k_init} real unknowns, M={fit.size * cache.num_bins} complex data; raster {args.granularity}^3", flush=True)

    target = torch.as_tensor(np.asarray(cache.weights[fit]).astype(np.complex64), device=device)
    y_train_held = spectra(np.asarray(cache.weights[held_train]))
    y_val_held = spectra(np.asarray(cache.weights[held_val]))
    y_norm = float((target.abs() ** 2).sum())
    fit_fields = {field: getattr(args, field) for field in FIT_FIELDS}
    fit_options = {name: getattr(options, name) for name in FIT_OPTIONS}
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "cgls_state.pt"

    def held_out(vector):
        p_train = spectra(op.forward(vector, held_train).cpu().numpy())
        p_val = spectra(op.forward(vector, held_val).cpu().numpy())
        return {
            "train64_full_g1": rel(y_train_held, p_train, masks, "full"),
            "train64_in_band_g1": rel(y_train_held, p_train, masks, IN_BAND),
            "val64_full_g1": rel(y_val_held, p_val, masks, "full"),
            "val64_in_band_g1": rel(y_val_held, p_val, masks, IN_BAND),
        }

    def check_source(label, source_fields, source_options):
        for field in FIT_FIELDS:
            if source_fields.get(field) != fit_fields[field]:
                raise ValueError(f"{label}: {field} is {source_fields.get(field)!r}, this leg has {fit_fields[field]!r}")
        for name in FIT_OPTIONS:
            if source_options.get(name) != fit_options[name]:
                raise ValueError(f"{label}: --{name.replace('_', '-')} is {source_options.get(name)!r}, this leg has {fit_options[name]!r}")

    def save_state(iteration, x, residual, direction, gamma):
        _sas.atomic_torch_save({
            "x": [t.detach().cpu() for t in x], "residual": residual.detach().cpu(),
            "direction": [t.detach().cpu() for t in direction], "gamma": gamma, "iteration": iteration,
            "history": history, "fit": fit, "fit_fields": fit_fields, "fit_options": fit_options,
            "continuation": continuation,
        }, state_path)

    started = time.time()
    history, continuation, start_iteration, new_rows = [], None, 0, 0
    q = s_grad = None
    if options.continue_state:
        source = torch.load(options.continue_state, map_location="cpu", weights_only=False)
        check_source(options.continue_state, source["fit_fields"], source["fit_options"])
        if not np.array_equal(np.asarray(source["fit"]), fit):
            raise ValueError(f"{options.continue_state}: the fit set differs from this leg's")
        x = [t.to(device) for t in source["x"]]
        residual = source["residual"].to(device)
        direction = [t.to(device) for t in source["direction"]]
        gamma = float(source["gamma"])
        history = [dict(row) for row in source["history"]]
        start_iteration = int(source["iteration"])
        drift = float((target - op.forward(x, fit) - residual).abs().norm()) / math.sqrt(y_norm)
        continuation = {"mode": "exact", "source": str(options.continue_state), "source_iteration": start_iteration,
                        "residual_drift": drift}
        print(f"continuing CGLS {options.continue_state} after iteration {start_iteration}: "
              f"|r - (b - Ax)| / |b| = {drift:.2e} ({time.time() - started:.0f}s)", flush=True)
        if drift > options.restart_tolerance:
            raise AssertionError(f"the saved residual does not match b - Ax: {drift:.3e}")
    elif options.restart_from:
        source = torch.load(options.restart_from, map_location="cpu", weights_only=False)
        provenance = json.loads((Path(options.restart_from).parent / "warm_start.json").read_text())
        if int(source["step"]) != 0:
            raise ValueError(f"{options.restart_from} is at step {source['step']}; --restart-from needs a warm start's step 0")
        check_source(options.restart_from, source["args"], provenance["options"])
        if provenance["fit_pings"] != fit.size or provenance["fit_first_last"] != [int(fit[0]), int(fit[-1])]:
            raise ValueError(f"{options.restart_from}: the fit set differs from this leg's")
        weights = [source["model_state_dict"][f"coefficient_field.scene.{name}"] for name in ("w_re", "w_im")]
        if any(t.shape[0] != k_init or float(t[:, 1:].abs().max()) > 0.0 for t in weights):
            raise ValueError(f"{options.restart_from} does not hold a DC-only field on {k_init} points")
        source_scale = float(provenance["scale"])
        x = [(t[:, :1].double() / source_scale).float().to(device) for t in weights]
        if any(xi.shape != p.shape for xi, p in zip(x, op.params)):
            raise AssertionError("the source field does not match the operator's parameter shapes")
        residual = target - op.forward(x, fit)
        restart_fit = float((residual.abs() ** 2).sum()) / y_norm
        history = [dict(row) for row in provenance["cgls_history"]]
        start_iteration = int(history[-1]["iteration"])
        recorded = float(history[-1]["fit_rel_mse"])
        check = {"fit_rel_mse": restart_fit, "recorded_fit_rel_mse": recorded, **held_out(x)}
        print(f"restart from {options.restart_from} (scale {source_scale:.6g}) after iteration {start_iteration}: fit "
              f"{restart_fit:.6f} against the recorded {recorded:.6f} | held-out TRAIN{options.readout_pings} full "
              f"{check['train64_full_g1']:.4f} in-band {check['train64_in_band_g1']:.4f} | VAL{options.readout_pings} full "
              f"{check['val64_full_g1']:.4f} in-band {check['val64_in_band_g1']:.4f} (linearized, g=1)", flush=True)
        if abs(restart_fit - recorded) > options.restart_tolerance:
            raise AssertionError(f"the source field does not reproduce its recorded fit: {restart_fit:.6f} vs {recorded:.6f}")
        s_grad = op.adjoint(residual, fit)
        direction = [v.clone() for v in s_grad]
        gamma = dot(s_grad, s_grad)
        continuation = {"mode": "restart", "source": str(options.restart_from), "source_iteration": start_iteration,
                        "source_scale": source_scale, "restart_check": check}
        print(f"restart pass (forward + adjoint) {time.time() - started:.0f}s; the CG direction restarts as A^H r", flush=True)
    else:
        x = [torch.zeros_like(p) for p in op.params]
        residual = target.clone()
        s_grad = op.adjoint(residual, fit)
        direction = [v.clone() for v in s_grad]
        gamma = dot(s_grad, s_grad)
        print(f"initial adjoint pass {time.time() - started:.0f}s", flush=True)
    for iteration in range(start_iteration + 1, start_iteration + options.iterations + 1):
        tick = time.time()
        q = op.forward(direction, fit)
        q_norm = float((q.abs() ** 2).sum())
        if q_norm <= 0.0 or gamma <= 0.0:
            break
        alpha = gamma / q_norm
        x = [xi + alpha * di for xi, di in zip(x, direction)]
        residual = residual - alpha * q
        s_grad = op.adjoint(residual, fit)
        gamma_new = dot(s_grad, s_grad)
        direction = [si + (gamma_new / gamma) * di for si, di in zip(s_grad, direction)]
        gamma = gamma_new
        row = {
            "iteration": iteration,
            "fit_rel_mse": float((residual.abs() ** 2).sum()) / y_norm,
            **held_out(x),
            "seconds": time.time() - tick,
            "job": os.environ.get("SLURM_JOB_ID"),
        }
        history.append(row)
        new_rows += 1
        save_state(iteration, x, residual, direction, gamma)
        print(f"CGLS it {iteration} ({row['seconds']:.0f}s): fit {row['fit_rel_mse']:.4f} | held-out TRAIN{options.readout_pings} full "
              f"{row['train64_full_g1']:.4f} in-band {row['train64_in_band_g1']:.4f} | VAL{options.readout_pings} full {row['val64_full_g1']:.4f} "
              f"in-band {row['val64_in_band_g1']:.4f} (linearized, g=1); state saved", flush=True)
        if options.time_budget and time.time() - started + row["seconds"] > options.time_budget:
            print(f"CGLS time budget: stopping after iteration {iteration} ({new_rows} in this leg); "
                  f"another would end at about {time.time() - started + row['seconds']:.0f}s > {options.time_budget:.0f}s", flush=True)
            break
    if not new_rows:
        raise RuntimeError("no CGLS iteration completed in this leg")
    fitted_re = x[0][:k_init, 0].detach().clone()
    fitted_im = x[1][:k_init, 0].detach().clone()

    def set_scale(scale):
        with torch.no_grad():
            scene.w_re.zero_()
            scene.w_im.zero_()
            scene.w_re[:k_init, 0] = fitted_re * scale
            scene.w_im[:k_init, 0] = fitted_im * scale

    # transfer check: the model's degree-3 field carrying the fitted DC reproduces the
    # linearized operator's prediction (lambertian_ratio 1, opacity 0) on two fit pings
    set_scale(1.0)
    linear_args = argparse.Namespace(**vars(args))
    linear_args.lambertian_ratio, linear_args.opacity_scale = 1.0, 0.0
    with torch.no_grad():
        check_pings = fit[:2]
        expected = op.forward(x, check_pings)
        rendered = torch.stack([
            _sas.render_one(model, calibration, cache, int(p), np.arange(cache.num_bins), linear_args, device)[2]["calibration_raw"]
            for p in check_pings
        ])
    transfer_error = float((rendered - expected).abs().norm() / expected.abs().norm().clamp_min(1e-30))
    print(f"transfer check (model field vs CGLS operator, linearized renderer): rel error {transfer_error:.2e}", flush=True)
    if transfer_error > 1e-4:
        raise AssertionError(f"the fitted field does not transfer to the model: {transfer_error:.3e}")
    del op, target, residual, s_grad, direction, q

    # 2. scale for the target minimum transmittance under the production renderer
    fit_rng = np.random.default_rng(args.seed)
    scale_pings = np.sort(fit_rng.choice(fit, size=min(options.scale_pings, fit.size), replace=False))
    gain_pings = np.sort(fit_rng.choice(fit, size=min(options.gain_pings, fit.size), replace=False))

    def t_min_at(scale):
        set_scale(scale)
        return float(production_pass(model, calibration, cache, scale_pings, args, device)[2].min())

    # T_all_min decreases with s: bracket [lo, hi] with T(lo) > target >= T(hi), then bisect geometrically
    lo = hi = 1.0
    if t_min_at(1.0) > options.target_t_min:
        while t_min_at(hi) > options.target_t_min:
            if hi >= 1e15:
                raise RuntimeError("transmittance never reaches the target; the fitted field is empty")
            lo, hi = hi, hi * 10.0
    else:
        while t_min_at(lo) <= options.target_t_min:
            if lo <= 1e-15:
                raise RuntimeError("transmittance stays below the target at every scale")
            hi, lo = lo, lo / 10.0
    while hi / lo > 1.02:
        mid = math.sqrt(lo * hi)
        if t_min_at(mid) > options.target_t_min:
            lo = mid
        else:
            hi = mid
    scale = lo
    set_scale(scale)
    raws, targets, t_mins, lamberts = production_pass(model, calibration, cache, scale_pings, args, device)
    corr = [abs(np.sum(np.conj(r) * t)) / math.sqrt(np.sum(np.abs(r) ** 2) * np.sum(np.abs(t) ** 2)) for r, t in zip(raws, targets)]
    print(f"scale s={scale:.4e}: T_all_min min/median {t_mins.min():.3f}/{np.median(t_mins):.3f}, "
          f"lambert_positive median {np.median(lamberts):.3f}, raw correlation median {np.median(corr):.3f} "
          f"({len(scale_pings)} TRAIN pings, production renderer)", flush=True)

    # 3. TRAIN-fitted global gain under the production renderer
    raws, targets, _t, _l = production_pass(model, calibration, cache, gain_pings, args, device)
    gain = complex(np.sum(np.conj(raws) * targets) / np.sum(np.abs(raws) ** 2))
    with torch.no_grad():
        calibration.log_mag.fill_(math.log(abs(gain)))
        calibration.phase.fill_(math.atan2(gain.imag, gain.real))
        calibration.initialized.fill_(True)
    gain_rel = float(np.sum(np.abs(gain * raws - targets) ** 2) / np.sum(np.abs(targets) ** 2))
    print(f"gain g={gain:.4e} fitted on {len(gain_pings)} TRAIN fit pings; their rel-MSE at g {gain_rel:.4f}", flush=True)

    # 4. learning rate from the scaled field
    magnitude = torch.sqrt(scene.w_re[:k_init, 0].detach() ** 2 + scene.w_im[:k_init, 0].detach() ** 2)
    median_w = float(magnitude.median())
    raw_lr = options.lr_fraction * median_w
    lr = float(f"{raw_lr:.0e}") if raw_lr > 0 else float(args.coefficient_lr)
    args.coefficient_lr = lr
    print(f"median |DC| of the scaled field {median_w:.4e}; coefficient_lr set to {lr:g} "
          f"({options.lr_fraction} x median, one significant digit)", flush=True)

    # 5. step-0 checkpoint with the trainer's own writer
    optimizer = _sas._optimizer_for_model(model, calibration, args)
    rng = np.random.default_rng(args.seed)
    payload = _sas.checkpoint(model, calibration, optimizer, 0, float("inf"), rng, [], args, cache)
    output.mkdir(parents=True, exist_ok=True)
    _sas.atomic_torch_save(payload, target_path)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    except OSError:
        commit = None
    provenance = {
        "script": "scripts_pvc_sas/rift_sas_warm_start.py", "git_head": commit, "trainer_argv": raw,
        "options": vars(options), "fit_pings": int(fit.size), "fit_first_last": [int(fit[0]), int(fit[-1])],
        "excluded_train_readout_pings": options.readout_pings, "k_real_unknowns": 2 * k_init,
        "cgls_history": history, "cgls_continuation": continuation, "cgls_state": str(state_path),
        "cgls_iterations_this_leg": new_rows,
        "cgls_seconds": time.time() - started, "transfer_rel_error": transfer_error, "scale": scale,
        "scale_pings": scale_pings.tolist(), "t_all_min_scale_pings": t_mins.tolist(),
        "lambert_positive_scale_pings": lamberts.tolist(), "raw_correlation_scale_pings": corr,
        "gain": [gain.real, gain.imag], "gain_pings": gain_pings.tolist(), "gain_rel_mse_on_gain_pings": gain_rel,
        "median_abs_dc": median_w, "coefficient_lr": lr, "checkpoint": str(target_path),
    }
    (output / "warm_start.json").write_text(json.dumps(provenance, indent=2))
    print(f"wrote step-0 checkpoint {target_path} and warm_start.json", flush=True)
    print(f"CONTINUE_COEFFICIENT_LR={lr:g}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
