#!/usr/bin/env python
"""PVC (Intel XPU) entry point for the sonar trainer ``train_sas.py`` (Package G).

Same CLI as ``train_sas.py`` (RIFT-SAS ``rift_sas``, adaptive RIFT-SAS
``adaptive_rift_sas`` and the independent SH-SAS ``sh_sas`` on a sonar cache),
so ``scripts_pvc_sas/run_airsas_comparison_pvc.sh`` substitutes the script name and
nothing else. The unchanged ``train_sas`` module is imported; its three
accelerator-specific functions are rebound to their ``rift_pvc_sas.training``
twins (``parse_args``: ``--device`` defaults to the accelerator device;
``seed_all``: seeds the accelerator; ``checkpoint``: adds ``xpu_rng_state`` and
``accelerator_backend`` next to the original ``cuda_rng_state``). ``main_copy``
is a verbatim copy of the original ``main`` with exactly these edits, pinned by
``rift_pvc_sas/tests/test_sas_pvc.py``:

* the resume block restores the RNG payload of the active backend
  (``twins.restore_device_rng_state``) instead of only ``cuda_rng_state``;
* ``selected_readout.json`` additionally carries ``peak_accelerator_memory_bytes``
  and ``accelerator_backend`` (``peak_cuda_memory_bytes`` keeps its meaning);
* ``STOP_REQUESTED`` is read from the trainer module, whose ``request_stop``
  signal handler sets it.

Recipes, ``SAVED_SCIENTIFIC_FIELDS``, the cache contract, selected-best
resolution, refinement, renderer, fields, checkpoint names and ``status.json``
semantics are the original's. Dtypes stay complex64/float32. By default this
entry point refuses to run on a non-XPU backend; set
``RIFT_PVC_ALLOW_BACKEND=cpu`` (tests) or ``=cuda`` to override.
"""
from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train_sas as _sas  # noqa: E402  (the unchanged trainer)
from train_sas import (  # noqa: E402  (unchanged helpers used by the copied main)
    _adaptive_scene,
    _apply_profile,
    _diagnostic_hook,
    _explicit_cli_fields,
    _historical_best,
    _model_parameters,
    _optimizer_for_model,
    _reconcile_saved_recipe,
    _resolve_selected_best,
    _validate_best_candidate,
    _validate_cache_contract,
    _validate_saved_model_box,
    atomic_json,
    atomic_torch_save,
    build_calibration,
    build_model,
    calibration_diagnostics,
    evaluate,
    load_sas_cache,
    render_one,
    request_stop,
    resolve_calibration_mode,
    save_history,
    select_bins,
    select_eval_bins,
    select_eval_indices,
)
from rift_pvc_sas import training as twins  # noqa: E402

# The three rebound names, as the copied main refers to them.
parse_args = twins.parse_args
seed_all = twins.seed_all
checkpoint = twins.checkpoint


def install():
    """Rebind the audited names inside the unchanged ``train_sas`` module (idempotent)."""
    return twins.install()


# --------------------------------------------------------------------------
# Copy of train_sas.main (checkout 6635e22) with the audited edits only.
# --------------------------------------------------------------------------
def main_copy(
    argv: Optional[Sequence[str]] = None, *, diagnostic_observer: object | None = None
) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    explicit_fields = _explicit_cli_fields(raw_argv)
    args = parse_args(raw_argv)
    profile_applied_fields = _apply_profile(args, explicit_fields)
    if args.opacity_normalize is None:
        args.opacity_normalize = False
    output = Path(args.checkpoint_root) / args.checkpoint_name
    output.mkdir(parents=True, exist_ok=True)
    resume = Path(args.resume) if args.resume else output / "checkpoint_latest.pt"
    if args.resume is not None and not resume.exists():
        raise FileNotFoundError(f"explicit --resume checkpoint does not exist: {resume}")
    if args.eval_only and not resume.exists():
        raise ValueError("--eval-only requires an existing --resume checkpoint")
    state = (
        torch.load(resume, map_location="cpu", weights_only=False)
        if resume.exists() else None
    )
    if state is not None and state.get("model_kind") != args.model:
        raise ValueError("checkpoint model kind does not match --model")
    if args.query_chunk <= 0:
        raise ValueError("--query-chunk must be positive")
    if state is not None:
        _reconcile_saved_recipe(
            args,
            state,
            explicit_fields,
            profile_applied_fields,
            eval_only=bool(args.eval_only),
        )
        if args.steps < int(state["step"]):
            raise ValueError(
                f"terminal --steps={args.steps} is below resumed checkpoint step {int(state['step'])}"
            )
    if args.grid_shape is not None and args.model != "rift_sas":
        raise ValueError("--grid-shape is allowed only with --model rift_sas")
    if args.ray_chunk < 0:
        raise ValueError("--ray-chunk must be nonnegative")
    if args.pings_per_step < 1:
        raise ValueError("--pings-per-step must be >= 1")
    if not 0.0 <= args.gain_init_corr_threshold <= 1.0:
        raise ValueError("--gain-init-corr-threshold must be in [0, 1]")
    if args.model in ("adaptive_rift_sas", "sh_sas"):
        if state is None:
            args.require_explicit_splits = True
        if not args.require_explicit_splits:
            raise ValueError("this comparison requires an AirSAS cache with explicit train/validation/test splits")
    if args.model == "adaptive_rift_sas":
        if args.sh_degree != 3:
            raise ValueError("adaptive_rift_sas comparison is fixed to SH degree 3")
        if args.probe_every <= 0 or args.refine_every <= 0:
            raise ValueError("adaptive probe/refinement intervals must be positive")
    args.calibration_mode = resolve_calibration_mode(args.model, args.calibration_mode, state)
    seed_all(args.seed)
    device = torch.device(args.device)
    cache = load_sas_cache(args.cache)
    if state is not None:
        _validate_cache_contract(state, cache)
        _validate_saved_model_box(state, cache)
    if args.require_explicit_splits and not cache.has_explicit_splits:
        raise ValueError("this comparison requires an AirSAS cache with explicit train/validation/test splits")
    if cache.has_explicit_splits:
        if args.max_pings > 0:
            raise ValueError("--max-pings cannot truncate a cache with explicit train/validation/test splits")
        train_indices = cache.train_indices
        validation_indices = cache.validation_indices
        test_indices = cache.test_indices
    else:
        if args.model in ("adaptive_rift_sas", "sh_sas"):
            raise ValueError("this comparison requires an AirSAS cache with explicit train/validation/test splits")
        num_pings = cache.num_pings if args.max_pings <= 0 else min(args.max_pings, cache.num_pings)
        train_indices = np.arange(num_pings, dtype=np.int64)
        validation_indices = np.arange(num_pings, dtype=np.int64)
        test_indices = np.empty(0, dtype=np.int64)
    if train_indices.size == 0 or validation_indices.size == 0:
        raise ValueError("training and validation cohorts must be non-empty")

    if state is not None and not args.eval_only and diagnostic_observer is None:
        selected_best = _resolve_selected_best(resume, state, cache)
        if selected_best is not None:
            selected_destination = output / "checkpoint_best.pt"
            selected_source = Path(selected_best["source"])
            if selected_source.resolve() != selected_destination.resolve():
                atomic_torch_save(selected_best["payload"], selected_destination)

    model = build_model(args, cache, device)
    calibration = build_calibration(args.calibration_mode, device, args.gain_init_corr_threshold)
    optimizer = _optimizer_for_model(model, calibration, args)
    rng = np.random.default_rng(args.seed)
    start, best_val, history = 0, float("inf"), []
    if state is not None:
        model.load_state_dict(state["model_state_dict"])
        calibration.load_state_dict(state["calibration_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start = int(state["step"])
        best_val = float(state["best_val_rel_mse"])
        history = list(state.get("history", []))
        rng.bit_generator.state = state["rng_state"]
        torch.set_rng_state(state["torch_rng_state"])
        twins.restore_device_rng_state(state, device=device)
        print(f"Resumed {args.model} at step {start}", flush=True)
    print(f"calibration_mode={args.calibration_mode}", flush=True)
    _diagnostic_hook(
        diagnostic_observer,
        "on_restore",
        model=model,
        calibration=calibration,
        optimizer=optimizer,
        rng=rng,
        cache=cache,
        args=args,
        device=device,
        state=state,
        start=start,
        best_val=best_val,
        history=history,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )

    if args.eval_only:
        role_indices = validation_indices if args.evaluation_role == "validation" else test_indices
        if role_indices.size == 0:
            raise ValueError(f"evaluation role {args.evaluation_role!r} is empty")
        metrics = evaluate(model, calibration, cache, role_indices, args, device)
        selected_eval_indices = select_eval_indices(role_indices, args.eval_pings)
        selected_eval_bins = select_eval_bins(cache, args)
        atomic_json(
            {
                **metrics,
                "evaluation_role": args.evaluation_role,
                "source_ids": cache.source_ids[selected_eval_indices].tolist(),
                "selected_source_ids": cache.source_ids[selected_eval_indices].tolist(),
                "selected_bin_ids": selected_eval_bins.tolist(),
                "checkpoint_step": start,
            },
            output / f"{args.evaluation_role}_eval.json",
        )
        print(f"{args.evaluation_role} evaluation rel_mse={metrics['rel_mse']:.6e}", flush=True)
        return

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    model.train()
    started = time.time()
    for step in range(start, args.steps):
        current = step + 1
        optimizer.zero_grad(set_to_none=True)
        scene = _adaptive_scene(model)
        prior_delta_grad = torch.zeros_like(scene.delta_raw) if scene is not None else None
        for _ping_index in range(args.pings_per_step):
            ping = int(rng.choice(train_indices))
            target_np = np.asarray(cache.weights[ping])
            bins = select_bins(rng, target_np, args.max_bins)
            loss, metrics, aux = render_one(
                model, calibration, cache, ping, bins, args, device,
                allow_calibration_init=True,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {current}")
            (loss / args.pings_per_step).backward()
            if scene is not None:
                # Incremental, not cumulative: accumulate_refinement_data_stats
                # wants the gradient contributed by *this* ping alone (its own
                # docstring warns against "dependence on optimizer
                # accumulation"), but pings within a step share one
                # zero_grad(), so delta_raw.grad keeps growing across them.
                current_delta_grad = (
                    scene.delta_raw.grad.detach().clone()
                    if scene.delta_raw.grad is not None else torch.zeros_like(scene.delta_raw)
                )
                data_delta = current_delta_grad - prior_delta_grad
                prior_delta_grad = current_delta_grad
                next_re = next_im = None
                if current % args.probe_every == 0 and (scene.active_mask & scene.order.lt(scene.max_degree)).any():
                    probe_loss, _probe_metrics, _probe_aux = render_one(
                        model, calibration, cache, ping, bins, args, device,
                        allow_calibration_init=False,
                        probe_next_band=True,
                    )
                    next_re, next_im = torch.autograd.grad(
                        probe_loss, (scene.w_re, scene.w_im), allow_unused=True
                    )
                    if next_re is None:
                        next_re = torch.zeros_like(scene.w_re)
                    if next_im is None:
                        next_im = torch.zeros_like(scene.w_im)
                scene.accumulate_refinement_data_stats(data_delta, next_re, next_im)
        log_now = current == 1 or current % args.log_every == 0
        diagnostics = calibration_diagnostics(aux, calibration) if log_now else None
        grad_norm = torch.nn.utils.clip_grad_norm_(
            _model_parameters(model, calibration), args.grad_clip
        )
        will_refine = scene is not None and current % args.refine_every == 0
        _diagnostic_hook(
            diagnostic_observer,
            "on_before_optimizer",
            model=model,
            calibration=calibration,
            optimizer=optimizer,
            cache=cache,
            args=args,
            device=device,
            scene=scene,
            step=current,
            ping=ping,
            bins=bins,
            loss=loss,
            metrics=metrics,
            aux=aux,
            grad_norm=grad_norm,
            will_refine=will_refine,
        )
        optimizer.step()
        _diagnostic_hook(
            diagnostic_observer,
            "on_after_optimizer",
            model=model,
            calibration=calibration,
            optimizer=optimizer,
            cache=cache,
            args=args,
            device=device,
            scene=scene,
            step=current,
            ping=ping,
            bins=bins,
            loss=loss,
            metrics=metrics,
            aux=aux,
            grad_norm=grad_norm,
            will_refine=will_refine,
        )
        if log_now:
            print(
                f"step {current}/{args.steps} loss={float(loss):.6e} rel_mse={float(metrics['rel_mse']):.6e} "
                f"grad={float(grad_norm):.3e} rays={int(aux['actual_rays'])}",
                flush=True,
            )
            diagnostic_text = " ".join(
                f"{key}={diagnostics[key]:.6e}" for key in (
                    "raw_rms", "target_rms", "pred_rms", "raw_corr_abs", "gain_abs",
                    "gain_phase", "T_all_min", "T_all_mean", "T_all_lt_1e3", "lambert_positive",
                )
            )
            print(f"sonar_diagnostics {diagnostic_text}", flush=True)
        if scene is not None and current % args.refine_every == 0:
            snapshot = scene.refinement_snapshot(
                max_level=args.split_max_level,
                min_spatial_exposure=1,
                min_angular_exposure=1,
                cooldown_events=args.cooldown_events,
                child_maturity_events=args.child_maturity_events,
            )
            _diagnostic_hook(
                diagnostic_observer,
                "on_before_refinement",
                model=model,
                calibration=calibration,
                optimizer=optimizer,
                cache=cache,
                args=args,
                device=device,
                scene=scene,
                step=current,
                snapshot=snapshot,
            )
            _n_split, _n_angular, _active, report = scene.apply_refinement_snapshot(
                snapshot,
                spatial_fraction=args.spatial_fraction,
                angular_fraction=args.angular_fraction,
                max_level=args.split_max_level,
                optimizer=optimizer,
                max_active=args.max_active,
            )
            _diagnostic_hook(
                diagnostic_observer,
                "on_after_refinement",
                model=model,
                calibration=calibration,
                optimizer=optimizer,
                cache=cache,
                args=args,
                device=device,
                scene=scene,
                step=current,
                snapshot=snapshot,
                n_split=_n_split,
                n_angular=_n_angular,
                active=_active,
                report=report,
            )
            print(report, flush=True)
        if current % args.eval_every == 0 or current == args.steps:
            val = evaluate(model, calibration, cache, validation_indices, args, device)
            history.append(
                {
                    "step": current,
                    "train_loss": float(loss.detach()),
                    "train_rel_mse": float(metrics["rel_mse"].detach()),
                    "val_rel_mse": val["rel_mse"],
                    "val_l1_real": val["l1_real"],
                    "val_l1_imag": val["l1_imag"],
                    "val_l1_mag": val["l1_mag"],
                    "elapsed_seconds": time.time() - started,
                }
            )
            print(f"validation step={current} rel_mse={val['rel_mse']:.6e} views={int(val['views'])}", flush=True)
            if val["complete"] and val["rel_mse"] < best_val:
                best_val = val["rel_mse"]
                atomic_torch_save(
                    checkpoint(model, calibration, optimizer, current, best_val, rng, history, args, cache),
                    output / "checkpoint_best.pt",
                )
            save_history(history, output / "history.csv")
        if current % args.checkpoint_every == 0 or current == args.steps or _sas.STOP_REQUESTED:
            atomic_torch_save(
                checkpoint(model, calibration, optimizer, current, best_val, rng, history, args, cache),
                output / "checkpoint_latest.pt",
            )
        if _sas.STOP_REQUESTED:
            atomic_json(
                {"done": False, "step": current, "reason": "signal", "model": args.model},
                output / "status.json",
            )
            return

    if diagnostic_observer is not None:
        _diagnostic_hook(
            diagnostic_observer,
            "on_finish",
            model=model,
            calibration=calibration,
            optimizer=optimizer,
            rng=rng,
            cache=cache,
            args=args,
            device=device,
            step=args.steps,
            best_val=best_val,
            history=history,
            checkpoint_writer=checkpoint,
            atomic_save=atomic_torch_save,
        )
        return

    final = checkpoint(model, calibration, optimizer, args.steps, best_val, rng, history, args, cache)
    atomic_torch_save(final, output / "checkpoint_final.pt")
    selected_path = output / "checkpoint_best.pt"
    if not selected_path.exists():
        raise RuntimeError(
            "no validated checkpoint_best.pt is available for final selection; "
            "the final model cannot be cited under a historical best metric"
        )
    selected = torch.load(selected_path, map_location=device, weights_only=False)
    selected_step, selected_metric = _historical_best(final)
    _validate_best_candidate(selected, final, cache, selected_step, selected_metric)
    model.load_state_dict(selected["model_state_dict"])
    calibration.load_state_dict(selected["calibration_state_dict"])
    model.eval()
    calibration.eval()
    voxels = torch.as_tensor(cache.voxels, dtype=torch.float32, device=device)
    density = model.dense_density(voxels).numpy().astype(np.float32)
    np.save(output / "density.npy", density)
    if test_indices.size:
        test_metrics = evaluate(model, calibration, cache, test_indices, args, device)
        atomic_json(test_metrics, output / "test_metrics.json")
    atomic_json(
        {
            "selected_checkpoint": str(selected_path.name if selected_path.exists() else "checkpoint_final.pt"),
            "selected_checkpoint_step": selected_step,
            "selected_checkpoint_rel_mse": selected_metric,
            "best_val_rel_mse": selected_metric,
            "test_role_used_after_selection": bool(test_indices.size),
            "selected_validation_source_ids": cache.source_ids[
                select_eval_indices(validation_indices, args.eval_pings)
            ].tolist(),
            "selected_test_source_ids": cache.source_ids[
                select_eval_indices(test_indices, args.eval_pings)
            ].tolist() if test_indices.size else [],
            "allocated_parameter_scalars": sum(parameter.numel() for parameter in model.parameters()),
            "active_parameter_scalars": (
                _adaptive_scene(model).active_parameter_count()
                if _adaptive_scene(model) is not None else None
            ),
            "elapsed_seconds": time.time() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            **twins.readout_telemetry(device),
        },
        output / "selected_readout.json",
    )
    atomic_json(
        {"done": True, "step": args.steps, "best_val_rel_mse": best_val, "model": args.model},
        output / "status.json",
    )


def main(argv: Optional[Sequence[str]] = None, *, diagnostic_observer: object | None = None) -> None:
    twins.check_backend()
    install()
    return main_copy(argv, diagnostic_observer=diagnostic_observer)


if __name__ == "__main__":
    main()
