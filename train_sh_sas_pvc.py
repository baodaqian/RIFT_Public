#!/usr/bin/env python3
"""Radar SH-SAS on PVC with the original CLI, recipe and sealed roles.

_main is copied verbatim from train_sh_sas.main with one edit: device RNG
restoration. Its wrapper binds the function to the original module namespace
so the unchanged signal handler and evaluate function share STOP_REQUESTED.
Device defaults, seeding, checkpoint RNGs and the probed tensor-division field
are installed by rift_pvc.sh_sas_training; the original files stay untouched.
"""
from __future__ import annotations

import types
from typing import Optional, Sequence

import train_sh_sas as _base
from rift_pvc import sh_sas_training as twins


def _main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    set_seed(args.seed)
    device = torch.device(args.device)

    print("SH-SAS independent implementation: arXiv:2509.11087 / 3DV 2026")
    print("Geometry supervision: disabled; initialization: random neural-field parameters")
    print(f"Using device: {device}")
    args.sealed_npz_protocol_contract = None
    if args.npz_role_manifest:
        from train import _load_sealed_npz_protocol_contract
        from rift.radar_fields_dataset import restrict_radar_fields_response_views
        public, contract = _load_sealed_npz_protocol_contract(
            args.npz_path, args.npz_role_manifest, num_train=args.num_train,
            num_val=args.num_val, num_test=args.num_test)
        args.sealed_npz_protocol_contract = contract
        train_indices = np.asarray(contract["role_ids"]["train"], dtype=np.int64)
        val_indices = np.asarray(contract["role_ids"]["validation"], dtype=np.int64)
        if contract.get('dataset_identity'):
            from rift.radar_fields_dataset import from_collection_arrays
            arrays = from_collection_arrays(public, contract)
        else:
            arrays = restrict_radar_fields_response_views(
                load_radar_fields_npz(args.npz_path, load_response=False),
                np.concatenate((train_indices, val_indices)))
    else:
        arrays = load_radar_fields_npz(args.npz_path)
        train_indices, val_indices, _test_indices = split_view_indices(
            arrays.num_views, args.num_train, args.num_val, args.num_test,
            args.seed, val_from_tail=args.val_from_tail)
    frequencies = torch.as_tensor(
        build_frequency_grid(arrays.metadata), dtype=torch.float32, device=device
    )
    kvector = (2.0 * torch.pi * frequencies) / cc
    print(
        f"Dataset: {arrays.num_views} views, {arrays.num_tx}x{arrays.num_rx} pairs, "
        f"{arrays.num_freq} frequencies; split train={len(train_indices)} val={len(val_indices)}"
    )

    checkpoint_dir = Path(args.checkpoint_root) / args.checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args, device)
    gain = GlobalComplexGain().to(device)
    print(f"Paper architecture: {model.paper_architecture}")
    print(
        f"Rendering: DC density/normals, Lambertian={args.lambertian}, "
        f"occlusion={args.occlusion}, key={args.opacity_key}, zeta={args.opacity_scale:g}, "
        f"opacity_normalize={args.opacity_normalize}"
    )
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(gain.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1.0e-15,
    )
    rng = np.random.default_rng(args.seed)
    start_step = 0
    best_val = float("inf")
    history = []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        from train import _validate_saved_sealed_npz_protocol_contract
        _validate_saved_sealed_npz_protocol_contract(
            checkpoint.get("sealed_npz_protocol_contract"), args.sealed_npz_protocol_contract)
        if checkpoint.get("sh_sas_contract_version") != CONTRACT_VERSION:
            raise ValueError("resume checkpoint has an incompatible SH-SAS contract version")
        if checkpoint.get("geometry_truth_used_for_training", True):
            raise ValueError("resume checkpoint does not certify the sensor-only training contract")
        model.load_state_dict(checkpoint["sh_sas_state_dict"])
        gain.load_state_dict(checkpoint["gain_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None and not args.eval_only:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"])
        best_val = float(checkpoint.get("best_val_rel_mse", checkpoint.get("loss", float("inf"))))
        history = list(checkpoint.get("history", []))
        rng.bit_generator.state = json.loads(checkpoint["numpy_rng_state_json"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        restore_device_rng_state(checkpoint)
        print(f"Resumed SH-SAS from step {start_step}, best val rel-MSE={best_val:.6e}")

    if args.eval_only:
        metrics = evaluate(
            model, gain, arrays, frequencies, kvector, val_indices, args, device
        )
        atomic_json(metrics, checkpoint_dir / "sh_sas_eval.json")
        print(
            f"Validation: rel-MSE={metrics['rel_mse']:.6%} "
            f"L1(Re/Im/Abs)={metrics['l1_real']:.3e}/{metrics['l1_imag']:.3e}/"
            f"{metrics['l1_abs']:.3e}"
        )
        return

    model.train()
    gain.train()
    prior_enabled = any(
        weight > 0
        for weight in (
            args.sparse_weight,
            args.density_tv_weight,
            args.scatter_tv_weight,
            args.phase_tv_weight,
        )
    )
    print(f"Paper Eq. (8) priors enabled: {prior_enabled} (paper disables them on simulated data)")

    for step_zero in range(start_step, args.steps):
        step = step_zero + 1
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        batch_views = rng.choice(
            train_indices,
            size=args.views_per_step,
            replace=len(train_indices) < args.views_per_step,
        )
        records = []
        term_totals = {key: 0.0 for key in ("data", "sparse", "density_tv", "scatter_tv", "phase_tv")}

        completed_views = 0
        for view_index in batch_views:
            tx_indices = select_axis_indices(rng, arrays.num_tx, args.train_tx, random=True)
            rx_indices = select_axis_indices(rng, arrays.num_rx, args.train_rx, random=True)
            freq_indices = select_frequency_indices(
                rng, arrays.num_freq, args.num_freq_wanted, random=True
            )
            loss, terms, metrics = view_objective(
                model,
                gain,
                arrays,
                frequencies,
                kvector,
                int(view_index),
                tx_indices,
                rx_indices,
                freq_indices,
                args,
                device,
                include_regularizers=True,
            )
            (loss / args.views_per_step).backward()
            completed_views += 1
            records.append(metrics)
            for key in term_totals:
                term_totals[key] += float(terms[key].detach()) / args.views_per_step
            if STOP_REQUESTED:
                break

        parameters = list(model.parameters()) + list(gain.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        train_metrics = combine_metrics(records)
        elapsed = time.time() - started
        print(
            f"Step [{step}/{args.steps}] data={term_totals['data']:.6e} "
            f"rel-MSE={train_metrics['rel_mse']:.4%} "
            f"zeta={args.opacity_scale:.3g} |g|={abs(gain.gain_value()):.3e} "
            f"grad={float(grad_norm):.3e} [{elapsed:.1f}s/step]",
            flush=True,
        )
        if prior_enabled:
            print(
                "  priors: "
                + " ".join(f"{key}={term_totals[key]:.3e}" for key in (
                    "sparse", "density_tv", "scatter_tv", "phase_tv"
                )),
                flush=True,
            )

        validation = None
        should_evaluate = (step % args.eval_every == 0 or step == args.steps) and not STOP_REQUESTED
        if should_evaluate:
            validation = evaluate(
                model, gain, arrays, frequencies, kvector, val_indices, args, device
            )
            if validation["complete"]:
                print(
                    f"Validation [step {step}] rel-MSE={validation['rel_mse']:.6%} "
                    f"L1(Re/Im/Abs)={validation['l1_real']:.3e}/"
                    f"{validation['l1_imag']:.3e}/{validation['l1_abs']:.3e}",
                    flush=True,
                )
                row = {
                    "step": step,
                    "train_rel_mse": train_metrics["rel_mse"],
                    "val_rel_mse": validation["rel_mse"],
                    "val_l1_real": validation["l1_real"],
                    "val_l1_imag": validation["l1_imag"],
                    "val_l1_abs": validation["l1_abs"],
                    "val_mse_real": validation["mse_real"],
                    "val_mse_imag": validation["mse_imag"],
                    "val_mse_abs": validation["mse_abs"],
                    "step_seconds": elapsed,
                }
                history.append(row)
                save_history(history, checkpoint_dir / "sh_sas_history.csv")

        is_best = bool(validation and validation["complete"] and validation["rel_mse"] < best_val)
        if is_best:
            best_val = validation["rel_mse"]
        should_checkpoint = (
            step % args.checkpoint_every == 0 or should_evaluate or STOP_REQUESTED
        )
        payload = None
        if should_checkpoint or is_best:
            payload = checkpoint_payload(
                model, gain, optimizer, step, best_val, rng, history, args
            )
        if is_best:
            atomic_torch_save(payload, checkpoint_dir / "checkpoint_best.pth.tar")
        if should_checkpoint:
            name = "checkpoint_final.pth.tar" if step == args.steps else "checkpoint_latest.pth.tar"
            atomic_torch_save(payload, checkpoint_dir / name)

        if STOP_REQUESTED:
            print(
                f"Stopped cleanly after {completed_views} view(s) and publishing checkpoint_latest.",
                flush=True,
            )
            return

    summary = {
        "status": "complete",
        "steps": int(args.steps),
        "best_val_rel_mse": float(best_val),
        "geometry_truth_used_for_training": False,
        "paper_architecture": model.paper_architecture,
    }
    atomic_json(summary, checkpoint_dir / "run_summary.json")
    print("SH-SAS training complete", flush=True)


def install():
    return twins.install()


def main(argv: Optional[Sequence[str]] = None) -> None:
    install()
    args = twins.parse_args(argv)
    twins.require_backend(args.device)
    _base.STOP_REQUESTED = False
    # Functions, including the signal callback, see the same live globals.
    run = types.FunctionType(_main.__code__, vars(_base), _main.__name__, _main.__defaults__)
    run(argv)


if __name__ == "__main__":
    main()
