#!/usr/bin/env python
"""PVC (Intel XPU) entry point for the Radar Fields baseline (Package E).

Same CLI as ``train_radar_fields.py``, so the PVC dataset frontends substitute
the script name and nothing else:

    train_rift_dataset_pvc.py --method radar_fields -> train_radar_fields_pvc.py --recipe
        source-adapted-v3 --npz-path ... --sealed-protocol ... --device xpu ...
    train_gotcha_dataset_pvc.py --method radar_fields -> run_gotcha(dataset=, output_dir=,
        config=, device=, resume=)

The unchanged ``train_radar_fields`` module is imported; its accelerator-specific
helpers are rebound to their ``rift_pvc.radar_fields_training`` twins
(``set_seed``, ``evaluate``, ``checkpoint_payload``, ``validate_resume_checkpoint``,
``build_model``, ``recipe_contract``, ``check_model_backend``), and the authors'
release runs through the tinycudann torch shim (``rift_pvc.tcnn_torch``,
backend id ``upstream-tcnn-torchshim``). ``main()`` is a copy of the original
``main()`` with one change: the resume block restores the RNG payload of the
active backend (``xpu_rng_state``) instead of only ``cuda_rng_state``. Everything
else (recipe, sealed roles, checkpoint schema, execution order) is the
original's.

CLI differences, both confined to this process:

* ``--device`` defaults to the accelerator device (``xpu``) instead of ``cuda``;
  the PVC frontend rewrites the ``--device cuda`` the CUDA planner emits.
* ``--model-backend`` additionally accepts ``upstream-tcnn-torchshim``; with the
  flag omitted a native recipe selects it on XPU (the original selects
  ``upstream-tcnn``). An explicit ``upstream-tcnn`` is refused off CUDA: that
  identity means the real tiny-cuda-nn.

By default this entry point refuses to run without an XPU device; set
``RIFT_PVC_ALLOW_BACKEND=cpu`` (tests, with ``RIFT_PVC_TCNN_SHIM=1``) or ``=cuda``
to override.
"""
from __future__ import annotations

import json
import math
import os
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train_radar_fields as _rf  # noqa: E402  (the unchanged trainer)
from train_radar_fields import (  # noqa: E402  (unchanged helpers used by the copied main)
    NORMALIZED_DB_INTENSITY_DOMAIN, NORMALIZED_DB_RELMSE_LABEL, OFFICIAL_REFERENCE_COMMIT, SOURCE_RECIPE,
    CoveredViewSampler, _RESUME_CONFIG_FIELDS, _compatible_value, atomic_torch_save, audited_view_tensors,
    checkpoint_provenance, combine_metrics, dataset_provenance, evenly_spaced_pairs, finish_audited_view,
    generate_dynamic_grid, grid_support_provenance, intensity_metrics, load_or_create_stats,
    load_radar_fields_npz, load_radar_fields_sealed_split_manifest, native_recipe, original_module,
    preflight_audited_continuation, preflight_sealed_resume_checkpoint, range_bin_centers, range_bin_size,
    recipe_name, released_batch_loss, render_bistatic_batch, resolve_diagnostic_view,
    restrict_radar_fields_response_views, role_provenance, sample_pairs, save_history,
    sealed_protocol_contract, signal_provenance, split_view_indices, validate_recipe_checkpoint, view_objective,
)
from rift_pvc import accelerator  # noqa: E402
from rift_pvc import tcnn_torch  # noqa: E402
from rift_pvc import radar_fields_training as twins  # noqa: E402
from rift_pvc.radar_fields_upstream import REAL_BACKEND, TORCHSHIM_BACKEND  # noqa: E402


# Literal capability declaration read by train_gotcha_dataset*.py via ast.literal_eval.
# Identical to train_radar_fields.GOTCHA_BACKEND; asserted against it in the PVC tests.
GOTCHA_BACKEND = {
    "schema": "rift_gotcha_backend_v1", "method": "radar_fields", "callable": "run_gotcha",
    "selection_unit": "pass_sector", "joint_passes": True,
    "native_frequency_policy": "ragged_exact", "polarizations": ["hh", "hv", "vh", "vv"],
    "metric_domain": "normalized_dB_range_power_intensity",
}


def run_gotcha(*, dataset, output_dir, config, device, resume):
    from rift_pvc.radar_fields_gotcha import run_gotcha as backend
    return backend(dataset=dataset, output_dir=output_dir, config=config, device=device, resume=resume)


def install():
    """Rebind the accelerator-specific names (see rift_pvc.radar_fields_training)."""
    twins.install()
    return _rf


def check_backend():
    allowed = {"xpu"} | {b.strip() for b in os.environ.get("RIFT_PVC_ALLOW_BACKEND", "").split(",") if b.strip()}
    backend = accelerator.backend()
    if backend not in allowed:
        raise RuntimeError(
            f"train_radar_fields_pvc.py is the PVC entry point and found backend {backend!r}; "
            "use train_radar_fields.py on CUDA, or set RIFT_PVC_ALLOW_BACKEND=cpu|cuda to override")
    return backend


def _flag_value(argv, flag):
    """``(value, argv_without_flag)`` for ``--flag X`` or ``--flag=X`` (``None`` when absent)."""
    kept, value, index = [], None, 0
    while index < len(argv):
        token = argv[index]
        if token == flag and index + 1 < len(argv):
            value, index = argv[index + 1], index + 2
            continue
        if token.startswith(flag + "="):
            value, index = token.split("=", 1)[1], index + 1
            continue
        kept.append(token)
        index += 1
    return value, kept


def _cli_error(message: str):
    print(f"train_radar_fields_pvc.py: error: {message}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def parse_args(argv=None):
    """``train_radar_fields.parse_args`` with the PVC device default and backend id."""
    install()   # the final identity is validated through the rebound recipe_contract
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    device_value, _ = _flag_value(argv, "--device")
    if device_value is None:
        # The unchanged parser's default is "cuda" or "cpu"; neither names the
        # PVC card. Injecting the flag keeps the parser itself unchanged.
        device_value = str(accelerator.device())
        argv = argv + ["--device", device_value]
    device_type = torch.device(device_value).type
    requested, stripped = _flag_value(argv, "--model-backend")
    if requested == TORCHSHIM_BACKEND:
        argv = stripped   # not among the unchanged parser's choices; re-applied below
    elif requested == REAL_BACKEND and device_type != "cuda":
        _cli_error(f"--model-backend {REAL_BACKEND} is the real tiny-cuda-nn backend and needs CUDA; "
                   f"on PVC omit the flag or pass --model-backend {TORCHSHIM_BACKEND}")
    args = _rf.parse_args(argv)
    if requested == TORCHSHIM_BACKEND:
        if recipe_name(args) == _rf.LEGACY_RECIPE:
            _cli_error("legacy-v1 requires its original Torch implementation")
        args.model_backend = TORCHSHIM_BACKEND
    elif args.model_backend == REAL_BACKEND and device_type != "cuda":
        # The unchanged default for native recipes, translated to the PVC identity.
        args.model_backend = TORCHSHIM_BACKEND
    _rf.recipe_contract(args)   # the twin validates the final identity
    return args


def main(argv=None):
    """Copy of ``train_radar_fields.main`` (2026-09-21 checkout) with the PVC resume block."""
    backend = check_backend()
    install()
    print(f"train_radar_fields_pvc.py: {accelerator.describe()}", flush=True)
    if backend == "xpu":
        print("train_radar_fields_pvc.py: PYTORCH_DEBUG_XPU_FALLBACK="
              f"{os.environ.get('PYTORCH_DEBUG_XPU_FALLBACK', 'unset')} SYCL_CACHE_PERSISTENT="
              f"{os.environ.get('SYCL_CACHE_PERSISTENT', 'unset')}", flush=True)
    args = parse_args(argv)
    print(f"train_radar_fields_pvc.py: model backend {args.model_backend}"
          + (f" {json.dumps(tcnn_torch.identity())}" if args.model_backend == TORCHSHIM_BACKEND else ""), flush=True)
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume")
    if args.eval_only and args.diagnose_padded_roi:
        raise ValueError("--eval-only and --diagnose-padded-roi are mutually exclusive")
    if args.steps <= 0 or args.view_batch <= 0:
        raise ValueError("--steps and --view-batch must be positive")
    if native_recipe(args) and not args.sealed_protocol:
        raise ValueError("audited-v2 requires --sealed-protocol and an explicit role manifest")
    if native_recipe(args) and not args.resume:
        output = Path(args.checkpoint_root) / args.checkpoint_name
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("audited-v2 requires a fresh output directory or explicit matching resume")
    if args.sealed_protocol:
        if not args.sealed_split_manifest:
            raise ValueError("--sealed-protocol requires --sealed-split-manifest")
        if args.num_train <= 0 or args.num_val <= 0 or args.num_test <= 0:
            raise ValueError(
                "--sealed-protocol requires positive --num-train, --num-val, and --num-test"
            )
    elif args.sealed_split_manifest:
        raise ValueError("--sealed-split-manifest requires --sealed-protocol")

    # Inspect a resume checkpoint before any dataset path can take the legacy
    # eager branch.  This is intentionally CPU-mapped and metadata-only from
    # the training perspective; the normal device-mapped restore still occurs
    # below after all continuation gates pass.  In particular, a sealed
    # checkpoint resumed without --sealed-protocol fails before opening an NPZ
    # response, creating a stats cache, a checkpoint directory, or a model.
    resume_preflight_checkpoint = None
    if args.resume:
        resume_preflight_checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        validate_recipe_checkpoint(resume_preflight_checkpoint, args)
        twins.validate_shim_identity(resume_preflight_checkpoint, args)
        # Reject changed objective/architecture before normalization response reads.
        if native_recipe(args):
            saved_args = resume_preflight_checkpoint.get("args", {})
            for key in _RESUME_CONFIG_FIELDS:
                if key not in saved_args or not _compatible_value(saved_args[key], getattr(args, key)):
                    raise ValueError(f"audited resume configuration mismatch: {key}")
        preflight_sealed_resume_checkpoint(
            resume_preflight_checkpoint,
            sealed_protocol_requested=args.sealed_protocol,
        )

    if native_recipe(args):
        twins.check_model_backend(args)  # Dependencies must fail before any response scan.

    signal.signal(signal.SIGTERM, _rf.request_stop)
    signal.signal(signal.SIGINT, _rf.request_stop)
    twins.set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Radar Fields reference commit: {OFFICIAL_REFERENCE_COMMIT}")
    print("Auxiliary geometry: disabled; initialization: random model parameters")
    print(f"Using device: {device}")

    # The diagnostic and sealed paths resolve only the response header plus
    # pose metadata first.  In sealed mode, explicit role IDs are validated
    # against that header and then installed as a lazy capability before any
    # response payload (including normalization data) can be streamed.
    if args.object or recipe_name(args) == SOURCE_RECIPE:
        from rift.rift_dataset import load_object_contract
        public, contract = load_object_contract(args.npz_path, args.sealed_split_manifest)
        from rift.radar_fields_dataset import from_collection_arrays
        arrays = from_collection_arrays(public, contract)
    else:
        arrays = load_radar_fields_npz(
            args.npz_path,
            load_response=not (args.diagnose_padded_roi or args.sealed_protocol),
        )
    sealed_protocol_info = None
    if args.sealed_protocol:
        sealed_split = load_radar_fields_sealed_split_manifest(
            args.sealed_split_manifest,
            arrays.num_views,
            response_shape=arrays._response_shape(),
            response_dtype=arrays.response_dtype,
            expected_num_train=args.num_train,
            expected_num_val=args.num_val,
            expected_num_test=args.num_test,
        )
        train_indices = np.asarray(sealed_split.train_indices, dtype=np.int64)
        val_indices = np.asarray(sealed_split.validation_indices, dtype=np.int64)
        test_indices = np.asarray(sealed_split.test_indices, dtype=np.int64)
        arrays = restrict_radar_fields_response_views(
            arrays, np.concatenate((train_indices, val_indices))
        )
    else:
        train_indices, val_indices, test_indices = split_view_indices(
            arrays.num_views,
            args.num_train,
            args.num_val,
            args.num_test,
            args.seed,
            val_from_tail=args.val_from_tail,
        )
    split_info = role_provenance(
        train_indices,
        val_indices,
        test_indices,
        test_payload_materialized=arrays.response_is_materialized,
    )
    if args.sealed_protocol:
        sealed_protocol_info = sealed_protocol_contract(sealed_split, split_info)
        from rift.rift_dataset import validate_manifest_object
        with open(args.sealed_split_manifest, encoding="utf-8") as handle:
            identity = validate_manifest_object(json.load(handle), arrays.metadata)
        if identity is not None:
            sealed_protocol_info["dataset_identity"] = identity
            if arrays.acquisition_identity:
                sealed_protocol_info["acquisition_identity"] = arrays.acquisition_identity
    dataset_info = dataset_provenance(arrays)

    # Both a changed manifest role/policy and a sealed-versus-legacy resume
    # are rejected while the dataset is still a restricted lazy header handle.
    # Do this before a cache lookup/recalibration or any model construction.
    if resume_preflight_checkpoint is not None and args.sealed_protocol:
        preflight_sealed_resume_checkpoint(
            resume_preflight_checkpoint,
            sealed_protocol_requested=True,
            current_dataset=dataset_info,
            current_split=split_info,
            current_sealed_protocol=sealed_protocol_info,
        )
        if native_recipe(args):
            preflight_audited_continuation(resume_preflight_checkpoint, args, train_indices,
                                          dataset_info, split_info, sealed_protocol_info)
    # Avoid retaining a full CPU-mapped checkpoint while normalization and the
    # model are prepared; the existing device-mapped restore below remains the
    # sole checkpoint object used for training continuation.
    del resume_preflight_checkpoint

    total_pairs = arrays.num_tx * arrays.num_rx
    val_pairs = evenly_spaced_pairs(total_pairs, args.val_pairs)
    checkpoint_dir = Path(args.checkpoint_root) / args.checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats = load_or_create_stats(
        str(checkpoint_dir / "radar_fields_power_stats.json"),
        arrays,
        train_indices,
        dynamic_range_db=args.dynamic_range_db,
        max_views=args.stats_max_views,
        sealed_protocol=args.sealed_protocol,
        dataset_identity=(sealed_protocol_info.get("dataset_identity") if sealed_protocol_info else None),
    )
    support_grid_info = grid_support_provenance(args)
    signal_info = signal_provenance(stats, args)
    print(
        f"Dataset: {arrays.num_views} views, {arrays.num_tx}x{arrays.num_rx} pairs, "
        f"{arrays.num_freq} frequencies; range bin={range_bin_size(arrays.metadata):.6f} m"
    )
    print(
        f"Split: train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}; "
        f"normalized-dB range-power peak={float(stats['peak_power']):.6e}",
    )
    if sealed_protocol_info is not None:
        print(
            "Sealed response protocol: explicit manifest roles bound before payload access; "
            "only train and validation responses are authorized during development.",
            flush=True,
        )

    xyz = generate_dynamic_grid(args.granularity, args.extent, device, jitter=False).reshape(-1, 3)
    ranges = range_bin_centers(arrays.metadata, device=device,
                              dtype=torch.float64 if native_recipe(args) else torch.float32)
    model = twins.build_model(args, device)
    parameters = (model.get_params(args.lr) if recipe_name(args) == SOURCE_RECIPE
                  and hasattr(model, "get_params") else model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.99), eps=1.0e-15)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 0.1 ** min(step / (800 if recipe_name(args) == SOURCE_RECIPE else max(args.steps, 1)), 1.0)
    )
    rng = np.random.default_rng(args.seed)
    start_step = 0
    best_val = float("inf")
    history = []
    view_sampler = CoveredViewSampler(train_indices) if native_recipe(args) else None
    checkpoint = None
    resume_validation = {
        "persisted_args_checked": [],
        "strict_resume_contract": False,
        "split_provenance_verified": False,
        "legacy_contract_unverified": False,
        "continuation_state_verified": False,
        "cuda_rng_state_verified": False,
        "power_stats_verified": False,
        "dataset_provenance_verified": False,
        "dataset_identity_verified": False,
        "sealed_protocol_verified": False,
    }

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        resume_validation = twins.validate_resume_checkpoint(
            checkpoint,
            args,
            stats,
            dataset_info,
            split_info,
            resume_device=device,
            current_sealed_protocol=sealed_protocol_info,
        )
        model.load_state_dict(checkpoint["radar_fields_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = int(checkpoint["step"])
        if view_sampler is not None:
            if "training_view_coverage" not in checkpoint:
                raise ValueError("audited checkpoint lacks optimizer view exposure state")
            view_sampler = CoveredViewSampler(train_indices, checkpoint["training_view_coverage"])
            if sum(view_sampler.counts.values()) != start_step * args.view_batch:
                raise ValueError("coverage state disagrees with completed optimizer steps")
        best_val = float(checkpoint.get("best_val_rel_mse", checkpoint.get("loss", float("inf"))))
        history = list(checkpoint.get("history", []))
        if checkpoint.get("numpy_rng_state_json") is not None:
            rng.bit_generator.state = json.loads(checkpoint["numpy_rng_state_json"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        # PVC: the original restores cuda_rng_state for a CUDA device only; the
        # twin restores the payload of the active backend (xpu_rng_state on PVC).
        twins.restore_device_rng_state(checkpoint, device, resume_validation)
        if resume_validation["legacy_contract_unverified"]:
            print(
                "WARNING: resumed a legacy checkpoint without a strict split/objective "
                "continuation contract; exact split provenance is not certified.",
                flush=True,
            )
        print(f"Resumed Radar Fields from step {start_step}, best val rel-MSE={best_val:.6e}")

    checkpoint_info = checkpoint_provenance(args.resume, checkpoint)

    if args.diagnose_padded_roi:
        if arrays.response_is_materialized:
            raise RuntimeError("diagnostic mode must not retain the full response payload")
        diagnostic_view = resolve_diagnostic_view(
            args.diagnostic_role,
            args.diagnostic_role_index,
            train_indices,
            val_indices,
        )
        diagnostic_pairs = evenly_spaced_pairs(total_pairs, args.diagnostic_pairs)
        was_training = model.training
        model.eval()  # do not update batch-norm state during a measurement-only probe
        _loss, _terms, metrics = view_objective(
            model,
            arrays,
            diagnostic_view,
            diagnostic_pairs,
            xyz,
            ranges,
            stats,
            args,
            mask_progress=1.0,
            device=device,
            diagnostic_padded_roi=True,
        )
        record = {
            "artifact_schema": "radar_fields_padded_roi_v2",
            "diagnostic": "padded_roi_existing_objective_measurement",
            "selected_role": args.diagnostic_role,
            "selected_role_index": int(args.diagnostic_role_index),
            "selected_view_id": diagnostic_view,
            "pair_count": int(diagnostic_pairs.size),
            "dataset": dataset_info,
            "checkpoint": checkpoint_info,
            "checkpoint_compatibility": resume_validation,
            "split": split_info,
            "support_grid": support_grid_info,
            "signal": signal_info,
            "sealed_protocol": sealed_protocol_info,
            "test_response_payload_materialized": False,
            **metrics,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        if args.diagnostic_json:
            output = Path(args.diagnostic_json)
            if output.exists():
                raise FileExistsError(
                    f"refusing to overwrite an existing diagnostic artifact: {output}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, output)
            print(f"wrote {output}", flush=True)
        if was_training:
            model.train()
        return

    if args.eval_only:
        metrics = twins.evaluate(model, arrays, val_indices, val_pairs, xyz, ranges, stats, args, device)
        print(
            f"Validation ({NORMALIZED_DB_RELMSE_LABEL}): rel-MSE={metrics['rel_mse']:.6%} RMSE={metrics['rmse']:.6f} "
            f"PSNR={metrics['psnr_db']:.3f} dB"
        )
        output = checkpoint_dir / "radar_fields_eval.json"
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "artifact_schema": "radar_fields_evaluation_v2",
                    "metric_domain": NORMALIZED_DB_INTENSITY_DOMAIN,
                    "reported_rel_mse_label": NORMALIZED_DB_RELMSE_LABEL,
                    "dataset": dataset_info,
                    "checkpoint": checkpoint_info,
                    "checkpoint_compatibility": resume_validation,
                    "split": split_info,
                    "support_grid": support_grid_info,
                    "signal": signal_info,
                    "sealed_protocol": sealed_protocol_info,
                    **metrics,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        print(f"wrote {output}")
        return

    model.train()
    for step_zero in range(start_step, args.steps):
        step = step_zero + 1
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        batch_views = (view_sampler.next(args.view_batch, rng, source_sampling=recipe_name(args) == SOURCE_RECIPE) if view_sampler is not None else
                       rng.choice(train_indices, size=args.view_batch, replace=len(train_indices) < args.view_batch))
        records = []
        term_totals = {"fft": 0.0, "occupancy": 0.0, "bimodal": 0.0}
        mask_progress = min(1.0, 0.05 + math.sin(step / max(args.steps - 1, 1) * math.pi / 2.0))

        if recipe_name(args) == SOURCE_RECIPE:
            batches = len(train_indices) // args.view_batch
            epoch = step_zero // batches + 1
            mask_progress = min(1.0, .05 + math.sin(epoch / (args.steps // batches - 1) * math.pi / 2))
        native_records = []
        for view in batch_views:
            pairs = (original_module("radarfields.sampler").get_azimuths(
                1, args.train_pairs, total_pairs, device).cpu().numpy()[0]
                if recipe_name(args) == SOURCE_RECIPE else sample_pairs(rng, total_pairs, args.train_pairs))
            if view_sampler is not None:
                record = audited_view_tensors(model, arrays, int(view), pairs, ranges, stats,
                                               args, mask_progress, device, defer_render=True)
                native_records.append(record)
                continue
            loss, terms, metrics = view_objective(
                model,
                arrays,
                int(view),
                pairs,
                xyz,
                ranges,
                stats,
                args,
                mask_progress=mask_progress,
                device=device,
            )
            (loss / args.view_batch).backward()
            records.append(metrics)
            for key in term_totals:
                term_totals[key] += float(terms[key].detach()) / args.view_batch

        if native_records:
            fields = render_bistatic_batch(model, [record["geometry"] for record in native_records],
                                           query_chunk=args.query_chunk, mask_progress=mask_progress)
            native_records = [finish_audited_view(record, field, args)
                              for record, field in zip(native_records, fields)]
            records = [intensity_metrics(record["prediction"].detach(), record["target"].detach())
                       for record in native_records]
            loss, terms = released_batch_loss(native_records, weight_fft=args.weight_fft,
                                             weight_occ=args.weight_occ, weight_bimodal=args.weight_bimodal,
                                             source_exact=recipe_name(args) == SOURCE_RECIPE)
            loss.backward()
            for record in records:
                record["loss"] = float(loss.detach())
            term_totals = {key: float(value.detach()) for key, value in terms.items()}

        if recipe_name(args) == SOURCE_RECIPE and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite released RF loss gradient; no silent numerical repair")
        optimizer.step()
        scheduler.step()
        train_metrics = combine_metrics(records)
        if view_sampler is not None:
            print(f"  optimizer coverage: unique={sum(n > 0 for n in view_sampler.counts.values())}/"
                  f"{len(train_indices)} min/max exposures={min(view_sampler.counts.values())}/"
                  f"{max(view_sampler.counts.values())}", flush=True)
        elapsed = time.time() - started
        print(
            f"Step [{step}/{args.steps}] loss={train_metrics['loss']:.6e} "
            f"normalized-dB intensity rel-MSE={train_metrics['rel_mse']:.4%} "
            f"fft={term_totals['fft']:.3e} occ={term_totals['occupancy']:.3e} "
            f"bim={term_totals['bimodal']:.3e} [{elapsed:.1f}s/step]",
            flush=True,
        )

        validation = None
        # A preemption checkpoint takes priority over validation.  The Slurm
        # signal window must never be consumed by a 200-view benchmark pass.
        should_evaluate = (step % args.eval_every == 0 or step == args.steps) and not _rf.STOP_REQUESTED
        if should_evaluate:
            validation = twins.evaluate(
                model, arrays, val_indices, val_pairs, xyz, ranges, stats, args, device
            )
            if validation["complete"]:
                print(
                    f"Validation [step {step}] normalized-dB intensity rel-MSE={validation['rel_mse']:.6%} "
                    f"RMSE={validation['rmse']:.6f} PSNR={validation['psnr_db']:.3f} dB",
                    flush=True,
                )
                row = {
                    "step": step,
                    "metric_domain": NORMALIZED_DB_INTENSITY_DOMAIN,
                    "train_rel_mse_label": NORMALIZED_DB_RELMSE_LABEL,
                    "val_rel_mse_label": NORMALIZED_DB_RELMSE_LABEL,
                    "train_loss": train_metrics["loss"],
                    "train_rel_mse": train_metrics["rel_mse"],
                    "val_rel_mse": validation["rel_mse"],
                    "val_rmse": validation["rmse"],
                    "val_psnr_db": validation["psnr_db"],
                }
                history.append(row)
                save_history(history, checkpoint_dir / "radar_fields_history.csv")

                if validation["rel_mse"] < best_val:
                    best_val = validation["rel_mse"]
                    payload = twins.checkpoint_payload(
                        model, optimizer, scheduler, step, best_val, xyz, stats, rng, history, args,
                        dataset_info=dataset_info,
                        split_info=split_info,
                        support_grid_info=support_grid_info,
                        signal_info=signal_info,
                        sealed_protocol_info=sealed_protocol_info,
                        view_sampler=view_sampler,
                    )
                    atomic_torch_save(payload, checkpoint_dir / "checkpoint_best.pth.tar")

        should_checkpoint = step % args.checkpoint_every == 0 or should_evaluate or _rf.STOP_REQUESTED
        if should_checkpoint:
            payload = twins.checkpoint_payload(
                model, optimizer, scheduler, step, best_val, xyz, stats, rng, history, args,
                dataset_info=dataset_info,
                split_info=split_info,
                support_grid_info=support_grid_info,
                signal_info=signal_info,
                sealed_protocol_info=sealed_protocol_info,
                view_sampler=view_sampler,
            )
            name = "checkpoint_final.pth.tar" if step == args.steps else "checkpoint_latest.pth.tar"
            atomic_torch_save(payload, checkpoint_dir / name)

        if _rf.STOP_REQUESTED:
            print("Stopped cleanly after publishing checkpoint_latest.", flush=True)
            return


if __name__ == "__main__":
    # Same convention as train_radar_fields.py: main() is called for its effects.
    main()
