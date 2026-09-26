#!/usr/bin/env python
"""Entrypoint and continuation regression gates for sealed ordinary-RIFT NPZ runs.

This creates a tiny synthetic NPZ archive and role manifest in a temporary
directory.  It does not open a PACE dataset, submit a job, or retain an
artifact.  Run it in an environment with the normal RIFT PyTorch dependency:

    python scripts/validate_train_npz_sealed_protocol.py

The existing lazy reader may consume/discard archive bytes while seeking an
authorized source row.  These gates deliberately assert the actual contract:
the training entrypoint never creates, exposes, or tensors a reserved-test or
unused response payload.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rift.npz_dataset as npz_dataset  # noqa: E402
import train as rift_train  # noqa: E402


CHECKS = 0


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)
    print(f"PASS {CHECKS:02d}: {message}")


def expect_value_error(action, required_text, message):
    try:
        action()
    except ValueError as exc:
        rejected = required_text in str(exc)
    else:
        rejected = False
    check(rejected, message)


def synthetic_npz(path: Path):
    n_views, n_freq = 6, 4
    values = np.arange(n_views * n_freq, dtype=np.float32).reshape(n_views, 1, 1, 1, n_freq)
    response = (0.05 + values + 1j * (0.25 + values)).astype(np.complex64)
    angles = np.linspace(0.3, 1.3, n_views, dtype=np.float32)
    viewpoints = np.stack((10.0 * np.cos(angles), 10.0 * np.sin(angles), np.ones_like(angles)), axis=1)
    tx_pos = viewpoints[:, None, :].copy()
    rx_pos = viewpoints[:, None, :].copy()
    rx_pos[..., 1] += 0.01
    metadata = {
        "radar_fc_hz": 10.0e9,
        "radar_bandwidth_hz": 1.0e9,
        "num_adc_samples": n_freq,
        "target_radius_m": 0.1,
    }
    np.savez_compressed(
        path,
        response=response,
        metadata_json=np.asarray(json.dumps(metadata)),
        viewpoint_positions=viewpoints,
        tx_pos=tx_pos,
        rx_pos=rx_pos,
    )
    return response.shape


def manifest_payload(shape):
    return {
        "schema_version": 1,
        "name": "synthetic_sealed_entrypoint_v1",
        "dataset": {
            "num_views": int(shape[0]),
            "response_shape": [int(value) for value in shape],
            "response_dtype": "complex64",
            # Historical provenance only: the sealed protocol validates this
            # archive's header and explicit role IDs, not a stale path/hash.
            "path_hint": "/historical/obsolete/source.npz",
            "sha256": "informational-only",
        },
        "split": {
            "strategy": "fixed_tail_subsampled",
            "num_train": 2,
            "num_validation": 1,
            "num_test": 1,
            "num_unused": 2,
            "complete_partition": True,
            "test_sealed": True,
            "unused_sealed": True,
            "indices_sha256": "informational-only",
            # Deliberately not sorted: the train dataset must retain manifest
            # order while the lazy reader may stream in source-ID order.
            "train_indices": [4, 1],
            "validation_indices": [5],
            "test_indices": [0],
            "unused_indices": [2, 3],
        },
    }


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stage_cli_defaults():
    adaptive = rift_train.parse_args(["--checkpoint-name", "adaptive"])
    check(
        adaptive.architecture == "adaptive" and adaptive.adaptive_capacity_v2
        and adaptive.npz_sealed_protocol and adaptive.npz_role_manifest is None,
        "adaptive default seals NPZ access and still requires a caller-supplied manifest/object",
    )
    legacy = rift_train.parse_args(["--checkpoint-name", "legacy", "--scene-repr", "grid_sh"])
    check(
        legacy.npz_sealed_protocol is False and legacy.npz_role_manifest is None,
        "explicit legacy scene selection keeps the legacy NPZ loader default",
    )
    sealed = rift_train.parse_args([
        "--checkpoint-name", "sealed",
        "--scene-repr", "grid_sh",
        "--data-format", "npz",
        "--npz-sealed-protocol",
        "--npz-role-manifest", "roles.json",
    ])
    check(
        sealed.npz_sealed_protocol and sealed.npz_role_manifest == "roles.json",
        "legacy architectures can still opt in explicitly to the sealed protocol",
    )


def sealed_cli(npz_path, manifest_path, checkpoint_root, checkpoint_name, *, epochs,
               adaptive=False, resume=None, sealed=True):
    """Return a deliberately tiny real ``train.py`` command.

    The only test instrumentation below pins the process to CPU and records
    access/call boundaries.  It does not substitute the sealed loader, model,
    optimizer, renderer, trainer, evaluator, or checkpoint writer.
    """
    argv = [
        "--checkpoint-name", checkpoint_name,
        "--checkpoint-root", str(checkpoint_root),
        "--data-format", "npz",
        "--npz-path", str(npz_path),
        "--num-train", "2",
        "--num-val", "1",
        "--num-test", "1",
        "--epochs", str(epochs),
        "--num-freq-wanted", "4",
        "--scene-repr", "point_sh" if adaptive else "grid",
        "--granularity", "2",
        "--extent", "0.1",
        "--num-rx", "1",
        "--num-tx", "1",
        "--bp-init", "0",
        "--init-scale", "0.01",
        "--no-learn-gain",
        "--forward-operator", "brute",
        "--step-every", "1",
        "--lr", "0.001",
        "--adam-eps", "1e-20",
        "--t0", "1",
        "--t-mult", "1",
    ]
    if sealed:
        argv.extend([
            "--npz-sealed-protocol",
            "--npz-role-manifest", str(manifest_path),
        ])
    if adaptive:
        # One small joint decision is enough to exercise the adaptive
        # checkpoint payload.  Both fractions are zero, so no test run spends
        # capacity on a topology change; it still collects the real data-fit
        # and next-band probe gradients.
        argv.extend([
            "--sh-max-degree", "1",
            "--sh-init-degree", "0",
            "--adaptive-capacity-v2",
            "--adaptive-refine-every", "1",
            "--adaptive-probe-every", "1",
            "--adaptive-spatial-fraction", "0",
            "--adaptive-angular-fraction", "0",
            "--max-points", "8",
            # Keep the contract independent of the total epoch target during
            # the one-epoch -> two-epoch continuation below.
            "--prune-end-epoch", "1",
        ])
    if resume is not None:
        argv.extend(["--resume", str(resume)])
    return argv


def new_trace():
    return {
        "metadata_load_response_flags": [],
        "response_requests": [],
        "build_attempts": [],
        "build_calls": [],
        "train_calls": [],
        "evaluate_calls": [],
    }


def run_main_with_trace(argv, trace=None):
    """Run the actual CLI while recording its sealed-data boundaries.

    CPU selection is the sole environment shim.  This makes the regression
    runnable on GPU workers without requiring a GPU allocation; all RIFT
    loader/model/render/training/checkpoint code remains the production code.
    """
    trace = new_trace() if trace is None else trace
    original_init_distributed = rift_train.init_distributed
    original_load_arrays = rift_train.load_npz_arrays
    original_iterator = npz_dataset.iter_npz_response_views
    original_build = rift_train.build_sealed_npz_dataloaders
    original_train = rift_train.train_sar
    original_evaluate = rift_train.evaluate

    def cpu_single_process():
        return 0, 1, torch.device("cpu")

    def recording_load_arrays(*args, **kwargs):
        trace["metadata_load_response_flags"].append(kwargs.get("load_response", True))
        return original_load_arrays(*args, **kwargs)

    def recording_iterator(arrays, requested):
        requested = tuple(int(value) for value in requested)
        trace["response_requests"].append(requested)
        yield from original_iterator(arrays, requested)

    def recording_build(*args, **kwargs):
        trace["build_attempts"].append({
            "num_train": kwargs.get("num_train"),
            "num_val": kwargs.get("num_val"),
            "num_test": kwargs.get("num_test"),
            "resume_contract": kwargs.get("resume_sealed_npz_protocol_contract"),
        })
        result = original_build(*args, **kwargs)
        train_loader, validation_loader, test_loader, contract = result
        trace["build_calls"].append({
            "train_indices": [int(value) for value in train_loader.dataset.indices],
            "validation_indices": [int(value) for value in validation_loader.dataset.indices],
            "test_loader_is_none": test_loader is None,
            "contract": contract,
        })
        return result

    def recording_train(*args, **kwargs):
        model = args[1]
        before = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if torch.is_floating_point(value) or torch.is_complex(value)
        }
        result = original_train(*args, **kwargs)
        changed = any(
            not torch.equal(before[name], model.state_dict()[name])
            for name in before
        )
        training_history, validation_history = result
        trace["train_calls"].append({
            "num_epochs": int(args[0]),
            "train_indices": [int(value) for value in args[2].dataset.indices],
            "validation_indices": [int(value) for value in args[3].dataset.indices],
            "state_changed": changed,
            "history_lengths": (len(training_history), len(validation_history)),
            "sealed_contract": kwargs.get("sealed_npz_protocol_contract"),
        })
        return result

    def recording_evaluate(*args, **kwargs):
        trace["evaluate_calls"].append({
            "validation_indices": [int(value) for value in args[1].dataset.indices],
            "data_format": kwargs.get("data_format"),
        })
        return original_evaluate(*args, **kwargs)

    rift_train.init_distributed = cpu_single_process
    rift_train.load_npz_arrays = recording_load_arrays
    npz_dataset.iter_npz_response_views = recording_iterator
    rift_train.build_sealed_npz_dataloaders = recording_build
    rift_train.train_sar = recording_train
    rift_train.evaluate = recording_evaluate
    try:
        rift_train.main(argv)
    finally:
        rift_train.init_distributed = original_init_distributed
        rift_train.load_npz_arrays = original_load_arrays
        npz_dataset.iter_npz_response_views = original_iterator
        rift_train.build_sealed_npz_dataloaders = original_build
        rift_train.train_sar = original_train
        rift_train.evaluate = original_evaluate
    return trace


def check_real_sealed_trace(trace, payload, *, label, epochs, resumed=False):
    """Assert that an unmodified ``main`` call used only authorized views."""
    roles = payload["split"]
    authorized = set(roles["train_indices"]) | set(roles["validation_indices"])
    reserved = set(roles["test_indices"]) | set(roles["unused_indices"])
    requested = {view_id for call in trace["response_requests"] for view_id in call}
    check(
        trace["metadata_load_response_flags"]
        and all(flag is False for flag in trace["metadata_load_response_flags"]),
        f"{label}: main reaches the NPZ archive through metadata/header-only access",
    )
    check(
        len(trace["build_calls"]) == 1
        and trace["build_calls"][0]["train_indices"] == roles["train_indices"]
        and trace["build_calls"][0]["validation_indices"] == roles["validation_indices"]
        and trace["build_calls"][0]["test_loader_is_none"],
        f"{label}: main constructs the real manifest-bound train/validation loaders and no test loader",
    )
    check(
        requested == authorized and requested.isdisjoint(reserved),
        f"{label}: no reserved-test or unused response is materialized",
    )
    check(
        len(trace["train_calls"]) == 1
        and trace["train_calls"][0]["num_epochs"] == epochs
        and trace["train_calls"][0]["state_changed"]
        and trace["train_calls"][0]["sealed_contract"] is not None,
        f"{label}: main performs a real sealed model/optimizer update",
    )
    expected_history = 1 if resumed else epochs
    check(
        trace["train_calls"][0]["history_lengths"] == (expected_history, expected_history)
        and len(trace["evaluate_calls"]) == expected_history,
        f"{label}: main performs the real validation pass for every executed epoch",
    )


def stage_basic_main_training(root, npz_path, manifest_path, payload):
    """Exercise the ordinary grid trainer through its public CLI."""
    checkpoint_root = root / "grid_checkpoints"
    checkpoint_name = "sealed_grid_fresh"
    trace = run_main_with_trace(sealed_cli(
        npz_path, manifest_path, checkpoint_root, checkpoint_name, epochs=1))
    check_real_sealed_trace(
        trace, payload, label="ordinary sealed grid run", epochs=1)

    final_path = checkpoint_root / checkpoint_name / "checkpoint_final.pth.tar"
    check(final_path.is_file(), "ordinary sealed grid run writes its real final checkpoint")
    saved = rift_train.load_tensor_checkpoint(final_path, map_location="cpu")
    contract = trace["build_calls"][0]["contract"]
    check(
        saved.get("sealed_npz_protocol_contract") == contract
        and saved.get("sealed_npz_protocol_contract", {}).get("response_access", {}).get(
            "reserved_test_materialized") is False,
        "ordinary RIFT checkpoint persists the complete sealed response-access contract",
    )
    return final_path, saved


def stage_cli_rejections(root, npz_path, manifest_path, final_path, saved):
    """Keep pre-loader rejection paths covered without stubbing ``main``."""
    legacy_state = dict(saved)
    legacy_state.pop("sealed_npz_protocol_contract")
    legacy_path = root / "legacy_checkpoint.pth.tar"
    torch.save(legacy_state, legacy_path)

    def reject_without_response(argv, required_text, message):
        trace = new_trace()
        expect_value_error(
            lambda: run_main_with_trace(argv, trace), required_text, message)
        check(
            not trace["metadata_load_response_flags"] and not trace["response_requests"],
            f"{message}: command reaches no NPZ response access",
        )

    legacy_argv = sealed_cli(
        npz_path, manifest_path, root / "invalid", "legacy_resume", epochs=1,
        resume=final_path, sealed=False)
    reject_without_response(
        legacy_argv,
        "cannot resume through a legacy",
        "main rejects a sealed checkpoint before its legacy NPZ loader can run",
    )
    sealed_legacy_argv = sealed_cli(
        npz_path, manifest_path, root / "invalid", "sealed_legacy_resume", epochs=1,
        resume=legacy_path)
    reject_without_response(
        sealed_legacy_argv,
        "sealed NPZ resume requires a checkpoint",
        "main rejects a legacy checkpoint before its sealed NPZ loader can run",
    )
    reject_without_response(
        [
            "--checkpoint-name", "bad_missing_manifest",
            "--data-format", "npz",
            "--npz-sealed-protocol",
        ],
        "requires --npz-role-manifest",
        "main rejects sealed mode without an explicit role manifest before data loading",
    )
    reject_without_response(
        [
            "--checkpoint-name", "bad_format",
            "--data-format", "csv",
            "--npz-sealed-protocol",
            "--npz-role-manifest", str(manifest_path),
        ],
        "requires --data-format npz",
        "main rejects sealed mode outside NPZ before data loading",
    )
    reject_without_response(
        sealed_cli(npz_path, manifest_path, root / "invalid", "legacy_split_flag", epochs=1)
        + ["--val-from-tail"],
        "do not also pass",
        "main rejects legacy split flags that conflict with manifest roles",
    )


def stage_adaptive_relocation(root, npz_path, manifest_path, payload):
    """Exercise fresh, relocated, and incompatible sealed adaptive CLI runs."""
    fresh_root = root / "adaptive_fresh"
    fresh_name = "sealed_adaptive_fresh"
    fresh_trace = run_main_with_trace(sealed_cli(
        npz_path, manifest_path, fresh_root, fresh_name, epochs=1, adaptive=True))
    check_real_sealed_trace(
        fresh_trace, payload, label="sealed adaptive-v2 fresh run", epochs=1)

    final_path = fresh_root / fresh_name / "checkpoint_final.pth.tar"
    saved = rift_train.load_tensor_checkpoint(final_path, map_location="cpu")
    check(
        saved.get("adaptive_capacity_v2") is True
        and isinstance(saved.get("adaptive_training_contract"), dict)
        and saved.get("rng_state", {}).get("version") == 2,
        "sealed adaptive-v2 CLI checkpoint retains its real topology, contract, and recovery RNG state",
    )

    # This direct check protects the historical behavior while the next two
    # calls prove the sealed exception through the actual CLI continuation.
    legacy_checkpoint = dict(saved)
    legacy_checkpoint.pop("sealed_npz_protocol_contract")
    changed_legacy_contract = copy.deepcopy(saved["adaptive_training_contract"])
    changed_legacy_contract["observations"]["train"]["source_path"] = "/other/legacy_source.npz"
    expect_value_error(
        lambda: rift_train.validate_adaptive_resume_config(
            legacy_checkpoint,
            legacy_checkpoint["adaptive_refinement"],
            expected_training_contract=changed_legacy_contract,
        ),
        "adaptive-capacity-v2 resume",
        "nonsealed adaptive recovery still treats a changed source path as trajectory-defining",
    )

    relocated_dir = root / "relocated"
    relocated_dir.mkdir()
    relocated_npz = relocated_dir / "renamed_tiny.npz"
    relocated_manifest = relocated_dir / "renamed_roles.json"
    shutil.copyfile(npz_path, relocated_npz)
    relocated_payload = copy.deepcopy(payload)
    relocated_payload["name"] = "same_roles_relocated_manifest"
    write_json(relocated_manifest, relocated_payload)

    resumed_root = root / "adaptive_relocated"
    resumed_name = "sealed_adaptive_relocated"
    resumed_trace = run_main_with_trace(sealed_cli(
        relocated_npz,
        relocated_manifest,
        resumed_root,
        resumed_name,
        epochs=2,
        adaptive=True,
        resume=final_path,
    ))
    check_real_sealed_trace(
        resumed_trace,
        payload,
        label="sealed adaptive-v2 relocated continuation",
        epochs=2,
        resumed=True,
    )
    resumed_final_path = resumed_root / resumed_name / "checkpoint_final.pth.tar"
    resumed_saved = rift_train.load_tensor_checkpoint(resumed_final_path, map_location="cpu")
    check(
        resumed_saved["epoch"] == 2
        and resumed_saved["sealed_npz_protocol_contract"]["source_path"]
        == str(relocated_npz.resolve())
        and resumed_saved["adaptive_training_contract"]["observations"]["train"]["source_path"]
        == str(relocated_npz.resolve()),
        "relocated sealed adaptive continuation runs and records its new paths as provenance",
    )
    check(
        any(
            not torch.equal(saved["model_state_dict"][name], resumed_saved["model_state_dict"][name])
            for name in ("w_re", "w_im", "delta_raw")
        ),
        "relocated sealed adaptive continuation performs a real post-recovery optimizer update",
    )

    incompatible_payload = copy.deepcopy(relocated_payload)
    incompatible_payload["name"] = "incompatible_ordered_roles"
    incompatible_payload["split"]["train_indices"] = [1, 4]
    incompatible_manifest = relocated_dir / "incompatible_roles.json"
    write_json(incompatible_manifest, incompatible_payload)
    incompatible_trace = new_trace()
    try:
        run_main_with_trace(sealed_cli(
            relocated_npz,
            incompatible_manifest,
            root / "adaptive_incompatible",
            "sealed_adaptive_incompatible",
            epochs=2,
            adaptive=True,
            resume=final_path,
        ), incompatible_trace)
    except ValueError as exc:
        incompatible_rejected = "sealed NPZ resume" in str(exc)
    else:
        incompatible_rejected = False
    check(
        incompatible_rejected,
        "actual main rejects an adaptive continuation with reordered sealed train IDs",
    )
    check(
        incompatible_trace["metadata_load_response_flags"] == [False]
        and not incompatible_trace["response_requests"]
        and not incompatible_trace["build_calls"]
        and not incompatible_trace["train_calls"]
        and not incompatible_trace["evaluate_calls"],
        "incompatible sealed adaptive resume stops after metadata binding and before any payload/model work",
    )


def main():
    stage_cli_defaults()
    with tempfile.TemporaryDirectory(prefix="rift_train_npz_sealed_") as temporary:
        root = Path(temporary)
        npz_path = root / "tiny.npz"
        shape = synthetic_npz(npz_path)
        manifest_path = root / "roles.json"
        payload = manifest_payload(shape)
        write_json(manifest_path, payload)
        final_path, saved = stage_basic_main_training(root, npz_path, manifest_path, payload)
        stage_cli_rejections(root, npz_path, manifest_path, final_path, saved)
        stage_adaptive_relocation(root, npz_path, manifest_path, payload)
    print(f"PASS: {CHECKS} train.py sealed-NPZ entrypoint/continuation checks")


if __name__ == "__main__":
    main()
