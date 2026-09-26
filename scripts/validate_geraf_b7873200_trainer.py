#!/usr/bin/env python
"""Helper-level Torch checks for the corrected GeRaF B7873200 runner.

This uses tiny CPU-only tensors and no B787 archive.  It tests pure pending
validation and finite-state helpers only.  The separate
``validate_geraf_b7873200_entrypoint.py`` test covers real checkpoint writing,
restore, and entrypoint interruption control flow.
"""

from __future__ import annotations

import copy
import io
import importlib.util
import os
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from rift.geraf_b7873200_adapter import B787ResponseHeader
from rift.geraf_b7873200_source import B7873200MetadataArrays


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _validation_tmp_parent() -> Path:
    configured = os.environ.get("RIFT_VALIDATION_TMPDIR")
    parent = ROOT if configured is None else Path(configured).expanduser().resolve()
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise RuntimeError(f"no usable validation temporary parent: {parent}")
    return parent

_TRAINER_SPEC = importlib.util.spec_from_file_location(
    "train_geraf_b7873200_under_test", ROOT / "train_geraf.py"
)
if _TRAINER_SPEC is None or _TRAINER_SPEC.loader is None:
    raise RuntimeError("cannot load the corrected B7873200 trainer")
trainer = importlib.util.module_from_spec(_TRAINER_SPEC)
sys.modules[_TRAINER_SPEC.name] = trainer
_TRAINER_SPEC.loader.exec_module(trainer)


class Gates:
    def __init__(self) -> None:
        self.count = 0

    def check(self, condition: bool, detail: str) -> None:
        if not condition:
            raise AssertionError(detail)
        self.count += 1
        print(f"PASS {self.count:02d}: {detail}", flush=True)


def _expect_floating_point(gates: Gates, detail: str, action: object) -> None:
    try:
        action()  # type: ignore[operator]
    except FloatingPointError:
        gates.check(True, detail)
    else:
        raise AssertionError(detail)


class FiniteForwardInfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, values: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.shape = tuple(values.shape)  # type: ignore[attr-defined]
        ctx.dtype = values.dtype  # type: ignore[attr-defined]
        ctx.device = values.device  # type: ignore[attr-defined]
        return values.new_tensor(1.0)

    @staticmethod
    def backward(ctx: object, upstream: torch.Tensor) -> tuple[torch.Tensor]:  # type: ignore[override]
        del upstream
        return (
            torch.full(  # type: ignore[attr-defined]
                ctx.shape, float("inf"), dtype=ctx.dtype, device=ctx.device  # type: ignore[attr-defined]
            ),
        )


def _commit_from_saved_pending(
    history: list[dict[str, object]],
    best: float,
    *,
    step: int,
    metrics: dict[str, float],
    interrupted_after_views: int | None,
) -> tuple[list[dict[str, object]], float, bool, int]:
    pending = trainer._begin_pending_validation(step, None)
    completed_views = 3 if interrupted_after_views is None else int(interrupted_after_views)
    if completed_views < 0 or completed_views > 3:
        raise ValueError("synthetic validation interruption must lie in [0, 3]")
    saved_history = copy.deepcopy(history)
    saved_best = float(best)
    if completed_views < 3:
        # This is the exact checkpoint payload at a cooperative stop before
        # validation, or after a partial validation.  The production runner
        # restarts validation from the frozen model state on resume.
        serialized = io.BytesIO()
        torch.save(
            {
                "step": step,
                "pending_validation_step": pending,
                "history": saved_history,
                "best_val_mse": saved_best,
            },
            serialized,
        )
        serialized.seek(0)
        checkpoint = torch.load(serialized, map_location="cpu", weights_only=False)
        if checkpoint["pending_validation_step"] != step or checkpoint["step"] != step:
            raise AssertionError("pending validation did not survive checkpoint serialization")
        pending = checkpoint["pending_validation_step"]
        saved_history = checkpoint["history"]
        saved_best = checkpoint["best_val_mse"]
    pending, saved_best, improved = trainer._commit_pending_validation(
        step=step,
        pending_validation_step=pending,
        metrics=metrics,
        history=saved_history,
        best_val_mse=saved_best,
    )
    if pending is not None:
        raise AssertionError("complete validation must clear its pending checkpoint state")
    return saved_history, saved_best, improved, int(improved)


def _valid_cli() -> SimpleNamespace:
    return SimpleNamespace(
        seed=trainer.B787_3200_SEED,
        phase_sign=-1.0,
        scene_extent=0.15,
        n_azimuth=32,
        n_elevation=32,
        n_depth=32,
        aperture_scale=1.0,
        sdf_softplus_beta=100.0,
        reflectivity_softplus_beta=1.0,
        init_tx_amplitude=1.0,
        init_inv_s=64.0,
        directional_exponent=1.0,
        min_distance=1.0e-6,
        steps=100,
        sdf_lr=1.0e-4,
        other_lr=1.0e-3,
        weight_decay=1.0e-2,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
        cosine_min_lr=0.0,
        gradient_clip_norm=0.0,
        mask_high_threshold=0.1,
        mask_low_ratio=0.1,
        mask_low_threshold=0.0,
        oversample=2,
        kernel_width=20,
        pair_chunk=32,
        point_chunk=4096,
        validation_every=10,
        checkpoint_every=10,
        checkpoint_seconds=600.0,
        log_every=10,
        sdf_levels=10,
    )


def _synthetic_target_recipe() -> dict[str, object]:
    return {
        "schema": trainer.B787_3200_CACHE_SCHEMA,
        "version": 1,
        "target_spec": {
            "native_readout": "complex magnitude |MF|",
            "phase_sign": -1.0,
            "backend": "range",
            "compute_dtype": "float32",
            "grid": {
                "scene_extent_m": 0.15,
                "n_azimuth": 1,
                "n_elevation": 1,
                "n_depth": 1,
                "aperture_scale": 1.0,
            },
            "operator": {
                "implementation": "range_nufft",
                "range_model": "none",
                "include_four_pi": False,
                "freq_chunk": None,
                "kernel_width": 4,
                "oversample": 2,
                "point_chunk": 1,
                "pair_chunk": 1,
            },
        },
    }


def _synthetic_metadata_arrays() -> B7873200MetadataArrays:
    viewpoints = np.asarray(((10.0, 0.0, 0.0), (0.0, 10.0, 0.0)), dtype=np.float64)
    coordinate = np.linspace(-0.05, 0.05, 16, dtype=np.float64)
    tx = np.zeros((2, 16, 3), dtype=np.float64)
    rx = np.zeros((2, 16, 3), dtype=np.float64)
    tx[:, :, 1] = coordinate
    rx[:, :, 2] = coordinate
    tx += viewpoints[:, None, :]
    rx += viewpoints[:, None, :]
    return B7873200MetadataArrays(
        path="synthetic-never-opened.npz",
        response=B787ResponseHeader((10_000, 16, 16, 1, 600), np.dtype(np.complex64)),
        viewpoint_positions=viewpoints,
        tx_pos=tx,
        rx_pos=rx,
        metadata={"synthetic": True},
    )


def _write_synthetic_preflight_target(
    root: Path,
    *,
    index: int,
    role: str,
    magnitude: float,
    arrays: B7873200MetadataArrays,
    args: SimpleNamespace,
    recipe: dict[str, object],
) -> Path:
    target_dir = root / trainer.B787_3200_CACHE_VIEW_DIRECTORY
    target_dir.mkdir(parents=True, exist_ok=True)
    geometry = trainer._expected_cached_geometry(
        arrays.viewpoint_positions[index], arrays.tx_pos[index], arrays.rx_pos[index], args
    )
    target = {
        "schema": np.asarray(trainer.B787_3200_TARGET_SCHEMA),
        "target_spec_json": np.asarray(trainer._target_spec_text(recipe["target_spec"])),
        "view_index": np.asarray(index, dtype=np.int64),
        "role": np.asarray(role),
        "geraf_mf_magnitude": np.asarray([[[magnitude]]], dtype=np.float32),
        "viewpoint_position": np.asarray(arrays.viewpoint_positions[index], dtype=np.float32),
        **geometry,
    }
    path = target_dir / f"view_{index:06d}.npz"
    np.savez(path, **target)
    return path


def _check_complete_cache_preflight(gates: Gates) -> None:
    """Exercise all-target geometry/peak checks without an archive or response payload."""

    args = _valid_cli()
    args.n_azimuth = args.n_elevation = args.n_depth = 1
    args.point_chunk = args.pair_chunk = 1
    args.kernel_width = 4
    arrays = _synthetic_metadata_arrays()
    recipe = _synthetic_target_recipe()
    with tempfile.TemporaryDirectory(
        prefix=".geraf_b7873200_preflight_", dir=_validation_tmp_parent()
    ) as temporary:
        root = Path(temporary)
        _write_synthetic_preflight_target(
            root, index=0, role="train", magnitude=2.0, arrays=arrays, args=args, recipe=recipe
        )
        validation_path = _write_synthetic_preflight_target(
            root, index=1, role="validation", magnitude=3.0, arrays=arrays, args=args, recipe=recipe
        )
        stats = {"geraf_mf_magnitude_peak": 2.0}
        trainer._validate_complete_b7873200_target_contents(
            root=root,
            recipe=recipe,
            stats=stats,
            arrays=arrays,
            train=(0,),
            validation=(1,),
            args=args,
        )
        gates.check(True, "all-target cache preflight accepts finite calibrated synthetic geometry")

        with np.load(validation_path, allow_pickle=False) as archive:
            altered = {name: np.asarray(archive[name]).copy() for name in archive.files}
        altered["azimuth_axis"][0] += np.float32(0.1)
        np.savez(validation_path, **altered)
        try:
            trainer._validate_complete_b7873200_target_contents(
                root=root,
                recipe=recipe,
                stats=stats,
                arrays=arrays,
                train=(0,),
                validation=(1,),
                args=args,
            )
        except ValueError:
            gates.check(True, "all-target cache preflight rejects altered frozen geometry before model construction")
        else:
            raise AssertionError("altered frozen geometry must fail complete-cache preflight")

        _write_synthetic_preflight_target(
            root, index=1, role="validation", magnitude=3.0, arrays=arrays, args=args, recipe=recipe
        )
        with np.load(validation_path, allow_pickle=False) as archive:
            altered = {name: np.asarray(archive[name]).copy() for name in archive.files}
        altered["geraf_mf_magnitude"] = np.zeros((1, 1), dtype=np.float32)
        np.savez(validation_path, **altered)
        try:
            trainer._validate_complete_b7873200_target_contents(
                root=root,
                recipe=recipe,
                stats=stats,
                arrays=arrays,
                train=(0,),
                validation=(1,),
                args=args,
            )
        except ValueError:
            gates.check(True, "all-target cache preflight rejects a malformed native-MF target shape")
        else:
            raise AssertionError("a malformed native-MF target shape must fail complete-cache preflight")

        _write_synthetic_preflight_target(
            root, index=1, role="validation", magnitude=3.0, arrays=arrays, args=args, recipe=recipe
        )
        with np.load(validation_path, allow_pickle=False) as archive:
            altered = {name: np.asarray(archive[name]).copy() for name in archive.files}
        altered["geraf_mf_magnitude"][0, 0, 0] = np.float32(-1.0)
        np.savez(validation_path, **altered)
        try:
            trainer._validate_complete_b7873200_target_contents(
                root=root,
                recipe=recipe,
                stats=stats,
                arrays=arrays,
                train=(0,),
                validation=(1,),
                args=args,
            )
        except ValueError:
            gates.check(True, "all-target cache preflight rejects an invalid native-MF target value")
        else:
            raise AssertionError("an invalid native-MF target value must fail complete-cache preflight")

        _write_synthetic_preflight_target(
            root, index=1, role="validation", magnitude=3.0, arrays=arrays, args=args, recipe=recipe
        )
        train_path = root / trainer.B787_3200_CACHE_VIEW_DIRECTORY / "view_000000.npz"
        with np.load(train_path, allow_pickle=False) as archive:
            altered = {name: np.asarray(archive[name]).copy() for name in archive.files}
        altered["geraf_mf_magnitude"] = np.asarray([[[1.0]]], dtype=np.float32)
        np.savez(train_path, **altered)
        try:
            trainer._validate_complete_b7873200_target_contents(
                root=root,
                recipe=recipe,
                stats=stats,
                arrays=arrays,
                train=(0,),
                validation=(1,),
                args=args,
            )
        except ValueError:
            gates.check(True, "all-target cache preflight recomputes and enforces the train-only normalizer")
        else:
            raise AssertionError("an altered train target and stale normalizer must fail complete-cache preflight")


def main() -> None:
    gates = Gates()
    _check_complete_cache_preflight(gates)
    prior_history: list[dict[str, object]] = [
        {"step": 5, "mf_magnitude_mse": 0.25, "mf_magnitude_relative_mse": 0.5}
    ]
    metrics = {"mf_magnitude_mse": 0.125, "mf_magnitude_relative_mse": 0.25}
    normal_history, normal_best, normal_improved, normal_best_saves = _commit_from_saved_pending(
        prior_history, 0.25, step=10, metrics=metrics, interrupted_after_views=None
    )
    before_history, before_best, before_improved, before_best_saves = _commit_from_saved_pending(
        prior_history, 0.25, step=10, metrics=metrics, interrupted_after_views=0
    )
    during_history, during_best, during_improved, during_best_saves = _commit_from_saved_pending(
        prior_history, 0.25, step=10, metrics=metrics, interrupted_after_views=2
    )
    after_history, after_best, after_improved, after_best_saves = _commit_from_saved_pending(
        prior_history, 0.25, step=10, metrics=metrics, interrupted_after_views=3
    )
    gates.check(
        normal_history == before_history == during_history == after_history
        and normal_best == before_best == during_best == after_best == 0.125
        and normal_improved is before_improved is during_improved is after_improved is True
        and normal_best_saves == before_best_saves == during_best_saves == after_best_saves == 1,
        "stops before, during, or just after validation preserve identical resumed history and one best selection",
    )
    pending = trainer._begin_pending_validation(10, None)
    history = copy.deepcopy(prior_history)
    cleared, best, _ = trainer._commit_pending_validation(
        step=10,
        pending_validation_step=pending,
        metrics=metrics,
        history=history,
        best_val_mse=0.25,
    )
    try:
        trainer._commit_pending_validation(
            step=10,
            pending_validation_step=cleared,
            metrics=metrics,
            history=history,
            best_val_mse=best,
        )
    except ValueError:
        gates.check(True, "a resumed validation cannot be committed twice")
    else:
        raise AssertionError("a resumed validation cannot be committed twice")

    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=0.0)
    finite_loss = FiniteForwardInfiniteBackward.apply(model.weight)
    gates.check(bool(torch.isfinite(finite_loss).item()), "the synthetic loss is finite before backward")
    finite_loss.backward()
    weight_before_rejection = model.weight.detach().clone()
    optimizer_before_rejection = copy.deepcopy(optimizer.state_dict())
    _expect_floating_point(
        gates,
        "a finite loss with an infinite backward gradient refuses the optimizer update",
        lambda: trainer._require_finite_gradients(model),
    )
    gates.check(
        torch.equal(model.weight.detach(), weight_before_rejection)
        and optimizer.state_dict() == optimizer_before_rejection,
        "the rejected non-finite gradient leaves model and optimizer state unchanged",
    )
    optimizer.zero_grad(set_to_none=True)

    ordinary_loss = model(torch.ones(1, 2)).square().sum()
    ordinary_loss.backward()
    trainer._require_finite_gradients(model)
    optimizer.step()
    scheduler.step()
    trainer._require_finite_optimization_state(model, optimizer, scheduler)
    gates.check(True, "a finite optimizer update remains checkpointable")
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    _expect_floating_point(
        gates,
        "a non-finite post-update model parameter cannot be checkpointed",
        lambda: trainer._require_finite_optimization_state(model, optimizer),
    )
    with torch.no_grad():
        model.weight.fill_(1.0)
    parameter = next(model.parameters())
    optimizer.state[parameter]["exp_avg"].fill_(float("inf"))
    _expect_floating_point(
        gates,
        "a non-finite post-update optimizer state cannot be checkpointed",
        lambda: trainer._require_finite_optimization_state(model, optimizer),
    )
    optimizer.state[parameter]["exp_avg"].zero_()
    optimizer.param_groups[0]["lr"] = float("nan")
    _expect_floating_point(
        gates,
        "a non-finite scheduler-visible learning rate cannot be checkpointed",
        lambda: trainer._require_finite_optimization_state(model, optimizer, scheduler),
    )
    optimizer.param_groups[0]["lr"] = 1.0e-3
    buffered_model = torch.nn.Linear(2, 1, bias=False)
    buffered_model.register_buffer("calibration", torch.tensor([float("nan")]))
    buffered_optimizer = torch.optim.AdamW(buffered_model.parameters(), lr=1.0e-3)
    _expect_floating_point(
        gates,
        "a non-finite registered model buffer cannot be resumed or checkpointed",
        lambda: trainer._require_finite_optimization_state(buffered_model, buffered_optimizer),
    )
    cli = _valid_cli()
    cli.cosine_min_lr = float("nan")
    try:
        trainer._validate_cli(cli)
    except ValueError:
        gates.check(True, "a non-finite cosine scheduler setting is rejected at launch")
    else:
        raise AssertionError("a non-finite cosine scheduler setting is rejected at launch")
    gates.check(
        trainer.CHECKPOINT_VERSION >= 2 and isinstance(np.asarray([normal_best]), np.ndarray),
        "the runner records the pending-validation checkpoint format revision",
    )
    print(f"GeRaF B7873200 trainer checks passed: {gates.count} checks.", flush=True)


if __name__ == "__main__":
    main()
