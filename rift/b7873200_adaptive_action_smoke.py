"""Narrow evidence adapter for the bounded B787 adaptive-RIFT action gate.

This module is deliberately not a second RIFT trainer.  It creates a complete
16/16/1000 engineering manifest from the established B787 development roles,
and observes the existing ``train.train_sar`` adaptive-v2 lifecycle through an
explicit opt-in callback.  Historical RIFT commands neither import nor invoke
this module.

The adapter is an engineering gate, not a production recipe or a result
evaluator.  In particular, its held-out role is only 16 views and its four
epochs contain exactly sixteen logical optimizer updates.
"""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

try:  # ``resource`` is unavailable on Windows, while PACE exposes it.
    import resource as _resource
except ImportError:  # pragma: no cover - platform capability branch
    _resource = None


B787_3200_CANONICAL_NPZ_PATH = (
    "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/"
    "b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz"
)
B787_3200_CANONICAL_MANIFEST_PATH = (
    "/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/"
    "b78710k_interp_seed42_train3200_val1000_test1000_v1.json"
)
B787_3200_CANONICAL_MANIFEST_NAME = "b78710k_interp_seed42_train3200_val1000_test1000_v1"
B787_3200_PARENT_COUNTS = (3_200, 1_000, 1_000, 4_800)
B787_3200_RESPONSE_SHAPE = [10_000, 16, 16, 1, 600]

ACTION_GATE_SCHEMA = "rift_b7873200_adaptive_action_gate_v1"
ACTION_GATE_MANIFEST_NAME = "b78710k_adaptive_rift_action_gate_subset16x16_v1"
ACTION_GATE_CHECKPOINT_NAME = "b78710k_adaptive_rift_action_gate_v1"
ACTION_GATE_EXECUTION_LABEL = "b78710k_adaptive_rift_action_gate16x16_v1"
ACTION_GATE_NUM_TRAIN = 16
ACTION_GATE_NUM_VALIDATION = 16
ACTION_GATE_NUM_TEST = 1_000
ACTION_GATE_EPOCHS = 4
ACTION_GATE_STEP_EVERY = 4
ACTION_GATE_EXPECTED_UPDATES = 16
ACTION_GATE_PROBE_COUNT = 4
ACTION_GATE_PRESERVATION_ATOL = 1.0e-7


def _resolved(path: str | os.PathLike[str]) -> str:
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def _require_integer_list(value: object, *, label: str, expected_count: int) -> list[int]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"{label} must be a list with {expected_count} IDs")
    result: list[int] = []
    for raw in value:
        if not isinstance(raw, int) or isinstance(raw, bool) or not 0 <= raw < 10_000:
            raise ValueError(f"{label} contains an invalid B787 source-view ID: {raw!r}")
        result.append(int(raw))
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate source-view IDs")
    return result


def _read_json_mapping(path: str | os.PathLike[str], *, label: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise ValueError(f"could not read {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _parent_roles(parent: Mapping[str, object]) -> dict[str, list[int]]:
    if parent.get("schema_version") != 1:
        raise ValueError("B787 action gate requires the established sealed manifest schema v1")
    if parent.get("name") != B787_3200_CANONICAL_MANIFEST_NAME:
        raise ValueError("B787 action gate requires the canonical interpolation manifest")
    split = parent.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("canonical B787 manifest lacks a split object")
    if split.get("strategy") != "fixed_tail_subsampled":
        raise ValueError("B787 action gate requires the fixed-tail-subsampled parent split")
    if split.get("complete_partition") is not True or split.get("test_sealed") is not True:
        raise ValueError("canonical B787 manifest must declare a complete sealed-test partition")
    expected = {
        "train": ("num_train", "train_indices", B787_3200_PARENT_COUNTS[0]),
        "validation": ("num_validation", "validation_indices", B787_3200_PARENT_COUNTS[1]),
        "test": ("num_test", "test_indices", B787_3200_PARENT_COUNTS[2]),
        "unused": ("num_unused", "unused_indices", B787_3200_PARENT_COUNTS[3]),
    }
    roles: dict[str, list[int]] = {}
    for role, (count_key, ids_key, count) in expected.items():
        if split.get(count_key) != count:
            raise ValueError(f"canonical B787 {role} count differs from {count}")
        roles[role] = _require_integer_list(split.get(ids_key), label=f"canonical {role}", expected_count=count)
    all_ids = [item for values in roles.values() for item in values]
    if len(set(all_ids)) != 10_000 or set(all_ids) != set(range(10_000)):
        raise ValueError("canonical B787 split is not a complete disjoint 10,000-view partition")
    return roles


def load_canonical_parent_manifest(
    npz_path: str | os.PathLike[str], parent_manifest_path: str | os.PathLike[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preflight the full parent contract before any response row is read."""

    if _resolved(npz_path) != _resolved(B787_3200_CANONICAL_NPZ_PATH):
        raise ValueError(
            "B787 action gate accepts only the canonical sphere10k archive at "
            f"{B787_3200_CANONICAL_NPZ_PATH}"
        )
    if _resolved(parent_manifest_path) != _resolved(B787_3200_CANONICAL_MANIFEST_PATH):
        raise ValueError(
            "B787 action gate accepts only the canonical interpolation manifest at "
            f"{B787_3200_CANONICAL_MANIFEST_PATH}"
        )

    # The generic sealed loader is intentionally the first source capable of
    # opening the archive.  It reads metadata and the response header only.
    from train import _load_sealed_npz_protocol_contract

    _arrays, contract = _load_sealed_npz_protocol_contract(
        npz_path,
        parent_manifest_path,
        num_train=B787_3200_PARENT_COUNTS[0],
        num_val=B787_3200_PARENT_COUNTS[1],
        num_test=B787_3200_PARENT_COUNTS[2],
    )
    if contract.get("response_shape") != B787_3200_RESPONSE_SHAPE:
        raise ValueError("canonical B787 archive header is not [10000,16,16,1,600]")
    if contract.get("response_dtype") != "complex64":
        raise ValueError("canonical B787 archive response dtype is not complex64")
    parent = _read_json_mapping(parent_manifest_path, label="canonical B787 role manifest")
    _parent_roles(parent)
    return parent, dict(contract)


def build_action_gate_manifest(parent: Mapping[str, object]) -> dict[str, Any]:
    """Derive the complete small-engineering partition without any response read."""

    roles = _parent_roles(parent)
    dataset = parent.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("canonical B787 manifest lacks a dataset object")
    if dataset.get("num_views") != 10_000:
        raise ValueError("canonical B787 manifest does not name 10,000 views")
    if list(dataset.get("response_shape", ())) != B787_3200_RESPONSE_SHAPE:
        raise ValueError("canonical B787 manifest has an unexpected response shape")
    if dataset.get("response_dtype") != "complex64":
        raise ValueError("canonical B787 manifest has an unexpected response dtype")

    train = roles["train"][:ACTION_GATE_NUM_TRAIN]
    validation = roles["validation"][:ACTION_GATE_NUM_VALIDATION]
    test = roles["test"]
    # Parent train/validation rows outside the selected engineering prefix are
    # deliberately sealed as unused here.  This is why the generic reader may
    # materialize only 16 + 16 response rows.
    unused = (
        roles["train"][ACTION_GATE_NUM_TRAIN:]
        + roles["unused"]
        + roles["validation"][ACTION_GATE_NUM_VALIDATION:]
    )
    expected_unused = 10_000 - len(train) - len(validation) - len(test)
    if len(unused) != expected_unused:
        raise AssertionError("derived B787 action-gate unused count is inconsistent")

    # Keep only the header fields consumed by the generic sealed protocol.
    # In particular, this adapter neither computes nor carries a source hash.
    derived_dataset = {
        "num_views": 10_000,
        "response_shape": list(B787_3200_RESPONSE_SHAPE),
        "response_dtype": "complex64",
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "name": ACTION_GATE_MANIFEST_NAME,
        "dataset": derived_dataset,
        "split": {
            "strategy": "parent_fixed_tail_prefix_engineering_action_gate_v1",
            "num_train": len(train),
            "num_validation": len(validation),
            "num_test": len(test),
            "num_unused": len(unused),
            "complete_partition": True,
            "test_sealed": True,
            "unused_sealed": True,
            "train_indices": train,
            "validation_indices": validation,
            "test_indices": test,
            "unused_indices": unused,
        },
        "engineering_subset": {
            "schema": ACTION_GATE_SCHEMA,
            "version": 1,
            "parent_manifest_name": B787_3200_CANONICAL_MANIFEST_NAME,
            "parent_strategy": "fixed_tail_subsampled",
            "selection": {
                "train": "ordered parent train[:16]",
                "validation": "ordered parent validation[:16]",
                "test": "all parent sealed-test IDs",
                "unused": "remaining parent train/validation plus parent unused IDs",
            },
            "reporting_status": "engineering_smoke_not_production_or_comparison",
        },
    }
    validate_action_gate_child_manifest(parent, payload)
    return payload


def _validate_action_gate_manifest(payload: Mapping[str, object]) -> None:
    split = payload.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("action-gate manifest lacks a split object")
    expected = {
        "train": ("num_train", "train_indices", ACTION_GATE_NUM_TRAIN),
        "validation": ("num_validation", "validation_indices", ACTION_GATE_NUM_VALIDATION),
        "test": ("num_test", "test_indices", ACTION_GATE_NUM_TEST),
        "unused": ("num_unused", "unused_indices", 8_968),
    }
    roles: list[int] = []
    for role, (count_key, ids_key, count) in expected.items():
        if split.get(count_key) != count:
            raise ValueError(f"action-gate {role} count is not {count}")
        roles.extend(_require_integer_list(split.get(ids_key), label=f"action-gate {role}", expected_count=count))
    if len(set(roles)) != 10_000 or set(roles) != set(range(10_000)):
        raise ValueError("action-gate roles are not a complete disjoint B787 partition")
    if split.get("complete_partition") is not True or split.get("test_sealed") is not True:
        raise ValueError("action-gate manifest must keep a complete sealed-test partition")
    if split.get("unused_sealed") is not True:
        raise ValueError("action-gate manifest must seal its unselected development IDs")


def validate_action_gate_child_manifest(
    parent: Mapping[str, object], payload: Mapping[str, object]
) -> None:
    """Prove that a complete child preserves the exact approved parent roles.

    The generic sealed-manifest validator deliberately checks only the child
    partition. This adapter additionally proves provenance before the child
    can authorize any B787 response row: the two 16-view roles are the ordered
    parent prefixes, all parent test IDs remain sealed test, and every other
    parent development ID remains sealed unused.
    """

    roles = _parent_roles(parent)
    _validate_action_gate_manifest(payload)
    if payload.get("schema_version") != 1 or payload.get("name") != ACTION_GATE_MANIFEST_NAME:
        raise ValueError("action-gate child has the wrong fixed engineering identity")
    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("action-gate child lacks its dataset header")
    if list(dataset.get("response_shape", ())) != B787_3200_RESPONSE_SHAPE:
        raise ValueError("action-gate child has an unexpected response header")
    if dataset.get("response_dtype") != "complex64" or dataset.get("num_views") != 10_000:
        raise ValueError("action-gate child has an unexpected response dataset")
    split = payload["split"]
    assert isinstance(split, Mapping)  # established by _validate_action_gate_manifest
    if split.get("strategy") != "parent_fixed_tail_prefix_engineering_action_gate_v1":
        raise ValueError("action-gate child has an unexpected role-derivation strategy")
    expected = {
        "train_indices": roles["train"][:ACTION_GATE_NUM_TRAIN],
        "validation_indices": roles["validation"][:ACTION_GATE_NUM_VALIDATION],
        "test_indices": roles["test"],
        "unused_indices": (
            roles["train"][ACTION_GATE_NUM_TRAIN:]
            + roles["unused"]
            + roles["validation"][ACTION_GATE_NUM_VALIDATION:]
        ),
    }
    for key, indices in expected.items():
        if split.get(key) != indices:
            raise ValueError(f"action-gate child {key} does not preserve the approved parent role order")
    engineering_subset = payload.get("engineering_subset")
    if not isinstance(engineering_subset, Mapping):
        raise ValueError("action-gate child lacks engineering provenance")
    if (
        engineering_subset.get("schema") != ACTION_GATE_SCHEMA
        or engineering_subset.get("version") != 1
        or engineering_subset.get("parent_manifest_name") != B787_3200_CANONICAL_MANIFEST_NAME
        or engineering_subset.get("parent_strategy") != "fixed_tail_subsampled"
        or engineering_subset.get("reporting_status")
        != "engineering_smoke_not_production_or_comparison"
    ):
        raise ValueError("action-gate child engineering provenance disagrees with its parent")
    selection = engineering_subset.get("selection")
    if selection != {
        "train": "ordered parent train[:16]",
        "validation": "ordered parent validation[:16]",
        "test": "all parent sealed-test IDs",
        "unused": "remaining parent train/validation plus parent unused IDs",
    }:
        raise ValueError("action-gate child selection provenance is incomplete or changed")


def write_action_gate_manifest(path: str | os.PathLike[str], payload: Mapping[str, object]) -> Path:
    """Publish one checked derived manifest, without changing an existing identity."""

    _validate_action_gate_manifest(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(json.dumps(payload))
    if destination.exists():
        existing = _read_json_mapping(destination, label="existing action-gate manifest")
        if existing != normalized:
            raise ValueError(
                "action-gate manifest path already contains a different configuration; "
                "choose a new engineering run root rather than overwrite it"
            )
        return destination
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(normalized, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return destination


def action_gate_train_argv(
    *, npz_path: str | os.PathLike[str], manifest_path: str | os.PathLike[str],
    checkpoint_root: str | os.PathLike[str], resume: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Return the fully frozen generic-trainer command for this one smoke."""

    argv = [
        "--checkpoint-name", ACTION_GATE_CHECKPOINT_NAME,
        "--checkpoint-root", os.fspath(checkpoint_root),
        "--execution-contract-label", ACTION_GATE_EXECUTION_LABEL,
        "--require-full-resume-state",
        "--data-format", "npz",
        "--npz-path", os.fspath(npz_path),
        "--npz-sealed-protocol",
        "--npz-role-manifest", os.fspath(manifest_path),
        "--num-train", str(ACTION_GATE_NUM_TRAIN),
        "--num-val", str(ACTION_GATE_NUM_VALIDATION),
        "--num-test", str(ACTION_GATE_NUM_TEST),
        "--num-tx", "16",
        "--num-rx", "16",
        "--num-freq-wanted", "600",
        "--epochs", str(ACTION_GATE_EPOCHS),
        "--step-every", str(ACTION_GATE_STEP_EVERY),
        "--loss", "complex",
        "--scene-repr", "point_sh",
        "--forward-operator", "range",
        "--range-model", "product",
        "--compute-dtype", "float64",
        "--granularity", "8",
        "--extent", "0.15",
        "--max-points", "640",
        "--adaptive-max-active", "640",
        "--sh-init-degree", "0",
        "--sh-max-degree", "1",
        "--bp-init", "16",
        "--init-scale", "0",
        "--phase-sign", "-1",
        "--lr", "0.003",
        "--pos-lr", "0.003",
        "--adam-eps", "1e-20",
        "--weight-decay", "0",
        "--t0", "10",
        "--t-mult", "2",
        "--seed", "42",
        "--adaptive-capacity-v2",
        "--adaptive-refine-every", "1",
        "--adaptive-probe-every", "1",
        "--adaptive-min-spatial-exposure", "1",
        "--adaptive-min-angular-exposure", "1",
        "--adaptive-spatial-fraction", "0.0015625",
        "--adaptive-angular-fraction", "0.0015625",
        "--adaptive-spatial-floor", "0",
        "--adaptive-angular-floor", "0",
        "--adaptive-cooldown-events", "0",
        "--adaptive-child-maturity-events", "1",
        "--split-max-level", "1",
        "--split-every", "0",
        "--grow-every", "0",
        "--prune-every", "0",
        "--l1-weight", "0",
        "--sh-smooth-weight", "0",
        "--mag-weight", "0",
        "--mag-warmup-epochs", "0",
        "--checkpoint-metric", "train",
    ]
    if resume is not None:
        argv.extend(["--resume", os.fspath(resume)])
    return argv


def _finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def _l2(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach()).item())


def _role_metrics_are_finite(metrics: Mapping[str, object]) -> bool:
    """Require the fixed train/validation coherent audit to be numerically usable."""

    if not isinstance(metrics, Mapping):
        return False
    required = (
        "error_squared_sum",
        "target_squared_sum",
        "prediction_squared_sum",
        "coherent_relative_mse",
        "coherent_relative_l2",
        "same_domain_zero_relative_mse",
        "same_domain_zero_relative_l2",
    )
    for role in ("train", "validation"):
        values = metrics.get(role)
        if not isinstance(values, Mapping):
            return False
        for key in required:
            value = values.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return False
        if float(values["target_squared_sum"]) <= 0.0:
            return False
    return True


class AdaptiveActionGateObserver:
    """Observe render preservation and post-event learning without steering it.

    The observer never changes a model, gradients, optimizer, scores, selected
    indices, or checkpoint schedule.  It only uses the first four selected
    training views to compare the existing renderer immediately before and
    after each immutable adaptive-v2 action.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        gain: torch.nn.Module | None,
        train_loader: Any,
        validation_loader: Any,
        criterion: torch.nn.Module,
        device: torch.device | str,
        num_freq_selected: int,
        phase_sign: float,
        forward_operator_name: str,
        compute_dtype: torch.dtype,
        data_format: str,
        op_kwargs: Mapping[str, object],
        occlusion: Mapping[str, object] | None,
        selected_train_ids: Sequence[int],
        selected_validation_ids: Sequence[int],
    ) -> None:
        if forward_operator_name != "range" or data_format != "npz":
            raise ValueError("B787 action observer requires the sealed NPZ range operator")
        if float(phase_sign) != -1.0 or op_kwargs.get("range_model") != "product":
            raise ValueError("B787 action observer requires phase -1 and product spreading")
        if int(num_freq_selected) != 600:
            raise ValueError("B787 action observer requires all 600 frequency bins")
        if len(selected_train_ids) != ACTION_GATE_NUM_TRAIN:
            raise ValueError("B787 action observer requires exactly 16 selected training IDs")
        if len(selected_validation_ids) != ACTION_GATE_NUM_VALIDATION:
            raise ValueError("B787 action observer requires exactly 16 selected validation IDs")
        if len(train_loader.dataset) != ACTION_GATE_NUM_TRAIN:
            raise ValueError("B787 action observer train loader is not the sealed 16-view subset")
        if list(int(value) for value in train_loader.dataset.indices) != list(selected_train_ids):
            raise ValueError("B787 action observer train IDs disagree with its derived manifest")
        if list(int(value) for value in validation_loader.dataset.indices) != list(selected_validation_ids):
            raise ValueError("B787 action observer validation IDs disagree with its derived manifest")

        self.model = model
        self.optimizer = optimizer
        self.gain = gain
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.criterion = criterion
        # ``init_distributed`` returns ``"cuda:0"`` for the ordinary
        # single-process CUDA path and a ``torch.device`` for several test and
        # distributed paths.  Keep the observer's local diagnostic state
        # canonical without changing the generic trainer's device contract.
        self.device = torch.device(device)
        self.num_freq_selected = int(num_freq_selected)
        self.phase_sign = float(phase_sign)
        self.compute_dtype = compute_dtype
        self.op_kwargs = dict(op_kwargs)
        self.occlusion = occlusion
        self.probe_ids = [int(value) for value in selected_train_ids[:ACTION_GATE_PROBE_COUNT]]
        self._probe_batches = [train_loader.dataset[index] for index in range(ACTION_GATE_PROBE_COUNT)]
        self.records: list[dict[str, Any]] = []
        self.pending: list[dict[str, Any]] = []
        self.optimizer_updates: list[dict[str, Any]] = []
        self._open_events: dict[int, dict[str, Any]] = {}
        self._start_time = time.monotonic()
        self._elapsed_before_resume = 0.0
        self._initial_metrics = self._role_metrics(train_loader, validation_loader)
        self._initial_parameters = self._parameter_snapshot()
        self._final_metrics: dict[str, Any] | None = None
        self._finished = False

    def _parameter_snapshot(self) -> dict[str, torch.Tensor]:
        result = {
            f"scene.{name}": value.detach().cpu().clone()
            for name, value in self.model.named_parameters()
        }
        if self.gain is not None:
            result.update({
                f"gain.{name}": value.detach().cpu().clone()
                for name, value in self.gain.named_parameters()
            })
        return result

    def _render_item(self, item: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """Render one already-authorized NPZ item through the production operator."""

        from train import (
            apply_occlusion,
            cc,
            get_kvector,
            reshape_measured_cubes,
        )
        from rift.range_operator import range_forward_operator

        (
            freqs_tensor,
            dphi_tensor,
            dtheta_tensor,
            magnitude_tensor,
            phase_tensor,
            rx_pos,
            tx_pos,
        ) = item
        # Match the batch_size=1 NPZ path in train_sar without creating a
        # loader that might reorder or request another response row. Dataset
        # __getitem__ returns the frequencies first and dphi before dtheta.
        magnitude_cube, phase_cube = reshape_measured_cubes(
            magnitude_tensor.unsqueeze(0), phase_tensor.unsqueeze(0), self.device, 16, 16
        )
        freqs_tensor = freqs_tensor.to(self.device)
        if int(freqs_tensor.numel()) != self.num_freq_selected:
            raise ValueError("B787 action observer requires exactly the archive's full 600-bin spectrum")
        # The generic trainer samples a frequency permutation even when it then
        # retains every bin. The engineering observer must not advance that
        # private training RNG merely by looking at a fixed full-spectrum view.
        freq_indices = torch.arange(freqs_tensor.shape[0], device=self.device)
        frame_data_mag = magnitude_cube[:, :, freq_indices]
        frame_data_phase = phase_cube[:, :, freq_indices]
        dtheta = dtheta_tensor.unsqueeze(0).to(self.device)
        dphi = dphi_tensor.unsqueeze(0).to(self.device)
        scatterer_pos, scatterer_weights = self.model.active_scatterers(dtheta, dphi)
        weights_view = apply_occlusion(
            self.model, scatterer_weights, rx_pos.to(self.device), tx_pos.to(self.device), self.occlusion
        )
        prediction = range_forward_operator(
            freqs_tensor,
            get_kvector(freqs_tensor, cc),
            rx_pos.to(self.device),
            tx_pos.to(self.device),
            scatterer_pos,
            weights_view,
            phase_sign=self.phase_sign,
            freq_indices=freq_indices,
            compute_dtype=self.compute_dtype,
            **self.op_kwargs,
        )
        if self.gain is not None:
            prediction = self.gain(prediction)
        observed = torch.polar(frame_data_mag, frame_data_phase).permute(2, 0, 1)
        return prediction, observed

    def _predictions(self) -> list[torch.Tensor]:
        was_training = self.model.training
        gain_training = self.gain.training if self.gain is not None else None
        self.model.eval()
        if self.gain is not None:
            self.gain.eval()
        try:
            with torch.no_grad():
                return [self._render_item(item)[0].detach().clone() for item in self._probe_batches]
        finally:
            self.model.train(was_training)
            if self.gain is not None and gain_training is not None:
                self.gain.train(gain_training)

    def _role_metrics(self, train_loader: Any, validation_loader: Any) -> dict[str, Any]:
        was_training = self.model.training
        gain_training = self.gain.training if self.gain is not None else None
        self.model.eval()
        if self.gain is not None:
            self.gain.eval()
        try:
            output: dict[str, Any] = {}
            with torch.no_grad():
                for label, loader in (("train", train_loader), ("validation", validation_loader)):
                    error_sq = torch.zeros((), dtype=torch.float64, device=self.device)
                    target_sq = torch.zeros((), dtype=torch.float64, device=self.device)
                    prediction_sq = torch.zeros((), dtype=torch.float64, device=self.device)
                    for index in range(len(loader.dataset)):
                        prediction, observed = self._render_item(loader.dataset[index])
                        delta = prediction - observed
                        error_sq += delta.abs().square().sum().double()
                        target_sq += observed.abs().square().sum().double()
                        prediction_sq += prediction.abs().square().sum().double()
                    numerator = float(error_sq.item())
                    denominator = float(target_sq.item())
                    rel_mse = numerator / denominator if denominator > 0.0 else None
                    output[label] = {
                        "native_readout": "coherent complex signal",
                        "samples": int(len(loader.dataset)),
                        "error_squared_sum": numerator,
                        "target_squared_sum": denominator,
                        "prediction_squared_sum": float(prediction_sq.item()),
                        "coherent_relative_mse": rel_mse,
                        "coherent_relative_l2": math.sqrt(rel_mse) if rel_mse is not None else None,
                        "same_domain_zero_relative_mse": 1.0 if denominator > 0.0 else None,
                        "same_domain_zero_relative_l2": 1.0 if denominator > 0.0 else None,
                    }
            return output
        finally:
            self.model.train(was_training)
            if self.gain is not None and gain_training is not None:
                self.gain.train(gain_training)

    def _scene_summary(self) -> dict[str, Any]:
        active = self.model.active_mask
        positions = self.model.positions()[active]
        orders = self.model.order[active]
        support_ok = bool(
            (positions >= self.model.support_min - 1.0e-6).all()
            and (positions <= self.model.support_max + 1.0e-6).all()
        )
        degree_ok = bool((orders >= 0).all() and (orders <= self.model.max_degree).all())
        return {
            "active_points": int(active.sum().item()),
            "allocated_slots": int(self.model.active_mask.numel()),
            "active_parameter_scalars": int(self.model.active_parameter_count()),
            "allocated_parameter_scalars": int(self.model.allocated_parameter_count()),
            "order_min": int(orders.min().item()) if orders.numel() else None,
            "order_max": int(orders.max().item()) if orders.numel() else None,
            "support_bounds_ok": support_ok,
            "degree_bounds_ok": degree_ok,
            "support_min_m": [float(value) for value in self.model.support_min.detach().cpu().tolist()],
            "support_max_m": [float(value) for value in self.model.support_max.detach().cpu().tolist()],
        }

    def _optimizer_finite(self) -> bool:
        for state in self.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and not _finite_tensor(value):
                    return False
        return True

    @staticmethod
    def _optimizer_rows_zero(
        optimizer: torch.optim.Optimizer,
        parameter: torch.Tensor,
        rows: torch.Tensor,
        column_mask: torch.Tensor | None = None,
    ) -> bool:
        if rows.numel() == 0:
            return True
        state = optimizer.state.get(parameter, {})
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if not torch.is_tensor(value) or value.shape != parameter.shape:
                continue
            selected = value[rows]
            if column_mask is not None:
                selected = selected[:, column_mask]
            if not bool((selected == 0).all()):
                return False
        return True

    def _model_finite(self) -> bool:
        parameters = list(self.model.parameters())
        if self.gain is not None:
            parameters.extend(self.gain.parameters())
        return all(_finite_tensor(value) for value in parameters)

    def _selection(self, snapshot: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = self.model._select_refinement_indices(
            snapshot["spatial_score"], snapshot["spatial_eligible"], 1.0 / 640.0
        )
        angular = self.model._select_refinement_indices(
            snapshot["angular_score"], snapshot["angular_eligible"], 1.0 / 640.0
        )
        active_before = int(self.model.active_mask.sum().item())
        free_parent_budget = int((~self.model.active_mask).sum().item()) // 7
        active_parent_budget = max(640 - active_before, 0) // 7
        spatial = spatial[:min(free_parent_budget, active_parent_budget)]
        return spatial, angular

    @staticmethod
    def _selected_detail(snapshot: Mapping[str, torch.Tensor], indices: torch.Tensor, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_indices": [int(value) for value in indices.detach().cpu().tolist()],
            f"{prefix}_scores": [float(value) for value in snapshot[f"{prefix}_score"][indices].detach().cpu().tolist()],
            f"{prefix}_exposures": [float(value) for value in snapshot[f"{prefix}_exposure"][indices].detach().cpu().tolist()],
            f"eligible_{prefix}_count": int(snapshot[f"{prefix}_eligible"].sum().item()),
        }

    def on_adaptive_event(self, phase: str, **context: Any) -> None:
        event = int(context["event"])
        if phase == "before":
            snapshot = context["snapshot"]
            if not isinstance(snapshot, Mapping):
                raise ValueError("adaptive action observer received a malformed snapshot")
            spatial, angular = self._selection(snapshot)
            active_mask = self.model.active_mask.detach().clone()
            orders = self.model.order.detach().clone()
            levels = self.model.level.detach().clone()
            record: dict[str, Any] = {
                "event": event,
                "epoch": int(context["epoch"]),
                "logical_optimizer_updates": int(context["logical_optimizer_updates"]),
                "probe_source_view_ids": list(self.probe_ids),
                "last_grad_norm": float(context["last_grad_norm"]),
                "active_before": int(active_mask.sum().item()),
                "scene_before": self._scene_summary(),
                **self._selected_detail(snapshot, spatial, "spatial"),
                **self._selected_detail(snapshot, angular, "angular"),
                "spatial_parent_levels_before": [
                    int(value) for value in levels[spatial].detach().cpu().tolist()
                ],
            }
            self._open_events[event] = {
                "record": record,
                "predictions": self._predictions(),
                "active_mask": active_mask,
                "orders": orders,
                "levels": levels,
            }
            return

        if phase != "after":
            raise ValueError(f"unknown adaptive action observer phase {phase!r}")
        pending = self._open_events.pop(event, None)
        if pending is None:
            raise RuntimeError("adaptive action observer received an after-event without before-event state")
        record = pending["record"]
        before_predictions: list[torch.Tensor] = pending["predictions"]
        after_predictions = self._predictions()
        if len(before_predictions) != len(after_predictions):
            raise RuntimeError("adaptive action observer probe count changed across an event")
        probes_finite = all(
            bool(torch.isfinite(before).all()) and bool(torch.isfinite(after).all())
            for before, after in zip(before_predictions, after_predictions)
        )
        if probes_finite:
            delta_sq = 0.0
            reference_sq = 0.0
            max_abs = 0.0
            for before, after in zip(before_predictions, after_predictions):
                difference = after - before
                delta_sq += float(difference.abs().square().sum().double().item())
                reference_sq += float(before.abs().square().sum().double().item())
                max_abs = max(max_abs, float(difference.abs().max().item()))
            difference_l2: float | None = math.sqrt(delta_sq)
            reference_l2: float | None = math.sqrt(reference_sq)
            relative_l2: float | None = math.sqrt(delta_sq / reference_sq) if reference_sq > 0.0 else None
        else:
            difference_l2 = None
            reference_l2 = None
            relative_l2 = None
            max_abs = None
        record["probe_full_coherent_predictions_finite"] = probes_finite
        record["probe_full_coherent_prediction_difference_l2"] = difference_l2
        record["probe_full_coherent_prediction_reference_l2"] = reference_l2
        record["probe_full_coherent_prediction_difference_relative_l2"] = relative_l2
        record["probe_full_coherent_prediction_difference_max_abs"] = max_abs
        record["probe_full_coherent_prediction_preserved"] = bool(
            probes_finite and max_abs is not None and max_abs <= ACTION_GATE_PRESERVATION_ATOL
        )
        record["probe_full_coherent_prediction_preservation_atol"] = ACTION_GATE_PRESERVATION_ATOL

        active_before: torch.Tensor = pending["active_mask"]
        orders_before: torch.Tensor = pending["orders"]
        levels_before: torch.Tensor = pending["levels"]
        child_indices = ((~active_before) & self.model.active_mask).nonzero(as_tuple=True)[0]
        expected_spatial = torch.as_tensor(
            record["spatial_indices"], device=self.model.level.device, dtype=torch.long
        )
        expected_angular = torch.as_tensor(
            record["angular_indices"], device=self.model.order.device, dtype=torch.long
        )
        changed_existing_orders = (
            active_before & (self.model.order > orders_before)
        ).nonzero(as_tuple=True)[0]
        changed_existing_levels = (
            active_before & (self.model.level > levels_before)
        ).nonzero(as_tuple=True)[0]
        # Angular parents are unlocked before a spatial split. A child of that
        # same parent inherits its newly unlocked order, so every allocated slot
        # is not an SH unlock. Prove the mutation against the selected
        # pre-existing parent rows only.
        newly_unlocked = expected_angular[
            self.model.order[expected_angular] == orders_before[expected_angular] + 1
        ]
        n_split = int(context["n_split"])
        n_grown = int(context["n_grown"])
        if child_indices.numel() != 7 * n_split:
            raise RuntimeError("adaptive action observer found an unexpected number of split siblings")
        if expected_spatial.numel() != n_split:
            raise RuntimeError("adaptive action observer found an unexpected spatial split selection")
        if expected_spatial.numel() and not torch.equal(
            self.model.level[expected_spatial], levels_before[expected_spatial] + 1,
        ):
            raise RuntimeError("adaptive action observer found a selected spatial parent without its level increment")
        if not torch.equal(
            torch.sort(changed_existing_levels).values,
            torch.sort(expected_spatial).values,
        ):
            raise RuntimeError("adaptive action observer found an unexpected existing-point spatial split")
        if expected_angular.numel() != n_grown or newly_unlocked.numel() != n_grown:
            raise RuntimeError("adaptive action observer found an unexpected SH unlock selection")
        if not torch.equal(
            torch.sort(changed_existing_orders).values,
            torch.sort(newly_unlocked).values,
        ):
            raise RuntimeError("adaptive action observer found an unexpected existing-point SH unlock")
        record.update({
            "n_split": n_split,
            "n_grown": n_grown,
            "active_after": int(context["n_active"]),
            "children": [int(value) for value in child_indices.detach().cpu().tolist()],
            "split_parent_indices": [int(value) for value in expected_spatial.detach().cpu().tolist()],
            "split_parent_levels_after": [
                int(value) for value in self.model.level[expected_spatial].detach().cpu().tolist()
            ],
            "unlocked_indices": [int(value) for value in newly_unlocked.detach().cpu().tolist()],
            "scene_after": self._scene_summary(),
            "model_parameters_finite": self._model_finite(),
            "optimizer_state_finite": self._optimizer_finite(),
        })
        child_birth_ok = True
        children_zero = True
        child_optimizer_zero = True
        if child_indices.numel():
            child_birth_ok = bool((self.model.refine_birth_event[child_indices] == event).all())
            children_zero = bool(
                (self.model.w_re[child_indices] == 0).all()
                and (self.model.w_im[child_indices] == 0).all()
            )
            child_optimizer_zero = (
                self._optimizer_rows_zero(self.optimizer, self.model.w_re, child_indices)
                and self._optimizer_rows_zero(self.optimizer, self.model.w_im, child_indices)
                and self._optimizer_rows_zero(self.optimizer, self.model.delta_raw, child_indices)
            )
        degree_one = self.model.basis_degree == 1
        band_zero = True
        band_optimizer_zero = True
        if newly_unlocked.numel():
            band_zero = bool(
                (self.model.w_re[newly_unlocked][:, degree_one] == 0).all()
                and (self.model.w_im[newly_unlocked][:, degree_one] == 0).all()
            )
            band_optimizer_zero = (
                self._optimizer_rows_zero(
                    self.optimizer, self.model.w_re, newly_unlocked, degree_one
                )
                and self._optimizer_rows_zero(
                    self.optimizer, self.model.w_im, newly_unlocked, degree_one
                )
            )
        record.update({
            "child_birth_event_ok": child_birth_ok,
            "children_zero_at_birth": children_zero,
            "child_optimizer_rows_zero_at_birth": child_optimizer_zero,
            "unlocked_band_zero_at_unlock": band_zero,
            "unlocked_band_optimizer_columns_zero": band_optimizer_zero,
        })
        self.records.append(record)
        if n_split or n_grown:
            self.pending.append({
                "event": event,
                "children": record["children"],
                "unlocked_indices": record["unlocked_indices"],
                "checked_after_update": False,
                "child_update_l2": None,
                "band_update_l2": None,
                "child_updated": n_split == 0,
                "band_updated": n_grown == 0,
            })

    def on_optimizer_step(self, **context: Any) -> None:
        """Record the first real optimizer update after each topology action."""

        grad_norm = float(context["grad_norm"])
        self.optimizer_updates.append({
            "epoch": int(context["epoch"]),
            "logical_optimizer_updates": int(context["logical_optimizer_updates"]),
            # This timer deliberately covers optimizer.step/zero_grad only;
            # forward/backward work remains in the epoch timing reported by
            # the generic trainer and must not be implied here.
            "optimizer_step_seconds": float(context["seconds"]),
            "grad_norm": grad_norm,
            "finite_grad_norm": math.isfinite(grad_norm),
            "model_parameters_finite": self._model_finite(),
            "optimizer_state_finite": self._optimizer_finite(),
        })

        if not self.pending:
            return
        for pending in self.pending:
            if pending["checked_after_update"]:
                continue
            child_indices = torch.as_tensor(pending["children"], device=self.model.w_re.device, dtype=torch.long)
            band_indices = torch.as_tensor(
                pending["unlocked_indices"], device=self.model.w_re.device, dtype=torch.long
            )
            child_norm = 0.0
            if child_indices.numel():
                child_norm = float(torch.sqrt(
                    self.model.w_re[child_indices].square().sum()
                    + self.model.w_im[child_indices].square().sum()
                ).item())
            band_norm = 0.0
            if band_indices.numel():
                degree_one = self.model.basis_degree == 1
                band_norm = float(torch.sqrt(
                    self.model.w_re[band_indices][:, degree_one].square().sum()
                    + self.model.w_im[band_indices][:, degree_one].square().sum()
                ).item())
            pending.update({
                "checked_after_update": True,
                "first_post_event_optimizer_update": int(context["logical_optimizer_updates"]),
                "child_update_l2": child_norm,
                "band_update_l2": band_norm,
                "child_updated": child_norm > 0.0 if child_indices.numel() else True,
                "band_updated": band_norm > 0.0 if band_indices.numel() else True,
            })

    def finish_training(self) -> None:
        """Mark the observation timeline complete before the final checkpoint saves."""

        if self._open_events:
            raise RuntimeError("adaptive action observer cannot finish with an open event")
        self._finished = True

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema": ACTION_GATE_SCHEMA,
            "version": 1,
            "probe_source_view_ids": list(self.probe_ids),
            "records": copy.deepcopy(self.records),
            "pending": copy.deepcopy(self.pending),
            "optimizer_updates": copy.deepcopy(self.optimizer_updates),
            "initial_metrics": copy.deepcopy(self._initial_metrics),
            "initial_parameters": {
                name: value.detach().cpu().clone()
                for name, value in self._initial_parameters.items()
            },
            "elapsed_seconds": float(
                self._elapsed_before_resume + (time.monotonic() - self._start_time)
            ),
            "finished": bool(self._finished),
        }

    def restore_checkpoint_state(self, state: Mapping[str, object]) -> None:
        if state.get("schema") != ACTION_GATE_SCHEMA or state.get("version") != 1:
            raise ValueError("resume checkpoint lacks the B787 adaptive action-observer state")
        if list(state.get("probe_source_view_ids", ())) != self.probe_ids:
            raise ValueError("resume checkpoint action-observer probes disagree with this run")
        records = state.get("records")
        pending = state.get("pending")
        optimizer_updates = state.get("optimizer_updates")
        initial_metrics = state.get("initial_metrics")
        initial_parameters = state.get("initial_parameters")
        if not isinstance(records, list) or not isinstance(pending, list) or not isinstance(optimizer_updates, list):
            raise ValueError("resume checkpoint action-observer state is malformed")
        if not isinstance(initial_metrics, Mapping) or not isinstance(initial_parameters, Mapping):
            raise ValueError("resume checkpoint lacks the action gate's initial audit baseline")
        if state.get("finished") is True:
            raise ValueError("a completed B787 adaptive action gate cannot resume")
        elapsed = state.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or float(elapsed) < 0.0:
            raise ValueError("resume checkpoint has an invalid action-observer elapsed time")
        restored_parameters: dict[str, torch.Tensor] = {}
        for name, value in initial_parameters.items():
            if not isinstance(name, str) or not torch.is_tensor(value):
                raise ValueError("resume checkpoint has malformed initial parameter evidence")
            restored_parameters[name] = value.detach().cpu().clone()
        if set(restored_parameters) != set(self._initial_parameters):
            raise ValueError("resume checkpoint initial parameter identities disagree with this action gate")
        for name, value in restored_parameters.items():
            if value.shape != self._initial_parameters[name].shape:
                raise ValueError("resume checkpoint initial parameter shape disagrees with this action gate")
        self.records = copy.deepcopy(records)
        self.pending = copy.deepcopy(pending)
        self.optimizer_updates = copy.deepcopy(optimizer_updates)
        self._initial_metrics = copy.deepcopy(dict(initial_metrics))
        self._initial_parameters = restored_parameters
        self._elapsed_before_resume = float(elapsed)
        # Construction happens before generic train_sar restores the model. Its
        # provisional constructor audit must not count as resumed wall time.
        self._start_time = time.monotonic()

    def finalize(self) -> dict[str, Any]:
        self._final_metrics = self._role_metrics(self.train_loader, self.validation_loader)
        current = self._parameter_snapshot()
        parameter_delta_sq = 0.0
        scene_parameter_delta_sq = 0.0
        gain_parameter_delta_sq = 0.0
        for name, before in self._initial_parameters.items():
            after = current.get(name)
            if after is None or after.shape != before.shape:
                raise RuntimeError(f"parameter identity changed during action gate: {name}")
            delta_sq = float((after - before).square().sum().item())
            parameter_delta_sq += delta_sq
            if name.startswith("scene."):
                scene_parameter_delta_sq += delta_sq
            elif name.startswith("gain."):
                gain_parameter_delta_sq += delta_sq
        action_spatial = [entry for entry in self.pending if entry["children"]]
        action_angular = [entry for entry in self.pending if entry["unlocked_indices"]]
        passed = {
            "expected_update_count": (
                len(self.optimizer_updates) == ACTION_GATE_EXPECTED_UPDATES
                and [int(entry["logical_optimizer_updates"]) for entry in self.optimizer_updates]
                == list(range(1, ACTION_GATE_EXPECTED_UPDATES + 1))
            ),
            "probe_prediction_preservation": bool(self.records) and all(
                bool(record.get("probe_full_coherent_prediction_preserved"))
                for record in self.records
            ),
            "spatial_action_with_later_child_update": any(
                bool(entry.get("child_updated")) and bool(entry.get("checked_after_update"))
                for entry in action_spatial
            ),
            "angular_action_with_later_band_update": any(
                bool(entry.get("band_updated")) and bool(entry.get("checked_after_update"))
                for entry in action_angular
            ),
            "finite_event_state": bool(self.records) and all(
                bool(record.get("model_parameters_finite"))
                and bool(record.get("optimizer_state_finite"))
                and bool(record.get("child_birth_event_ok"))
                and bool(record.get("children_zero_at_birth"))
                and bool(record.get("child_optimizer_rows_zero_at_birth"))
                and bool(record.get("unlocked_band_zero_at_unlock"))
                and bool(record.get("unlocked_band_optimizer_columns_zero"))
                and bool(record.get("probe_full_coherent_predictions_finite"))
                and bool(record.get("scene_after", {}).get("support_bounds_ok"))
                and bool(record.get("scene_after", {}).get("degree_bounds_ok"))
                for record in self.records
            ),
            # A lazy global-gain initialization can legitimately change the
            # gain before its first optimizer step. The scene-only delta is the
            # relevant evidence that the actual inverse representation moved.
            "nonzero_scene_parameter_change": scene_parameter_delta_sq > 0.0,
            "fixed_role_metrics_finite": (
                _role_metrics_are_finite(self._initial_metrics)
                and _role_metrics_are_finite(self._final_metrics)
            ),
            "completed_observer_timeline": bool(self._finished),
            "finite_logical_updates": (
                len(self.optimizer_updates) == ACTION_GATE_EXPECTED_UPDATES
                and all(
                    bool(entry.get("finite_grad_norm"))
                    and bool(entry.get("model_parameters_finite"))
                    and bool(entry.get("optimizer_state_finite"))
                    for entry in self.optimizer_updates
                )
            ),
        }
        torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
        elapsed = self._elapsed_before_resume + (time.monotonic() - self._start_time)
        allocated = int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0
        reserved = int(torch.cuda.max_memory_reserved(self.device)) if self.device.type == "cuda" else 0
        return {
            "schema": ACTION_GATE_SCHEMA,
            "version": 1,
            "engineering_status": "not_production_not_comparison_not_convergence_evidence",
            "probe_source_view_ids": list(self.probe_ids),
            "initial_metrics": self._initial_metrics,
            "final_metrics": self._final_metrics,
            "parameter_change_l2": math.sqrt(parameter_delta_sq),
            "scene_parameter_change_l2": math.sqrt(scene_parameter_delta_sq),
            "gain_parameter_change_l2": math.sqrt(gain_parameter_delta_sq),
            "events": copy.deepcopy(self.records),
            "post_event_updates": copy.deepcopy(self.pending),
            "optimizer_updates": copy.deepcopy(self.optimizer_updates),
            "final_scene": self._scene_summary(),
            "checks": passed,
            "pass": bool(all(passed.values())),
            "wall_seconds_from_initial_metric": float(elapsed),
            "peak_torch_allocated_bytes": allocated,
            "peak_torch_reserved_bytes": reserved,
            "process_max_rss_kib": (
                int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
                if _resource is not None else None
            ),
        }


def write_action_gate_report(path: str | os.PathLike[str], payload: Mapping[str, object]) -> Path:
    """Atomically publish a data-free JSON report beside the smoke checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return destination


def require_action_gate_pass(report: Mapping[str, object]) -> None:
    checks = report.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("adaptive action-gate report lacks checks")
    failed = [name for name, value in checks.items() if value is not True]
    if failed:
        raise RuntimeError("B787 adaptive action gate did not pass: " + ", ".join(failed))
