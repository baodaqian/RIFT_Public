"""Run the manager-agnostic Camry full-polarization RIFT experiment.

The command owns no scheduler or deployment state.  It accepts four native
``NativeShard`` archives, inventories the complete P1/sector-002 TRAIN panel,
and writes a fail-closed preflight manifest.  ``--dry-run`` stops after that
manifest and deliberately imports no Torch/core trainer, so it cannot allocate
GPU tensors.  A real run builds the bounded implicit-lattice support, performs
the matched BP and DC-CGLS24 initialization, then delegates the fixed L=3,
four-gain, 150-epoch lifecycle to :mod:`rift.gotcha_camry_fullpol_core`.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLARIZATIONS = ("hh", "hv", "vh", "vv")
PROTOCOL_SCHEMA = "rift_gotcha_camry_fullpol_protocol_v1"
DRIVER_SCHEMA = "rift_gotcha_camry_fullpol_driver_v1"
SPEED_OF_LIGHT_M_S = 299_792_458.0
MAX_ACTIVE_SITES = 8_192
SH_BASIS_COUNT = 16
EPOCHS = 150
SEED = 42
SCOUT_BLOCK_WIDTH = 16
SCOUT_INTERNAL_BLOCK_QUOTA = 2_048
SCOUT_MAX_BISECTIONS = 4
SCOUT_MAX_FINAL_LEAVES = MAX_ACTIVE_SITES
RESOURCE_QOS = "inferno"
RESOURCE_PARTITION = "gpu-a100"
RESOURCE_GPU = "A100"
DEVICE_SCHEMA = "rift_gotcha_camry_fullpol_device_v1"
CHECKPOINT_NAMES = (
    "checkpoint_bp.npz",
    "checkpoint_cgls.pt",
    "checkpoint_initialization.pt",
    "checkpoint_best.pt",
    "checkpoint_latest.pt",
    "checkpoint_final.pt",
)


def _load_module(name: str, path: Path):
    """Load the NumPy-only acquisition adapter without eager package imports."""

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# This adapter is intentionally importable in the desktop's Torch-free authoring
# runtime.  The core trainer is imported lazily only for a non-dry run.
ACQ = _load_module("gotcha_camry_fullpol_acquisition", PROJECT_ROOT / "rift" / "gotcha_acquisition.py")
NATIVE_ROI = _load_module("gotcha_camry_fullpol_native_roi", PROJECT_ROOT / "rift" / "gotcha_native_complex_roi_fit.py")


@dataclass(frozen=True)
class FrozenCamryConfig:
    """The non-negotiable experiment settings mirrored by the JSON protocol."""

    epochs: int = EPOCHS
    seed: int = SEED
    sh_degree: int = 3
    basis_kind: str = "real_sh"
    sh_order: str = "degree_major_m_minus_l_to_l"
    coefficients_are_complex: bool = True
    max_active_sites: int = MAX_ACTIVE_SITES
    scene_lr: float = 3.0e-3
    gain_lr: float = 3.0e-3
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1.0e-8
    weight_decay: float = 0.0
    scheduler_t0: int = 10
    scheduler_tmult: int = 2
    scheduler_eta_min: float = 1.0e-6

    def validate(self) -> None:
        if self.epochs != EPOCHS or self.seed != SEED:
            raise ValueError("Camry full-pol config is frozen at 150 epochs and seed 42")
        if (self.sh_degree, self.basis_kind, self.sh_order, self.coefficients_are_complex) != (
            3,
            "real_sh",
            "degree_major_m_minus_l_to_l",
            True,
        ):
            raise ValueError("Camry full-pol SH basis contract changed")
        if self.max_active_sites != MAX_ACTIVE_SITES:
            raise ValueError("Camry support capacity is frozen at K<=8192")
        if self.scene_lr != 3.0e-3 or self.gain_lr != 3.0e-3:
            raise ValueError("Camry optimizer learning rates are frozen at 3e-3")
        if self.adam_betas != (0.9, 0.999) or self.adam_eps != 1.0e-8 or self.weight_decay != 0.0:
            raise ValueError("Camry AdamW settings changed")
        if (self.scheduler_t0, self.scheduler_tmult, self.scheduler_eta_min) != (10, 2, 1.0e-6):
            raise ValueError("Camry cosine-restart scheduler settings changed")

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "seed": self.seed,
            "sh_degree": self.sh_degree,
            "basis_kind": self.basis_kind,
            "sh_order": self.sh_order,
            "coefficients_are_complex": self.coefficients_are_complex,
            "max_active_sites": self.max_active_sites,
            "scene_lr": self.scene_lr,
            "gain_lr": self.gain_lr,
            "adam_betas": list(self.adam_betas),
            "adam_eps": self.adam_eps,
            "weight_decay": self.weight_decay,
            "scheduler": {
                "name": "CosineAnnealingWarmRestarts",
                "T0": self.scheduler_t0,
                "T_mult": self.scheduler_tmult,
                "eta_min": self.scheduler_eta_min,
                "step_rule": "once_after_each_completed_epoch",
            },
        }


FROZEN_CONFIG = FrozenCamryConfig()
FROZEN_CONFIG.validate()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _relative_error_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a metrics payload into the four-channel TRAIN comparison form."""

    by_polarization = {
        polarization: float(metrics["range_relmse_by_polarization"][polarization])
        for polarization in POLARIZATIONS
    }
    macro = float(metrics["range_macro_relmse"])
    if not np.isfinite(macro) or any(not np.isfinite(value) for value in by_polarization.values()):
        raise ValueError("TRAIN relative-error reporting contains a non-finite value")
    return {"by_polarization": by_polarization, "macro": macro}


def _optimizer_improvement(cgls_initial: Mapping[str, Any], best: Mapping[str, Any]) -> dict[str, float | str]:
    """Report signed CGLS-initial-to-best Q reduction; positive means improvement."""

    q_initial = float(cgls_initial["range_macro_relmse"])
    q_best = float(best["range_macro_relmse"])
    if not np.isfinite(q_initial) or not np.isfinite(q_best):
        raise ValueError("cannot report optimizer improvement from non-finite TRAIN metrics")
    absolute = q_initial - q_best
    relative = absolute / max(abs(q_initial), np.finfo(np.float64).tiny)
    by_polarization = {}
    for polarization in POLARIZATIONS:
        q_initial_p = float(cgls_initial["range_relmse_by_polarization"][polarization])
        q_best_p = float(best["range_relmse_by_polarization"][polarization])
        if not np.isfinite(q_initial_p) or not np.isfinite(q_best_p):
            raise ValueError(f"cannot report {polarization.upper()} optimizer improvement from non-finite TRAIN metrics")
        signed_absolute = q_initial_p - q_best_p
        by_polarization[polarization] = {
            "signed_absolute_improvement": float(signed_absolute),
            "signed_relative_improvement": float(signed_absolute / max(abs(q_initial_p), np.finfo(np.float64).tiny)),
        }
    return {
        "definition": "signed reduction (Q_cgls_initial - Q_best); positive means Adam improved TRAIN macro Q",
        "signed_absolute_improvement": float(absolute),
        "signed_relative_improvement": float(relative),
        "by_polarization": by_polarization,
    }


def _scout_bounds(block_count: int) -> tuple[int, int, int, int]:
    """Return safe (roots, parent, terminal-leaf, scored-candidate) bounds.

    A legal split has two through eight children, so it adds at least one and
    at most seven terminal leaves while evaluating at most eight new child
    representatives.  The leaf cap therefore bounds the total number of
    refined parents by ``cap - roots`` regardless of edge-block arity or depth.
    """

    blocks = int(block_count)
    if blocks <= 0:
        raise ValueError("scout block count must be positive")
    roots = min(SCOUT_INTERNAL_BLOCK_QUOTA, blocks)
    parent_bound = max(SCOUT_MAX_FINAL_LEAVES - roots, 0)
    terminal_bound = min(SCOUT_MAX_FINAL_LEAVES, roots + 7 * parent_bound)
    scored_bound = blocks + 8 * parent_bound
    return int(roots), int(parent_bound), int(terminal_bound), int(scored_bound)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_pairs)
    if not isinstance(payload, dict):
        raise ValueError("Camry full-pol protocol must be a JSON object")
    validate_protocol(payload)
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate only static/frozen settings; archive metadata is runtime data."""

    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "unexpected Camry full-pol protocol schema")
    _require(protocol.get("stage") == "full_train_fit", "protocol stage must be full_train_fit")
    _require(protocol.get("test_sealed") is True, "TEST must remain sealed")
    selection = protocol.get("selection")
    _require(isinstance(selection, Mapping), "selection contract is missing")
    for key, expected in {
        "pass_id": 1,
        "sector_id": 2,
        "role": "train",
        "polarizations": list(POLARIZATIONS),
        "frequency_policy": "all_native_stored_exact",
        "cross_polarization_intersection": "not_used; each channel selected independently",
        "frequency_thinning": False,
        "pulse_subsampling": False,
        "canonical_identity": "(pass_id, polarization, sector_id, pulse_index)",
        "identity_order": "ascending canonical identity per channel",
    }.items():
        _require(selection.get(key) == expected, f"selection.{key} changed")
    spatial = protocol.get("spatial")
    _require(isinstance(spatial, Mapping), "spatial contract is missing")
    for key, expected in {
        "target_id": "toyota_camry",
        "cube_side_m": 10.0,
        "h_space": "c/(2*f_max) exact float64",
        "N": "floor(10/h_space)+1",
        "point_definition": "(i-(N-1)/2)*h_space",
        "candidate_lattice": "implicit; never materialize N^3",
        "max_active_sites": MAX_ACTIVE_SITES,
        "max_allocated_sites": MAX_ACTIVE_SITES,
        "support_policy": "fixed legal integer lattice indices for all CGLS/Adam epochs",
    }.items():
        _require(spatial.get(key) == expected, f"spatial.{key} changed")
    _require(spatial.get("placement_equation") == "p_native = R @ p_local + t", "spatial placement equation changed")
    _require(
        spatial.get("placement_qualification") == "footprint_derived_vehicle_axis_inference_not_independently_registered",
        "spatial placement qualification changed",
    )
    _require(spatial.get("physical_registration_claim") is False, "spatial registration claim changed")
    _require(np.array_equal(np.asarray(spatial.get("translation_m"), dtype=np.float64), np.asarray([20.66, -18.71, 0.02], dtype=np.float64)), "spatial translation changed")
    _require(
        np.array_equal(
            np.asarray(spatial.get("rotation"), dtype=np.float64),
            np.asarray(
                [[-0.05442654550121675, 0.9985177770800097, 0.0], [-0.9985177770800097, -0.05442654550121675, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
        ),
        "spatial rotation changed",
    )
    model = protocol.get("model")
    _require(isinstance(model, Mapping), "model contract is missing")
    for key, expected in {
        "sh_degree": 3,
        "basis_kind": "real_sh",
        "sh_order": "degree_major_m_minus_l_to_l",
        "basis_count": SH_BASIS_COUNT,
        "coefficients_are_complex": True,
        "heads": list(POLARIZATIONS),
        "gains": list(POLARIZATIONS),
        "gain_scope": "one constant GlobalComplexGain per complete channel",
        "integrated_point_weights": True,
        "voxel_volume_scaling": False,
    }.items():
        _require(model.get(key) == expected, f"model.{key} changed")
    training = protocol.get("training")
    _require(isinstance(training, Mapping), "training contract is missing")
    for key, expected in {
        "epochs": EPOCHS,
        "seed": SEED,
        "optimizer": "AdamW",
        "scene_lr": 3.0e-3,
        "gain_lr": 3.0e-3,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "logical_updates": "150*max(n_p), one combined step per ragged batch index",
        "cgls": "DC/Y00 only, zero start, at most 24 iterations per channel",
        "cgls_diagnostics": [0, 8, 16, 24],
        "gain_initialization": "one full-channel projected ragged call before Adam",
        "rms_normalization": "one prediction-preserving normalization before optimizer",
    }.items():
        _require(training.get(key) == expected, f"training.{key} changed")
    checkpoint = protocol.get("checkpoints")
    _require(isinstance(checkpoint, Mapping), "checkpoint contract is missing")
    _require(
        checkpoint.get("required") == [
            "checkpoint_bp.npz",
            "checkpoint_cgls.pt",
            "checkpoint_initialization.pt",
            "checkpoint_best.pt",
            "checkpoint_latest.pt",
            "checkpoint_final.pt",
        ],
        "checkpoint set changed",
    )
    _require(checkpoint.get("best_selection") == "strict finite minimum full-panel TRAIN macro Q among epochs 1..150", "best selection changed")
    _require(checkpoint.get("test_opened") is False, "checkpoint TEST contract changed")
    validation = protocol.get("validation")
    _require(isinstance(validation, Mapping), "validation contract is missing")
    cuda_test = "tests.test_gotcha_camry_fullpol_core.CamryTrainingContractTest.test_cuda_checkpoint_round_trip_restores_rng_optimizer_devices_and_resume_readiness"
    _require(validation.get("same_allocation_before_fit") is True, "CUDA validation must run in the fit allocation")
    _require(validation.get("runtime_validator") == "scripts/validate_gotcha_camry_fullpol_runtime.py", "runtime validator path changed")
    _require(validation.get("runtime_command") == "python3 -B scripts/validate_gotcha_camry_fullpol_runtime.py", "runtime validator command changed")
    _require(validation.get("expected_core_test_count") == 19, "expected core test count changed")
    _require(validation.get("cuda_required_test") == cuda_test, "CUDA checkpoint validation test changed")
    _require(validation.get("cuda_skip_is_failure") is True, "CUDA validation skip must fail closed")
    outputs = protocol.get("outputs")
    _require(isinstance(outputs, Mapping), "output artifact contract is missing")
    for key in ("preflight", "protocol_echo", "config", "device", "resource", "support_search", "bp_metrics", "cgls_metrics", "initialization_metrics", "metrics", "support_readout", "gain_values", "training_history", "checkpoint_manifest", "status"):
        _require(isinstance(outputs.get(key), str) and outputs.get(key), f"outputs.{key} is missing")
    resource = protocol.get("resource")
    _require(isinstance(resource, Mapping), "resource contract is missing")
    for key, expected in {
        "qos": RESOURCE_QOS,
        "partition": RESOURCE_PARTITION,
        "gpu": RESOURCE_GPU,
        "cpus": 8,
        "ram_gib": 64,
        "tmp_gib": 16,
        "walltime": "2-00:00:00",
    }.items():
        _require(resource.get(key) == expected, f"resource.{key} changed")


def select_execution_device(
    torch_module: Any,
    requested_device: Any = None,
    *,
    protocol: Mapping[str, Any],
    test_device: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Select a device explicitly and fail closed for the production protocol.

    CPU execution is available only to callers that explicitly inject a test
    device.  A real protocol run defaults to CUDA and additionally checks that
    the allocated device identifies as an A100, matching the frozen resource
    contract.  The small Torch-module seam keeps this contract testable without
    importing Torch in the desktop's Torch-free authoring runtime.
    """

    if test_device and requested_device is None:
        raise ValueError("test_device requires an explicit device injection")
    raw_device = "cuda" if requested_device is None else requested_device
    device = torch_module.device(raw_device)
    device_type = str(getattr(device, "type", "")).lower()
    requested_text = str(raw_device)
    if device_type == "cpu":
        if not test_device:
            raise RuntimeError("production Camry full-pol runs require CUDA/A100; CPU is test-only")
        return device, {
            "schema": DEVICE_SCHEMA,
            "mode": "explicit_test_device",
            "requested_device": requested_text,
            "actual_device": "cpu",
            "device_type": "cpu",
            "device_index": None,
            "cuda_available": bool(getattr(torch_module.cuda, "is_available", lambda: False)()),
            "gpu_name": None,
            "a100_compatible": False,
            "gpu_properties": {},
            "test_opened": False,
        }
    if device_type != "cuda":
        raise ValueError(f"Camry full-pol execution device must be CUDA, got {device_type or device!r}")
    if not bool(torch_module.cuda.is_available()):
        raise RuntimeError("Camry full-pol production run requires torch.cuda.is_available()")
    device_count = int(torch_module.cuda.device_count())
    if device_count != 1:
        raise RuntimeError(f"production Camry full-pol runs require exactly one visible CUDA device, got {device_count}")
    current_index = int(torch_module.cuda.current_device())
    index = getattr(device, "index", None)
    if index is None:
        index = current_index
    index = int(index)
    if index != current_index:
        raise RuntimeError(f"production Camry full-pol run must use the current CUDA device {current_index}, got {index}")
    gpu_name = str(torch_module.cuda.get_device_name(index))
    if not gpu_name:
        raise RuntimeError("CUDA device name is unavailable")
    resource = protocol.get("resource")
    if not isinstance(resource, Mapping) or resource.get("gpu") != RESOURCE_GPU:
        raise ValueError("production protocol must request an A100 GPU")
    a100_compatible = "a100" in gpu_name.lower()
    if not test_device and not a100_compatible:
        raise RuntimeError(f"production Camry full-pol run requires an A100-compatible GPU, got {gpu_name!r}")
    properties = torch_module.cuda.get_device_properties(index)
    property_names = ("total_memory", "major", "minor", "multi_processor_count")
    gpu_properties = {
        name: (None if getattr(properties, name, None) is None else int(getattr(properties, name)))
        for name in property_names
    }
    return device, {
        "schema": DEVICE_SCHEMA,
        "mode": "production" if not test_device else "explicit_test_device",
        "requested_device": requested_text,
        "actual_device": f"cuda:{index}",
        "device_type": "cuda",
        "device_index": index,
        "cuda_available": True,
        "gpu_name": gpu_name,
        "a100_compatible": a100_compatible,
        "gpu_count": device_count,
        "gpu_properties": gpu_properties,
        "test_opened": False,
    }


def _identity_key(identity: Any) -> tuple[int, str, int, int]:
    try:
        return (
            int(identity.pass_id),
            str(identity.polarization).lower(),
            int(identity.sector_id),
            int(identity.pulse_index),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("native identities must expose pass/polarization/sector/pulse") from exc


def resolve_archive_paths(
    archive_paths: Sequence[str | Path] | None,
    archive_root: str | Path | None,
) -> tuple[Path, ...]:
    if archive_paths and archive_root is not None:
        raise ValueError("provide --archive four times or --archive-root, not both")
    if archive_root is not None:
        root = Path(archive_root)
        paths = tuple(root / "shards" / f"pass1_{polarization}.npz" for polarization in POLARIZATIONS)
    elif archive_paths is not None:
        paths = tuple(Path(path) for path in archive_paths)
    else:
        raise ValueError("four --archive paths or --archive-root are required")
    if len(paths) != 4:
        raise ValueError("exactly four NativeShard archive paths are required")
    resolved = tuple(path.resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("NativeShard archive paths must be distinct")
    return resolved


def _selected_header_inventory(shard: Any) -> dict[str, Any]:
    polarization = str(getattr(shard, "polarization", "")).lower()
    if polarization not in POLARIZATIONS or int(getattr(shard, "pass_id", -1)) != 1:
        raise ValueError("Camry archives must be one P1 shard for HH/HV/VH/VV")
    identities = tuple(getattr(shard, "observation_ids", ()))
    roles = tuple(str(value).lower() for value in np.asarray(getattr(shard, "role", ())).tolist())
    if len(identities) == 0 or len(identities) != len(roles):
        raise ValueError(f"{polarization.upper()} identity/role vectors are incomplete")
    keys = tuple(_identity_key(identity) for identity in identities)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{polarization.upper()} archive contains duplicate identities")
    if any(key[0] != 1 or key[1] != polarization for key in keys):
        raise ValueError(f"{polarization.upper()} archive identity header is inconsistent")
    if any(role == "test" for role in roles):
        raise ValueError(f"{polarization.upper()} archive exposes sealed TEST rows")
    if any(role not in {"train", "validation"} for role in roles):
        raise ValueError(f"{polarization.upper()} archive contains an unknown role")
    frequencies = np.asarray(getattr(shard, "frequencies_hz"), dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0 or not np.isfinite(frequencies).all():
        raise ValueError(f"{polarization.upper()} archive has an invalid native frequency vector")
    if frequencies.size > 1 and not np.all(np.diff(frequencies) > 0.0):
        raise ValueError(f"{polarization.upper()} native frequencies are not strictly increasing")
    selected = tuple(
        sorted((identity for identity, role in zip(identities, roles) if role == "train" and int(identity.sector_id) == 2), key=_identity_key)
    )
    if not selected:
        raise ValueError(f"{polarization.upper()} archive has no P1/sector-002 TRAIN records")
    autofocus = getattr(shard, "autofocus", None)
    phase_reference = getattr(shard, "phase_reference", None)
    if polarization in {"hh", "vv"}:
        if autofocus is None or getattr(autofocus, "official_available", False) is not True or bool(getattr(autofocus, "applied", True)):
            raise ValueError(f"{polarization.upper()} archive lacks unapplied own source-AF provenance")
        if getattr(phase_reference, "reference_range_field", None) != "r0" or getattr(phase_reference, "geometry_contract", None) != "paired_monostatic_tx_equals_rx_same_observation":
            raise ValueError(f"{polarization.upper()} archive lacks native paired-monostatic r0 provenance")
        if polarization == "vv" and (
            getattr(phase_reference, "frequency_values", None) != "native_stored_exact"
            or getattr(autofocus, "mode", None) != "raw_channel_own_arrays_unapplied"
            or getattr(autofocus, "source_shard_id", None) != "pass1_vv"
            or getattr(autofocus, "range_field", None) not in {"r_correct_raw", "af.r_correct"}
            or getattr(autofocus, "phase_field", None) not in {"ph_correct_raw", "af.ph_correct"}
        ):
            raise ValueError("VV archive source-AF provenance is not the native pass1_vv representation")
        range_correction = np.asarray(getattr(shard, "r_correct_raw"), dtype=np.float64)
        phase_correction = np.asarray(getattr(shard, "ph_correct_raw"), dtype=np.float64)
        if range_correction.shape != (len(identities),) or phase_correction.shape != (len(identities),):
            raise ValueError(f"{polarization.upper()} correction vectors are not row-aligned")
        if not np.isfinite(range_correction).all() or not np.isfinite(phase_correction).all():
            raise ValueError(f"{polarization.upper()} correction vectors contain non-finite values")
    else:
        if autofocus is None or getattr(autofocus, "official_available", True) is not False or bool(getattr(autofocus, "applied", True)):
            raise ValueError(f"{polarization.upper()} archive must retain raw no-source-AF provenance")
        if np.asarray(getattr(shard, "r_correct_raw")).size or np.asarray(getattr(shard, "ph_correct_raw")).size:
            raise ValueError(f"{polarization.upper()} archive must not carry correction arrays")
    phase_reference_metadata = {
        key: getattr(phase_reference, key, None)
        for key in (
            "schema",
            "frequency_unit",
            "position_unit",
            "range_unit",
            "phase_unit",
            "frequency_values",
            "reference_range_field",
            "geometry_contract",
            "correction_application_convention",
        )
    }
    autofocus_metadata = {
        key: getattr(autofocus, key, None)
        for key in (
            "schema",
            "mode",
            "official_available",
            "applied",
            "source_shard_id",
            "range_field",
            "phase_field",
        )
    }
    return {
        "shard_id": str(getattr(shard, "shard_id")),
        "polarization": polarization,
        "pass_id": 1,
        "view_count": int(getattr(shard, "view_count")),
        "native_frequency_count": int(frequencies.size),
        "native_fmin_hz": float(frequencies[0]),
        "native_fmax_hz": float(frequencies[-1]),
        "selected_record_count": len(selected),
        "selected_native_sample_count": int(len(selected) * frequencies.size),
        "selected_record_ids": [list(_identity_key(identity)) for identity in selected],
        "source_af_representation": "own_source_af_once" if polarization in {"hh", "vv"} else "raw_unapplied",
        "phase_reference": phase_reference_metadata,
        "autofocus": autofocus_metadata,
        "test_opened": False,
    }


def preflight_archives(
    paths: Sequence[Path],
    *,
    expected_scene_id: str = "gotcha_v1_joint8_fullpol",
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Load and inventory native headers; return shards only after all gates pass."""

    shards: list[Any] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"NativeShard path must be a regular file: {path}")
        shards.append(
            ACQ.load_native_shard(
                path,
                expected_pass_id=1,
                expected_scene_id=expected_scene_id,
            )
        )
    inventories = [_selected_header_inventory(shard) for shard in shards]
    by_pol = {entry["polarization"]: entry for entry in inventories}
    if set(by_pol) != set(POLARIZATIONS) or len(by_pol) != len(POLARIZATIONS):
        raise ValueError("the four archives must contain exactly HH/HV/VH/VV")
    native_fmax = float(max(entry["native_fmax_hz"] for entry in inventories))
    h_space = float(np.float64(SPEED_OF_LIGHT_M_S) / np.float64(2.0 * native_fmax))
    N = int(math.floor(10.0 / h_space)) + 1
    edge_margin = float((10.0 - (N - 1) * h_space) / 2.0)
    candidate_count = int(N) ** 3
    total_samples = int(sum(entry["selected_native_sample_count"] for entry in inventories))
    U = int(max(entry["selected_record_count"] for entry in inventories))
    blocks_per_axis = int(math.ceil(N / SCOUT_BLOCK_WIDTH))
    block_count = blocks_per_axis ** 3
    K = MAX_ACTIVE_SITES
    coefficient_bytes = int(len(POLARIZATIONS) * K * SH_BASIS_COUNT * 16)
    top_block_count, refinement_parent_bound, scout_terminal_leaf_bound, scout_scoring_bound = _scout_bounds(block_count)
    resource = {
        "qos": RESOURCE_QOS,
        "partition": RESOURCE_PARTITION,
        "gpu": RESOURCE_GPU,
        "selected_native_samples_total": total_samples,
        "records_per_channel": {entry["polarization"]: entry["selected_record_count"] for entry in inventories},
        "logical_batches_per_epoch_U": U,
        "logical_updates": int(EPOCHS * U),
        "record_visits": int(EPOCHS * sum(entry["selected_record_count"] for entry in inventories)),
        "fmax_hz": native_fmax,
        "h_space_m": h_space,
        "N": N,
        "candidate_count_N3_implicit": candidate_count,
        "dense_N3_materialized": False,
        "scout": {
            "block_width": SCOUT_BLOCK_WIDTH,
            "blocks_per_axis": blocks_per_axis,
            "representative_blocks": block_count,
            "internal_block_quota": SCOUT_INTERNAL_BLOCK_QUOTA,
            "top_blocks_scored": top_block_count,
            "max_bisections": SCOUT_MAX_BISECTIONS,
            "max_final_leaves": SCOUT_MAX_FINAL_LEAVES,
            "support_allocation_cap": SCOUT_MAX_FINAL_LEAVES,
            "refinement_parent_bound": refinement_parent_bound,
            "terminal_leaf_bound": scout_terminal_leaf_bound,
            "scored_candidate_evaluation_bound": scout_scoring_bound,
            "representatives_rescored": False,
        },
        "work_estimate_direct_terms": {
            "scout_dc_adjoint": int(scout_scoring_bound * total_samples),
            "active_bp_forward_adjoint": int(2 * K * total_samples),
            "cgls_three_pass_recurrence": int(3 * K * total_samples * 24),
            "training_forward_and_adjoint": int(2 * EPOCHS * K * total_samples),
            "epoch_end_full_panel_evaluations": int(EPOCHS * K * total_samples),
            "subtotal_before_scout_and_io": int((2 * EPOCHS + EPOCHS) * K * total_samples + 3 * K * total_samples * 24),
            "cgls_accounting_note": "single three-pass CGLS recurrence term; counted once in subtotal",
        },
        "memory_estimate_bytes": {
            "complex128_scene_coefficients_all_channels_at_K8192": coefficient_bytes,
            "two_adam_moments_scene_lower_bound": int(2 * coefficient_bytes),
            "support_points_and_indices": int(K * (3 * 8 + 3 * 8)),
            "checkpoint_model_optimizer_state_lower_bound": int(4 * coefficient_bytes),
            "scout_terminal_leaf_indices_upper_bound": int(scout_terminal_leaf_bound * 3 * 8),
            "scout_scored_candidate_indices_upper_bound": int(scout_scoring_bound * 3 * 8),
            "scout_chunk_geometry_upper_bound": int(128 * 3 * 8),
            "per_channel_gain_state": int(4 * 3 * 8),
            "note": "kernel/autograd buffers dominate; estimate is before GPU allocation",
        },
        "resource_ceiling": {
            "qos": RESOURCE_QOS,
            "partition": RESOURCE_PARTITION,
            "gpu": RESOURCE_GPU,
            "cpus": 8,
            "ram_gib": 64,
            "tmp_gib": 16,
            "walltime": "2-00:00:00",
        },
    }
    report = {
        "schema": f"{DRIVER_SCHEMA}.preflight",
        "protocol_schema": PROTOCOL_SCHEMA,
        "archive_paths": [str(path) for path in paths],
        "expected_scene_id": expected_scene_id,
        "selection": {
            "pass_id": 1,
            "sector_id": 2,
            "role": "train",
            "polarizations": list(POLARIZATIONS),
            "independent_channel_selection": True,
            "cross_polarization_intersection_used": False,
            "frequency_thinning": False,
            "pulse_subsampling": False,
            "test_opened": False,
        },
        "channels": {polarization: by_pol[polarization] for polarization in POLARIZATIONS},
        "spatial": {
            "target_id": "toyota_camry",
            "cube_side_m": 10.0,
            "placement": {
                "name": NATIVE_ROI.CAMRY_PLACEMENT.name,
                "rotation": np.asarray(NATIVE_ROI.CAMRY_PLACEMENT.rotation, dtype=np.float64),
                "translation_m": np.asarray(NATIVE_ROI.CAMRY_PLACEMENT.translation_m, dtype=np.float64),
                "equation": "p_native = R @ p_local + t",
            },
            "fmax_hz": native_fmax,
            "h_space_definition": "float64(c)/(float64(2)*float64(fmax_hz))",
            "h_space_m": h_space,
            "N": N,
            "origin_m": [-0.5 * float(N - 1) * h_space] * 3,
            "edge_margin_m": edge_margin,
            "centered_points": "(i-(N-1)/2)*h_space",
            "candidate_count": candidate_count,
            "implicit": True,
            "allocated_site_cap": MAX_ACTIVE_SITES,
        },
        "resource": resource,
        "status": "PASS",
        "test_opened": False,
    }
    return report, tuple(shards)


def _block_representatives(N: int, width: int = SCOUT_BLOCK_WIDTH) -> tuple[np.ndarray, tuple[tuple[int, int, int], ...]]:
    """Enumerate one legal lattice representative per block, never an N^3 grid."""

    if N <= 0 or width <= 0 or width > 16:
        raise ValueError("N must be positive and scout block width must lie in 1..16")
    blocks = int(math.ceil(N / width))
    indices: list[tuple[int, int, int]] = []
    block_ids: list[tuple[int, int, int]] = []
    for bx in range(blocks):
        for by in range(blocks):
            for bz in range(blocks):
                starts = (bx * width, by * width, bz * width)
                stops = tuple(min(start + width, N) for start in starts)
                indices.append(tuple((start + stop - 1) // 2 for start, stop in zip(starts, stops)))
                block_ids.append((bx, by, bz))
    return np.asarray(indices, dtype=np.int64), tuple(block_ids)


def _root_region(
    block: tuple[int, int, int],
    N: int,
    width: int = SCOUT_BLOCK_WIDTH,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if N <= 0 or width <= 0 or width > 16:
        raise ValueError("N must be positive and scout block width must lie in 1..16")
    ranges = []
    for axis in block:
        start = int(axis) * width
        stop = min(start + width, N)
        if start >= stop:
            raise ValueError("scout block is outside the candidate lattice")
        ranges.append((start, stop))
    return tuple(ranges)  # type: ignore[return-value]


def _split_region(
    region: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], ...]:
    children: list[list[tuple[int, int]]] = [[]]
    for start, stop in region:
        split = start + (stop - start) // 2
        axis_children = [(start, split), (split, stop)] if split > start and split < stop else [(start, stop)]
        children = [prefix + [child] for prefix in children for child in axis_children]
    return tuple(tuple(child) for child in children)  # type: ignore[return-value]


def _region_representative(
    region: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> tuple[int, int, int]:
    return tuple((start + stop - 1) // 2 for start, stop in region)  # type: ignore[return-value]


def _legal_refinement_children(
    region: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    scored_indices: set[tuple[int, int, int]],
) -> tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], ...]:
    """Return children whose representatives have not already been scored."""

    children = _split_region(region)
    if len(children) <= 1:
        return ()
    child_indices = [_region_representative(child) for child in children]
    if len(set(child_indices)) != len(child_indices):
        return ()
    if any(index in scored_indices for index in child_indices):
        return ()
    return children


def _refined_representatives(
    block_ids: Sequence[tuple[int, int, int]],
    N: int,
    width: int = SCOUT_BLOCK_WIDTH,
    *,
    max_bisections: int = 1,
) -> np.ndarray:
    """Return terminal representatives after a bounded deterministic bisection."""

    if max_bisections < 0 or max_bisections > SCOUT_MAX_BISECTIONS:
        raise ValueError("scout bisections must lie in 0..4")
    regions = [_root_region(block, N, width) for block in block_ids]
    for _ in range(max_bisections):
        regions = [child for region in regions for child in _split_region(region)]
    output = {_region_representative(region) for region in regions}
    if not output:
        raise ValueError("scout refinement produced no legal candidates")
    return np.asarray(sorted(output), dtype=np.int64)


def _one_bisection_children(
    block: tuple[int, int, int],
    N: int,
    width: int = SCOUT_BLOCK_WIDTH,
) -> np.ndarray:
    """Return the eight-or-fewer legal child representatives for one block."""

    return _refined_representatives((block,), N, width, max_bisections=1)


def _scout_scores(core: Any, panel: Any, lattice: Any, projectors: Mapping[str, Any], indices: np.ndarray, *, chunk_size: int = 128) -> np.ndarray:
    """Score candidate sites with the all-channel DC projected adjoint."""

    scores = np.zeros((indices.shape[0],), dtype=np.float64)
    target_adjoint: dict[str, Any] = {}
    target_energy: dict[str, float] = {}
    for polarization in POLARIZATIONS:
        target = projectors[polarization].target()
        target_adjoint[polarization] = projectors[polarization].apply_adjoint(target)
        target_energy[polarization] = float(sum(np.vdot(value, value).real for value in target.values))
        if not np.isfinite(target_energy[polarization]) or target_energy[polarization] <= 0.0:
            raise ValueError(f"{polarization.upper()} support scout target energy is non-positive/non-finite")
    for start in range(0, indices.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), indices.shape[0])
        points_native = lattice.points_native(indices[start:stop])
        operator = core.CamryL3SparseOperator(points_native, point_chunk_size=min(256, stop - start))
        chunk_score = np.zeros((stop - start,), dtype=np.float64)
        for polarization in POLARIZATIONS:
            values = np.asarray(operator.adjoint_dc(panel.records(polarization), target_adjoint[polarization]), dtype=np.complex128)
            if values.shape != (stop - start,):
                raise ValueError(f"{polarization.upper()} support scout adjoint must have shape [{stop - start}]")
            chunk_score += (np.abs(values) ** 2) / target_energy[polarization]
        scores[start:stop] = chunk_score
    if not np.isfinite(scores).all() or np.any(scores < 0.0):
        raise FloatingPointError("all-channel support scout produced non-finite/negative scores")
    return scores


def scout_shared_support(core: Any, panel: Any, lattice: Any, projectors: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the deterministic terminal-leaf support search.

    Every implicit width-16-or-smaller block is scored once.  The top 2,048
    blocks are retained as roots.  Best-first terminal leaves are bisected up
    to four levels while the 8,192-leaf cap permits it; unselected coarse roots
    remain terminal leaves, while a refined root is replaced by its children.
    Each child batch is scored exactly once and root representatives are never
    rescored.  Edge blocks may have variable arity, so the live terminal-leaf
    cap—not an assumed seven-leaf increment—controls stopping.  The proved
    worst-case parent bound is cap minus retained roots; smaller synthetic
    lattices and edge-heavy selections may consume deeper levels.
    """

    reps, blocks = _block_representatives(lattice.N)
    rep_scores = _scout_scores(core, panel, lattice, projectors, reps)
    order = np.argsort(-rep_scores, kind="stable")
    top_count, refinement_parent_bound, terminal_leaf_bound, scored_candidate_bound = _scout_bounds(reps.shape[0])
    top_order = tuple(int(index) for index in order[:top_count])
    leaves: list[dict[str, Any]] = [
        {
            "region": _root_region(blocks[index], lattice.N),
            "depth": 0,
            "index": np.asarray(reps[index], dtype=np.int64),
            "score": float(rep_scores[index]),
            "order": rank,
        }
        for rank, index in enumerate(top_order)
    ]
    refinement_parent_count = 0
    scored_children = 0
    scoring_passes = 1
    actual_bisections = 0
    stopping_reason = "max_bisections_reached"
    next_leaf_order = top_count
    scored_index_keys = {tuple(int(value) for value in index) for index in reps}
    for depth in range(SCOUT_MAX_BISECTIONS):
        split_capacity = max((SCOUT_MAX_FINAL_LEAVES - len(leaves)) // 7, 0)
        if split_capacity <= 0:
            stopping_reason = "terminal_leaf_cap_reached"
            break
        eligible = [
            (position, leaf, children)
            for position, leaf in enumerate(leaves)
            if int(leaf["depth"]) == depth
            for children in (_legal_refinement_children(leaf["region"], scored_index_keys),)
            if children
        ]
        if not eligible:
            stopping_reason = "no_legal_refinement_at_next_depth"
            break
        eligible.sort(key=lambda item: (-float(item[1]["score"]), int(item[1]["order"])))
        selected_positions = {position for position, _, _ in eligible[: min(split_capacity, len(eligible))]}
        parent_specs = [
            (position, leaf, children)
            for position, leaf in enumerate(leaves)
            if position in selected_positions
            for children in (_legal_refinement_children(leaf["region"], scored_index_keys),)
        ]
        child_regions = [child for _, _, children in parent_specs for child in children]
        child_indices = np.asarray([_region_representative(region) for region in child_regions], dtype=np.int64)
        child_scores = _scout_scores(core, panel, lattice, projectors, child_indices)
        scoring_passes += 1
        scored_children += int(child_indices.shape[0])
        replacements: dict[int, list[dict[str, Any]]] = {}
        score_offset = 0
        for position, leaf, children in parent_specs:
            replacement = []
            for child in children:
                replacement.append(
                    {
                        "region": child,
                        "depth": depth + 1,
                        "index": child_indices[score_offset],
                        "score": float(child_scores[score_offset]),
                        "order": next_leaf_order,
                    }
                )
                next_leaf_order += 1
                score_offset += 1
            replacements[position] = replacement
        scored_index_keys.update(tuple(int(value) for value in index) for index in child_indices)
        leaves = [replacement for position, leaf in enumerate(leaves) for replacement in (replacements[position] if position in replacements else [leaf])]
        refinement_parent_count += len(parent_specs)
        actual_bisections = depth + 1
    if actual_bisections == SCOUT_MAX_BISECTIONS and len(leaves) < SCOUT_MAX_FINAL_LEAVES:
        stopping_reason = "maximum_bisections_reached"
    candidates = np.asarray([leaf["index"] for leaf in leaves], dtype=np.int64)
    candidate_scores = np.asarray([leaf["score"] for leaf in leaves], dtype=np.float64)
    terminal_leaf_count = int(len(leaves))
    if terminal_leaf_count > SCOUT_MAX_FINAL_LEAVES:
        raise RuntimeError("support scout exceeded the frozen terminal-leaf cap")
    selected = core.rank_candidate_sites(candidates, candidate_scores, max_active=min(MAX_ACTIVE_SITES, terminal_leaf_count))
    return selected, {
        "method": "all_four_channel_dc_projected_adjoint",
        "block_width": SCOUT_BLOCK_WIDTH,
        "representative_count": int(reps.shape[0]),
        "internal_block_quota": int(top_count),
        "top_blocks_scored": int(top_count),
        "max_bisections": SCOUT_MAX_BISECTIONS,
        "refinement_bisections": int(actual_bisections),
        "refinement_parent_count": int(refinement_parent_count),
        "terminal_leaf_count": terminal_leaf_count,
        "refinement_parent_bound": refinement_parent_bound,
        "terminal_leaf_bound": terminal_leaf_bound,
        "scored_candidate_evaluation_bound": scored_candidate_bound,
        "coarse_terminal_leaf_count": int(sum(int(leaf["depth"]) == 0 for leaf in leaves)),
        "scored_candidate_count": int(reps.shape[0] + scored_children),
        "scoring_passes": int(scoring_passes),
        "refinement_stopping_reason": stopping_reason,
        "refinement_leaf_budget_justification": "each legal split replaces one terminal leaf with two through eight children, adding one through seven leaves; the parent bound is cap minus roots, and coarse roots not selected remain terminal leaves",
        "representatives_rescored": False,
        "selected_count": int(selected.shape[0]),
        "candidate_lattice_implicit": True,
        "dense_N3_materialized": False,
    }


def _complex_projection(predicted: Sequence[np.ndarray], target: Sequence[np.ndarray]) -> complex:
    numerator = sum(np.vdot(a, b) for a, b in zip(predicted, target))
    denominator = float(sum(np.vdot(a, a).real for a in predicted))
    return complex(numerator / max(denominator, 1.0e-30))


def matched_isotropic_bp(core: Any, panel: Any, operator: Any, projectors: Mapping[str, Any]) -> dict[str, Any]:
    """Compute the matched isotropic DC BP and its recorded per-pol scale."""

    output: dict[str, Any] = {}
    for polarization in POLARIZATIONS:
        records = panel.records(polarization)
        projector = projectors[polarization]
        target = projector.target()
        native_target = projector.apply_adjoint(target)
        coeff = operator.adjoint_dc(records, native_target)
        predicted = projector.apply_forward(operator.forward_dc(records, coeff))
        gain = _complex_projection(predicted.values, target.values)
        residual = [gain * a - b for a, b in zip(predicted.values, target.values)]
        numerator = float(sum(np.vdot(value, value).real for value in residual))
        denominator = float(sum(np.vdot(value, value).real for value in target.values))
        output[polarization] = {
            "coefficients_dc": coeff,
            "scalar_fit": {"real": gain.real, "imag": gain.imag},
            "range_relmse": numerator / denominator,
            "record_count": len(records),
            "native_sample_count": int(sum(value.size for value in target.values)),
        }
    return output


def _save_bp_checkpoint(path: Path, bp: Mapping[str, Any], lattice: Any, panel: Any, support_indices: Any) -> None:
    arrays: dict[str, Any] = {}
    for polarization in POLARIZATIONS:
        coefficients = np.asarray(bp[polarization]["coefficients_dc"], dtype=np.complex128)
        scalar = bp[polarization].get("scalar_fit")
        if not isinstance(scalar, Mapping):
            raise ValueError(f"{polarization.upper()} BP result is missing scalar_fit")
        gain = complex(float(scalar.get("real", np.nan)), float(scalar.get("imag", np.nan)))
        if not np.isfinite(gain.real) or not np.isfinite(gain.imag):
            raise ValueError(f"{polarization.upper()} BP scalar_fit is non-finite")
        # Keep the historical key as the fitted coefficient so rereading this
        # checkpoint reproduces the quoted fitted BP baseline exactly.
        arrays[f"{polarization}_coefficients_dc"] = gain * coefficients
        arrays[f"{polarization}_coefficients_dc_unfitted"] = coefficients
        arrays[f"{polarization}_scalar_fit"] = np.asarray([gain.real, gain.imag], dtype=np.float64)
    arrays["test_opened"] = np.asarray(False)
    arrays["schema"] = np.asarray("rift_gotcha_camry_fullpol_bp_v1")
    arrays["lattice_N"] = np.asarray(int(lattice.N), dtype=np.int64)
    arrays["h_space_m"] = np.asarray(float(lattice.h_space_m), dtype=np.float64)
    arrays["record_counts_json"] = np.asarray(json.dumps(panel.record_counts, sort_keys=True))
    arrays["support_indices_ijk"] = lattice.validate_ijk(support_indices)
    arrays["support_points_local_m"] = lattice.points_local(support_indices)
    with path.open("wb") as handle:
        np.savez(handle, **arrays)


def _support_readout(trainer: Any) -> dict[str, Any]:
    model = trainer.model
    coefficients = {
        polarization: model.effective_coefficients(polarization).detach().cpu().numpy()
        for polarization in POLARIZATIONS
    }
    energy = np.sum(np.stack([np.abs(coefficients[p]) ** 2 for p in POLARIZATIONS], axis=0), axis=(0, 2))
    order = np.argsort(-energy, kind="stable")
    top = []
    indices = model.site_indices_ijk.detach().cpu().numpy()
    points = model.points_local_m.detach().cpu().numpy()
    for index in order[: min(256, len(order))]:
        top.append({"index": int(index), "lattice_ijk": indices[int(index)].tolist(), "point_local_m": points[int(index)].tolist(), "effective_energy": float(energy[int(index)])})
    return {
        "schema": "rift_gotcha_camry_fullpol_support_readout_v1",
        "same_selected_best_checkpoint_for_all_channels": True,
        "active_count": int(model.point_count),
        "allocated_capacity": int(model.point_count),
        "lattice_spacing_m": float(trainer.lattice.h_space_m),
        "support_local_frame": "Camry accepted native placement local frame",
        "energy_sum": float(np.sum(energy)),
        "top_sites": top,
        "test_opened": False,
    }


def run_experiment(
    protocol: Mapping[str, Any],
    paths: Sequence[Path],
    output: Path,
    *,
    resume: bool = False,
    preflight: Mapping[str, Any] | None = None,
    shards: Sequence[Any] | None = None,
    device: Any = None,
    test_device: bool = False,
) -> dict[str, Any]:
    """Run the real Torch path after successful archive preflight."""

    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise RuntimeError("Torch is required for a non-dry Camry full-pol run; use --dry-run in this authoring runtime") from exc
    from rift import gotcha_camry_fullpol_core as core  # noqa: PLC0415

    execution_device, device_report = select_execution_device(
        torch,
        device,
        protocol=protocol,
        test_device=bool(test_device),
    )
    if preflight is None or shards is None:
        preflight, loaded_shards = preflight_archives(paths)
        shards = loaded_shards
    panel = core.select_camry_train_panel(shards)
    lattice = core.build_camry_candidate_lattice(panel)
    projectors = {polarization: core.CamryRangeProjector(panel.records(polarization)) for polarization in POLARIZATIONS}
    output.mkdir(parents=True, exist_ok=True)

    latest_path = output / "checkpoint_latest.pt"
    if resume:
        if not latest_path.is_file():
            raise ValueError("--resume requires checkpoint_latest.pt in the output directory")
        for name in ("checkpoint_bp.npz", "checkpoint_cgls.pt", "checkpoint_initialization.pt", "bp_metrics.json", "cgls_metrics.json", "initialization_metrics.json", "support_search.json"):
            if not (output / name).is_file():
                raise ValueError(f"--resume requires the preserved {name} artifact")
        try:
            payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(latest_path, map_location="cpu")
        support_indices = np.asarray(payload.get("selected_site_indices"), dtype=np.int64)
        lattice.validate_ijk(support_indices)
        with (output / "support_search.json").open("r", encoding="utf-8") as handle:
            support_search = json.load(handle)
        support_search = {**support_search, "method": "restored_from_latest_checkpoint", "selected_count": int(support_indices.shape[0]), "reran_bp": False, "reran_cgls": False, "reran_gain_initialization": False}
    else:
        support_indices, support_search = scout_shared_support(core, panel, lattice, projectors)
    _write_json(output / "support_search.json", support_search)
    points_local = lattice.points_local(support_indices)
    model = core.CamryL3FullPolModel(points_local, lattice=lattice, site_indices_ijk=support_indices)
    model.to(execution_device)
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    buffer_devices = sorted({str(buffer.device) for buffer in model.buffers()})
    if any(torch.device(value).type != execution_device.type for value in parameter_devices + buffer_devices):
        raise RuntimeError("Camry model parameters/buffers did not move to the selected execution device")
    device_report = {
        **device_report,
        "model_parameter_devices": parameter_devices,
        "model_buffer_devices": buffer_devices,
    }
    resource_output = {**dict(preflight["resource"]), "execution": device_report}
    _write_json(output / "preflight.json", preflight)
    _write_json(output / "protocol_echo.json", protocol)
    _write_json(
        output / "config.json",
        {"schema": f"{DRIVER_SCHEMA}.config", **FROZEN_CONFIG.as_dict(), "execution": device_report, "test_opened": False},
    )
    _write_json(output / "device.json", device_report)
    _write_json(output / "resource.json", resource_output)
    trainer = core.CamryFullPolTrainer(panel, model)
    initial_metrics: Mapping[str, Any]
    if resume:
        trainer.build_optimizer()
        trainer.load_checkpoint(latest_path)
        bp = None
        cgls = None
        with (output / "initialization_metrics.json").open("r", encoding="utf-8") as handle:
            initialization_payload = json.load(handle)
        initial_metrics = initialization_payload.get("initial_metrics", {})
        _relative_error_summary(initial_metrics)
    else:
        operator = core.CamryL3SparseOperator(lattice.points_native(support_indices))
        bp = matched_isotropic_bp(core, panel, operator, projectors)
        retained_by_pol = {polarization: float(projectors[polarization].retained_energy_fraction()) for polarization in POLARIZATIONS}
        native_energy_by_pol = {
            polarization: float(sum(np.vdot(record.response_selected, record.response_selected).real for record in panel.records(polarization)))
            for polarization in POLARIZATIONS
        }
        retained_pooled = sum(retained_by_pol[p] * native_energy_by_pol[p] for p in POLARIZATIONS) / max(sum(native_energy_by_pol.values()), np.finfo(np.float64).tiny)
        _save_bp_checkpoint(output / "checkpoint_bp.npz", bp, lattice, panel, support_indices)
        _write_json(
            output / "bp_metrics.json",
            {
                "schema": f"{DRIVER_SCHEMA}.bp_metrics",
                "range_relmse_by_polarization": {polarization: float(bp[polarization]["range_relmse"]) for polarization in POLARIZATIONS},
                "range_macro_relmse": float(np.mean([bp[polarization]["range_relmse"] for polarization in POLARIZATIONS])),
                "scalar_fit": {polarization: dict(bp[polarization]["scalar_fit"]) for polarization in POLARIZATIONS},
                "range_projector_retained_energy_fraction_by_polarization": retained_by_pol,
                "range_projector_retained_energy_fraction_pooled": float(retained_pooled),
                "test_opened": False,
            },
        )
        cgls = core.run_dc_cgls24(panel, operator, projectors)
        core.save_dc_cgls_checkpoint(output / "checkpoint_cgls.pt", panel, lattice, cgls)
        _write_json(
            output / "cgls_metrics.json",
            {
                "schema": f"{DRIVER_SCHEMA}.cgls_metrics",
                "max_iterations": int(cgls.max_iterations),
                "termination": dict(cgls.termination),
                "diagnostics": {polarization: list(cgls.diagnostics[polarization]) for polarization in POLARIZATIONS},
                "test_opened": False,
            },
        )
        init_report = trainer.initialize_from_dc_cgls(cgls)
        initial_metrics = init_report["initial_metrics"]
        _write_json(
            output / "initialization_metrics.json",
            {
                "schema": f"{DRIVER_SCHEMA}.initialization_metrics",
                "gain_init": {polarization: _json_ready(init_report["gain_init"][polarization]) for polarization in POLARIZATIONS},
                "rms_scales": dict(init_report["rms_scales"]),
                "initial_metrics": init_report["initial_metrics"],
                "Q0": init_report["Q0"],
                "S_group": init_report["S_group"],
                "S_energy": init_report["S_energy"],
                "test_opened": False,
            },
        )
        trainer.build_optimizer()
        trainer.save_checkpoint(output / "checkpoint_initialization.pt", epoch=0, best_epoch=None, best_q=None, phase="initialization")
    result = trainer.fit(epochs=EPOCHS, seed=SEED, output_dir=output)
    best_path = output / "checkpoint_best.pt"
    if not best_path.is_file():
        raise RuntimeError("training completed without a finite best checkpoint")
    trainer.load_checkpoint(best_path)
    best_metrics = trainer.metrics()
    bp_metrics_path = output / "bp_metrics.json"
    with bp_metrics_path.open("r", encoding="utf-8") as handle:
        bp_metrics = json.load(handle)
    best_q_by_pol = best_metrics["range_relmse_by_polarization"]
    best_q = float(best_metrics["range_macro_relmse"])
    bp_q = None if bp_metrics is None else float(bp_metrics["range_macro_relmse"])
    train_relative_errors = {
        "bp": _relative_error_summary(bp_metrics),
        "cgls_initial": _relative_error_summary(initial_metrics),
        "best": _relative_error_summary(best_metrics),
        "final": _relative_error_summary(result["final_metrics"]),
    }
    retention_by_pol = {
        polarization: float(best_metrics["range_projector_retained_energy_fraction_by_polarization"][polarization])
        for polarization in POLARIZATIONS
    }
    success_gate = {
        "evaluated": True,
        "macro_Q_le_0.20": bool(best_q <= 0.20),
        "Q_le_half_Q_BP": bool(best_q <= 0.5 * bp_q),
        "every_Qp_le_0.20": bool(all(float(best_q_by_pol[p]) <= 0.20 for p in POLARIZATIONS)),
        "passed": bool(best_q <= 0.20 and best_q <= 0.5 * bp_q and all(float(best_q_by_pol[p]) <= 0.20 for p in POLARIZATIONS)),
    }
    _write_json(
        output / "metrics.json",
        {
            "schema": f"{DRIVER_SCHEMA}.metrics",
            "best": best_metrics,
            "final": result["final_metrics"],
            "bp": bp_metrics,
            "cgls_initial": initial_metrics,
            "train_relative_errors": train_relative_errors,
            "optimizer_improvement": _optimizer_improvement(initial_metrics, best_metrics),
            "range_projector_retained_energy_fractions": {
                "definition": "||B_p y_p||^2 / ||y_p||^2 over the full accepted Camry cube projector",
                "by_polarization": retention_by_pol,
                "pooled": float(best_metrics["range_projector_retained_energy_fraction_pooled"]),
            },
            "success_gate": success_gate,
            "test_opened": False,
        },
    )
    _write_json(output / "support_readout.json", _support_readout(trainer))
    _write_json(
        output / "training_history.json",
        {
            "history": result["history"],
            "optimizer_updates": result["optimizer_updates"],
            "optimizer_diagnostics": result["optimizer_diagnostics"],
            "epochs": result["epochs"],
            "test_opened": False,
        },
    )
    _write_json(output / "support_search.json", support_search)
    gain_report = {
        "schema": f"{DRIVER_SCHEMA}.gain_values",
        "polarizations": list(POLARIZATIONS),
        "gains": {},
        "all_polarizations_present": True,
        "all_initialized": True,
        "test_opened": False,
    }
    for polarization in POLARIZATIONS:
        gain_module = model.gains[polarization]
        initialized = bool(gain_module.initialized.item())
        if not initialized:
            raise RuntimeError(f"{polarization.upper()} learned global complex gain is not initialized")
        gain_report["gains"][polarization] = {
            "value": _json_ready(model.gain_values()[polarization]),
            "log_mag": float(gain_module.log_mag.detach().cpu().item()),
            "phase": float(gain_module.phase.detach().cpu().item()),
            "initialized": initialized,
        }
    _write_json(output / "gain_values.json", gain_report)
    _write_json(output / "checkpoint_manifest.json", {
        "schema": f"{DRIVER_SCHEMA}.checkpoint_manifest",
        "required": list(CHECKPOINT_NAMES),
        "present": {name: (output / name).is_file() for name in CHECKPOINT_NAMES},
        "best_epoch": result.get("best_epoch"),
        "best_q": result.get("best_q"),
        "same_best_state_for_signal_and_support": True,
        "test_opened": False,
    })
    return {"status": "PASS", "output": str(output), "best_epoch": result.get("best_epoch"), "best_q": result.get("best_q"), "optimizer_updates": result.get("optimizer_updates"), "support_search": support_search, "success_gate": success_gate}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", help="NativeShard NPZ path; provide exactly four in HH/HV/VH/VV order")
    parser.add_argument("--archive-root", help="Root containing shards/pass1_{hh,hv,vh,vv}.npz")
    parser.add_argument("--protocol", default=str(PROJECT_ROOT / "protocols" / "gotcha_camry_fullpol_v1.json"))
    parser.add_argument("--output", required=True, help="Fresh PACE-side output directory for a real run, or local manifest directory for dry-run")
    parser.add_argument("--dry-run", action="store_true", help="Inventory headers and estimate work without importing Torch or allocating GPU tensors")
    parser.add_argument("--resume", action="store_true", help="Resume checkpoint_latest.pt without rerunning BP/CGLS/gain initialization")
    parser.add_argument("--device", help="Explicit Torch device; production defaults to CUDA and never silently falls back to CPU")
    parser.add_argument("--test-device", action="store_true", help="Allow the explicitly injected --device for CPU/device-contract tests; never use for production")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        paths = resolve_archive_paths(args.archive, args.archive_root)
        preflight, shards = preflight_archives(paths)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "protocol_echo.json", protocol)
        _write_json(output / "preflight.json", preflight)
        _write_json(output / "resource.json", preflight["resource"])
        if args.dry_run:
            _write_json(
                output / "device.json",
                {
                    "schema": DEVICE_SCHEMA,
                    "mode": "dry_run",
                    "device_selected": False,
                    "actual_device": None,
                    "test_opened": False,
                },
            )
            _write_json(output / "status.json", {"schema": f"{DRIVER_SCHEMA}.status", "status": "PASS", "mode": "dry_run", "torch_imported": False, "gpu_tensors_allocated": False, "test_opened": False})
            print(json.dumps({"status": "PASS", "mode": "dry_run", "output": str(output), "record_counts": {key: value["selected_record_count"] for key, value in preflight["channels"].items()}, "fmax_hz": preflight["spatial"]["fmax_hz"], "N": preflight["spatial"]["N"]}, sort_keys=True))
            return 0
        result = run_experiment(
            protocol,
            paths,
            output,
            resume=bool(args.resume),
            preflight=preflight,
            shards=shards,
            device=args.device,
            test_device=bool(args.test_device),
        )
        _write_json(output / "status.json", {"schema": f"{DRIVER_SCHEMA}.status", **result, "mode": "resume" if args.resume else "fresh", "test_opened": False})
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, FloatingPointError, RuntimeError) as exc:
        print(f"Camry full-pol preflight/run failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
