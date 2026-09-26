"""Six-scene contract and lifecycle foundation for corrected SE Stage 2 v3.

The registry in this module is the only supported source of scene, Stage-1,
measurement, and output paths.  It deliberately excludes every public-radar
scene.  The proven geometric implementation remains in the preserved A320
module and is exposed lazily here so importing the contract does not require a
CUDA/PyTorch runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from importlib import import_module
import math
import os
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, NoReturn, Optional

import numpy as np


CAMPAIGN_IDENTITY = "rift_sugavanam_ertin_stage2_stabilized_v3"
METHOD_NAME = "Sugavanam--Ertin valid-zero stabilized derivative v3"
POLICY_IDENTITY = "stage1_closed_init_oriented_offsets_valid_projection_v3"
IMPLEMENTATION_KIND = "stabilized derivative; not the raw paper reproduction"

OUTPUT_ROOT = "/storage/scratch1/1/dbao31/rift_sugavanam_ertin_stage2_stabilized_v3"
MESH10K_ROOT = "/storage/scratch1/1/dbao31/rift_mesh10k_baselines_20260830_v1"
MESH10K_DATA_ROOT = (
    "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/"
    "mesh_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k_data/objects"
)
PROJECT_ROOT = "/storage/project/r-jromberg3-0/dbao31/RIFT"

STAGE1_EPOCH = 150
STAGE1_GRANULARITY = 48
STAGE1_EXTENT = 0.15
STAGE1_THRESHOLD_FRACTION = 0.15
STAGE1_GRID_SHAPE = (48, 48, 48)

PROJECTION_ITERATIONS = 24
PROJECTION_TOLERANCE = 1.0e-4
PROJECTION_MIN_ACCEPTANCE = 0.60
CLEAN_TERM_EXIT_CODE = 143


class ContractViolation(ValueError):
    """A fail-closed violation of the frozen v3 experiment contract."""


class LifecyclePhase(str, Enum):
    INITIALIZATION = "initialization"
    TRAINING = "training"
    EXPORT_PENDING = "export_pending"
    COMPLETE = "complete"


class StatusState(str, Enum):
    RUNNING = "running"
    CLEAN_INTERRUPTED = "clean_interrupted"
    FAILED = "failed"
    COMPLETE = "complete"


class OutputDisposition(str, Enum):
    NOT_COMPLETE = "not_complete"
    COMPLETE_NOOP = "complete_noop"


@dataclass(frozen=True)
class SceneContract:
    """Immutable identity and provenance for one permitted homemade scene."""

    key: str
    label: str
    acquisition: str
    measurement_path: str
    stage1_checkpoint: str
    output_dir: str

    @property
    def manager_identity(self) -> str:
        return f"{CAMPAIGN_IDENTITY}_{self.key}"

    @property
    def artifact_identity(self) -> str:
        return f"{self.key}_sugavanam_ertin_stage2_stabilized_v3"

    @property
    def latest_checkpoint(self) -> str:
        return f"{self.output_dir}/checkpoint_latest.pth.tar"

    @property
    def final_checkpoint(self) -> str:
        return f"{self.output_dir}/checkpoint_final.pth.tar"

    @property
    def surface_path(self) -> str:
        return f"{self.output_dir}/surface_reconstruction.npz"

    @property
    def summary_path(self) -> str:
        return f"{self.output_dir}/run_summary.json"

    @property
    def status_path(self) -> str:
        return f"{self.output_dir}/status.json"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _mesh10k_contract(key: str, label: str, filename: str) -> SceneContract:
    return SceneContract(
        key=key,
        label=label,
        acquisition="sphere10k",
        measurement_path=f"{MESH10K_DATA_ROOT}/{filename}",
        stage1_checkpoint=f"{MESH10K_ROOT}/{key}/se/scatter/checkpoint_final.pth.tar",
        output_dir=f"{OUTPUT_ROOT}/{key}",
    )


SCENE_CONTRACTS: Mapping[str, SceneContract] = MappingProxyType({
    "a320": _mesh10k_contract(
        "a320",
        "A320",
        "airliner_a320_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz",
    ),
    "x59": _mesh10k_contract(
        "x59",
        "X-59",
        "supersonic_x59_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz",
    ),
    "firetruck": _mesh10k_contract(
        "firetruck",
        "fire truck",
        "firetruck_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz",
    ),
    "racecar": _mesh10k_contract(
        "racecar",
        "race car",
        "race_car_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz",
    ),
    "loader": _mesh10k_contract(
        "loader",
        "loader",
        "loader_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz",
    ),
    "b787_sphere2k": SceneContract(
        key="b787_sphere2k",
        label="B787",
        acquisition="sphere2k",
        measurement_path=f"{PROJECT_ROOT}/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz",
        stage1_checkpoint=(
            f"{PROJECT_ROOT}/training_checkpoints/"
            "b787_sugavanam_ertin_scatter/checkpoint_final.pth.tar"
        ),
        output_dir=f"{OUTPUT_ROOT}/b787_sphere2k",
    ),
})

SCENE_KEYS = tuple(SCENE_CONTRACTS)
PUBLIC_SCENE_KEYS = frozenset(
    {
        "camry",
        "cvdomes",
        "gotcha",
        "gotcha_p2",
        "honda",
        "jeep",
        "rpd_00",
        "rpd_11",
        "rps_11",
    }
)


@dataclass(frozen=True)
class Stage1StructuralAudit:
    scene_key: str
    checkpoint_path: str
    epoch: int
    granularity: int
    extent: float
    active_count: int
    retained_count: int
    threshold_fraction: float
    threshold_absolute: float
    maximum_magnitude: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def get_scene_contract(scene_key: str) -> SceneContract:
    """Resolve one frozen scene; arbitrary and public scenes fail closed."""
    if not isinstance(scene_key, str) or scene_key not in SCENE_CONTRACTS:
        if isinstance(scene_key, str) and scene_key.lower() in PUBLIC_SCENE_KEYS:
            raise ContractViolation(f"public scene {scene_key!r} is excluded from corrected SE v3")
        raise ContractViolation(
            f"unsupported corrected-SE scene {scene_key!r}; expected one of {SCENE_KEYS}"
        )
    return SCENE_CONTRACTS[scene_key]


def validate_scene_contract(contract: SceneContract) -> SceneContract:
    """Require the exact immutable registry object for one permitted scene."""
    if not isinstance(contract, SceneContract):
        raise ContractViolation("corrected SE v3 requires a registered SceneContract")
    canonical = SCENE_CONTRACTS.get(contract.key)
    if canonical is None or contract is not canonical:
        raise ContractViolation(
            f"scene contract is not the frozen corrected-SE registry entry for {contract.key!r}"
        )
    return canonical


def validate_claimed_paths(
    contract: SceneContract,
    *,
    measurement_path: str,
    stage1_checkpoint: str,
    output_dir: str,
) -> None:
    """Reject path overrides even when the caller supplies a valid scene key."""
    validate_scene_contract(contract)
    expected = {
        "measurement_path": contract.measurement_path,
        "stage1_checkpoint": contract.stage1_checkpoint,
        "output_dir": contract.output_dir,
    }
    actual = {
        "measurement_path": measurement_path,
        "stage1_checkpoint": stage1_checkpoint,
        "output_dir": output_dir,
    }
    changed = [name for name in expected if actual[name] != expected[name]]
    if changed:
        raise ContractViolation(f"path override is forbidden for {contract.key}: {changed}")


def _array(value: object, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        result = np.asarray(value)
    except Exception as error:  # pragma: no cover - defensive around foreign tensors
        raise ContractViolation(f"Stage-1 {name} is not array-like") from error
    return result


def validate_stage1_final_state(
    state: Mapping[str, object],
    contract: SceneContract,
    *,
    source_path: str,
) -> Stage1StructuralAudit:
    """Validate a final grid checkpoint without pinning scene-derived magnitudes."""
    validate_scene_contract(contract)
    if source_path != contract.stage1_checkpoint:
        raise ContractViolation("Stage-1 source is not the scene contract's exact final checkpoint")
    if os.path.basename(source_path) != "checkpoint_final.pth.tar":
        raise ContractViolation("corrected SE v3 accepts only checkpoint_final.pth.tar")
    if state.get("scene_repr") != "grid":
        raise ContractViolation("Stage-1 scene_repr must be grid")
    if int(state.get("epoch", -1)) != STAGE1_EPOCH:
        raise ContractViolation(f"Stage-1 epoch must be {STAGE1_EPOCH}")
    if int(state.get("granularity", -1)) != STAGE1_GRANULARITY:
        raise ContractViolation(f"Stage-1 granularity must be {STAGE1_GRANULARITY}")
    try:
        extent = float(state.get("extent", float("nan")))
    except (TypeError, ValueError) as error:
        raise ContractViolation("Stage-1 extent is not numeric") from error
    if not math.isclose(extent, STAGE1_EXTENT, rel_tol=0.0, abs_tol=1.0e-12):
        raise ContractViolation(f"Stage-1 extent must be {STAGE1_EXTENT}")

    model_state = state.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ContractViolation("Stage-1 model_state_dict is missing")
    required = ("w_re", "w_im", "active_mask", "grid_positions")
    missing = [name for name in required if name not in model_state]
    if missing:
        raise ContractViolation(f"Stage-1 model state is missing {missing}")

    real = _array(model_state["w_re"], "w_re")
    imag = _array(model_state["w_im"], "w_im")
    active = _array(model_state["active_mask"], "active_mask")
    positions = _array(model_state["grid_positions"], "grid_positions")
    if real.shape != STAGE1_GRID_SHAPE or imag.shape != STAGE1_GRID_SHAPE:
        raise ContractViolation(f"Stage-1 weights must both have shape {STAGE1_GRID_SHAPE}")
    if active.shape != STAGE1_GRID_SHAPE or active.dtype.kind != "b":
        raise ContractViolation("Stage-1 active_mask must be a boolean 48^3 array")
    if positions.ndim < 2 or positions.shape[-1] != 3 or positions.size != 48**3 * 3:
        raise ContractViolation("Stage-1 grid_positions must contain exactly 48^3 xyz positions")

    for name, value in model_state.items():
        try:
            array = _array(value, str(name))
        except ContractViolation:
            continue
        if array.dtype.kind in "fc" and not bool(np.isfinite(array).all()):
            raise ContractViolation(f"Stage-1 model tensor {name!r} is non-finite")

    magnitude = np.sqrt(np.square(real, dtype=np.float64) + np.square(imag, dtype=np.float64))
    active_values = magnitude[active]
    if active_values.size == 0 or not bool(np.isfinite(active_values).all()):
        raise ContractViolation("Stage-1 has no finite active scattering values")
    maximum = float(active_values.max())
    if maximum <= 0.0:
        raise ContractViolation("Stage-1 maximum active magnitude must be positive")
    threshold = STAGE1_THRESHOLD_FRACTION * maximum
    retained = int(np.count_nonzero(active & (magnitude >= threshold)))
    if retained < 3:
        raise ContractViolation("Stage-1 threshold retains fewer than three scattering centres")
    return Stage1StructuralAudit(
        scene_key=contract.key,
        checkpoint_path=contract.stage1_checkpoint,
        epoch=STAGE1_EPOCH,
        granularity=STAGE1_GRANULARITY,
        extent=extent,
        active_count=int(np.count_nonzero(active)),
        retained_count=retained,
        threshold_fraction=STAGE1_THRESHOLD_FRACTION,
        threshold_absolute=threshold,
        maximum_magnitude=maximum,
    )


def validate_stage1_final_checkpoint(
    contract: SceneContract,
    load_checkpoint: Callable[[str], Mapping[str, object]],
    *,
    is_file: Callable[[str], bool] = os.path.isfile,
) -> Stage1StructuralAudit:
    """Load only the exact final path resolved by the frozen scene registry."""
    validate_scene_contract(contract)
    path = contract.stage1_checkpoint
    if not is_file(path):
        raise ContractViolation(f"required Stage-1 final is unavailable: {path}")
    return validate_stage1_final_state(load_checkpoint(path), contract, source_path=path)


def identity_record(contract: SceneContract) -> dict[str, object]:
    """Canonical identity/provenance fields shared by all v3 artifacts."""
    validate_scene_contract(contract)
    return {
        "campaign_identity": CAMPAIGN_IDENTITY,
        "manager_identity": contract.manager_identity,
        "artifact_identity": contract.artifact_identity,
        "scene_key": contract.key,
        "acquisition": contract.acquisition,
        "method": METHOD_NAME,
        "policy": POLICY_IDENTITY,
        "implementation_kind": IMPLEMENTATION_KIND,
        "measurement_path": contract.measurement_path,
        "stage1_checkpoint": contract.stage1_checkpoint,
        "output_dir": contract.output_dir,
        "ground_truth_geometry_used": False,
    }


def validate_identity_record(record: Mapping[str, object], contract: SceneContract) -> None:
    expected = identity_record(contract)
    missing = [name for name in expected if name not in record]
    changed = [name for name, value in expected.items() if name in record and record[name] != value]
    if missing or changed:
        raise ContractViolation(
            f"artifact identity mismatch for {contract.key}: missing={missing}, changed={changed}"
        )


def _phase(value: object) -> LifecyclePhase:
    try:
        return LifecyclePhase(str(value))
    except ValueError as error:
        raise ContractViolation(f"invalid corrected-SE lifecycle phase {value!r}") from error


def _progress(record: Mapping[str, object]) -> tuple[int, int]:
    try:
        raw_step = record["step"]
        raw_max_steps = record["max_steps"]
    except KeyError as error:
        raise ContractViolation("lifecycle record requires integer step/max_steps") from error
    if isinstance(raw_step, np.ndarray) and raw_step.shape == ():
        raw_step = raw_step.item()
    if isinstance(raw_max_steps, np.ndarray) and raw_max_steps.shape == ():
        raw_max_steps = raw_max_steps.item()
    integer_types = (int, np.integer)
    if (
        not isinstance(raw_step, integer_types)
        or isinstance(raw_step, (bool, np.bool_))
        or not isinstance(raw_max_steps, integer_types)
        or isinstance(raw_max_steps, (bool, np.bool_))
    ):
        raise ContractViolation("lifecycle record requires integer step/max_steps")
    step = int(raw_step)
    max_steps = int(raw_max_steps)
    if max_steps <= 0 or step < 0 or step > max_steps:
        raise ContractViolation(f"invalid lifecycle progress {step}/{max_steps}")
    return step, max_steps


def _require_nonempty_mapping(
    record: Mapping[str, object], name: str, *, where: str
) -> Mapping[str, object]:
    value = record.get(name)
    if not isinstance(value, Mapping) or not value:
        raise ContractViolation(f"{where} requires a nonempty {name} mapping")
    return value


def _validate_resumable_payload(
    state: Mapping[str, object], contract: SceneContract, phase: LifecyclePhase
) -> None:
    """Require enough durable state to resume the declared phase exactly."""
    for name in (
        "model_state_dict",
        "optimizer_state_dict",
        "rng_state",
        "stage1_audit",
        "model_config",
        "args",
    ):
        _require_nonempty_mapping(state, name, where="resumable checkpoint")
    if state.get("sample_rng_state") is None:
        raise ContractViolation("resumable checkpoint requires sample_rng_state")

    stage1_audit = state["stage1_audit"]
    try:
        audited_extent = float(stage1_audit.get("extent", float("nan")))
    except (TypeError, ValueError) as error:
        raise ContractViolation("resumable checkpoint Stage-1 extent is not numeric") from error
    if (
        stage1_audit.get("scene_key") != contract.key
        or stage1_audit.get("checkpoint_path") != contract.stage1_checkpoint
        or stage1_audit.get("epoch") != STAGE1_EPOCH
        or stage1_audit.get("granularity") != STAGE1_GRANULARITY
        or not math.isclose(
            audited_extent,
            STAGE1_EXTENT,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ContractViolation("resumable checkpoint Stage-1 audit changed")

    args = state["args"]
    claimed = {
        "scene_key": contract.key,
        "measurement_path": contract.measurement_path,
        "stage1_checkpoint": contract.stage1_checkpoint,
        "output_dir": contract.output_dir,
    }
    changed = [name for name, expected in claimed.items() if args.get(name) != expected]
    if changed:
        raise ContractViolation(f"resumable checkpoint arguments changed: {changed}")

    if phase is LifecyclePhase.INITIALIZATION:
        _require_nonempty_mapping(
            state, "initialization_state", where="initialization checkpoint"
        )
    elif phase is LifecyclePhase.TRAINING:
        _require_nonempty_mapping(
            state, "scheduler_state_dict", where="training checkpoint"
        )
        _require_nonempty_mapping(state, "initialization", where="training checkpoint")
        if not isinstance(state.get("history"), list):
            raise ContractViolation("training checkpoint requires list history")
    elif phase is LifecyclePhase.EXPORT_PENDING:
        _require_nonempty_mapping(state, "initialization", where="export checkpoint")
        _require_nonempty_mapping(state, "final_gate", where="export checkpoint")
        if not isinstance(state.get("history"), list):
            raise ContractViolation("export checkpoint requires list history")


def validate_checkpoint_identity(
    state: Mapping[str, object],
    contract: SceneContract,
    *,
    expected_phase: Optional[LifecyclePhase] = None,
    require_resumable: bool = False,
) -> LifecyclePhase:
    validate_identity_record(state, contract)
    phase = _phase(state.get("phase"))
    step, max_steps = _progress(state)
    resume_allowed = state.get("resume_allowed")
    if not isinstance(resume_allowed, (bool, np.bool_)):
        raise ContractViolation("checkpoint resume_allowed must be boolean")
    role = state.get("checkpoint_role")
    expected_role = "final" if phase is LifecyclePhase.COMPLETE else "latest"
    if role != expected_role:
        raise ContractViolation(
            f"{phase.value} checkpoint role must be {expected_role!r}, got {role!r}"
        )
    if expected_phase is not None and phase is not expected_phase:
        raise ContractViolation(f"checkpoint phase {phase.value} is not {expected_phase.value}")
    if phase is LifecyclePhase.COMPLETE:
        if bool(resume_allowed):
            raise ContractViolation("complete checkpoint cannot be resumable")
        if step != max_steps:
            raise ContractViolation("complete checkpoint must be at max_steps")
    elif bool(resume_allowed):
        _validate_resumable_payload(state, contract, phase)
    if require_resumable:
        if phase is LifecyclePhase.COMPLETE or not bool(resume_allowed):
            raise ContractViolation("checkpoint is not a resumable lifecycle checkpoint")
    return phase


def validate_status_record(status: Mapping[str, object], contract: SceneContract) -> LifecyclePhase:
    validate_identity_record(status, contract)
    phase = _phase(status.get("phase"))
    step, max_steps = _progress(status)
    try:
        state = StatusState(str(status.get("state")))
    except ValueError as error:
        raise ContractViolation(f"invalid corrected-SE status {status.get('state')!r}") from error
    resume_allowed = status.get("resume_allowed")
    if not isinstance(resume_allowed, (bool, np.bool_)):
        raise ContractViolation("status resume_allowed must be boolean")
    exit_code = status.get("exit_code")
    if state is StatusState.CLEAN_INTERRUPTED:
        if phase is LifecyclePhase.COMPLETE or not bool(resume_allowed):
            raise ContractViolation("clean interruption must be resumable and incomplete")
        if exit_code != CLEAN_TERM_EXIT_CODE:
            raise ContractViolation("clean interruption must carry exit code 143")
        if status.get("checkpoint_path") != contract.latest_checkpoint:
            raise ContractViolation("clean interruption must point to the scene's latest checkpoint")
    elif state is StatusState.COMPLETE:
        if phase is not LifecyclePhase.COMPLETE or bool(resume_allowed) or exit_code != 0:
            raise ContractViolation("complete status requires complete phase, exit 0, and no resume")
        if step != max_steps:
            raise ContractViolation("complete status must be at max_steps")
        if status.get("checkpoint_path") != contract.final_checkpoint:
            raise ContractViolation("complete status must point to the scene's final checkpoint")
    elif state is StatusState.FAILED:
        if bool(resume_allowed):
            raise ContractViolation("terminal failure cannot be resumable")
        if phase is LifecyclePhase.COMPLETE:
            raise ContractViolation("terminal failure cannot claim complete phase")
        if (
            not isinstance(exit_code, (int, np.integer))
            or isinstance(exit_code, (bool, np.bool_))
            or int(exit_code) == 0
        ):
            raise ContractViolation("terminal failure requires a nonzero integer exit code")
    else:
        if phase is LifecyclePhase.COMPLETE:
            raise ContractViolation("complete phase cannot have running status")
        if bool(resume_allowed):
            raise ContractViolation("running status cannot advertise a manager-safe resume")
        if exit_code is not None:
            raise ContractViolation("running status cannot carry a terminal exit code")
    return phase


def clean_term_status(
    contract: SceneContract,
    phase: LifecyclePhase,
    *,
    step: int,
    max_steps: int,
) -> dict[str, object]:
    """Build the status that must be durable before a clean exit 143."""
    if phase is LifecyclePhase.COMPLETE:
        raise ContractViolation("complete work cannot be clean-interrupted")
    record = {
        **identity_record(contract),
        "phase": phase.value,
        "state": StatusState.CLEAN_INTERRUPTED.value,
        "step": step,
        "max_steps": max_steps,
        "resume_allowed": True,
        "checkpoint_path": contract.latest_checkpoint,
        "exit_code": CLEAN_TERM_EXIT_CODE,
    }
    validate_status_record(record, contract)
    return record


def exit_after_clean_term(status: Mapping[str, object], contract: SceneContract) -> NoReturn:
    """Validate the durable status and terminate with the manager resume code."""
    validate_status_record(status, contract)
    if status.get("state") != StatusState.CLEAN_INTERRUPTED.value:
        raise ContractViolation("exit 143 is reserved for a validated clean interruption")
    raise SystemExit(CLEAN_TERM_EXIT_CODE)


COMPLETE_ARTIFACTS = frozenset(
    {
        "checkpoint_final.pth.tar",
        "surface_reconstruction.npz",
        "run_summary.json",
        "status.json",
    }
)
_COMPLETE_PAYLOAD_ARTIFACTS = COMPLETE_ARTIFACTS - {"status.json"}
FAILURE_ARTIFACTS = frozenset(
    {
        "terminal_failure.json",
        "checkpoint_projection_failure.pth.tar",
        "interruption_without_checkpoint.json",
    }
)


def complete_artifact_disposition(present_names: Iterable[str]) -> OutputDisposition:
    """Classify a terminal artifact set without treating partial export as complete."""
    names = frozenset(str(name) for name in present_names)
    failure = sorted(names & FAILURE_ARTIFACTS)
    if failure:
        raise ContractViolation(f"terminal failure evidence blocks completion: {failure}")
    terminal_present = names & _COMPLETE_PAYLOAD_ARTIFACTS
    if COMPLETE_ARTIFACTS.issubset(names):
        return OutputDisposition.COMPLETE_NOOP
    if terminal_present:
        missing = sorted(COMPLETE_ARTIFACTS - names)
        raise ContractViolation(f"partial complete artifact set; missing {missing}")
    return OutputDisposition.NOT_COMPLETE


def _is_strict_true(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.ndarray) and value.shape == ():
        scalar = value.item()
        return isinstance(scalar, (bool, np.bool_)) and bool(scalar)
    return False


def validate_idempotent_complete_noop(
    present_names: Iterable[str],
    *,
    final_checkpoint: Mapping[str, object],
    surface_metadata: Mapping[str, object],
    summary: Mapping[str, object],
    status: Mapping[str, object],
    contract: SceneContract,
) -> OutputDisposition:
    """Prove that a rerun may exit successfully without mutating a complete run."""
    disposition = complete_artifact_disposition(present_names)
    if disposition is not OutputDisposition.COMPLETE_NOOP:
        raise ContractViolation("output is not a complete idempotent no-op")
    phase = validate_checkpoint_identity(
        final_checkpoint,
        contract,
        expected_phase=LifecyclePhase.COMPLETE,
    )
    if phase is not LifecyclePhase.COMPLETE or final_checkpoint.get("checkpoint_role") != "final":
        raise ContractViolation("complete output lacks a final-role checkpoint")
    if bool(final_checkpoint.get("resume_allowed")):
        raise ContractViolation("final checkpoint must not permit resume")
    validate_identity_record(surface_metadata, contract)
    validate_identity_record(summary, contract)
    if _phase(surface_metadata.get("phase")) is not LifecyclePhase.COMPLETE:
        raise ContractViolation("surface metadata is not complete")
    if _phase(summary.get("phase")) is not LifecyclePhase.COMPLETE:
        raise ContractViolation("summary metadata is not complete")
    if not _is_strict_true(surface_metadata.get("validity_passed")):
        raise ContractViolation("surface field-validity gate did not pass")
    if not _is_strict_true(surface_metadata.get("topology_passed")):
        raise ContractViolation("surface topology gate did not pass")
    surface_audit = final_checkpoint.get("surface_audit")
    if not isinstance(surface_audit, Mapping):
        raise ContractViolation("final checkpoint lacks surface audit")
    validity = surface_audit.get("validity")
    topology = surface_audit.get("topology")
    if not isinstance(validity, Mapping) or not _is_strict_true(validity.get("passed")):
        raise ContractViolation("checkpoint field-validity audit did not pass")
    if not isinstance(topology, Mapping) or not _is_strict_true(topology.get("passed")):
        raise ContractViolation("checkpoint topology audit did not pass")
    if summary.get("checkpoint_final") != contract.final_checkpoint:
        raise ContractViolation("summary final-checkpoint path changed")
    if summary.get("surface_reconstruction") != contract.surface_path:
        raise ContractViolation("summary surface path changed")
    validate_status_record(status, contract)
    final_progress = _progress(final_checkpoint)
    status_progress = _progress(status)
    summary_progress = _progress(summary)
    surface_progress = _progress(surface_metadata)
    if not (
        final_progress == status_progress == summary_progress == surface_progress
    ):
        raise ContractViolation("complete artifacts disagree on step/max_steps")
    return disposition


# These names are implemented and validated in the preserved A320 module.  Lazy
# delegation keeps this contract importable in CPU-only tooling that has NumPy
# but not the full Torch/SciPy training environment.
GEOMETRY_EXPORT_NAMES = frozenset(
    {
        "ProjectionAcceptanceError",
        "ProjectionResult",
        "analytic_sphere_sdf",
        "closed_anchor_losses",
        "closed_field_spec",
        "deterministic_boundary_shell",
        "deterministic_inner_anchors",
        "evaluate_field_grid",
        "field_validity_from_array",
        "orient_normals_outward",
        "oriented_normal_loss",
        "project_to_zero_level_strict",
        "protected_shell_gate",
        "ramped_weight",
        "refresh_iso_points_strict",
        "sample_roi",
        "signed_offset_loss",
        "signed_offset_samples",
        "strict_field_gate",
        "topology_contract",
    }
)


def __getattr__(name: str) -> object:
    if name not in GEOMETRY_EXPORT_NAMES:
        raise AttributeError(name)
    preserved = import_module("rift.sugavanam_ertin_a320_stabilized")
    value = getattr(preserved, name)
    globals()[name] = value
    return value


__all__ = [
    "CAMPAIGN_IDENTITY",
    "CLEAN_TERM_EXIT_CODE",
    "COMPLETE_ARTIFACTS",
    "ContractViolation",
    "FAILURE_ARTIFACTS",
    "GEOMETRY_EXPORT_NAMES",
    "IMPLEMENTATION_KIND",
    "LifecyclePhase",
    "METHOD_NAME",
    "OutputDisposition",
    "POLICY_IDENTITY",
    "PROJECTION_ITERATIONS",
    "PROJECTION_MIN_ACCEPTANCE",
    "PROJECTION_TOLERANCE",
    "PUBLIC_SCENE_KEYS",
    "SCENE_CONTRACTS",
    "SCENE_KEYS",
    "STAGE1_EPOCH",
    "STAGE1_EXTENT",
    "STAGE1_GRANULARITY",
    "STAGE1_GRID_SHAPE",
    "STAGE1_THRESHOLD_FRACTION",
    "SceneContract",
    "Stage1StructuralAudit",
    "StatusState",
    "clean_term_status",
    "complete_artifact_disposition",
    "exit_after_clean_term",
    "get_scene_contract",
    "identity_record",
    "validate_checkpoint_identity",
    "validate_claimed_paths",
    "validate_idempotent_complete_noop",
    "validate_identity_record",
    "validate_scene_contract",
    "validate_stage1_final_checkpoint",
    "validate_stage1_final_state",
    "validate_status_record",
    *sorted(GEOMETRY_EXPORT_NAMES),
]
