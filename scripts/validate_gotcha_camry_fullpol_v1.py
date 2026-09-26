"""Static/data-free validator for the Camry full-polarization package.

This validator never imports Torch, opens TEST, writes manager state, or submits
a job.  With four ``--archive`` paths it additionally runs the NumPy-only
NativeShard metadata/selection preflight and prints the actual channel counts.
"""

from __future__ import annotations

import argparse
import ast
import json
import importlib.util
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = PROJECT_ROOT / "scripts" / "run_gotcha_camry_fullpol_v1.py"
CORE_PATH = PROJECT_ROOT / "rift" / "gotcha_camry_fullpol_core.py"
SOURCE_AF_PATH = PROJECT_ROOT / "rift" / "gotcha_source_af.py"
RUNTIME_VALIDATOR_PATH = PROJECT_ROOT / "scripts" / "validate_gotcha_camry_fullpol_runtime.py"


def _load_runner():
    import importlib.util
    import sys

    name = "gotcha_camry_fullpol_runner_for_validation"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {DRIVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_static(protocol_path: str | Path) -> dict[str, Any]:
    runner = _load_runner()
    protocol = runner.load_protocol(protocol_path)
    source = DRIVER_PATH.read_text(encoding="utf-8")
    core = CORE_PATH.read_text(encoding="utf-8")
    source_af = SOURCE_AF_PATH.read_text(encoding="utf-8")
    runtime_validator = RUNTIME_VALIDATOR_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(DRIVER_PATH))
    # A dry-run must not import Torch or allocate GPU tensors.  Keep the check
    # structural so it remains usable without optional runtime dependencies.
    top_level_torch_import = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_torch_import.extend(alias.name for alias in node.names if alias.name == "torch")
        elif isinstance(node, ast.ImportFrom) and node.module == "torch":
            top_level_torch_import.append("torch")
    if top_level_torch_import:
        raise ValueError("runner must not import Torch at module scope")
    source_text_lower = source.lower()
    if "np.empty((n, n, n" in source_text_lower or "np.zeros((n, n, n" in source_text_lower:
        raise ValueError("runner contains a dense N^3 allocation")
    if "candidate_lattice_implicit" not in source_text_lower or "dense_n3_materialized" not in source_text_lower:
        raise ValueError("runner must disclose an implicit candidate lattice")
    required_runner_tokens = (
        "preflight_archives",
        "scout_shared_support",
        "matched_isotropic_bp",
        "run_dc_cgls24",
        "checkpoint_initialization.pt",
        "checkpoint_latest.pt",
        "--resume",
        "select_execution_device",
        "device.json",
        "representatives_rescored",
        "scout_terminal_leaf_bound",
        "scored_candidate_evaluation_bound",
        "train_relative_errors",
        "signed_relative_improvement",
        "range_projector_retained_energy_fraction",
        "all_initialized",
        "optimizer_diagnostics",
    )
    missing_runner = [token for token in required_runner_tokens if token not in source]
    if missing_runner:
        raise ValueError(f"runner is missing required integration tokens: {missing_runner}")
    if "scout_candidate_bound" in source or '"active_operator_cgls"' in source:
        raise ValueError("runner contains a stale scout bound or duplicate CGLS ledger term")
    required_core_tokens = (
        "MAX_ACTIVE_SITES = 8_192",
        "CamryRangeProjector",
        "run_dc_cgls24",
        "CamryVVTrainSourceAFScope",
        "nn.ModuleDict",
        "total_epochs != 150",
        '"h_m": None',
        "checkpoint_best.pt",
        "checkpoint_final.pt",
        "retained_energy_fraction",
        "_cpu_rng_tensor",
        "_checkpoint_rng_states",
        "_verify_optimizer_state_devices",
        "_optimizer_step_with_diagnostics",
        "optimizer_diagnostics",
    )
    missing_core = [token for token in required_core_tokens if token not in core]
    if missing_core:
        raise ValueError(f"core is missing required frozen-contract tokens: {missing_core}")
    required_source_af_tokens = ("CamryVVTrainSourceAFScope", "raw_channel_own_arrays_unapplied", "source_shard_id")
    missing_source_af = [token for token in required_source_af_tokens if token not in source_af]
    if missing_source_af:
        raise ValueError(f"source-AF module is missing required provenance tokens: {missing_source_af}")
    required_runtime_tokens = (
        "EXPECTED_CORE_TEST_COUNT = 19",
        'REQUIRED_GPU_LABEL = "A100"',
        'REQUIRED_GPU_TOKEN = "a100"',
        "torch.cuda.is_available",
        "get_device_name",
        "current_device",
        "tests.test_gotcha_camry_fullpol_core",
        "result.skipped",
        "result.failures",
        "result.errors",
        "failure_details",
        "error_details",
        "stream.getvalue()",
        "PASS_RUNTIME",
    )
    missing_runtime = [token for token in required_runtime_tokens if token not in runtime_validator]
    if missing_runtime:
        raise ValueError(f"runtime validator is missing required fail-closed tokens: {missing_runtime}")
    if protocol.get("success_gate", {}).get("test_sealed") is not True:
        raise ValueError("protocol success gate must preserve TEST sealing")
    if protocol.get("source", {}).get("metadata_inventory") is None:
        raise ValueError("protocol must require runtime input provenance inventory")
    validation = protocol.get("validation", {})
    expected_cuda_test = "tests.test_gotcha_camry_fullpol_core.CamryTrainingContractTest.test_cuda_checkpoint_round_trip_restores_rng_optimizer_devices_and_resume_readiness"
    if (
        validation.get("same_allocation_before_fit") is not True
        or validation.get("runtime_validator") != "scripts/validate_gotcha_camry_fullpol_runtime.py"
        or validation.get("runtime_command") != "python3 -B scripts/validate_gotcha_camry_fullpol_runtime.py"
        or validation.get("expected_core_test_count") != 19
        or validation.get("cuda_required_test") != expected_cuda_test
        or validation.get("cuda_skip_is_failure") is not True
    ):
        raise ValueError("protocol must require the fail-closed full CUDA runtime validator")
    resource = protocol.get("resource", {})
    if (
        resource.get("qos") != "inferno"
        or resource.get("partition") != "gpu-a100"
        or resource.get("gpu") != "A100"
        or resource.get("cpus") != 8
        or resource.get("ram_gib") != 64
        or resource.get("tmp_gib") != 16
        or resource.get("walltime") != "2-00:00:00"
    ):
        raise ValueError("protocol must request qos=inferno, partition=gpu-a100, A100, 8 CPUs, 64GiB, tmp16GiB, and 48h")
    torch_available = importlib.util.find_spec("torch") is not None
    return {
        "schema": "rift_gotcha_camry_fullpol_static_validation_v1",
        "status": "PASS_STATIC" if torch_available else "PASS_STATIC_TORCH_UNAVAILABLE",
        "protocol": str(Path(protocol_path).resolve()),
        "protocol_schema": protocol["schema"],
        "torch_available": torch_available,
        "torch_imported": False,
        "dense_N3_materialized": False,
        "test_opened": False,
        "archive_accessed": False,
        "static_checks": {
            "runner_contract": True,
            "core_contract": True,
            "source_af_contract": True,
            "protocol_contract": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(PROJECT_ROOT / "protocols" / "gotcha_camry_fullpol_v1.json"))
    parser.add_argument("--archive", action="append", help="optional NativeShard paths; provide all four for runtime metadata preflight")
    parser.add_argument("--archive-root", help="optional root containing shards/pass1_{hh,hv,vh,vv}.npz")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_static(args.protocol)
        if args.archive or args.archive_root:
            runner = _load_runner()
            paths = runner.resolve_archive_paths(args.archive, args.archive_root)
            preflight, _ = runner.preflight_archives(paths)
            result["runtime_preflight"] = preflight
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Camry full-pol static validation failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
