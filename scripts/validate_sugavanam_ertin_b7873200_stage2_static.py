"""Static/source checks for the isolated corrected B7873200 SE Stage-2 lane.

This validator intentionally needs neither PyTorch nor B787 data.  The actual
Torch lifecycle/recovery suite belongs on an allocated PACE node once the
source is reconciled and deployed.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]


def load_under_fake_rift(name: str, path: Path):
    package = sys.modules.get("rift")
    if package is None:
        package = types.ModuleType("rift")
        package.__path__ = [str(ROOT / "rift")]
        sys.modules["rift"] = package
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def functions_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def main() -> None:
    stage1 = load_under_fake_rift(
        "rift.sugavanam_ertin_b7873200_stage1", ROOT / "rift" / "sugavanam_ertin_b7873200_stage1.py"
    )
    stage2 = load_under_fake_rift(
        "rift.sugavanam_ertin_b7873200_stage2_v1", ROOT / "rift" / "sugavanam_ertin_b7873200_stage2_v1.py"
    )
    contract = stage2.B7873200Stage2Contract()
    assert contract.canonical_npz_path == stage1.B787_3200_CANONICAL_NPZ_PATH
    assert contract.stage1_final_bundle.endswith(stage1.STAGE1_FINAL_BUNDLE_FILENAME)
    recipe = stage2.default_stage2_recipe()
    assert stage2.validate_stage2_recipe(recipe) == recipe
    assert recipe["steps"] == 5000 and recipe["init_steps"] == 200
    assert recipe["scatter_threshold"] == 0.15 and recipe["seed"] == 42
    try:
        changed = dict(recipe)
        changed["steps"] = 15
        stage2.validate_stage2_recipe(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("changed Stage-2 recipe was accepted")

    # Lifecycle JSON intentionally excludes binary acquisition arrays while the
    # checkpoint provenance retains them.  The input is synthetic but carries
    # the exact semantic fields the transformation needs.
    fake_stage1 = {
        "schema": stage1.STAGE1_SCHEMA,
        "recipe_id": stage1.STAGE1_RECIPE_ID,
        "role": "checkpoint_final",
        "sealed_protocol_identity": {"role_ids": {"train": [0], "validation": [1]}},
        "stage1_recipe": {"schema": stage1.STAGE1_SCHEMA, "acquisition_identity": {"response_payload_materialized": False}, "scene": {"granularity": 48}},
        "structural_audit": {"retained_count": 3},
    }
    lifecycle = stage2.lifecycle_contract_record(
        {"contract": contract.identity_record(), "stage1_record": fake_stage1, "stage2_recipe": recipe}
    )
    assert "acquisition_identity" not in lifecycle["stage1"]["stage1_recipe_without_acquisition_arrays"]
    assert lifecycle["ground_truth_geometry_used"] is False

    runtime_path = ROOT / "rift" / "sugavanam_ertin_stage2_runtime_v1.py"
    trainer_path = ROOT / "train_sugavanam_ertin_stage2.py"
    runtime_functions = functions_in(runtime_path)
    trainer_functions = functions_in(trainer_path)
    for name in (
        "atomic_json_dump",
        "atomic_torch_save",
        "prepare_run_root",
        "promote_complete_package",
        "record_terminal_failure",
        "same_value",
    ):
        assert name in runtime_functions, name
    for name in (
        "_cosine_learning_rate",
        "_validate_scheduler_state",
        "_validate_checkpoint",
        "_export_surface",
        "_validate_complete_package",
        "_run",
        "main",
    ):
        assert name in trainer_functions, name

    source = trainer_path.read_text(encoding="utf-8")
    for forbidden in ("radarsplat", "geraf", "response_view", "hashlib", "sha256"):
        assert forbidden not in source.lower(), forbidden
    assert "load_validated_stage1_source(contract)" in source
    assert "promote_complete_package" in source
    assert "CLEAN_INTERRUPTION_EXIT_CODE" in source
    assert "checkpoint_role\"] = \"final\"" in source
    assert "resumed_export_pending" in source
    validator_source = (ROOT / "scripts" / "validate_sugavanam_ertin_b7873200_stage2_v1.py").read_text(encoding="utf-8")
    for required in (
        "actual_entrypoint_recovery_checks",
        "os.kill(os.getpid(), signal.SIGTERM)",
        "final-init clean lifecycle",
        "export-pending clean lifecycle",
        "uninterrupted final checkpoint",
    ):
        assert required in validator_source, required
    print("SE_B7873200_STAGE2_STATIC_PASS: 31 contract/lifecycle source checks", flush=True)


if __name__ == "__main__":
    main()
