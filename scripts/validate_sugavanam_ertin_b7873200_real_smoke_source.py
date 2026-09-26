#!/usr/bin/env python3
"""Tiny source-only gate for the combined B787 Sugavanam--Ertin smoke.

This validator parses source and checks the declared identities and resource
envelope.  It does not import Torch, open the B787 archive, materialize a
response, create checkpoints, or run a training step.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE_MODULE = ROOT / "rift" / "sugavanam_ertin_b7873200_real_smoke.py"
SMOKE_DRIVER = ROOT / "train_sugavanam_ertin_smoke.py"
TRAINER = ROOT / "train.py"
STAGE2_DRIVER = ROOT / "train_sugavanam_ertin_stage2.py"
STAGE2_CONTRACT = ROOT / "rift" / "sugavanam_ertin_b7873200_stage2_v1.py"
SDF_MODULE = ROOT / "rift" / "sugavanam_ertin.py"
LAUNCHER = ROOT / "slurm" / "validate_sugavanam_ertin_b7873200_real_smoke_v1.sbatch"
RUNTIME_CHECKS = ROOT / "scripts" / "validate_sugavanam_ertin_b7873200_real_smoke_runtime.py"


def source(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if "\r" in text:
        raise AssertionError(f"{path.name} must use LF line endings")
    ast.parse(text, filename=str(path))
    if "hashlib" in text or "sha256" in text.lower():
        raise AssertionError(f"{path.name} contains forbidden hash validation")
    return text


def shell_source(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if "\r" in text:
        raise AssertionError(f"{path.name} must use LF line endings")
    if "hashlib" in text or "sha256" in text.lower():
        raise AssertionError(f"{path.name} contains forbidden hash validation")
    return text


def require(text: str, fragment: str) -> None:
    if fragment not in text:
        raise AssertionError(f"missing source fragment: {fragment!r}")


def check_smoke_recipe_constructor_constraints(driver_text: str, sdf_text: str) -> None:
    tree = ast.parse(driver_text, filename=str(SMOKE_DRIVER))
    recipe_function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_smoke_stage2_recipe"
    )
    return_node = next(node for node in recipe_function.body if isinstance(node, ast.Return))
    if not isinstance(return_node.value, ast.Dict):
        raise AssertionError("smoke Stage-2 recipe is not a literal mapping")
    literal = {}
    for key, value in zip(return_node.value.keys, return_node.value.values):
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
            literal[key.value] = value.value
    if literal.get("schema") != "rift_sugavanam_ertin_b7873200_stage2_engineering_smoke_recipe_v3":
        raise AssertionError("smoke Stage-2 recipe revision is not v3")
    if literal.get("recipe_revision") != 3 or literal.get("n_fourier") != 5 or literal.get("hidden_dim") != 128:
        raise AssertionError("smoke Stage-2 recipe fixed architecture changed")
    if literal.get("init_steps") != 1000 or literal.get("init_lr") != 5.0e-4 or literal.get("init_batch") != 2048 or literal.get("init_log_every") != 100:
        raise AssertionError("smoke Stage-2 initialization tuning changed")
    if literal.get("sdf_skip_layer_index") != 4 or literal.get("sdf_min_layers") != 5 or literal.get("n_layers") != 5:
        raise AssertionError("smoke Stage-2 recipe is incompatible with its layer-4 skip")
    require(sdf_text, "n_layers < 5")
    require(sdf_text, "if i == 4:")
    require(sdf_text, "h = torch.cat((h, encoded), dim=-1)")


def main() -> None:
    module = source(SMOKE_MODULE)
    driver = source(SMOKE_DRIVER)
    trainer = source(TRAINER)
    stage2 = source(STAGE2_DRIVER)
    stage2_contract = source(STAGE2_CONTRACT)
    sdf = source(SDF_MODULE)
    runtime = source(RUNTIME_CHECKS)
    launcher = shell_source(LAUNCHER)
    check_smoke_recipe_constructor_constraints(driver, sdf)

    for fragment in (
        "SMOKE_MANIFEST_NAME",
        "SMOKE_TRAIN_COUNT = 16",
        "SMOKE_VALIDATION_COUNT = 16",
        "SMOKE_TEST_COUNT = 1_000",
        "SMOKE_STAGE1_EPOCHS = 2",
        "SMOKE_STAGE1_EXPECTED_UPDATES = 32",
        "SMOKE_STAGE1_RECIPE_REVISION = 3",
        "SMOKE_STAGE1_EXTENT = 0.15",
        "SMOKE_STAGE1_GRANULARITY = 16",
        "parent_fixed_tail_prefix_engineering_se_v1",
        "compute_train_only_signal_normalization",
        "complex_signal_statistics",
        "analytic_sdf_shell_diagnostic",
        "smoke_stage1_checkpoint_dir",
        "assert_restricted_roles",
        "range_model",
    ):
        require(module, fragment)

    for fragment in (
        "--forward-operator",
        "\"range\"",
        "\"product\"",
        "\"--phase-sign\", \"-1\"",
        "\"--num-freq-wanted\", \"600\"",
        "\"--extent\", str(SMOKE_STAGE1_EXTENT)",
        "\"--granularity\", str(SMOKE_STAGE1_GRANULARITY)",
    ):
        require(module, fragment)

    for fragment in (
        "SMOKE_STAGE1_BUNDLE_FIELD",
        "stage2._run(",
        "engineering_override=True",
        "import copy",
        "import numpy as np",
        "def _resource_failure_guard",
        "_RESOURCE_CONTEXT[\"stage1\"]",
        "_RESOURCE_CONTEXT[\"stage2\"]",
        '_RESOURCE_CONTEXT["phase"] = "export"',
        "engineering_observer_factory=engineering_observer_factory",
        "geometry_feasibility = analytic_sdf_shell_diagnostic",
        "protected_shell_depth",
        "json.dumps(geometry_feasibility, sort_keys=True)",
        "_validate_resource_snapshot",
        '"fit_evidence": stage1_audit["observed_fit"]',
        "test_and_unused_response_materialized",
        "production_clearance",
        "historical_baseline_unchanged",
    ):
        require(driver, fragment)

    for fragment in (
        "engineering_observer_factory=None",
        "return_metrics=False",
        "before_optimizer_step",
        "on_epoch_end",
        "engineering_observer=engineering_observer",
    ):
        require(trainer, fragment)

    for fragment in (
        "def _capture_rng(*, include_cuda: bool)",
        "def _restore_rng(payload: Mapping[str, object], *, require_cuda: bool)",
        "require_cuda_rng=device.type == \"cuda\"",
        "_restore_rng(state[\"rng_state\"], require_cuda=device.type == \"cuda\")",
    ):
        require(stage2, fragment)
    require(stage2_contract, "def provenance_output_identity")

    for fragment in (
        "test_numeric_normalization",
        "test_geometry_feasibility_g8_vs_g16",
        "test_stage1_path_agreement",
        "test_smoke_stage2_recipe_constructor_and_backward",
        "test_observer_runtime_names",
        "test_validate_generic_stage1_final_composition",
        "test_record_to_lifecycle",
        "test_cuda_rng_contract",
        "test_resource_acceptance",
    ):
        require(runtime, fragment)

    for fragment in (
        "#SBATCH --qos=inferno",
        "#SBATCH --partition=gpu-rtx6000",
        "#SBATCH --gres=gpu:rtx_6000:1",
        "#SBATCH --cpus-per-task=6",
        "#SBATCH --mem=32G",
        "#SBATCH --tmp=12G",
        "#SBATCH --time=01:00:00",
        "#SBATCH --job-name=mgr_se_b7873200_real_smoke_v4",
        "SMOKE_ROOT=/storage/scratch1/1/dbao31/rift_b7873200_sugavanam_ertin_real_smoke_v4",
        "SMOKE_RUN_NAME=b78710k_sugavanam_ertin_stage1_stage2_engineering_smoke_v3",
        "STAGE1_CHECKPOINT_NAME=sugavanam_ertin_b7873200_stage1_engineering_smoke_v3",
        "--device cuda",
        "RIFT_SITE=",
        "EXPORT_SITE=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/sensor_fusion/lib/python3.10/site-packages",
        "from skimage.measure import marching_cubes",
        "One combined real-data",
        "AutoResume=false",
        "SE_B7873200_REAL_SMOKE_LIFECYCLE_COMPLETE",
    ):
        require(launcher, fragment)
    for forbidden in ("pip install", "conda install", "sbatch", "srun", "train_sugavanam_ertin_stage1.py"):
        if forbidden in launcher:
            raise AssertionError(f"launcher contains forbidden action: {forbidden!r}")

    print("SE_B7873200_REAL_SMOKE_SOURCE_STATIC_PASS: identity, sealing, RNG, and one-job envelope", flush=True)


if __name__ == "__main__":
    main()
