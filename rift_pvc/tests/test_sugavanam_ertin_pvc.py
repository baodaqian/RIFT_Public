"""Package C (Sugavanam--Ertin) PVC adaptation: CPU-side contract tests.

These assert the things the dispatch contract requires of a ``_pvc`` package:
identical CLI and run-function signatures, an argument list that differs from
the original only in the script name, the scientific recipe/gate imported
rather than reimplemented, and the recorded H100 initialization outcome
reproduced exactly. Device work lives in ``test_sugavanam_ertin_xpu.py``.
"""
import argparse
import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift import sugavanam_ertin_collection as se_collection_cuda  # noqa: E402
from rift import sugavanam_ertin_paper_workflow as workflow_cuda  # noqa: E402
from rift.sugavanam_ertin_paper import PaperSDF, initialization_audit  # noqa: E402
from rift_pvc import accelerator  # noqa: E402
from rift_pvc import sugavanam_ertin_collection as se_collection_pvc  # noqa: E402
from rift_pvc import sugavanam_ertin_paper_workflow as workflow_pvc  # noqa: E402

CAMPAIGN = Path("/scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645")
DATASET_ROOT = CAMPAIGN / "inputs/RIFT_dataset"

# The exact audit recorded by the H100 production runs (jobs 2150127/2150133),
# whose code snapshot predates --initialization_std and therefore ran the
# literal paper std = 1.0. Reproduced here as a gate-integrity fixture.
H100_DEGENERATE = {
    "b787": dict(kind="rift_collection", extent=0.15, positive_fraction=0.94140625),
    "gotcha": dict(kind="gotcha_native", extent=5.0, positive_fraction=0.9609375),
}


# --------------------------------------------------------------------------
# Contract rule 2: identical CLI and run-function signatures
# --------------------------------------------------------------------------

def test_run_signature_matches_original_except_device_default():
    original = inspect.signature(workflow_cuda.run)
    pvc = inspect.signature(workflow_pvc.run)
    assert list(original.parameters) == list(pvc.parameters)
    assert original.parameters["device"].default == "cuda"
    assert pvc.parameters["device"].default is None
    for name in original.parameters:
        if name != "device":
            assert original.parameters[name].default == pvc.parameters[name].default


def test_run_gotcha_signature_is_identical():
    import train_sugavanam_ertin as entry_cuda
    import train_sugavanam_ertin_pvc as entry_pvc
    expected = inspect.signature(entry_cuda.run_gotcha)
    assert inspect.signature(entry_pvc.run_gotcha) == expected
    assert inspect.signature(workflow_pvc.run_gotcha) == expected
    # the shared dispatcher reads this backend record off the entry point
    assert entry_pvc.GOTCHA_BACKEND == entry_cuda.GOTCHA_BACKEND


def _parser_arguments(parser):
    return [(a.option_strings, a.dest, a.nargs, a.const, a.choices, type(a).__name__)
            for a in parser._actions]


def test_cli_argument_list_is_identical():
    """The PVC parser accepts exactly the original flags (help text may differ)."""
    pvc = workflow_pvc.build_parser()
    # Rebuild the original's parser from its own main() by capturing --help.
    import contextlib, io
    original_help, pvc_help = io.StringIO(), io.StringIO()
    for target, buf in ((workflow_cuda.main, original_help), (workflow_pvc.main, pvc_help)):
        with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
            target(["--help"])
    original_flags = {t for t in original_help.getvalue().split() if t.startswith("--")}
    pvc_flags = {t for t in pvc_help.getvalue().split() if t.startswith("--")}
    assert original_flags == pvc_flags
    assert {"--recipe", "--object", "--config", "--resume", "--device",
            "--dry-run", "--check-initialization", "--stage1-only"} <= pvc_flags
    assert isinstance(pvc, argparse.ArgumentParser)


def test_device_flag_still_accepts_explicit_values():
    args = workflow_pvc.build_parser().parse_args(["--device", "cpu"])
    assert workflow_pvc.resolve_device(args.device) == torch.device("cpu")
    assert workflow_pvc.resolve_device(None).type == accelerator.backend()
    assert workflow_pvc.resolve_device("xpu") == torch.device("xpu")


# --------------------------------------------------------------------------
# Contract rule 2: same argument list, only the script name replaced
# --------------------------------------------------------------------------

def test_commands_differ_only_in_the_script_name():
    kwargs = dict(dataset_root=DATASET_ROOT, output_root="/tmp/se_pvc_test",
                  recipe="paper-v1", config="protocols/se_g40_readout48.json",
                  device="xpu", check_initialization=True)
    cuda = se_collection_cuda.commands_for("b787", **kwargs)
    pvc = se_collection_pvc.commands_for("b787", **kwargs)
    assert len(cuda) == len(pvc) == 1
    assert len(cuda[0]) == len(pvc[0])
    differing = [i for i, (a, b) in enumerate(zip(cuda[0], pvc[0])) if a != b]
    assert differing == [1], f"expected only the script name to differ, got {differing}"
    assert pvc[0][1].endswith("train_sugavanam_ertin_pvc.py")
    assert cuda[0][1].endswith("train_sugavanam_ertin.py")


def test_collection_reexports_original_flag_registration():
    assert se_collection_pvc.add_arguments is se_collection_cuda.add_arguments
    assert se_collection_pvc.RECIPES == se_collection_cuda.RECIPES
    assert se_collection_pvc.DEFAULT_RECIPE == se_collection_cuda.DEFAULT_RECIPE == "paper-v1"


def test_collection_preserves_original_validation():
    for kwargs, message in (
        (dict(recipe="nope"), "Unknown Sugavanam--Ertin recipe"),
        (dict(resume="auto"), "explicit --resume"),
        (dict(recipe="legacy-full", check_initialization=True), "belong to paper-v1"),
        (dict(recipe="legacy-full", stage1_only=True), "requires paper-v1"),
    ):
        with pytest.raises(ValueError):
            se_collection_pvc.commands_for("b787", dataset_root=DATASET_ROOT,
                                           output_root="/tmp/se_pvc_test", **kwargs)


# --------------------------------------------------------------------------
# Rule 4: the recipe, the gate and the checkpoint contract are NOT reimplemented
# --------------------------------------------------------------------------

def test_scientific_functions_are_the_originals_not_copies():
    for name in ("make_recipe", "plan", "grid_points", "extract_cloud",
                 "validate_resume", "refresh_surface_cpu_generator", "_model_config"):
        assert getattr(workflow_pvc, name) is getattr(workflow_cuda, name), name
    assert workflow_pvc.SCHEMA == workflow_cuda.SCHEMA


def test_recipes_are_identical_for_every_production_protocol():
    for protocol in ("protocols/se_g40_readout48.json", "protocols/se_paper_std1.json"):
        config = json.loads((ROOT / protocol).read_text())
        for kind in ("rift_collection", "gotcha_native"):
            assert workflow_pvc.make_recipe(kind, config) == workflow_cuda.make_recipe(kind, config)


def test_production_protocol_selects_std_005_and_passes_the_gate():
    """std = 0.05 is the production value; a large std zeroes the gradients."""
    config = json.loads((ROOT / "protocols/se_g40_readout48.json").read_text())
    for tag, fixture in H100_DEGENERATE.items():
        recipe, audit = workflow_pvc.initialization_gate_reference(
            fixture["kind"], fixture["extent"], config=config)
        assert recipe["initialization_std"] == 0.05
        assert recipe["fidelity"] == "user_requested_gaussian_initialization_std_override"
        assert audit["status"] == "initialization_probe_passed", tag
        assert audit["mean_gradient_norm"] > 0.0
        assert audit["saturated_fraction"] == 0.0
        assert audit["nonzero_spatial_gradients"] == audit["samples"]


def test_literal_paper_std1_reproduces_the_recorded_h100_gate_exactly():
    """Gate integrity: the published-initialization check must still fire."""
    config = json.loads((ROOT / "protocols/se_paper_std1.json").read_text())
    for tag, fixture in H100_DEGENERATE.items():
        recipe, audit = workflow_pvc.initialization_gate_reference(
            fixture["kind"], fixture["extent"], config=config)
        assert "initialization_std" not in recipe, "std=1.0 keeps the literal recipe identity"
        assert recipe["fidelity"] == "paper_equations_declared_author_gaps"
        assert audit["status"] == "initialization_degenerate", tag
        assert audit["mean_gradient_norm"] == 0.0
        assert audit["saturated_fraction"] == 1.0
        assert audit["nonzero_spatial_gradients"] == 0
        assert audit["positive_fraction"] == fixture["positive_fraction"]
        assert audit["initialization"] == "standard_gaussian_weights_and_biases"


@pytest.mark.skipif(not CAMPAIGN.is_dir(), reason="H100 campaign outputs not reachable")
@pytest.mark.parametrize("tag,relative", [
    ("b787", "outputs/rift/train2400/1t1r_2010b2bbe725/b787/sugavanam_ertin/initialization.json"),
    ("gotcha", "outputs/gotcha/camry/d3d8822490914f54/sugavanam_ertin/initialization.json"),
])
def test_gate_audit_matches_the_h100_record_field_for_field(tag, relative):
    record = CAMPAIGN / relative
    if not record.is_file():
        pytest.skip(f"{record} absent")
    h100 = json.loads(record.read_text())["initialization_audit"]
    fixture = H100_DEGENERATE[tag]
    _, audit = workflow_pvc.initialization_gate_reference(
        fixture["kind"], fixture["extent"], config={"initialization_std": 1.0})
    assert {k: audit[k] for k in h100} == h100


# --------------------------------------------------------------------------
# Entry point behaviour
# --------------------------------------------------------------------------

def test_entry_point_refuses_the_sealed_legacy_full_cuda_lane():
    import train_sugavanam_ertin_pvc as entry_pvc
    with pytest.raises(SystemExit) as excinfo:
        entry_pvc.main(["--recipe", "legacy-full"])
    assert "legacy-full" in str(excinfo.value)
    assert "train_sugavanam_ertin.py" in str(excinfo.value)


def test_entry_point_refuses_a_non_pvc_backend(monkeypatch):
    import train_sugavanam_ertin_pvc as entry_pvc
    monkeypatch.setenv("RIFT_ACCELERATOR", "cpu")
    monkeypatch.delenv("RIFT_PVC_ALLOW_BACKEND", raising=False)
    with pytest.raises(RuntimeError, match="PVC entry point"):
        entry_pvc.check_backend()
    monkeypatch.setenv("RIFT_PVC_ALLOW_BACKEND", "cpu")
    assert entry_pvc.check_backend() == "cpu"


def test_stage1_entry_point_reexports_the_untouched_contract():
    import train_sugavanam_ertin_stage1 as original
    import train_sugavanam_ertin_stage1_pvc as pvc
    assert pvc.build_parser is original.build_parser
    assert pvc.parse_args is original.parse_args
    assert pvc.GENERIC_FINAL_PATH == original.GENERIC_FINAL_PATH
    assert pvc.CHECKPOINT_NAME == original.CHECKPOINT_NAME


def test_cuda_modules_are_not_mutated_by_importing_the_pvc_package():
    """Isolation rule: importing rift_pvc must not touch the CUDA pipeline."""
    assert workflow_cuda.run.__module__ == "rift.sugavanam_ertin_paper_workflow"
    assert inspect.signature(workflow_cuda.run).parameters["device"].default == "cuda"
    assert se_collection_cuda.commands_for("b787", dataset_root=DATASET_ROOT,
        output_root="/tmp/se_pvc_test")[0][1].endswith("train_sugavanam_ertin.py")


# --------------------------------------------------------------------------
# Stage 2 (audit C.5): the sampling adaptation and the rebinding
# --------------------------------------------------------------------------

def test_stage2_sampling_proxy_only_changes_generator_and_draws():
    from rift_pvc import sugavanam_ertin_stage2_sampling as sampling
    proxy = sampling.CpuGeneratorTorch(torch)
    assert sampling.CpuGeneratorTorch.OVERRIDDEN == ("Generator", "randint", "rand", "randn", "randperm")
    # everything else is the real torch
    for name in ("Tensor", "optim", "load", "save", "cuda", "float32", "manual_seed",
                 "zeros", "nn", "autograd", "linspace"):
        assert getattr(proxy, name) is getattr(torch, name), name
    assert proxy.randint is not torch.randint
    assert proxy.rand is not torch.rand


def test_stage2_xpu_generator_constructed_on_cpu_and_draw_state_recovers():
    from rift_pvc.sugavanam_ertin_stage2_sampling import CpuGeneratorTorch
    proxy = CpuGeneratorTorch()
    generator = proxy.Generator(device="xpu:0").manual_seed(73)
    assert generator.device.type == "cpu"
    before = generator.get_state()
    draws = [proxy.rand(4, generator=generator), proxy.randn(4, generator=generator),
             proxy.randint(20, (4,), generator=generator), proxy.randperm(10, generator=generator)]
    generator.set_state(before)
    reference = [torch.rand(4, generator=generator), torch.randn(4, generator=generator),
                 torch.randint(20, (4,), generator=generator), torch.randperm(10, generator=generator)]
    for actual, expected in zip(draws, reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_stage2_strict_refresh_preserves_cpu_algorithm_and_gates():
    from rift.sugavanam_ertin_a320_stabilized import refresh_iso_points_strict as original
    from rift_pvc.sugavanam_ertin_stage2_refresh import refresh_iso_points_strict as pvc
    class Sphere(torch.nn.Module):
        def forward(self, points):
            return points.norm(dim=-1) - .05
    seeds = torch.randn(128, 3, generator=torch.Generator().manual_seed(1))
    seeds = torch.nn.functional.normalize(seeds, dim=-1) * .05
    a, aa = original(Sphere(), seeds, .15, .002, 32, torch.Generator().manual_seed(2))
    b, ba = pvc(Sphere(), seeds, .15, .002, 32, torch.Generator().manual_seed(2))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert aa == ba and aa["status"] == "passed"
    assert aa["min_acceptance"] == .6


def test_stage2_cpu_generator_draws_are_identical_to_a_plain_cpu_draw():
    """The proxy must not change the numbers, only where they are allocated."""
    from rift_pvc import sugavanam_ertin_stage2_sampling as sampling
    proxy = sampling.CpuGeneratorTorch(torch)
    cpu = torch.device("cpu")
    expected = torch.randint(100, (16,), generator=torch.Generator().manual_seed(7))
    got = proxy.randint(100, (16,), generator=torch.Generator().manual_seed(7), device=cpu)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    expected_roi = sampling.sample_roi(8, 0.15, torch.Generator().manual_seed(9), cpu)
    from rift.sugavanam_ertin_validzero import sample_roi as original_sample_roi
    original = original_sample_roi(8, 0.15, torch.Generator().manual_seed(9), cpu)
    torch.testing.assert_close(expected_roi, original, rtol=0, atol=0)
    with pytest.raises(ValueError):
        sampling.sample_roi(0, 0.15, torch.Generator(), cpu)


def test_stage2_draws_on_cpu_predicate():
    from rift_pvc import sugavanam_ertin_stage2_sampling as sampling
    cpu_generator = torch.Generator()
    assert not sampling.draws_on_cpu(cpu_generator, torch.device("cpu"))
    assert not sampling.draws_on_cpu(None, torch.device("xpu"))
    assert sampling.draws_on_cpu(cpu_generator, torch.device("xpu"))
    assert sampling.draws_on_cpu(cpu_generator, torch.device("cuda"))


def test_stage2_install_rebinds_then_restores_the_original_module():
    import train_sugavanam_ertin_stage2 as stage2
    import train_sugavanam_ertin_stage2_pvc as stage2_pvc
    from rift_pvc import sugavanam_ertin_stage2_sampling as sampling
    originals = dict(stage2_pvc._ORIGINAL)
    try:
        stage2_pvc.install()
        assert stage2.sample_roi is sampling.sample_roi
        from rift_pvc.sugavanam_ertin_stage2_refresh import refresh_iso_points_strict
        assert stage2.refresh_iso_points_strict is refresh_iso_points_strict
        assert isinstance(stage2.torch, sampling.CpuGeneratorTorch)
        assert stage2._set_seed is stage2_pvc._set_seed
        assert stage2.build_parser is stage2_pvc.build_parser
    finally:
        stage2_pvc.uninstall()
    for name, value in originals.items():
        assert getattr(stage2, name) is value, f"{name} was not restored"


def test_stage2_rng_payload_schema_has_no_xpu_key():
    """_validate_rng_state rejects keys outside the closed CUDA-era set."""
    import numpy as np
    import random as _random
    import train_sugavanam_ertin_stage2 as stage2
    payload = {"python": _random.getstate(), "numpy": np.random.get_state(),
               "torch_cpu": torch.get_rng_state()}
    stage2._validate_rng_state(payload)  # the CPU/XPU lane payload is valid
    with pytest.raises(stage2.RuntimeContractError):
        stage2._validate_rng_state({**payload, "torch_xpu": [torch.get_rng_state()]})


def test_stage2_device_default_follows_the_active_backend():
    import train_sugavanam_ertin_stage2_pvc as stage2_pvc
    args = stage2_pvc.build_parser().parse_args([])
    assert torch.device(args.device).type == accelerator.backend()


def test_stage2_sealed_inputs_are_not_on_this_cluster():
    """Documents why this entry point has no end-to-end smoke on ACES."""
    from rift.sugavanam_ertin_b7873200_stage1 import STAGE1_BUNDLE_PATH
    from rift.sugavanam_ertin_b7873200_stage2_v1 import OUTPUT_DIR
    assert STAGE1_BUNDLE_PATH.startswith("/storage/"), STAGE1_BUNDLE_PATH
    assert OUTPUT_DIR.startswith("/storage/"), OUTPUT_DIR
    assert not Path(STAGE1_BUNDLE_PATH).exists()
