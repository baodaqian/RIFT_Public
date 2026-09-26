"""Adaptive default, explicit legacy recipes, object binding, and unified CLI."""
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

import train
from rift import rift_dataset as dataset
from rift import adaptive_training_workflow as workflow
from rift.b7873200_adaptive_fullscale import fullscale_train_argv
from train_rift_dataset import METHODS, commands_for


OBJECTS = [spec["object_id"] for spec in dataset.catalog()["objects"]]


def test_adaptive_default_matches_reviewed_recipe():
    default = train.parse_args(["--checkpoint-name", "probe"])
    reviewed = train.parse_args(fullscale_train_argv(npz_path="data.npz", manifest_path="roles.json",
                                                    checkpoint_root="checkpoints"))
    assert default.architecture == reviewed.architecture == "adaptive"
    assert default.scene_repr == "point_sh" and default.adaptive_capacity_v2
    assert default.max_points == 262144 and default.granularity == 48
    assert default.sh_init_degree == 0 and default.sh_max_degree == 3
    assert default.adaptive_refine_every == 10 and default.adaptive_probe_every == 16
    assert default.forward_operator == "range" and default.compute_dtype == "float64"
    ignored = {"checkpoint_name", "checkpoint_root", "npz_path", "npz_role_manifest", "execution_contract_label"}
    assert {key: value for key, value in vars(default).items() if key not in ignored} == {
        key: value for key, value in vars(reviewed).items() if key not in ignored}


@pytest.mark.parametrize("architecture", ["grid", "grid_sh", "point_sh", "mlp"])
def test_explicit_legacy_scene_defaults_unchanged(architecture):
    old = train.parse_args(["--checkpoint-name", "probe", "--scene-repr", architecture])
    new = train.parse_args(["--checkpoint-name", "probe", "--architecture", architecture])
    assert vars(old) == vars(new)
    assert old.scene_repr == architecture and not old.adaptive_capacity_v2
    assert old.data_format == "csv" and old.forward_operator == "brute"
    assert old.max_points == 0 and old.adaptive_refine_every == 0
    assert old.num_train == 1000 and old.epochs == 100


def test_explicit_options_override_adaptive_defaults():
    args = train.parse_args(["--checkpoint-name", "probe", "--architecture", "adaptive",
                             "--epochs", "7", "--lr", "0.001", "--max-points", "120000",
                             "--adaptive-max-active", "120000", "--adaptive-refine-every", "3"])
    assert args.epochs == 7 and args.lr == 0.001 and args.max_points == 120000
    assert args.adaptive_refine_every == 3 and args.adaptive_capacity_v2
    with pytest.raises(SystemExit):
        train.parse_args(["--checkpoint-name", "bad", "--architecture", "adaptive", "--scene-repr", "grid"])


@pytest.mark.parametrize("name", OBJECTS + ["a320", "x59", "racecar"])
def test_named_object_cli_binds_geometry_and_roles_without_reading(name, tmp_path):
    args = train.parse_args(["--checkpoint-name", "probe", "--object", name, "--dataset-root", str(tmp_path)])
    npz, manifest = dataset.object_paths(tmp_path, name)
    assert (Path(args.npz_path), Path(args.npz_role_manifest)) == (npz, manifest)
    assert args.data_format == "npz" and args.npz_sealed_protocol
    assert (args.num_train, args.num_val, args.num_test) == (3200, 1000, 1000)
    assert args.extent == 0.15 and args.phase_sign == -1 and args.num_freq_wanted == 600
    assert args.architecture == "adaptive"
    fixed = train.parse_args(["--checkpoint-name", "probe", "--object", name,
                              "--dataset-root", str(tmp_path), "--architecture", "grid_sh"])
    assert fixed.scene_repr == "grid_sh" and not fixed.adaptive_capacity_v2
    assert fixed.extent == 0.15 and fixed.npz_sealed_protocol


def test_named_object_rejects_conflicting_paths_and_swapped_files(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Conflicting"):
        train.parse_args(["--checkpoint-name", "bad", "--object", "a320", "--npz-path", str(tmp_path / "wrong.npz")])
    monkeypatch.setattr(dataset, "load_object_contract", lambda *_: ({}, dataset._object_contract("b787")))
    with pytest.raises(ValueError, match="Selected object"):
        train.main(["--checkpoint-name", "bad", "--object", "a320"])


def test_adaptive_reporting_dispatch_keeps_one_public_entrypoint(monkeypatch):
    run = Mock(return_value=7)
    monkeypatch.setattr(workflow, "main", run)
    args = ["--workflow", "adaptive-fullscale", "--checkpoint-root", "unused", "--report-only"]
    assert train.main(args) == 7
    run.assert_called_once_with(args[2:])
    with pytest.raises(ValueError, match="owns its observer"):
        train.main(args, adaptive_event_observer_factory=lambda **_: None)
    assert not (dataset.PROJECT_ROOT / "train_adaptive_rift_b787.py").exists()
    assert 'if __name__ == "__main__"' not in Path(workflow.__file__).read_text()


@pytest.mark.parametrize("name", OBJECTS)
def test_dataset_rift_means_adaptive_and_grid_remains_explicit(name, tmp_path):
    kwargs = {"dataset_root": tmp_path, "output_root": tmp_path / "runs"}
    default = commands_for(name, "rift", **kwargs)[0]
    adaptive = commands_for(name, "adaptive_rift", **kwargs)[0]
    grid = commands_for(name, "rift_grid", **kwargs)[0]
    assert default == adaptive
    assert Path(default[1]).name == Path(grid[1]).name == "train.py"
    args = train.parse_args(default[2:])
    fixed = train.parse_args(grid[2:])
    assert args.architecture == "adaptive" and args.scene_repr == "point_sh" and args.adaptive_capacity_v2
    assert fixed.architecture == "grid_sh" and not fixed.adaptive_capacity_v2
    assert args.checkpoint_name == "rift" and fixed.checkpoint_name == "rift_grid"


def test_named_trainers_are_object_neutral_and_reference_is_separate(tmp_path):
    assert not list(dataset.PROJECT_ROOT.glob("train*b787*.py"))
    import train_sugavanam_ertin as maintained
    import train_sugavanam_ertin_reference as reference
    assert hasattr(maintained, "FullStage1Observer") and not hasattr(reference, "FullStage1Observer")
    for method in METHODS:
        for command in commands_for("loader", method, dataset_root=tmp_path, output_root=tmp_path):
            assert Path(command[1]).is_file()


@pytest.mark.parametrize("name", OBJECTS)
def test_fullscale_reporting_can_bind_each_real_object_metadata_only(name, tmp_path, monkeypatch):
    from rift import npz_dataset
    npz, manifest = dataset.object_paths(dataset.DEFAULT_ROOT, name)
    if not npz.exists() or not manifest.exists():
        pytest.skip("Local radar archives are not distributed with source")
    monkeypatch.setattr(npz_dataset._LazyResponseReader, "iter_response_views", Mock(side_effect=AssertionError("payload")))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    args = workflow.parse_args(["--object", name, "--checkpoint-root", str(tmp_path)])
    root = workflow._validate_cli(args)
    assert root.parent == tmp_path / name and not root.exists()
    contract = workflow._validate_parent_header(args.npz_path, args.parent_role_manifest)
    assert contract["dataset_identity"] == dataset.object_identity(name)
