"""Unified smoke CLI and object-bound engineering subsets, without real training."""
import ast
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from rift import rift_dataset as dataset
from rift import sugavanam_ertin_b7873200_real_smoke as protocol
import train_sugavanam_ertin_smoke as smoke


OBJECTS = [row["object_id"] for row in dataset.catalog()["objects"]]


def test_only_one_root_driver_per_smoke_family():
    root = dataset.PROJECT_ROOT
    assert sorted(path.name for path in root.glob("train_spinr_style*smoke*.py")) == [
        "train_spinr_style_smoke.py"]
    assert sorted(path.name for path in root.glob("train_sugavanam_ertin*smoke*.py")) == [
        "train_sugavanam_ertin_smoke.py"]
    for name in ("spinr_smoke_runtime.py", "sugavanam_ertin_stabilized_smoke_runtime.py"):
        source = (root / "rift" / name).read_text()
        ast.parse(source)
        assert 'if __name__ == "__main__"' not in source


@pytest.mark.parametrize("name", OBJECTS + ["a320", "x59", "racecar"])
def test_unified_object_cli_and_separate_output_identity(name, tmp_path, monkeypatch):
    args = smoke.parse_args(["--object", name, "--dataset-root", str(tmp_path / "data"),
                             "--checkpoint-root", str(tmp_path / "runs")])
    canonical = dataset.object_spec(name)["object_id"]
    assert args.object == canonical
    assert (Path(args.npz_path), Path(args.parent_role_manifest)) == dataset.object_paths(tmp_path / "data", name)
    monkeypatch.setattr(smoke, "collection_manifest", lambda _: True)
    monkeypatch.setattr(smoke, "load_canonical_parent", lambda *_: (
        dataset.role_manifest(canonical), dataset._object_contract(canonical)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    root, *_ = smoke._validate_cli(args)
    assert root.name == f"rift_dataset_{canonical}_se_smoke_v1"
    assert not root.exists()  # Validation alone cannot create a run or fit a scene.


def test_conflicting_named_archive_and_swapped_named_object_fail(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Conflicting"):
        smoke.parse_args(["--object", "a320", "--npz-path", str(tmp_path / "b787.npz"),
                          "--checkpoint-root", str(tmp_path)])
    args = smoke.parse_args(["--object", "a320", "--checkpoint-root", str(tmp_path)])
    monkeypatch.setattr(smoke, "collection_manifest", lambda _: True)
    monkeypatch.setattr(smoke, "load_canonical_parent", lambda *_: (
        dataset.role_manifest("b787"), dataset._object_contract("b787")))
    with pytest.raises(ValueError, match="does not match"):
        smoke._validate_cli(args)


@pytest.mark.parametrize("name", OBJECTS)
def test_exact_collection_smoke_subset(name):
    parent = dataset.role_manifest(name)
    child = protocol.build_child_manifest(parent)
    protocol.validate_child_manifest(parent, child)
    assert child["dataset"] == parent["dataset"]
    assert child["engineering_subset"]["schema"] == protocol.COLLECTION_SMOKE_SCHEMA
    assert child["engineering_subset"]["parent_manifest_name"] == parent["name"]
    assert child["name"] == f"rift_dataset_{name}_se_smoke16x16_v1"
    assert child["split"]["train_indices"] == parent["split"]["train_indices"][:16]
    assert child["split"]["validation_indices"] == parent["split"]["validation_indices"][:16]
    assert child["split"]["test_indices"] == parent["split"]["test_indices"]
    joined = sum((child["split"][key] for key in (
        "train_indices", "validation_indices", "test_indices", "unused_indices")), [])
    assert len(joined) == len(set(joined)) == 10000
    assert len(child["split"]["unused_indices"]) == 8968
    for field, value in (("num_train", 16.0), ("test_sealed", False),
                         ("train_indices", parent["split"]["train_indices"][:17])):
        invalid = copy.deepcopy(child)
        invalid["split"][field] = value
        with pytest.raises(ValueError, match="collection smoke child"):
            protocol.validate_child_manifest(parent, invalid)
    invalid = copy.deepcopy(child)
    invalid["dataset"]["object_id"] = "b787" if name != "b787" else "loader"
    with pytest.raises(ValueError, match="collection smoke child"):
        protocol.validate_child_manifest(parent, invalid)


def test_legacy_stabilized_dispatch_preserves_distinct_recipe(monkeypatch):
    from rift import sugavanam_ertin_stabilized_smoke_runtime as legacy
    run = Mock()
    monkeypatch.setattr(legacy, "main", run)
    assert smoke.dispatch_main(["--recipe", "legacy-a320-stabilized", "--object", "a320",
                                "--steps", "120"]) == 0
    run.assert_called_once_with(["--steps", "120"])
    with pytest.raises(ValueError, match="historical A320"):
        smoke.dispatch_main(["--recipe", "legacy-a320-stabilized", "--object", "b787"])
    assert run.call_count == 1
    args = smoke.parse_args(["--checkpoint-root", "/tmp/unused_smoke_test_root"])
    assert args.npz_path == protocol.B787_CANONICAL_NPZ
    assert args.parent_role_manifest == protocol.B787_PARENT_MANIFEST


@pytest.mark.parametrize("name", OBJECTS)
def test_real_child_ingress_is_metadata_only_and_sealed(name, tmp_path, monkeypatch):
    from rift import npz_dataset
    import train

    npz, parent_path = dataset.object_paths(dataset.DEFAULT_ROOT, name)
    if not npz.is_file() or not parent_path.is_file():
        pytest.skip("Local radar archives are not distributed with source")

    def forbidden(*args, **kwargs):
        raise AssertionError("No real response read or conversion is permitted")

    monkeypatch.setattr(npz_dataset, "_validated_response_stream", forbidden)
    parent, parent_contract = protocol.load_canonical_parent(npz, parent_path)
    child = protocol.build_child_manifest(parent)
    path = tmp_path / "child.json"
    path.write_text(json.dumps(child))
    arrays, contract = protocol.load_collection_smoke_inputs(npz, path)
    assert arrays["response"] is None
    assert contract["dataset_identity"] == dataset.object_identity(name)
    assert len(arrays["_lazy_response_reader"].allowed_view_indices) == 32
    for role in ("reserved_test", "unused"):
        with pytest.raises(PermissionError):
            npz_dataset.get_npz_response_view(arrays, contract["role_ids"][role][0])
    _, observed = train._load_sealed_npz_protocol_contract(npz, path, num_train=16, num_val=16, num_test=1000)
    assert observed == contract
    with pytest.raises(ValueError, match="16/16/1000"):
        train._load_sealed_npz_protocol_contract(npz, path, num_train=3200, num_val=1000, num_test=1000)
    # Smoke subsets must not masquerade as full production collection contracts.
    with pytest.raises(ValueError):
        dataset.load_object_contract(npz, path)
    other = "b787" if name != "b787" else "loader"
    other_child = protocol.build_child_manifest(dataset.role_manifest(other))
    path.write_text(json.dumps(other_child))
    with pytest.raises(ValueError, match="does not match NPZ"):
        protocol.load_collection_smoke_inputs(npz, path)
    dataset.validate_checkpoint_object({"sealed_npz_protocol_contract": contract}, parent_contract)
    with pytest.raises(ValueError, match="does not match"):
        dataset.validate_checkpoint_object({"sealed_npz_protocol_contract": contract}, dataset._object_contract(other))


def test_new_stage2_provenance_binds_actual_object(tmp_path):
    identity = dataset.object_identity("loader")
    actual = smoke._stage2_provenance(stage1_record={}, recipe={}, output_dir=tmp_path,
                                     npz_path="/selected/loader.npz", dataset_identity=identity)
    assert actual["contract"]["canonical_npz_path"] == "/selected/loader.npz"
    assert actual["contract"]["dataset_identity"] == identity
    dataset.validate_checkpoint_object(actual, dataset._object_contract("loader"))


def test_stage2_recipe_constructor_and_backward_on_cpu():
    from rift.sugavanam_ertin import FourierFeatureSDF
    recipe = smoke._smoke_stage2_recipe()
    model = FourierFeatureSDF(extent=0.15, **{key: recipe[key] for key in (
        "n_fourier", "fourier_scale", "hidden_dim", "n_layers", "seed")})
    values = model(torch.zeros((4, 3)))
    values.square().mean().backward()
    assert bool(torch.isfinite(values).all())
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
