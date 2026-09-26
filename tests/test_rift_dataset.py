"""Collection, sealed-role, command routing, and cross-object resume contracts."""
import copy
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

from rift.rift_dataset import (catalog, collection_contract, manifest_name,
                               metadata_object_id, object_identity, object_paths,
                               object_spec, role_ids, role_manifest, validate_manifest_object)
from train_rift_dataset import METHODS, commands_for


OBJECTS = [spec["object_id"] for spec in catalog()["objects"]]


def metadata(name):
    return {"target_type": "b787" if name == "b787" else "mesh", "target_id": name,
            "experiment": "sphere10k", "radar_fc_hz": 1e10, "radar_bandwidth_hz": 3e9,
            "viewpoint_sampling": "fibonacci_sphere", "target_position_m": [0.0, 0.0, 0.0],
            "num_adc_samples": 600, "num_chirps_cpi": 1, "scaled_max_extent_m": 0.1}


def contract(name):
    return {"schema": "rift_npz_sealed_protocol_v1", "version": 1, "data_format": "npz",
            "response_shape": [10000, 16, 16, 1, 600], "response_dtype": "complex64",
            "role_manifest_name": manifest_name(name), "split_strategy": "fixed_tail_subsampled",
            "dataset_identity": object_identity(name), "role_ids": role_ids(),
            "response_access": {"train_materialized": True, "validation_materialized": True,
                                "reserved_test_materialized": False, "unused_materialized": False}}


def test_catalog_and_partition():
    assert len(set(OBJECTS)) == 6
    assert object_spec("a320")["object_id"] == "airliner_a320"
    assert object_spec("racecar")["object_id"] == "race_car"
    roles = role_ids()
    assert {key: len(value) for key, value in roles.items()} == {
        "train": 3200, "validation": 1000, "reserved_test": 1000, "unused": 4800}
    joined = sum(roles.values(), [])
    assert len(joined) == len(set(joined)) == 10000
    assert set(joined) == set(range(10000))


@pytest.mark.parametrize("name", OBJECTS)
def test_metadata_binding(name):
    assert metadata_object_id(metadata(name)) == name
    assert validate_manifest_object(role_manifest(name), metadata(name)) == object_identity(name)
    other = "b787" if name != "b787" else "loader"
    with pytest.raises(ValueError, match="object does not match"):
        validate_manifest_object(role_manifest(name), metadata(other))
    changed = role_manifest(name)
    changed["split"]["train_indices"].reverse()
    with pytest.raises(ValueError, match="registered"):
        validate_manifest_object(changed, metadata(name))


@pytest.mark.parametrize("name", OBJECTS)
def test_all_baseline_contract_adapters_preserve_object(name):
    from rift.geraf_b7873200_protocol import b7873200_sealed_protocol_identity
    from rift.radarsplat_b7873200_protocol import b7873200_sealed_identity
    from rift.sugavanam_ertin_b7873200_stage1 import canonical_sealed_identity, _metadata_identity
    from rift.spinr_style import _canonical_b787_metadata
    expected = contract(name)
    for adapter in (collection_contract, b7873200_sealed_protocol_identity,
                    b7873200_sealed_identity, canonical_sealed_identity):
        assert adapter(expected) == expected
        changed = copy.deepcopy(expected)
        changed["response_access"]["reserved_test_materialized"] = True
        with pytest.raises(ValueError):
            adapter(changed)
    for adapter in (_metadata_identity, _canonical_b787_metadata):
        identity = adapter(metadata(name))
        assert identity.get("target_id", "b787") == name


def test_missing_object_identity_cannot_become_legacy():
    changed = contract("b787")
    del changed["dataset_identity"]
    with pytest.raises(ValueError, match="missing"):
        collection_contract(changed)


def test_cross_object_resume_rejected():
    from train import _validate_saved_sealed_npz_protocol_contract
    a, b = contract("b787"), contract("loader")
    _validate_saved_sealed_npz_protocol_contract(a, a)
    with pytest.raises(ValueError, match="resume would change"):
        _validate_saved_sealed_npz_protocol_contract(a, b)
    legacy = {key: value for key, value in a.items() if key != "dataset_identity"}
    with pytest.raises(ValueError):
        _validate_saved_sealed_npz_protocol_contract(legacy, a)


@pytest.mark.parametrize("name", OBJECTS)
def test_generic_preflight_reads_no_responses(name, tmp_path):
    import train
    path = tmp_path / "roles.json"
    path.write_text(json.dumps(role_manifest(name)))
    arrays = {"response": None, "response_shape": (10000, 16, 16, 1, 600),
              "response_dtype": np.dtype("complex64"), "meta": metadata(name)}
    with patch.object(train, "load_npz_arrays", return_value=arrays) as loader:
        observed, identity = train._load_sealed_npz_protocol_contract(
            "not-opened.npz", path, num_train=3200, num_val=1000, num_test=1000)
        loader.assert_called_once_with("not-opened.npz", load_response=False)
        assert observed["response"] is None
        assert collection_contract(identity) == contract(name)


@pytest.mark.parametrize("name", OBJECTS)
@pytest.mark.parametrize("method", METHODS)
def test_every_generated_command_parses(name, method, tmp_path):
    commands = commands_for(name, method, dataset_root=tmp_path / "dataset", output_root=tmp_path / "runs")
    expected_npz, expected_manifest = object_paths(tmp_path / "dataset", name)
    assert str(expected_npz) in commands[0]
    assert str(expected_manifest) in commands[0]
    for command in commands:
        script = Path(command[1])
        module_name = f"scripts.{script.stem}" if script.parent.name == "scripts" else script.stem
        module = importlib.import_module(module_name)
        with patch.object(sys, "argv", [command[1], *command[2:]]):
            parsed = module.parse_args()
        if method in ("rift", "isotropic", "rift_grid"):
            assert parsed.npz_sealed_protocol
            assert parsed.num_test == 1000


def test_resume_matrix_guardrails(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        commands_for("b787", "rift", dataset_root=tmp_path, output_root=tmp_path, resume="auto")


@pytest.mark.parametrize("name", OBJECTS)
def test_real_collection_metadata_only(name):
    from rift.rift_dataset import DEFAULT_ROOT
    from rift.npz_dataset import _LazyResponseReader
    from rift.geraf_b7873200_source import load_b7873200_metadata_source
    from rift.geraf_b7873200_adapter import load_b7873200_sealed_power_arrays
    from rift.radarsplat_b7873200_protocol import load_b7873200_sealed_identity as radarsplat
    from rift.sugavanam_ertin_b7873200_stage1 import (
        load_b7873200_sealed_identity as se, build_b7873200_acquisition_identity,
        default_stage1_recipe, validate_stage1_recipe)
    from train_spinr_style import preflight_b787_development_inputs
    npz, manifest = object_paths(DEFAULT_ROOT, name)
    if not npz.exists() or not manifest.exists():
        pytest.skip("Local radar archives are not distributed with the source checkout")
    with patch.object(_LazyResponseReader, "iter_response_views", side_effect=AssertionError("payload read")):
        assert load_b7873200_metadata_source(npz, manifest).identity == contract(name)
        assert radarsplat(npz, manifest) == contract(name)
        arrays, identity = se(npz, manifest)
        acquisition = build_b7873200_acquisition_identity(arrays, identity)
        recipe = default_stage1_recipe(identity, acquisition)
        validate_stage1_recipe(recipe, identity, acquisition)
        assert arrays["response"] is None
        _, spinr_contract, _ = preflight_b787_development_inputs(
            npz_path=str(npz), manifest_path=str(manifest), resume_checkpoint=None)
        assert collection_contract(spinr_contract) == contract(name)
        # Mapping an uncompressed array does not read its response payload.
        power, power_contract = load_b7873200_sealed_power_arrays(npz, manifest)
        assert power_contract == contract(name)
        for role in ("reserved_test", "unused"):
            with pytest.raises(PermissionError):
                power.response_view(identity["role_ids"][role][0])


def test_radar_fields_resume_identity_keeps_object(tmp_path):
    from train_radar_fields import _normalized_sealed_protocol_contract
    from rift.radar_fields_dataset import load_radar_fields_sealed_split_manifest
    normalized = []
    for name in ("b787", "loader"):
        manifest = tmp_path / f"{name}.json"
        manifest.write_text(json.dumps(role_manifest(name)))
        split = load_radar_fields_sealed_split_manifest(
            str(manifest), 10000, response_shape=[10000, 16, 16, 1, 600], response_dtype="complex64")
        protocol = split.protocol_contract()
        protocol["dataset_identity"] = object_identity(name)
        normalized.append(_normalized_sealed_protocol_contract(protocol, label="test"))
    assert normalized[0] != normalized[1]


def test_sh_sas_measured_view_uses_role_guard():
    import torch
    from types import SimpleNamespace
    from unittest.mock import Mock
    from train_sh_sas import measured_view
    raw = (np.arange(2 * 3 * 1 * 4).reshape(2, 3, 1, 4) * (1 + 2j)).astype(np.complex64)
    arrays = SimpleNamespace(response=None, response_view=Mock(return_value=raw))
    actual = measured_view(arrays, 13, np.arange(2), np.arange(3), np.arange(4), torch.device("cpu"))
    assert np.array_equal(actual.numpy(), raw.mean(axis=2).transpose(2, 1, 0))
    arrays.response_view.assert_called_once_with(13)
    arrays.response_view.side_effect = PermissionError("sealed")
    with pytest.raises(PermissionError):
        measured_view(arrays, 987, np.arange(2), np.arange(3), np.arange(4), torch.device("cpu"))
