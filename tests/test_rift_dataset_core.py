"""Object-bound public data ingress; no real radar payloads or training."""
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from rift import npz_dataset
from rift import rift_dataset as dataset


OBJECTS = [row["object_id"] for row in dataset.catalog()["objects"]]


def metadata(name="loader"):
    return {"target_type": "b787" if name == "b787" else "mesh", "target_id": name,
            "experiment": "sphere10k", "viewpoint_sampling": "fibonacci_sphere",
            "target_position_m": [0.0, 0.0, 0.0], "radar_fc_hz": 1e10,
            "radar_bandwidth_hz": 3e9, "num_adc_samples": 600, "num_chirps_cpi": 1,
            "scaled_max_extent_m": 0.1, "raw_dimensions_model_units": [2.0, 4.0, 10.0],
            "scaled_dimensions_m": [0.02, 0.04, 0.1], "scale_factor": 0.01}


@pytest.fixture
def source(tmp_path, monkeypatch):
    """Mock header/pose loader, with an archive containing no response payload."""
    # Import before patching: train binds load_npz_arrays at module import.
    # A late import inside this fixture would retain its mock in later tests.
    import train

    npz, manifest = dataset.object_paths(tmp_path, "loader")
    npz.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dataset.role_manifest("loader")))
    with zipfile.ZipFile(npz, "w") as archive:
        archive.writestr("response.npy", b"not a payload")
    viewpoints = np.zeros((10000, 3), dtype=np.float64)
    viewpoints[:, 0] = 10.0
    arrays = {"response": None, "response_shape": (10000, 16, 16, 1, 600),
              "response_dtype": np.dtype("complex64"), "meta": metadata(),
              "viewpoint_positions": viewpoints,
              "tx_pos": np.repeat(viewpoints[:, None, :], 16, axis=1),
              "rx_pos": np.repeat(viewpoints[:, None, :], 16, axis=1),
              "_lazy_response_reader": npz_dataset._LazyResponseReader(
                  str(npz), (10000, 16, 16, 1, 600), np.dtype("complex64"))}

    def load(path, *, load_response=True):
        assert Path(path) == npz
        assert load_response is False
        return arrays

    def no_stream(*args, **kwargs):
        raise AssertionError("No response payload may be accessed in this suite")

    monkeypatch.setattr(npz_dataset, "load_npz_arrays", load)
    monkeypatch.setattr(npz_dataset, "_validated_response_stream", no_stream)
    return npz, manifest, arrays


@pytest.mark.parametrize("name", OBJECTS + ["a320", "x59", "racecar"])
def test_named_input_resolution(name, tmp_path):
    expected = dataset.object_paths(tmp_path, name)
    assert dataset.resolve_object_inputs(object_name=name, dataset_root=tmp_path) == expected
    assert dataset.resolve_object_inputs(object_name=name, dataset_root=tmp_path,
                                         npz_path=expected[0], role_manifest_path=expected[1]) == expected
    with pytest.raises(ValueError, match="Conflicting"):
        dataset.resolve_object_inputs(object_name=name, dataset_root=tmp_path, npz_path="wrong.npz")


def test_explicit_input_resolution_and_symlinks(source, tmp_path):
    npz, manifest, _ = source
    alias = tmp_path / "alias.npz"
    alias.symlink_to(npz)
    assert dataset.resolve_object_inputs(object_name="loader", dataset_root=tmp_path,
                                         npz_path=alias) == (npz, manifest)
    assert dataset.resolve_object_inputs(npz_path=npz, role_manifest_path=manifest) == (npz, manifest)
    with pytest.raises(ValueError, match="both"):
        dataset.resolve_object_inputs(npz_path=npz)


def test_public_loader_is_lazy_and_narrows_roles(source):
    npz, manifest, original = source
    arrays, contract = dataset.load_object_contract(npz, manifest)
    assert arrays["response"] is None
    assert dataset.collection_contract(contract) == dataset._object_contract("loader")
    assert original["_lazy_response_reader"].allowed_view_indices is None
    assert arrays["_lazy_response_reader"].allowed_view_indices == frozenset(
        contract["role_ids"]["train"] + contract["role_ids"]["validation"])
    for role in ("reserved_test", "unused"):
        with pytest.raises(PermissionError):
            npz_dataset.get_npz_response_view(arrays, contract["role_ids"][role][0])
    only_val, _ = dataset.load_object_contract(npz, manifest, response_roles=("val",))
    with pytest.raises(PermissionError):
        npz_dataset.get_npz_response_view(only_val, contract["role_ids"]["train"][0])
    assert len(only_val["_lazy_response_reader"].allowed_view_indices) == 1000


def test_test_role_needs_explicit_opt_in_and_unused_is_never_allowed(source):
    npz, manifest, _ = source
    for opt_in in (False, 1, "true"):
        with pytest.raises(PermissionError):
            dataset.load_object_contract(npz, manifest, response_roles=("test",), allow_reserved_test=opt_in)
    # Construct only the capability; never read a reserved-test response.
    arrays, contract = dataset.load_object_contract(
        npz, manifest, response_roles=("test",), allow_reserved_test=True)
    assert arrays["_lazy_response_reader"].allowed_view_indices == frozenset(contract["role_ids"]["reserved_test"])
    with pytest.raises(ValueError, match="unused"):
        dataset.load_object_contract(npz, manifest, response_roles=("unused",), allow_reserved_test=True)
    for roles in ((), "validation"):
        with pytest.raises(ValueError, match="sequence"):
            dataset.load_object_contract(npz, manifest, response_roles=roles)


@pytest.mark.parametrize("field,value", [
    ("response_shape", (10000, 16, 16, 10, 600)), ("response_dtype", np.dtype("complex128")),
    ("response", np.zeros(1)), ("viewpoint_positions", np.zeros((10000, 3), dtype=np.float32)),
    ("tx_pos", np.zeros((10000, 16, 3), dtype=np.float32)),
    ("rx_pos", np.full((10000, 16, 3), np.nan)),
])
def test_incompatible_header_or_poses_rejected_before_payload(source, field, value):
    npz, manifest, arrays = source
    arrays[field] = value
    with pytest.raises(ValueError, match="RIFT dataset"):
        dataset.load_object_contract(npz, manifest)


def test_wrong_radius_and_compressed_archives_rejected(source):
    npz, manifest, arrays = source
    arrays["viewpoint_positions"][0, 0] = 9.0
    with pytest.raises(ValueError, match="10 m sphere"):
        dataset.load_object_contract(npz, manifest)
    arrays["viewpoint_positions"][0, 0] = 10.0
    with zipfile.ZipFile(npz, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("response.npy", b"not a payload")
    with pytest.raises(ValueError, match="uncompressed"):
        dataset.load_object_contract(npz, manifest)


def test_mismatched_or_swapped_object_rejected(source, tmp_path):
    npz, manifest, arrays = source
    arrays["meta"] = metadata("b787")
    with pytest.raises(ValueError, match="object does not match"):
        dataset.load_object_contract(npz, manifest)
    manifest.write_text(json.dumps(dataset.role_manifest("b787")))
    with pytest.raises(ValueError, match="selected object"):
        dataset.load_object("loader", tmp_path)


@pytest.mark.parametrize("bad_value", [True, 1.0, "1"])
def test_manifest_schema_requires_integer_one(source, bad_value):
    npz, manifest, _ = source
    changed = dataset.role_manifest("loader")
    changed["schema_version"] = bad_value
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="schema_version"):
        dataset.load_object_contract(npz, manifest)


@pytest.mark.parametrize("field,value", [("version", True), ("version", 1.0),
                                          ("response_shape", [10000.0, 16, 16, 1, 600])])
def test_contract_rejects_equal_but_wrong_typed_fields(field, value):
    changed = dataset._object_contract("loader")
    changed[field] = value
    with pytest.raises(ValueError, match="registered"):
        dataset.collection_contract(changed)


def test_role_ids_and_flags_reject_coercion(source):
    npz, manifest, _ = source
    changed = dataset.role_manifest("loader")
    changed["split"]["train_indices"][0] = float(changed["split"]["train_indices"][0])
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="PCG64"):
        dataset.load_object_contract(npz, manifest)
    contract = dataset._object_contract("loader")
    contract["response_access"]["reserved_test_materialized"] = 0
    with pytest.raises(ValueError, match="registered"):
        dataset.collection_contract(contract)


@pytest.mark.parametrize("containers", [(), ("sealed_npz_protocol_contract",), ("sealed_protocol_contract",),
    ("run_identity", "sealed_protocol_identity"), ("recipe", "sealed_protocol_identity"),
    ("cache_recipe", "sealed_protocol_identity"), ("sugavanam_ertin_b7873200_stage1", "sealed_protocol_identity"),
    ("generic_final_state", "contract"), ("contract", "provenance", "stage1_record", "sealed_protocol_identity")])
def test_checkpoint_containers_are_object_bound(containers):
    contract = dataset._object_contract("loader")
    checkpoint = {"dataset_identity": dataset.object_identity("loader")}
    for key in reversed(containers):
        checkpoint = {key: checkpoint}
    assert dataset.validate_checkpoint_object(checkpoint, contract) == dataset.object_identity("loader")
    with pytest.raises(ValueError, match="does not match"):
        dataset.validate_checkpoint_object(checkpoint, dataset._object_contract("b787"))
    checkpoint["dataset_identity"] = dataset.object_identity("b787")
    with pytest.raises(ValueError, match="does not match"):
        dataset.validate_checkpoint_object(checkpoint, contract)


def test_checkpoint_does_not_infer_object_from_args_or_weights():
    contract = dataset._object_contract("loader")
    for record in ({}, {"args": {"dataset_identity": contract["dataset_identity"]}},
                   {"scene_state": {"dataset_identity": contract["dataset_identity"]}}):
        with pytest.raises(ValueError, match="lacks"):
            dataset.validate_checkpoint_object(record, contract)


def test_mesh_transform_preserves_orientation_and_metric_scale():
    vertices = np.array([[3.0, -2.0, 5.0], [5.0, 2.0, 15.0]])
    expected = np.array([[-0.01, -0.02, -0.05], [0.01, 0.02, 0.05]])
    assert np.allclose(dataset.transform_mesh_vertices(vertices, metadata()), expected)
    shaped = vertices.reshape(1, 2, 3)
    assert dataset.transform_mesh_vertices(shaped, metadata()).shape == shaped.shape
    for changed in ({**metadata(), "scale_factor": -1},
                    {**metadata(), "scaled_dimensions_m": [2, 4, 10]},
                    {**metadata(), "raw_dimensions_model_units": [4, 2, 10]}):
        with pytest.raises(ValueError):
            dataset.transform_mesh_vertices(vertices, changed)


def test_geometry_never_falls_back_to_b787(source, tmp_path):
    (tmp_path / "dataset_manifest.json").write_text(json.dumps({"objects": [
        {"object_id": "loader", "geometry": None}, {"object_id": "b787", "geometry": "B787.stl"}]}))
    mesh = tmp_path / "loader.stl"
    mesh.touch()
    with pytest.raises(FileNotFoundError, match="No local registered mesh"):
        dataset.geometry_reference("loader", tmp_path)
    info = dataset.geometry_reference("loader", tmp_path, mesh_path=mesh)
    assert info["mesh_path"] == mesh
    assert info["dataset_identity"] == dataset.object_identity("loader")
    assert info["transform"] == {"centering": "source_aabb_center", "scale_factor": 0.01,
                                  "units": "metres", "input_frame": "source_model_units"}
    with pytest.raises(FileNotFoundError, match="unavailable"):
        dataset.geometry_reference("loader", tmp_path, mesh_path=tmp_path / "missing.stl")


def test_radar_fields_stats_bind_object_on_creation_and_reuse(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rift import radar_fields_dataset as rf

    path = str(tmp_path / "stats.json")
    arrays = SimpleNamespace(metadata=metadata(), num_views=10000)
    identity = dataset.object_identity("loader")
    train_ids = dataset.role_ids()["train"]
    calls = []

    def peak(*args, **kwargs):
        calls.append(True)
        return 1.25

    monkeypatch.setattr(rf, "estimate_power_peak", peak)
    created = rf.load_or_create_stats(path, arrays, train_ids, 60, sealed_protocol=True, dataset_identity=identity)
    assert created["dataset_identity"] == identity
    assert json.loads(Path(path).read_text())["dataset_identity"] == identity
    assert created["normalization_stats_cache_reused"] is False
    reused = rf.load_or_create_stats(path, arrays, train_ids, 60, sealed_protocol=True, dataset_identity=identity)
    assert reused["dataset_identity"] == identity
    assert reused["normalization_stats_cache_reused"] is True
    assert calls == [True]
    safe_cache = json.loads(Path(path).read_text())
    for identity_value in (None, dataset.object_identity("b787")):
        unsafe = dict(safe_cache)
        if identity_value is None:
            del unsafe["dataset_identity"]
        else:
            unsafe["dataset_identity"] = identity_value
        Path(path).write_text(json.dumps(unsafe))
        before = Path(path).read_bytes()
        with pytest.raises(ValueError, match="dataset_identity"):
            rf.load_or_create_stats(path, arrays, train_ids, 60, sealed_protocol=True, dataset_identity=identity)
        assert Path(path).read_bytes() == before
        assert calls == [True]  # No response rescan or cache overwrite.
    with pytest.raises(ValueError, match="dataset_identity"):
        rf.load_or_create_stats(str(tmp_path / "new.json"), arrays, train_ids, 60,
                                sealed_protocol=True, dataset_identity=dataset.object_identity("b787"))
    assert not (tmp_path / "new.json").exists()
    assert calls == [True]


def test_radar_fields_resume_rejects_missing_or_wrong_stats_identity(tmp_path):
    from rift.radar_fields_dataset import load_radar_fields_sealed_split_manifest
    from train_radar_fields import preflight_sealed_resume_checkpoint

    manifest = tmp_path / "loader.json"
    manifest.write_text(json.dumps(dataset.role_manifest("loader")))
    protocol = load_radar_fields_sealed_split_manifest(
        str(manifest), 10000, response_shape=[10000, 16, 16, 1, 600], response_dtype="complex64").protocol_contract()
    protocol["dataset_identity"] = dataset.object_identity("loader")
    for stats in (None, {}, {"dataset_identity": dataset.object_identity("b787")}):
        # Object stats gate must run before later split/dataset provenance validation.
        with pytest.raises(ValueError, match="dataset_identity"):
            preflight_sealed_resume_checkpoint(
                {"sealed_protocol_contract": protocol, "power_stats": stats},
                sealed_protocol_requested=True, current_sealed_protocol=protocol)


@pytest.mark.parametrize("name", OBJECTS)
def test_real_metadata_only_public_ingress_and_baselines(name, monkeypatch):
    from rift.geraf_b7873200_protocol import load_b7873200_sealed_protocol_identity as geraf
    from rift.radarsplat_b7873200_protocol import load_b7873200_sealed_identity as radarsplat
    from rift.sugavanam_ertin_b7873200_stage1 import load_b7873200_sealed_identity as se
    import train

    npz, manifest = dataset.object_paths(dataset.DEFAULT_ROOT, name)
    if not npz.exists() or not manifest.exists():
        pytest.skip("Local radar archives are not distributed with the source checkout")

    def forbidden(*args, **kwargs):
        raise AssertionError("Public collection ingress must not read responses or call the root trainer")

    monkeypatch.setattr(train, "_load_sealed_npz_protocol_contract", forbidden)
    monkeypatch.setattr(npz_dataset._LazyResponseReader, "iter_response_views", forbidden)
    arrays, contract = dataset.load_object(name)
    assert arrays["response"] is None
    expected = dataset.collection_contract(contract)
    assert geraf(npz, manifest) == expected
    assert radarsplat(npz, manifest) == expected
    se_arrays, se_contract = se(npz, manifest)
    assert se_contract == expected
    assert se_arrays["response"] is None
