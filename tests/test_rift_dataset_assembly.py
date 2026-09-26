"""Mesh-frame assembly/refresh contracts; no production radar payload access."""
import copy
import json

import numpy as np
import pytest

from rift.rift_dataset import catalog, geometry_transform
from scripts import prepare_rift_dataset as assembly


def test_geometry_refresh_is_opt_in_and_cannot_change_scientific_metadata(tmp_path):
    path = tmp_path / "dataset_manifest.json"
    original = {"dataset_id": "rift_dataset_v1", "objects": [{
        "object_id": "b787", "archive": "objects/b787.npz", "role_manifest": "splits/b787.json",
        "metadata": {"scale_factor": .01}, "geometry": "meshes/B787.stl",
        "geometry_transform": {"centering": "already_centered", "scale_factor": 1.0}}]}
    path.write_text(json.dumps(original))
    before = path.read_bytes()
    current = copy.deepcopy(original)
    current["objects"][0]["geometry_transform"] = geometry_transform("b787", {"scale_factor": .01})
    with pytest.raises(ValueError, match="Refusing"):
        assembly.write_new_or_equal(path, current)
    assembly.write_new_or_equal(path, current, refresh_geometry=True, check_only=True)
    assert path.read_bytes() == before
    for key, replacement in (("object_id", "loader"), ("archive", "different.npz"),
                              ("role_manifest", "different.json"), ("metadata", {"scale_factor": 1.0})):
        changed = copy.deepcopy(current)
        changed["objects"][0][key] = replacement
        with pytest.raises(ValueError, match="Refusing"):
            assembly.write_new_or_equal(path, changed, refresh_geometry=True)
        assert path.read_bytes() == before
    assembly.write_new_or_equal(path, current, refresh_geometry=True)
    assert json.loads(path.read_text()) == current
    # A repeated refresh is idempotent and does not rewrite the file.
    mtime = path.stat().st_mtime_ns
    assembly.write_new_or_equal(path, current)
    assert path.stat().st_mtime_ns == mtime


def test_check_only_does_not_create_manifest_or_parent(tmp_path):
    path = tmp_path / "absent" / "manifest.json"
    assembly.write_new_or_equal(path, {"objects": []}, check_only=True)
    assert not path.parent.exists()


def test_assembly_records_same_source_frame_for_all_six_objects(tmp_path, monkeypatch):
    source_root = tmp_path / "sources"
    mesh_root = tmp_path / "original_meshes"
    mesh_root.mkdir()
    output = tmp_path / "collection"
    arrays_by_path = {}
    # Stub the giant response header only; real tiny ZIPs exercise stored-entry
    # and viewpoint access, manifests and relative-link creation/idempotence.
    for spec in catalog()["objects"]:
        name = spec["object_id"]
        source = source_root / spec["source"]
        source.parent.mkdir(parents=True, exist_ok=True)
        np.savez(source, response=np.empty(0, dtype=np.complex64), viewpoint_angles=np.zeros((1, 2)))
        (mesh_root / spec["geometry_filename"]).touch()
        meta = {"target_type": "b787" if name == "b787" else "mesh", "target_id": name,
                "experiment": "sphere10k", "viewpoint_sampling": "fibonacci_sphere",
                "target_position_m": [0., 0., 0.], "radar_fc_hz": 1e10,
                "radar_bandwidth_hz": 3e9, "num_adc_samples": 600, "num_chirps_cpi": 1,
                "scaled_max_extent_m": .1, "scale_factor": .01}
        # The original B787 NPZ has no source_stl_filename key.
        if name != "b787":
            meta["source_stl_filename"] = spec["geometry_filename"]
        arrays_by_path[source] = {"response_shape": (10000, 16, 16, 1, 600),
            "response_dtype": np.dtype("complex64"), "meta": meta,
            "viewpoint_positions": np.array([[10., 0., 0.]]),
            "tx_pos": np.zeros((1, 16, 3)), "rx_pos": np.zeros((1, 16, 3))}

    def metadata_only(path, *, load_response):
        assert load_response is False
        return arrays_by_path[path]

    monkeypatch.setattr(assembly, "load_npz_arrays", metadata_only)
    assembly.prepare(source_root, output, mesh_root=mesh_root, check_only=True)
    assert not output.exists()
    result = assembly.prepare(source_root, output, mesh_root=mesh_root)
    assert result["response_payloads_read"] is False
    manifest_path = output / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["objects"]:
        assert row["geometry_transform"] == geometry_transform(row["object_id"], row["metadata"])
        assert row["geometry_transform"]["input_frame"] == "source_model_units"
        assert (output / row["geometry"]).resolve() == mesh_root / (output / row["geometry"]).name
        assert (output / row["archive"]).is_symlink()
    before = manifest_path.read_bytes()
    assembly.prepare(source_root, output, mesh_root=mesh_root)
    assert manifest_path.read_bytes() == before
    # A bad split is rejected before even restoring a missing mesh link.
    missing_link = output / manifest["objects"][0]["geometry"]
    missing_link.unlink()
    split = output / manifest["objects"][0]["role_manifest"]
    split.write_text(json.dumps({"wrong": "split"}))
    with pytest.raises(ValueError, match="Refusing"):
        assembly.prepare(source_root, output, mesh_root=mesh_root, refresh_geometry=True)
    assert not missing_link.exists()
    assert manifest_path.read_bytes() == before
