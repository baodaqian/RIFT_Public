"""Adaptive geometry readout contracts; tiny CPU scenes, never radar responses."""
import copy
import csv
import json
import struct

import numpy as np
import pytest
import torch

from rift import rift_dataset as dataset
from rift.sparse_scene import AdaptivePointSHScene
from scripts import eval_b787_geometry_metrics as metrics
from scripts import render_b787_vs_stl as render
from scripts.eval_scene_geometry import deposit_points


def point_checkpoint():
    return {
        "scene_repr": "point_sh", "epoch": 7, "extent": None,
        "execution_contract": {"scene": {"extent_m": 1.0}},
        "sealed_npz_protocol_contract": dataset._object_contract("loader"),
        "model_state_dict": {
            "anchors": torch.tensor([[-.5, .5, -.5], [0., 0., 0.], [9., 9., 9.]], dtype=torch.float64),
            "cell_half": torch.full((3, 1), .1, dtype=torch.float64),
            "delta_raw": torch.zeros(3, 3, dtype=torch.float64),
            "w_re": torch.tensor([[1., 100., 100., 100.], [1., 2., 0., 0.], [1e6, 1e6, 1e6, 1e6]], dtype=torch.float64),
            "w_im": torch.tensor([[2., 100., 100., 100.], [0., 0., 0., 0.], [1e6, 1e6, 1e6, 1e6]], dtype=torch.float64),
            "active_mask": torch.tensor([True, True, False]),
            "order": torch.tensor([0, 1, 1]), "basis_degree": torch.tensor([0, 1, 1, 1]),
            "support_bounds_enabled": torch.tensor(False),
        },
    }


def load(tmp_path, checkpoint, **kwargs):
    path = tmp_path / "scene.pt"
    torch.save(checkpoint, path)
    return render.load_energy_field(path, **kwargs)


def test_conserves_only_active_unlocked_energy_and_voxel_centers(tmp_path):
    info = {}
    volume, grid, epoch = load(tmp_path, point_checkpoint(), point_grid=2, readout_info=info)
    expected = np.full((2, 2, 2), 5 / 8)
    expected[0, 1, 0] += 5
    np.testing.assert_allclose(volume, expected, rtol=0, atol=1e-14)
    assert (grid, epoch) == (2, 7)
    assert info["energy_conserved"] and info["source_energy"] == info["readout_energy"] == 10
    assert info["active_scatterers"] == 2 and info["spatial_readout"] == "point_sh_cic_voxel_centers_v1"
    dense = render.trilinear_sample_centers(1., grid, 8)
    np.testing.assert_allclose(dense[[0, -1]], [-.5, .5])


def test_learned_positions_match_actual_model_and_support_clamp(tmp_path):
    model = AdaptivePointSHScene(torch.tensor([[-.5, -.5, -.5], [.5, .5, .5]]),
                                 .5, "cpu", max_degree=1, capacity=3,
                                 enforce_support_bounds=True, compact_sh_eval=True)
    with torch.no_grad():
        model.anchors[0] = torch.tensor([-.99, -.2, .3])
        model.delta_raw[0] = torch.tensor([-3., .3, -.7])
        model.delta_raw[1] = torch.tensor([.1, .5, .2])
        model.w_re[0, 0] = 2
        model.w_im[1, 0] = 3
        model.w_re[:, 1:] = 1e3  # locked coefficients must not leak into support
    checkpoint = point_checkpoint()
    checkpoint["model_state_dict"] = model.state_dict()
    expected = deposit_points(model.positions().detach()[model.active_mask].double(),
                              torch.tensor([4., 9.], dtype=torch.float64), 1., 8)
    before = copy.deepcopy(checkpoint["model_state_dict"])
    actual, _, _ = load(tmp_path, checkpoint, point_grid=8)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-12, atol=1e-12)
    for key, value in checkpoint["model_state_dict"].items():
        assert torch.equal(value, before[key])


@pytest.mark.parametrize("position", [[1., 1., 1.], [-1., -1., -1.], [1., 0., -.5]])
def test_scene_face_deposition_conserves_energy(tmp_path, position):
    checkpoint = point_checkpoint()
    sd = checkpoint["model_state_dict"]
    sd["active_mask"][1] = False
    sd["anchors"][0] = torch.tensor(position)
    volume, _, _ = load(tmp_path, checkpoint, point_grid=4)
    assert volume.min() >= 0 and volume.sum() == pytest.approx(5)


def test_inactive_nan_slots_are_ignored_and_empty_scene_is_zero(tmp_path):
    checkpoint = point_checkpoint()
    sd = checkpoint["model_state_dict"]
    for key in ("anchors", "cell_half", "delta_raw", "w_re", "w_im"):
        sd[key][2] = float("nan")
    volume, _, _ = load(tmp_path, checkpoint, point_grid=2)
    assert np.isfinite(volume).all() and volume.sum() == 10
    sd["active_mask"][:] = False
    volume, _, _ = load(tmp_path, checkpoint, point_grid=2)
    assert not volume.any()


@pytest.mark.parametrize("field,value,fragment", [
    ("anchors", torch.zeros(2, 3), "shapes"),
    ("cell_half", torch.ones(3, 1) * -.1, "nonnegative"),
    ("delta_raw", torch.full((3, 3), float("nan")), "finite"),
    ("w_re", torch.full((3, 4), float("inf")), "finite"),
    ("active_mask", torch.tensor([1, 1, 0]), "boolean"),
    ("order", torch.tensor([2, 1, 1]), "outside"),
    ("order", torch.tensor([0., 1., 1.]), "integer"),
    ("basis_degree", torch.tensor([1, 0, 1, 1]), "ordering"),
    ("support_bounds_enabled", torch.tensor(1), "boolean"),
])
def test_malformed_point_state_fails_closed(tmp_path, field, value, fragment):
    checkpoint = point_checkpoint()
    checkpoint["model_state_dict"][field] = value
    with pytest.raises(ValueError, match=fragment):
        load(tmp_path, checkpoint, point_grid=2)


def test_extent_identity_and_outside_positions_are_not_silently_overridden(tmp_path):
    checkpoint = point_checkpoint()
    with pytest.raises(ValueError, match="extent"):
        load(tmp_path, checkpoint, extent=.15)
    with pytest.raises(ValueError, match="identity"):
        load(tmp_path, checkpoint, expected_contract=dataset._object_contract("b787"))
    checkpoint["model_state_dict"]["anchors"][0, 0] = 1.1
    with pytest.raises(ValueError, match="outside"):
        load(tmp_path, checkpoint)
    checkpoint["execution_contract"] = {}
    with pytest.raises(ValueError, match="extent"):
        load(tmp_path, checkpoint, extent=1.)


def test_support_bounds_checked_against_declared_extent(tmp_path):
    checkpoint = point_checkpoint()
    sd = checkpoint["model_state_dict"]
    sd.update(support_bounds_enabled=torch.tensor(True), support_min=torch.tensor([-2., -1., -1.]),
              support_max=torch.ones(3))
    with pytest.raises(ValueError, match="bounds"):
        load(tmp_path, checkpoint)


@pytest.mark.parametrize("grid", [0, 1, 2.5, True])
def test_invalid_point_grid_rejected(tmp_path, grid):
    with pytest.raises(ValueError, match="point-grid"):
        load(tmp_path, point_checkpoint(), point_grid=grid)


def test_grid_fields_keep_native_shape_and_values(tmp_path):
    w = torch.arange(8.).reshape(2, 2, 2, 1)
    checkpoint = {"scene_repr": "grid_sh", "epoch": 1,
                  "model_state_dict": {"w_re": w, "w_im": torch.ones_like(w)}}
    info = {}
    field, grid, _ = load(tmp_path, checkpoint, point_grid=7, readout_info=info)
    np.testing.assert_array_equal(field, (w.square() + 1).squeeze(-1).numpy())
    assert grid == 2 and info["spatial_readout"] == "native_voxel_centers"


def test_primary_oracle_csv_and_renderer_accept_adaptive_checkpoint(tmp_path, monkeypatch):
    checkpoint = point_checkpoint()
    path = tmp_path / "adaptive.pt"
    torch.save(checkpoint, path)
    triangle = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]])
    contract = dataset._object_contract("loader")
    inputs = lambda args: ({"scaled_dimensions_m": [.1, .1, .1], "scale_factor": .1}, triangle, contract)
    monkeypatch.setattr(metrics, "collection_geometry_inputs", inputs)
    monkeypatch.setattr(render, "collection_geometry_inputs", inputs)
    monkeypatch.setattr(metrics, "sample_surface_points", lambda *a: triangle)
    monkeypatch.setattr(metrics, "sample_volume_points", lambda *a: triangle)
    monkeypatch.setattr(metrics, "inside_mask", lambda *a: np.ones((2, 2, 2), dtype=bool))
    output = tmp_path / "geometry.csv"
    metrics.main(["--object", "loader", "--checkpoints", str(path), "--extent", "1",
                  "--point-grid", "4", "--upsample", "2", "--fixed-threshold", ".2",
                  "--thresholds", ".5", "--gt-grid", "2", "--csv", str(output)])
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["threshold_role"] for row in rows] == ["fixed_primary", "oracle_diagnostic"]
    for row in rows:
        assert row["scene_repr"] == "point_sh" and row["object_id"] == "loader"
        assert row["spatial_readout"] == "point_sh_cic_voxel_centers_v1"
        assert (row["readout_grid"], row["sampled_grid"]) == ("4", "8")
        assert np.isfinite(float(row["cd_surface"])) and row["checkpoint"] == str(path)
    render.main(["--object", "loader", "--checkpoint", str(path), "--extent", "1",
                 "--point-grid", "4", "--upsample", "2", "--out-prefix", str(tmp_path / "figure")])
    for suffix in ("overlay", "stl_only", "support"):
        assert (tmp_path / f"figure_{suffix}.png").stat().st_size > 0
    render.plt.close("all")


def test_invalid_object_rejected_before_truth_sampling(tmp_path, monkeypatch):
    path = tmp_path / "adaptive.pt"
    checkpoint = point_checkpoint()
    checkpoint["sealed_npz_protocol_contract"] = dataset._object_contract("b787")
    torch.save(checkpoint, path)
    monkeypatch.setattr(metrics, "collection_geometry_inputs", lambda args: ({}, np.zeros((3, 3)), dataset._object_contract("loader")))
    monkeypatch.setattr(metrics, "sample_surface_points", lambda *a: pytest.fail("sampled truth before identity check"))
    with pytest.raises(ValueError, match="identity"):
        metrics.main(["--object", "loader", "--checkpoints", str(path), "--extent", "1", "--fixed-threshold", ".2"])


@pytest.mark.parametrize("binary", [False, True])
def test_ascii_and_solid_prefixed_binary_stl(tmp_path, binary):
    path = tmp_path / "source.stl"
    vertices = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
    if binary:
        record = struct.pack("<12fH", 0., 0., 1., *np.asarray(vertices).reshape(-1), 0)
        path.write_bytes(b"solid binary header".ljust(80, b" ") + struct.pack("<I", 1) + record)
    else:
        path.write_text("solid source\nfacet normal 0 0 1\nouter loop\n"
                        "vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid source\n")
    np.testing.assert_array_equal(render.load_stl_vertices(path), vertices)


def test_truncated_stl_rejected(tmp_path):
    path = tmp_path / "broken.stl"
    path.write_bytes(b"binary".ljust(80, b" ") + struct.pack("<I", 2) + b"short")
    with pytest.raises(ValueError, match="STL"):
        render.load_stl_vertices(path)


def test_restored_mesh_resolution_is_exact_metadata_basename(tmp_path, monkeypatch):
    (tmp_path / "meshes").mkdir()
    (tmp_path / "dataset_manifest.json").write_text(json.dumps({"objects": [{"object_id": "loader"}]}))
    metadata = {"source_stl_filename": "original_loader.stl", "scale_factor": .01}
    monkeypatch.setattr(dataset, "load_object", lambda *a, **k: ({"meta": metadata}, dataset._object_contract("loader")))
    (tmp_path / "meshes" / "B787.stl").touch()
    with pytest.raises(FileNotFoundError, match="No local registered mesh"):
        dataset.geometry_reference("loader", tmp_path)
    target = tmp_path / "meshes" / "original_loader.stl"
    target.touch()
    assert dataset.geometry_reference("loader", tmp_path)["mesh_path"] == target
    for bad in ("../original_loader.stl", "..\\original_loader.stl", ""):
        metadata["source_stl_filename"] = bad
        with pytest.raises(FileNotFoundError, match="No local registered mesh"):
            dataset.geometry_reference("loader", tmp_path)


@pytest.mark.parametrize("name", [row["object_id"] for row in dataset.catalog()["objects"]])
def test_every_object_uses_original_mesh_transform(name, tmp_path, monkeypatch):
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    spec = dataset.object_spec(name)
    original = meshes / spec["geometry_filename"]
    original.touch()
    (tmp_path / "dataset_manifest.json").write_text(json.dumps({"objects": [
        {"object_id": name, "geometry": f"meshes/{original.name}"}]}))
    meta = {"scale_factor": .01}
    monkeypatch.setattr(dataset, "load_object", lambda *a, **k: ({"meta": meta}, dataset._object_contract(name)))
    reference = dataset.geometry_reference(name, tmp_path)
    assert reference["mesh_path"] == original
    assert reference["transform"] == {"centering": "source_aabb_center", "scale_factor": .01,
                                       "units": "metres", "input_frame": "source_model_units"}
    assert reference["metadata"]["scale_factor"] == .01  # original metadata is untouched
    # Neither an explicit path nor the former special filename can change units.
    scaled = meshes / "_b787_scaled_0p10m.stl"
    scaled.touch()
    assert dataset.geometry_reference(name, tmp_path, mesh_path=scaled)["transform"] == reference["transform"]


def test_prescaled_geometry_is_rejected_in_original_mesh_protocol():
    meta = {"raw_dimensions_model_units": [2., 4., 10.], "scale_factor": .01,
            "scaled_dimensions_m": [.02, .04, .1], "target_position_m": [0., 0., 0.]}
    vertices = np.array([[-.01, -.02, -.05], [.01, .02, .05]])
    with pytest.raises(ValueError, match="dimensions"):
        dataset.transform_mesh_vertices(vertices, meta)  # no implicit unit inference
    with pytest.raises(ValueError, match="original model units"):
        dataset.transform_mesh_vertices(vertices, meta, input_frame="scene_metres")


def test_collection_b787_uses_same_centering_and_scaling_as_other_objects(monkeypatch):
    contract = dataset._object_contract("b787")
    meta = {"raw_dimensions_model_units": [2., 4., 10.], "scale_factor": .01,
            "scaled_dimensions_m": [.02, .04, .1], "target_position_m": [0., 0., 0.]}
    vertices = np.array([[3., -2., 5.], [5., 2., 15.]])
    before = vertices.copy()
    monkeypatch.setattr(dataset, "load_object_contract", lambda *a, **k: ({}, contract))
    monkeypatch.setattr(dataset, "geometry_reference", lambda *a, **k: {
        "mesh_path": "unused.stl", "metadata": meta, "dataset_identity": dataset.object_identity("b787"),
        "transform": dataset.geometry_transform("b787", meta)})
    monkeypatch.setattr(render, "load_stl_vertices", lambda *a: vertices)
    args = render.parse_args(["--object", "b787", "--checkpoint", "unused.pt"])
    _, actual, _ = render.collection_geometry_inputs(args)
    np.testing.assert_allclose(actual, [[-.01, -.02, -.05], [.01, .02, .05]])
    np.testing.assert_array_equal(vertices, before)
