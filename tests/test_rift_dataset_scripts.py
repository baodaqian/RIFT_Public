"""Collection script boundaries, using metadata stubs and small synthetic fields."""
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch
import train  # Import before fixtures patch module-level NPZ helpers.

from rift import npz_dataset
from rift import rift_dataset as dataset
from scripts import prepare_geraf_b7873200_targets as prepare_geraf
from scripts import prepare_radarsplat_b7873200_targets as prepare_splat
from scripts import eval_rift_dataset_model_free as model_free
from scripts import eval_b787_range_power as power
from scripts import eval_geraf_complex_response as geraf
from scripts import readout_radar_fields_b7873200_native as rf
from scripts import readout_radarsplat_b7873200_fullscale as splat
from scripts import render_b787_vs_stl as render
from scripts import eval_b787_geometry_metrics as geometry


CLI_CASES = [
    (prepare_geraf.parse_args, ["--cache-root", "cache"], "role_manifest"),
    (prepare_splat.parse_args, ["--cache-root", "cache"], "role_manifest"),
    (model_free.parse_args, ["--method", "fsh", "--output-dir", "out"], "role_manifest"),
    (power.parse_args, ["--checkpoint", "checkpoint", "--label", "sample", "--stats", "stats"], "role_manifest"),
    (geraf._parse_args, ["--checkpoint", "checkpoint", "--cache-root", "cache", "--power-stats", "stats"], "role_manifest"),
    (rf.parse_args, ["--checkpoint", "checkpoint", "--output", "out", "--config", "recipe"], "sealed_split_manifest"),
    (render.parse_args, ["--checkpoint", "checkpoint"], None),
    (geometry.parse_args, ["--checkpoints", "checkpoint", "--fixed-threshold", "0.2"], None),
]
ALIASES = ["a320", "x59", "firetruck", "racecar", "loader", "b787"]


@pytest.mark.parametrize("name", ALIASES)
@pytest.mark.parametrize("parse,base,manifest_attr", CLI_CASES)
def test_named_cli_resolves_each_scene(name, parse, base, manifest_attr, tmp_path):
    args = parse([*base, "--object", name, "--dataset-root", str(tmp_path)])
    expected_npz, expected_manifest = dataset.object_paths(tmp_path, name)
    if manifest_attr is not None:
        assert Path(args.npz_path) == expected_npz
        assert Path(getattr(args, manifest_attr)) == expected_manifest
    else:
        assert args.object == dataset.object_spec(name)["object_id"]
        assert args.stl is None  # Never inherit the historical B787 mesh.


@pytest.mark.parametrize("parse,base,manifest_attr", CLI_CASES)
def test_conflicting_named_and_explicit_paths_fail(parse, base, manifest_attr, tmp_path):
    with pytest.raises(ValueError, match="Conflicting"):
        parse([*base, "--object", "loader", "--dataset-root", str(tmp_path),
               "--npz-path", str(tmp_path / "b787.npz")])


@pytest.mark.parametrize("parse,base,flag", [
    (power.parse_args, CLI_CASES[3][1], "--role"),
    (geraf._parse_args, CLI_CASES[4][1], "--role"),
    (rf.parse_args, CLI_CASES[5][1], "--roles"),
])
def test_collection_test_is_never_implicit(parse, base, flag, tmp_path):
    argv = [*base, "--object", "loader", "--dataset-root", str(tmp_path)]
    args = parse(argv)
    assert getattr(args, "role", None) == "validation" or getattr(args, "roles", None) == ["val"]
    for role in ("test", "reserved_test"):
        with pytest.raises(SystemExit):
            parse([*argv, flag, role])
        allowed = parse([*argv, flag, role, "--allow-reserved-test"])
        assert allowed.allow_reserved_test is True  # Parse only; never evaluate test.


def test_collection_requires_explicit_normalization_and_fixed_geometry():
    with pytest.raises(SystemExit):
        power.parse_args(["--checkpoint", "checkpoint", "--label", "x", "--object", "loader"])
    with pytest.raises(SystemExit):
        geraf._parse_args(["--object", "loader"])
    with pytest.raises(SystemExit):
        rf.parse_args(["--checkpoint", "checkpoint", "--output", "out", "--object", "loader"])
    with pytest.raises(SystemExit):
        geometry.parse_args(["--checkpoints", "checkpoint", "--object", "loader"])
    fixed = geometry.parse_args(["--checkpoints", "checkpoint", "--object", "loader", "--fixed-threshold", ".2"])
    assert fixed.thresholds == [.2]


def test_historical_cli_defaults_are_retained(tmp_path):
    assert prepare_geraf.parse_args(["--cache-root", "cache"]).npz_path == prepare_geraf.B787_3200_CANONICAL_NPZ_PATH
    assert prepare_splat.parse_args(["--cache-root", "cache"]).role_manifest == prepare_splat.B787_3200_CANONICAL_MANIFEST_PATH
    assert power.parse_args(["--checkpoint", "c", "--label", "x"]).stats == power.DEFAULT_STATS
    assert geraf._parse_args([]).checkpoint == geraf.DEFAULT_CHECKPOINT
    old_rf = rf.parse_args(["--checkpoint", "c", "--output", "out", "--npz-path", "old.npz",
                            "--sealed-split-manifest", str(tmp_path / "old.json")])
    assert old_rf.roles == ["val", "test"]
    assert old_rf.config == rf.DEFAULT_CONFIG
    assert render.parse_args(["--checkpoint", "c", "--npz-path", "old.npz"]).stl == "data/B787.stl"
    assert geometry.parse_args(["--checkpoints", "c"]).thresholds == list(geometry.DEFAULT_THRESHOLDS)
    explicit = model_free.parse_args(["--npz-path", "scene.npz", "--role-manifest", "split.json",
                                     "--method", "fsh", "--output-dir", "out"])
    assert explicit.npz_path.name == "scene.npz"


def forbidden(*args, **kwargs):
    raise AssertionError("A response or target payload was accessed before identity validation")


@pytest.fixture
def source(tmp_path, monkeypatch):
    npz_path, manifest = dataset.object_paths(tmp_path, "loader")
    npz_path.parent.mkdir()
    manifest.parent.mkdir()
    manifest.write_text(json.dumps(dataset.role_manifest("loader")))
    with zipfile.ZipFile(npz_path, "w") as archive:
        archive.writestr("response.npy", b"metadata fixture, no radar payload")
    positions = np.zeros((10000, 3), dtype=np.float64)
    positions[:, 0] = 10.0
    meta = dict(target_type="mesh", target_id="loader", experiment="sphere10k",
                viewpoint_sampling="fibonacci_sphere", target_position_m=[0, 0, 0],
                radar_fc_hz=1e10, radar_bandwidth_hz=3e9, num_adc_samples=600,
                num_chirps_cpi=1, scaled_max_extent_m=.1, scale_factor=.01,
                raw_dimensions_model_units=[2., 4., 10.], scaled_dimensions_m=[.02, .04, .1])
    arrays = dict(response=None, response_shape=(10000, 16, 16, 1, 600),
                  response_dtype=np.dtype("complex64"), meta=meta,
                  viewpoint_positions=positions, tx_pos=np.repeat(positions[:, None], 16, axis=1),
                  rx_pos=np.repeat(positions[:, None], 16, axis=1),
                  _lazy_response_reader=npz_dataset._LazyResponseReader(str(npz_path), (10000,16,16,1,600), np.dtype("complex64")))
    def metadata_only(path, *, load_response=True):
        assert Path(path) == npz_path
        assert load_response is False
        return arrays
    monkeypatch.setattr(npz_dataset, "load_npz_arrays", metadata_only)
    monkeypatch.setattr(npz_dataset, "_validated_response_stream", forbidden)
    (tmp_path / "dataset_manifest.json").write_text(json.dumps({"objects": [
        {"object_id": "loader", "geometry": None}, {"object_id": "b787", "geometry": "B787.stl"}]}))
    return tmp_path, arrays, dataset._object_contract("loader")


@pytest.mark.parametrize("bad", ["checkpoint", "stats", "missing_stats"])
def test_power_rejects_foreign_inputs_before_response_access(source, monkeypatch, bad):
    root, _, contract = source
    checkpoint = {"sealed_npz_protocol_contract": contract, "extent": .15}
    stats = {"dataset_identity": contract["dataset_identity"]}
    if bad == "checkpoint":
        checkpoint["sealed_npz_protocol_contract"] = dataset._object_contract("b787")
    elif bad == "stats":
        stats["dataset_identity"] = dataset.object_identity("b787")
    else:
        stats = {}
    stats_path = root / "stats.json"
    stats_path.write_text(json.dumps(stats))
    monkeypatch.setattr(power, "load_scene", lambda *a: (checkpoint, None, None, None))
    monkeypatch.setattr(power, "load_radar_fields_npz", forbidden)
    with pytest.raises(ValueError, match="identity"):
        power.main(["--object", "loader", "--dataset-root", str(root), "--checkpoint", "c",
                    "--stats", str(stats_path), "--label", "test", "--device", "cpu"])


def test_registered_roles_and_checkpoint_bound_resume(source, tmp_path):
    root, _, contract = source
    args = power.parse_args([*CLI_CASES[3][1], "--object", "loader", "--dataset-root", str(root)])
    bound, ids = power.collection_evaluation_inputs(args,
        {"sealed_npz_protocol_contract": contract}, {"dataset_identity": contract["dataset_identity"]})
    np.testing.assert_array_equal(ids, dataset.role_ids()["validation"])
    assert not set(ids) & set(dataset.role_ids()["reserved_test"])
    cache_path = tmp_path / "metric.npz"
    positions = np.zeros((2, 3))
    provenance = {"dataset_identity": contract["dataset_identity"], "checkpoint": "a", "stats": {"peak": 1.0}}
    cache = power.load_or_initialize_cache(cache_path, ids[:2], positions, 3, False, provenance=provenance)
    np.savez(cache_path, **cache)
    for changed in ({**provenance, "checkpoint": "b"}, {**provenance, "stats": {"peak": 2.0}},
                    {**provenance, "dataset_identity": dataset.object_identity("b787")}):
        with pytest.raises(ValueError, match="different object"):
            power.load_or_initialize_cache(cache_path, ids[:2], positions, 3, True, provenance=changed)


@pytest.mark.parametrize("bad", ["checkpoint", "cache", "native_stats", "power_stats"])
def test_geraf_preflights_all_bound_inputs_before_cache_targets(source, monkeypatch, bad):
    root, _, contract = source
    cache_root = root / "cache"
    cache_root.mkdir()
    foreign = dataset._object_contract("b787")
    checkpoint = {"run_identity": {"sealed_protocol_identity": foreign if bad == "checkpoint" else contract}}
    (cache_root / geraf.B787_3200_CACHE_RECIPE_FILENAME).write_text(json.dumps(
        {"sealed_protocol_identity": foreign if bad == "cache" else contract}))
    (cache_root / geraf.B787_3200_CACHE_STATS_FILENAME).write_text(json.dumps(
        {"dataset_identity": (foreign if bad == "native_stats" else contract)["dataset_identity"]}))
    power_stats = root / "power.json"
    power_stats.write_text(json.dumps({"dataset_identity": (foreign if bad == "power_stats" else contract)["dataset_identity"]}))
    monkeypatch.setattr(torch, "load", lambda *a, **k: checkpoint)
    monkeypatch.setattr(geraf.tg, "verify_prepared_b7873200_cache", forbidden)
    with pytest.raises(ValueError, match="identity"):
        geraf.main(["--object", "loader", "--dataset-root", str(root), "--checkpoint", "c",
                    "--cache-root", str(cache_root), "--power-stats", str(power_stats), "--device", "cpu"])


@pytest.mark.parametrize("bad", ["checkpoint", "stats"])
def test_rf_rejects_foreign_checkpoint_and_embedded_stats(source, monkeypatch, bad):
    root, _, contract = source
    checkpoint = {"sealed_protocol_contract": contract,
                  "power_stats": {"dataset_identity": contract["dataset_identity"]}}
    if bad == "checkpoint":
        checkpoint["sealed_protocol_contract"] = dataset._object_contract("b787")
    else:
        checkpoint["power_stats"]["dataset_identity"] = dataset.object_identity("b787")
    monkeypatch.setattr(torch, "load", lambda *a, **k: checkpoint)
    monkeypatch.setattr(rf, "load_json", lambda *a: {})
    monkeypatch.setattr(rf, "load_radar_fields_npz", forbidden)
    with pytest.raises(ValueError, match="identity"):
        rf.main(["--object", "loader", "--dataset-root", str(root), "--checkpoint", "c",
                 "--config", "recipe", "--output", str(root / "out.json")])


def test_splat_provenance_comes_from_bound_object(source, monkeypatch):
    from rift.radarsplat_b7873200_protocol import RECIPE_FILENAME, STATS_FILENAME
    root, _, contract = source
    cache_root = root / "cache"
    cache_root.mkdir()
    (cache_root / RECIPE_FILENAME).write_text(json.dumps({"sealed_protocol_identity": contract}))
    (cache_root / STATS_FILENAME).write_text(json.dumps({"dataset_identity": contract["dataset_identity"]}))
    args = SimpleNamespace(cache_root=cache_root, checkpoint_dir=root, object=None, dataset_root=root)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"cache_recipe": {"sealed_protocol_identity": contract}})
    actual = splat.collection_readout_provenance(args)
    assert actual["dataset_identity"] == dataset.object_identity("loader")
    assert actual["dataset_npz_path"] == str(dataset.object_paths(root, "loader")[0])
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"cache_recipe": {"sealed_protocol_identity": dataset._object_contract("b787")}})
    with pytest.raises(ValueError, match="identity"):
        splat.collection_readout_provenance(args)


def test_geometry_missing_or_wrong_mesh_never_uses_b787(source, monkeypatch):
    root, arrays, _ = source
    args = render.parse_args(["--object", "loader", "--dataset-root", str(root), "--checkpoint", "c"])
    (root / "B787.stl").touch()
    with pytest.raises(FileNotFoundError, match="No local registered mesh"):
        render.collection_geometry_inputs(args)
    args.stl = root / "loader.stl"
    args.stl.touch()
    monkeypatch.setattr(render, "load_stl_vertices", lambda p: np.array([[0., 0., 0.], [4., 2., 10.]]))
    with pytest.raises(ValueError, match="dimensions"):
        render.collection_geometry_inputs(args)
    monkeypatch.setattr(render, "load_stl_vertices", lambda p: np.array([[3., -2., 5.], [5., 2., 15.]]))
    _, vertices, _ = render.collection_geometry_inputs(args)
    np.testing.assert_allclose(vertices, [[-.01, -.02, -.05], [.01, .02, .05]])


def test_geometry_checkpoint_and_sample_coordinates(tmp_path):
    contract = dataset._object_contract("loader")
    path = tmp_path / "grid.pt"
    checkpoint = {"sealed_npz_protocol_contract": contract, "extent": .15, "epoch": 2,
        "model_state_dict": {"w_re": torch.ones((2, 2, 2, 1)), "w_im": torch.zeros((2, 2, 2, 1))}}
    torch.save(checkpoint, path)
    energy, size, epoch = render.load_energy_field(path, expected_contract=contract, extent=.15)
    assert energy.shape == (2, 2, 2) and size == 2 and epoch == 2
    with pytest.raises(ValueError, match="identity"):
        render.load_energy_field(path, expected_contract=dataset._object_contract("b787"), extent=.15)
    with pytest.raises(ValueError, match="extent"):
        render.load_energy_field(path, expected_contract=contract, extent=1.5)
    centers = render.trilinear_sample_centers(.15, 48, 192)
    bounds = render.image_extent_from_centers(centers)
    pitch = (bounds[1] - bounds[0]) / len(centers)
    np.testing.assert_allclose(bounds[0] + (np.arange(len(centers)) + .5) * pitch, centers)
    assert bounds[1] < .15  # Dense interpolation retains the native center interval.


def test_model_free_uses_public_ingress_and_preserves_output_identity(source, monkeypatch):
    root, _, _ = source
    monkeypatch.setattr(train, "_load_sealed_npz_protocol_contract", forbidden)
    calls = []
    def record(arrays, contract, output):
        allowed = arrays["_lazy_response_reader"].allowed_view_indices
        assert not allowed.intersection(contract["role_ids"]["reserved_test"])
        calls.append(contract["dataset_identity"])
    monkeypatch.setattr(model_free, "finite_sh", record)
    argv = ["--object", "loader", "--dataset-root", str(root), "--method", "fsh", "--output-dir", str(root / "out")]
    model_free.main(argv)
    assert calls == [dataset.object_identity("loader")]
    identity = root / "out/model_free_identity.json"
    saved = json.loads(identity.read_text())
    saved["sealed_protocol_identity"]["dataset_identity"] = dataset.object_identity("b787")
    identity.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="another object"):
        model_free.main(argv)
    assert len(calls) == 1


def test_collection_power_stats_accept_object_specific_train_peak():
    contract = dataset._object_contract("loader")
    train = contract["role_ids"]["train"]
    stats = {"dataset_identity": contract["dataset_identity"], "peak_power": 2.5,
        "dynamic_range_db": 60., "train_view_indices": train, "train_view_count": len(train),
        "normalization_scan_view_indices": train[:8], "normalization_provenance": {
            "version": 1, "normalization_scan_role": "train", "train_view_indices": train,
            "normalization_scan_view_indices": train[:8]}}
    assert power.validate_normalization_stats(stats, {"sealed_npz_protocol_contract": contract},
                                              num_views=10000) == (2.5, 60.)
    stats["normalization_provenance"]["normalization_scan_role"] = "test"
    with pytest.raises(ValueError, match="train-role"):
        power.validate_normalization_stats(stats, {"sealed_npz_protocol_contract": contract}, num_views=10000)


def test_collection_checkpoint_roles_cannot_be_reassigned(source):
    root, _, contract = source
    changed = json.loads(json.dumps(contract))
    changed["role_ids"]["validation"], changed["role_ids"]["reserved_test"] = (
        changed["role_ids"]["reserved_test"], changed["role_ids"]["validation"])
    args = power.parse_args([*CLI_CASES[3][1], "--object", "loader", "--dataset-root", str(root)])
    with pytest.raises(ValueError, match="registered"):
        power.collection_evaluation_inputs(args, {"sealed_npz_protocol_contract": changed},
                                            {"dataset_identity": contract["dataset_identity"]})


def test_preparation_stats_bind_object_and_fit_only_training_targets(tmp_path, monkeypatch):
    contract = dataset._object_contract("loader")
    monkeypatch.setattr(prepare_geraf, "_validate_target", lambda *a, role, **k: np.full((2,2,2), 2. if role == "train" else 9.))
    monkeypatch.setattr(prepare_geraf, "write_b7873200_target_cache_protocol", lambda *a: None)
    prepare_geraf._finalize(tmp_path, identity=contract, target_spec={}, train=[1], validation=[2], expected_shape=(2,2,2))
    stats = json.loads((tmp_path / prepare_geraf.STATS_FILENAME).read_text())
    assert stats["dataset_identity"] == dataset.object_identity("loader")
    assert stats["geraf_mf_magnitude_peak"] == 2.
    monkeypatch.setattr(prepare_splat, "load_target", lambda root, i, role, **k: {
        "radarsplat_mf_power": np.full((2,2), 2. if role == "train" else 9.)})
    stats = prepare_splat._stats_from_complete_cache(tmp_path,
        {"target_spec": {"grid": {}}, "sealed_protocol_identity": contract},
        {"train": [1], "validation": [2]}, .001)
    assert stats["dataset_identity"] == dataset.object_identity("loader")
    assert stats["train_peak_power"] == 2.


def test_geraf_collection_selection_uses_its_own_best_history(monkeypatch):
    contract = dataset._object_contract("loader")
    run = {"sealed_protocol_identity": contract, "steps": 50000, "seed": 42,
        "validation": {"every_optimizer_steps": 1000, "selection_metric": "mf_magnitude_mse"},
        "grid": {"scene_extent": .15, "aperture_scale": 1., "n_azimuth": 32, "n_elevation": 32, "n_depth": 32},
        "render": {"lensless_correction": True, "detach_start_cdf": True, "directional_exponent": 1., "min_distance": 1e-6},
        "operator": {"backend": "range_nufft", "compute_dtype": "float64", "phase_sign": -1.,
                     "oversample": 2, "kernel_width": 20, "pair_chunk": 32, "point_chunk": 4096}}
    row = {"step": 1000, "views": 1000, "voxels": 1000*32**3,
           "mf_magnitude_mse": .25, "mf_magnitude_relative_mse": .4}
    checkpoint = {"step": 1000, "best_val_mse": .25, "history": [row], "run_identity": run,
        "model_config": {}, "model_state_dict": {}, "acquisition_record": {},
        "cache_recipe": {}, "target_manifest": {}, "target_stats": {}}
    cache = SimpleNamespace(sealed_identity=contract, acquisition_record={}, recipe={}, target_manifest={}, stats={})
    monkeypatch.setattr(geraf.tg, "run_identity", lambda **kwargs: run)
    # This test isolates selection; model/recipe consistency has its own gate.
    monkeypatch.setattr(geraf.tg, "model_config_from_args", lambda args: {})
    monkeypatch.setattr(geraf, "acquisition_records_equal", lambda a,b: a == b)
    assert geraf._validate_checkpoint_selection(checkpoint, cache, SimpleNamespace()) == row
    row["mf_magnitude_relative_mse"] = -.1
    with pytest.raises(ValueError, match="nonnegative"):
        geraf._validate_checkpoint_selection(checkpoint, cache, SimpleNamespace())
    row["mf_magnitude_relative_mse"] = .4
    checkpoint["best_val_mse"] = .2
    with pytest.raises(ValueError, match="magnitude MSE"):
        geraf._validate_checkpoint_selection(checkpoint, cache, SimpleNamespace())


def test_geraf_collection_aggregate_has_no_b787_target_norm():
    summary = {"views_complete": 1000, "views_total": 1000,
        "coherent_sample_count": geraf.EXPECTED_VALIDATION_SAMPLES,
        "normalized_range_power_element_count": geraf.EXPECTED_POWER_ELEMENT_COUNT,
        "normalized_range_power_target_squared_norm": 2.5,
        "coherent_complex_rel_mse": .3, "normalized_range_power_rel_mse": .4}
    geraf.validate_production_aggregate(summary, collection_identity=dataset._object_contract("loader"))
    with pytest.raises(ValueError, match="target squared norm"):
        geraf.validate_production_aggregate(summary)
    summary["coherent_sample_count"] -= 1
    with pytest.raises(ValueError, match="sample count"):
        geraf.validate_production_aggregate(summary, collection_identity=dataset._object_contract("loader"))


@pytest.mark.parametrize("sweep", [False, True])
def test_geometry_primary_and_oracle_rows_keep_checkpoint_provenance(tmp_path, monkeypatch, capsys, sweep):
    import csv
    contract = dataset._object_contract("loader")
    triangle = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]])
    monkeypatch.setattr(geometry, "collection_geometry_inputs", lambda args: ({}, triangle, contract))
    monkeypatch.setattr(geometry, "load_energy_field", lambda *a, **k: (np.arange(8.).reshape(2,2,2), 2, 1))
    monkeypatch.setattr(geometry, "sample_surface_points", lambda *a: triangle)
    monkeypatch.setattr(geometry, "sample_volume_points", lambda *a: triangle)
    monkeypatch.setattr(geometry, "inside_mask", lambda *a: np.ones((2,2,2), dtype=bool))
    output = tmp_path / "geometry.csv"
    argv = ["--object", "loader", "--checkpoints", str(tmp_path / "a.pt"), str(tmp_path / "b.pt"),
            "--labels", "same", "same", "--fixed-threshold", ".2", "--min-points", "1",
            "--gt-grid", "2", "--csv", str(output)]
    if sweep:
        argv += ["--thresholds", ".5"]
    geometry.main(argv)
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == (4 if sweep else 2)
    for checkpoint in ("a.pt", "b.pt"):
        own = [row for row in rows if Path(row["checkpoint"]).name == checkpoint]
        assert len(own) == (2 if sweep else 1)
        assert own[0]["object_id"] == "loader"
        assert own[0]["threshold_role"] == "fixed_primary"
        if sweep:
            assert own[1]["threshold_role"] == "oracle_diagnostic"
    assert ("ground-truth-oracle threshold diagnostic" in capsys.readouterr().out) == sweep
