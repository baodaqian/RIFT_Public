"""Nested training roles, sealed ingress and checkpoint separation; no real fitting."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from rift import rift_dataset as collection
from rift.gotcha_dataset import GOTCHADataset, digest, training_sectors, validate_checkpoint
from rift.npz_dataset import get_npz_response_view
from tests.test_rift_dataset_core import source
from tests.test_gotcha_dataset import write_shard, tiny_region


@pytest.mark.parametrize("count", [1, 32, 128, 1600, 2400, 3199, 3200])
def test_collection_nested_roles_preserve_holdouts(count):
    parent = collection.role_ids()
    roles = collection.role_ids(count)
    assert roles["train"] == parent["train"][:count]
    assert roles["validation"] == parent["validation"]
    assert roles["reserved_test"] == parent["reserved_test"]
    assert roles["unused"] == parent["train"][count:] + parent["unused"]
    assert len(set().union(*map(set, roles.values()))) == 10000
    contract = collection._object_contract("loader", count)
    assert collection.collection_contract(contract) == contract


@pytest.mark.parametrize("bad", [0, -1, 3201, True, 2400.0, "2400"])
def test_collection_invalid_counts(bad):
    with pytest.raises(ValueError, match="num_train"):
        collection.role_ids(bad)


def test_selected_fingerprints():
    assert collection.training_selection(2400)["train_ids_sha256"] == "f7e42dbc7a99f9651638b2a238fe77470031b2ead7c864a8eb550eb0be6d2a8b"
    sectors = training_sectors(num_train=1500)
    assert [len(sectors[p]) for p in range(1, 9)] == [187, 187, 188, 188, 188, 187, 187, 188]
    assert [p for p in sectors if 307 in sectors[p]] == [3, 4, 5, 8]
    assert digest([[p, s] for p in sectors for s in sectors[p]]) == "bd4cb1c8e1d6e2cad6cc49cf7e28f54cafbf529d2d7f6b6a9dfbadc1fb79c09a"


def test_collection_excluded_responses_denied_at_public_and_trainer_ingress(source):
    import train
    npz, manifest, original = source
    before = manifest.read_bytes()
    arrays, contract = collection.load_object_contract(npz, manifest, num_train=2400)
    assert manifest.read_bytes() == before
    assert original["_lazy_response_reader"].allowed_view_indices is None
    allowed = arrays["_lazy_response_reader"].allowed_view_indices
    assert len(allowed) == 3400
    for role in ("unused", "reserved_test"):
        with pytest.raises(PermissionError):
            get_npz_response_view(arrays, contract["role_ids"][role][0])
    trainer_arrays, trainer_contract = train._load_sealed_npz_protocol_contract(
        npz, manifest, num_train=2400, num_val=1000, num_test=1000)
    assert trainer_contract == contract
    assert trainer_arrays["_lazy_response_reader"].allowed_view_indices == allowed
    train._validate_saved_sealed_npz_protocol_contract(contract, contract)
    for incompatible in (collection._object_contract("loader"), collection._object_contract("loader", 1600)):
        with pytest.raises(ValueError, match="resume would change"):
            train._validate_saved_sealed_npz_protocol_contract(incompatible, contract)


def test_subset_manifest_roundtrip_tampering_and_expansion(source):
    npz, manifest, _ = source
    selected = collection.role_manifest("loader", 2400)
    manifest.write_text(json.dumps(selected))
    _, contract = collection.load_object_contract(npz, manifest)
    assert len(contract["role_ids"]["train"]) == 2400
    with pytest.raises(ValueError, match="expand"):
        collection.load_object_contract(npz, manifest, num_train=3200)
    for mutation in ("ids", "hash"):
        changed = copy.deepcopy(selected)
        if mutation == "ids":
            changed["split"]["train_indices"][0], changed["split"]["unused_indices"][0] = (
                changed["split"]["unused_indices"][0], changed["split"]["train_indices"][0])
        else:
            changed["training_selection"]["train_ids_sha256"] = "wrong"
        manifest.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="PCG64"):
            collection.load_object_contract(npz, manifest)


@pytest.mark.parametrize("count", [8, 9, 1500, 1501, 1999, 2000])
def test_gotcha_balanced_nested_selection(count):
    smaller, larger = training_sectors(num_train=count), training_sectors(num_train=min(count+1, 2000))
    assert sum(map(len, smaller.values())) == count
    assert max(map(len, smaller.values())) - min(map(len, smaller.values())) <= 1
    assert all(set(smaller[p]) <= set(larger[p]) for p in smaller)


@pytest.mark.parametrize("count", [0, 7, 2001, True, 1500.0, "1500"])
def test_gotcha_invalid_counts(count):
    with pytest.raises(ValueError, match="num_train"):
        training_sectors(num_train=count)


def test_native_subset_row_access_pulse_plans_and_resume(tmp_path):
    from rift.spinr_gotcha_training import PulsePlan
    for p in range(1, 9):
        for pol in ("hh", "vv"):
            write_shard(tmp_path / "New_Transfer/shards" / f"pass{p}_{pol}.npz", pass_id=p, pol=pol)
    selected = GOTCHADataset(tmp_path, polarizations=("hh", "vv"), region=tiny_region(), num_train=1500)
    parent = GOTCHADataset(tmp_path, polarizations=("hh", "vv"), region=tiny_region())
    assert selected.viewpoints("validation") == parent.viewpoints("validation")
    assert len(selected.viewpoints("train")) == 1500
    assert selected.identity != parent.identity
    assert selected.summary()["response_payload_read"] is False
    for (p, pol), shard in selected.shards.items():
        dropped = set(parent.splits_by_pass[p]["train"]) - set(selected.splits_by_pass[p]["train"])
        for s in dropped:
            with pytest.raises(PermissionError):
                shard.read(int(shard.sector_rows[s][0]))
        assert shard.response_reads == 0
        for s in selected.splits_by_pass[p]["test"]:
            with pytest.raises(PermissionError):
                shard.read(int(shard.sector_rows[s][0]))
    saved = dict(schema="rift_gotcha_checkpoint_v1", dataset_contract=selected.contract,
                 dataset_identity=selected.identity, recipe={})
    validate_checkpoint(saved, selected, {})
    with pytest.raises(ValueError, match="split changed"):
        validate_checkpoint(saved, parent, {})
    # The train-only pulse plan must never enumerate discarded native rows.
    plan = PulsePlan(selected)
    assert len(plan.records) == 3000
    for shard_index, row in plan.records:
        assert selected.shards[plan.keys[shard_index]].row_roles[row] == "train"


def test_canonical_frontend_defaults_and_parent_override(source, tmp_path):
    import train_rift_dataset as rift_cli
    import train_gotcha_dataset as gotcha_cli
    args = rift_cli.parse_args(["--dataset-root", str(tmp_path), "--object", "loader", "--dry-run"])
    assert args.num_train == 2400
    plan = rift_cli.make_plan(args)
    entry = plan["plans"][0]
    assert len(entry["dataset_identity"]["role_ids"]["train"]) == 2400
    assert "/train2400/" in entry["output_dir"]
    assert not Path(entry["role_manifest_path"]).exists()
    command = entry["commands"][0]
    assert command[command.index("--num-train")+1] == "2400"
    assert gotcha_cli.parse_args([]).num_train == 1500
    assert gotcha_cli.parse_args(["--num-train", "2000"]).num_train == 2000
    assert gotcha_cli.parse_args(["--passes", "1"]).num_train == 250


def test_subset_manifest_publication_is_recoverable_and_never_overwrites(tmp_path, monkeypatch):
    import train_rift_dataset as cli
    path = tmp_path / "train2400/loader/role_manifest.json"
    manifest = collection.role_manifest("loader", 2400)
    link = cli.os.link
    def interrupted(*_):
        raise OSError("simulated publish interruption")
    monkeypatch.setattr(cli.os, "link", interrupted)
    with pytest.raises(OSError, match="interruption"):
        cli.write_selected_manifest(path, manifest)
    assert list(path.parent.iterdir()) == []
    monkeypatch.setattr(cli.os, "link", link)
    cli.write_selected_manifest(path, manifest)
    inode = path.stat().st_ino
    cli.write_selected_manifest(path, manifest)
    assert path.stat().st_ino == inode
    with pytest.raises(ValueError, match="Existing subset manifest changed"):
        cli.write_selected_manifest(path, collection.role_manifest("loader", 1600))
    assert json.loads(path.read_text()) == manifest
    assert list(path.parent.iterdir()) == [path]


def test_spinr_subset_epoch_cursor_and_recipe():
    import train_spinr_style as spinr
    ids = collection.role_ids(2400)["train"]
    assert len(spinr.epoch_view_batches(ids, epoch=0)) == 600
    recipe = spinr._recipe_identity("paper-v1-direct", 2400)
    assert recipe["batching"]["updates_per_epoch"] == 600
    coverage = spinr.optimization_coverage(ids, 1, 600)
    assert coverage["optimizer_updates"] == 1200
    assert coverage["view_exposures"] == 4800
    assert spinr.optimization_coverage(ids, 2, 0) == coverage
    with pytest.raises(ValueError, match="cursor"):
        spinr.optimization_coverage(ids, 1, 601)
    spinr._validate_canonical_contract(collection._object_contract("loader", 2400))


def test_subset_adaptive_observer_keeps_parent_gates_and_exposure_recipe(source, tmp_path, monkeypatch):
    import torch
    import train
    import train_rift_dataset as cli
    from rift import adaptive_training_workflow as workflow
    from rift.collection_adaptive import observer_type
    from rift.b7873200_adaptive_fullscale import AdaptiveFullScaleObserver, FULLSCALE_SCHEMA
    observer = observer_type(collection._object_contract("loader", 2400))
    assert observer.num_train == 2400 and observer.expected_updates == 360000
    assert observer.schema != FULLSCALE_SCHEMA
    assert AdaptiveFullScaleObserver.num_train == 3200
    assert AdaptiveFullScaleObserver.expected_updates == 480000
    with pytest.raises(ValueError, match="observer state"):
        observer.report_from_checkpoint_state({"schema": FULLSCALE_SCHEMA, "version": 2}, "checkpoint_final.pth.tar")
    command = cli.commands_for("loader", "rift", dataset_root=tmp_path, output_root=tmp_path/"runs", num_train=2400)[0]
    args = train.parse_args(command[2:])
    parent = train.parse_args(cli.commands_for("loader", "rift", dataset_root=tmp_path, output_root=tmp_path/"runs")[0][2:])
    for key, value in vars(parent).items():
        if key not in ("num_train", "checkpoint_root", "npz_role_manifest", "execution_contract_label"):
            assert getattr(args, key) == value
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    args = workflow.parse_args(["--object", "loader", "--dataset-root", str(tmp_path),
        "--num-train", "2400", "--checkpoint-root", str(tmp_path/"runs")])
    output = workflow._validate_cli(args)
    assert "train2400" in output.name and not output.exists()
    assert args.observer_type.schema == observer.schema
    assert len(workflow._validate_parent_header(args.npz_path, args.parent_role_manifest, args.num_train)["role_ids"]["train"]) == 2400


@pytest.mark.parametrize("name", [o["object_id"] for o in collection.catalog()["objects"]])
def test_real_subset_metadata_reaches_all_collection_ingress(name, tmp_path, monkeypatch):
    """Actual headers and all owner ingress; raw payload access is forbidden."""
    from rift import npz_dataset
    from rift.geraf_source_data import RIFTSourceData
    from rift.radarsplat_b7873200_protocol import load_b7873200_sealed_identity as splat
    from rift.sugavanam_ertin_b7873200_stage1 import load_b7873200_sealed_identity as se
    from rift.radar_fields_dataset import load_radar_fields_sealed_split_manifest
    from train_spinr_style import preflight_b787_development_inputs
    import train
    npz, parent = collection.object_paths(collection.DEFAULT_ROOT, name)
    if not npz.is_file() or not parent.is_file():
        pytest.skip("Local source archives are not distributed")
    def forbidden(*a, **kw):
        raise AssertionError("Response access during subset metadata preflight")
    monkeypatch.setattr(npz_dataset._LazyResponseReader, "iter_response_views", forbidden)
    original = np.lib.npyio.NpzFile.__getitem__
    def guard(archive, key):
        if key in ("response", "response.npy"):
            forbidden()
        return original(archive, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", guard)
    manifest = tmp_path / "role_manifest.json"
    manifest.write_text(json.dumps(collection.role_manifest(name, 2400)))
    expected = collection._object_contract(name, 2400)
    assert splat(npz, manifest) == expected
    arrays, contract = se(npz, manifest)
    assert contract == expected and arrays["response"] is None
    arrays, contract, acquisition = preflight_b787_development_inputs(npz_path=str(npz), manifest_path=str(manifest), resume_checkpoint=None)
    assert collection.collection_contract(contract) == expected
    assert len(acquisition["authorized_view_ids"]) == 3400
    geraf = RIFTSourceData(npz, manifest)
    assert len(geraf.views("train")) == 2400 and len(geraf.views("validation")) == 1000
    _, contract = train._load_sealed_npz_protocol_contract(npz, manifest, num_train=2400, num_val=1000, num_test=1000)
    assert collection.collection_contract(contract) == expected
    rf = load_radar_fields_sealed_split_manifest(str(manifest), 10000,
        response_shape=[10000, 16, 16, 1, 600], response_dtype="complex64", expected_num_train=2400)
    assert list(rf.train_indices) == expected["role_ids"]["train"]
