"""SE routes through both canonical dataset frontends."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import train_rift_dataset as collection
import train_rift_dataset as shared_collection
import train_gotcha_dataset as gotcha
import train_gotcha_dataset as shared_gotcha
from rift.rift_dataset import object_paths, object_spec
from tests.se_dataset_fixtures import write_shard, tiny_region


@pytest.mark.parametrize("name", ["a320", "x59", "firetruck", "racecar", "loader", "b787"])
def test_collection_commands_bind_each_object_and_only_the_se_trainer(tmp_path, name):
    command, = collection.commands_for(name, "se", dataset_root=tmp_path, output_root=tmp_path/"runs")
    npz, manifest = object_paths(tmp_path, name)
    assert Path(command[1]).name == "train_sugavanam_ertin.py"
    assert command[command.index("--npz-path")+1] == str(npz)
    assert command[command.index("--parent-role-manifest")+1] == str(manifest)
    assert command[command.index("--recipe")+1] == "paper-v1"
    assert Path(command[command.index("--checkpoint-root")+1]) == tmp_path/"runs"/object_spec(name)["object_id"]/"sugavanam_ertin"


def test_entrypoints_list_without_importing_other_trainers(monkeypatch, capsys):
    for name in ("train_radarsplat", "train_geraf",
                 "train_radar_fields", "train_spinr_style", "tests.test_gotcha_dataset"):
        monkeypatch.setitem(sys.modules, name, None)
    importlib.reload(collection)
    importlib.reload(gotcha)
    assert collection.main(["--list"]) == 0
    assert "sugavanam_ertin" in json.loads(capsys.readouterr().out)["methods"]
    gotcha.main(["--list"])
    assert json.loads(capsys.readouterr().out)["sugavanam_ertin"]["polarizations"] == ["hh"]


def test_canonical_default_outputs_and_configuration_forwarding(tmp_path):
    assert collection.parse_args([]).output_root.name == "RIFT_dataset"
    assert gotcha.parse_args([]).output_root.name == "GOTCHA_dataset"
    config, checkpoint = tmp_path/"config.json", tmp_path/"run"/"checkpoint_latest.pt"
    command, = collection.commands_for("loader", "se", dataset_root=tmp_path, output_root=tmp_path,
        se_config=config, resume=str(checkpoint), device="cpu", check_initialization=True)
    assert command[command.index("--resume")+1] == str(checkpoint)
    assert command[command.index("--config")+1] == str(config)
    assert command[command.index("--device")+1] == "cpu" and "--check-initialization" in command
    for kw in (dict(resume="auto"), dict(se_recipe="legacy-full", se_config=config)):
        with pytest.raises(ValueError):
            collection.commands_for("loader", "se", dataset_root=tmp_path, output_root=tmp_path, **kw)


def test_collection_dry_run_reads_metadata_without_launch_or_writes(tmp_path, monkeypatch):
    preflights = []
    monkeypatch.setattr(shared_collection, "preflight_object", lambda root, name, num_train, antenna_selection=None: preflights.append(name) or {"object_id": name})
    monkeypatch.setattr(shared_collection.subprocess, "run", lambda *a, **kw: pytest.fail("dry-run launched a subprocess"))
    output = tmp_path/"output"
    assert collection.main(["--object", "a320", "loader", "--method", "se", "--dry-run", "--output-root", str(output)]) == 0
    assert len(preflights) == 2 and not output.exists()
    with pytest.raises(ValueError, match="one object"):
        collection.make_plan(collection.parse_args(["--object", "a320", "loader", "--resume", "/tmp/checkpoint.pt"]))


def test_collection_fitting_requires_allocation_and_checks_all_destinations(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_collection, "preflight_object", lambda root, name, num_train, antenna_selection=None: {})
    monkeypatch.setattr(shared_collection.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected launch"))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="allocation"):
        collection.main(["--object", "a320", "--method", "se", "--output-root", str(tmp_path)])
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic_fixture")
    from rift.antenna_selection import acquisition_label, selection
    occupied = tmp_path/"train2400"/acquisition_label(selection(1, 1))/"loader"/"sugavanam_ertin"
    occupied.mkdir(parents=True)
    (occupied/"checkpoint.pt").write_text("synthetic")
    with pytest.raises(ValueError, match="new output"):
        collection.main(["--object", "a320", "loader", "--method", "se", "--output-root", str(tmp_path)])


def test_collection_probe_propagates_nonzero_status_without_fitting(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_collection, "preflight_object", lambda root, name, num_train, antenna_selection=None: {})
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        assert "--check-initialization" in command
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr(shared_collection.subprocess, "run", child)
    assert collection.main(["--object", "a320", "--method", "se", "--check-initialization", "--output-root", str(tmp_path)]) == 2
    assert len(calls) == 1


@pytest.fixture
def gotcha_argv(tmp_path):
    root = tmp_path/"data"
    write_shard(root/"New_Transfer"/"shards"/"pass1_hh.npz", nf=5)
    region = tiny_region().as_dict()
    region.pop("name")
    regions = tmp_path/"regions.json"
    regions.write_text(json.dumps(dict(schema="rift_gotcha_regions_v1", regions={"fixture": region})))
    return ["--method", "se", "--dataset-root", str(root), "--passes", "1", "--region", "fixture",
            "--region-config", str(regions), "--output-root", str(tmp_path/"output")]


def test_gotcha_dry_run_has_own_full_recipe_and_preserves_sealed_native_counts(gotcha_argv, monkeypatch):
    monkeypatch.setattr(shared_gotcha, "dispatch", lambda *a: pytest.fail("planning attempted fitting"))
    args = gotcha.parse_args(gotcha_argv+["--dry-run"])
    dataset, combined = gotcha.make_plan(args)
    report = combined["plans"][0]
    assert report["se"]["native_view_counts"] == dict(train=250, validation=55)
    assert report["se"]["recipe"]["azimuth_bins"] == 72
    assert combined["dataset"]["viewpoints"]["test"] == 55
    assert not combined["dataset"]["response_payload_read"]
    assert sum(s.response_reads for s in dataset.shards.values()) == 0
    assert not args.output_root.exists()
    assert gotcha.main(gotcha_argv+["--dry-run"])["plans"][0]["method"] == "sugavanam_ertin"


def test_gotcha_rejects_incompatible_selection_and_configs_before_opening_data(tmp_path, monkeypatch):
    config = tmp_path/"bad.json"
    config.write_text(json.dumps({"radarsplat": {}}))
    monkeypatch.setattr(shared_gotcha, "GOTCHADataset", lambda *a, **kw: pytest.fail("invalid config opened data"))
    with pytest.raises(ValueError, match="polarization"):
        gotcha.make_plan(gotcha.parse_args(["--method", "se", "--polarizations", "vv"]))
    with pytest.raises(ValueError, match="selected method names"):
        gotcha.make_plan(gotcha.parse_args(["--method", "se", "--method-config", str(config)]))


def test_gotcha_dispatch_is_se_only_and_propagates_interruption(gotcha_argv, tmp_path, monkeypatch):
    from rift import sugavanam_ertin_paper_workflow as backend
    calls = []
    def dispatch(**kwargs):
        calls.append(kwargs)
        return dict(status="interrupted")
    monkeypatch.setattr(backend, "run_gotcha", dispatch)
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic_fixture")
    checkpoint = tmp_path/"checkpoint_latest.pt"
    checkpoint.write_text("synthetic checkpoint; backend is mocked")
    with pytest.raises(SystemExit) as exc:
        gotcha.main(gotcha_argv+["--resume", str(checkpoint), "--device", "cpu"])
    assert exc.value.code == 143
    assert len(calls) == 1 and calls[0]["resume"] == checkpoint
    assert calls[0]["output_dir"].name == "sugavanam_ertin"
    assert sum(s.response_reads for s in calls[0]["dataset"].shards.values()) == 0
