"""Sugavanam--Ertin-owned collection recipe and CLI contracts.

Moved from shared dataset tests at the user's request; SE maintains this file.
"""
import json
from pathlib import Path
import pytest
from tests.test_rift_dataset_core import source

from rift.rift_dataset import catalog, object_paths, role_manifest
from rift import sugavanam_ertin_collection as owner
from train_rift_dataset import commands_for


def test_se_full_commands_use_selected_object(tmp_path):
    import train_sugavanam_ertin as se
    _, manifest = object_paths(tmp_path, "loader")
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(role_manifest("loader")))
    command = commands_for("loader", "se", dataset_root=tmp_path, output_root=tmp_path / "runs",
                           se_recipe="legacy-full")[0]
    args = se.parse_args(command[2:])
    paths = se._paths(args)
    paths.update(npz_path=Path(args.npz_path), role_manifest=Path(args.parent_role_manifest))
    assert paths["root"].name == "rift_dataset_se_full_v1"
    stage1 = se._stage1_argv(paths, None)
    assert stage1[stage1.index("--npz-path") + 1] == args.npz_path
    assert stage1[stage1.index("--npz-role-manifest") + 1] == args.parent_role_manifest


@pytest.mark.parametrize("name", [s["object_id"] for s in catalog()["objects"]])
@pytest.mark.parametrize("recipe", owner.RECIPES)
def test_se_commands(name, recipe, tmp_path):
    from train_sugavanam_ertin import parse_args
    kwargs = dict(dataset_root=tmp_path/"data", output_root=tmp_path/"runs")
    direct = owner.commands_for(name, recipe=recipe, **kwargs)
    assert direct == commands_for(name, "se", se_recipe=recipe, **kwargs)
    npz, manifest = object_paths(kwargs["dataset_root"], name)
    parsed = parse_args(direct[0][2:])
    assert parsed.recipe == recipe
    assert Path(parsed.npz_path) == npz
    assert Path(parsed.parent_role_manifest) == manifest


def test_se_resume_and_default(tmp_path):
    kwargs = dict(dataset_root=tmp_path, output_root=tmp_path/"runs")
    command = owner.commands_for("a320", resume=tmp_path/"saved.pt", **kwargs)[0]
    assert command[command.index("--recipe")+1] == owner.DEFAULT_RECIPE == "paper-v1"
    assert command[command.index("--resume")+1] == str(tmp_path/"saved.pt")
    with pytest.raises(ValueError, match="explicit"):
        owner.commands_for("a320", resume="auto", **kwargs)


def test_single_pair_collection_operator_keeps_se_gates(source,monkeypatch):
    import numpy as np
    import torch
    from rift.antenna_selection import selection
    from rift.sugavanam_ertin_acquisition import CollectionAcquisition
    from rift import npz_dataset
    npz,manifest,_=source
    manifest.write_text(json.dumps(role_manifest('loader',2400,selection(1,1))))
    acquisition=CollectionAcquisition(npz_path=npz,manifest=manifest)
    assert acquisition.train_sample_counts==[600]*2400
    assert acquisition.arrays['tx_pos'].shape==(10000,1,3)
    raw=np.ones((1,1,1,600),dtype=np.complex64)
    monkeypatch.setattr(npz_dataset,'get_npz_response_view',lambda a,i:raw)
    view=next(acquisition.observations(acquisition.keys['train'][0],role='train'))
    assert view['response'].shape==(600,1,1)
    xyz=torch.tensor([[0.,0.,0.],[.01,.01,0.]],dtype=torch.float64)
    weights=torch.tensor([1.+.5j,.3-.2j],dtype=torch.complex128,requires_grad=True)
    result=acquisition.render(xyz,weights,view,point_chunk=2,pair_chunk=1)
    assert result.shape==(600,1,1)
    result.abs().square().sum().backward()
    assert torch.isfinite(weights.grad).all() and weights.grad.abs().sum()>0
    # Existing literal initialization and stage-1 convergence gates are untouched.
