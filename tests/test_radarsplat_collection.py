"""RadarSplat-owned collection recipe and CLI contracts."""
import pytest
from tests.test_rift_dataset_core import source

from rift.rift_dataset import catalog, object_paths
from rift import radarsplat_collection as owner


@pytest.mark.parametrize("name", [s["object_id"] for s in catalog()["objects"]])
@pytest.mark.parametrize("recipe", owner.RECIPES)
def test_radarsplat_commands(name, recipe, tmp_path):
    from train_radarsplat import parse_args
    from scripts.prepare_radarsplat_b7873200_targets import parse_args as parse_targets
    kwargs = dict(dataset_root=tmp_path/"data", output_root=tmp_path/"runs")
    direct = owner.commands_for(name, recipe=recipe, **kwargs)
    npz, manifest = object_paths(kwargs["dataset_root"], name)
    assert str(npz) in direct[0] and str(manifest) in direct[0]
    targets = parse_targets(direct[0][2:])
    parsed = parse_args(direct[1][2:])
    assert parsed.fidelity_profile == recipe
    assert parsed.steps == (2000 if recipe in ("upstream", "budget48") else 480000)
    assert parsed.init_num_gaussians == (112000 if recipe == "budget48" else 20000 if recipe == "upstream" else 2048)
    assert not parsed.resume
    if recipe != "legacy":
        assert targets.grid_policy == "scene_support"


def test_radarsplat_resume_and_default(tmp_path):
    from train_radarsplat import parse_args
    kwargs = dict(dataset_root=tmp_path, output_root=tmp_path/"runs")
    command = owner.commands_for("a320", resume="auto", **kwargs)[1]
    parsed = parse_args(command[2:])
    assert parsed.fidelity_profile == owner.DEFAULT_RECIPE == "budget48"
    assert parsed.resume
    with pytest.raises(ValueError, match="own output"):
        owner.commands_for("a320", resume="wrong.pt", **kwargs)
    with pytest.raises(ValueError, match="Unknown RadarSplat"):
        owner.commands_for("a320", recipe="wrong", **kwargs)


def test_canonical_entrypoint_metadata_plan(monkeypatch, tmp_path):
    import sys
    import train_rift_dataset as entry
    import train_rift_dataset as shared
    from train_radarsplat import parse_args
    monkeypatch.setattr(shared, "preflight_object", lambda root, name, num_train, antenna_selection=None: {"object_id": name})
    monkeypatch.setitem(sys.modules, "train_sugavanam_ertin", None)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    args = entry.parse_args(["--object", "a320", "--method", "rs", "--output-root", str(tmp_path/"out"), "--dry-run"])
    plan = entry.make_plan(args)
    assert not plan["response_payload_read"]
    train = plan["plans"][0]["commands"][1]
    assert parse_args(train[2:]).steps == 2000
    assert not (tmp_path/"out").exists()
    with pytest.raises(RuntimeError, match="allocation"):
        entry.main(["--object", "a320", "--method", "rs", "--output-root", str(tmp_path/"out")])
    with pytest.raises(ValueError, match="one object"):
        entry.make_plan(entry.parse_args(["--resume", "auto"]))


def test_selected_acquisition_record_and_source_cache_identity(tmp_path, monkeypatch):
    import json
    import numpy as np
    from rift.antenna_selection import selection
    from rift.rift_dataset import _object_contract
    from rift.radarsplat_b7873200_acquisition import acquisition_payload, write_acquisition_record, load_acquisition_record
    from rift.radarsplat_b7873200_protocol import expected_cache_recipe
    from scripts.prepare_radarsplat_b7873200_targets import parse_args, _target_spec
    acquisition=selection(1,1)
    identity=_object_contract('loader',2400,acquisition,'a'*64)
    args=parse_args(['--npz-path',str(tmp_path/'source.npz'),'--role-manifest',str(tmp_path/'roles.json'),
                     '--cache-root',str(tmp_path/'cache')])
    recipe=expected_cache_recipe(identity,_target_spec(args,10.,.2))
    assert recipe['acquisition']['response_shape']==[10000,1,1,1,600]
    assert recipe['single_pair_sensor_adaptation']['azimuth_beamwidth_deg']==10.
    assert recipe['sealed_protocol_identity']['antenna_selection']==acquisition
    ids=identity['role_ids']['train'][:1]+identity['role_ids']['validation'][:1]
    poses=np.array([[[10.,.01,.02]],[[10.,.03,.04]]])
    payload=acquisition_payload(view_indices=ids,frequency_hz=np.arange(600)*5e6+8.5e9,
        viewpoint_positions=np.array([[10.,0.,0.]]*2),tx_pos=poses,rx_pos=poses+.01,
        metadata={'rift_antenna_selection':acquisition},response_shape=(10000,1,1,1,600),response_dtype='complex64')
    write_acquisition_record(tmp_path,**payload)
    record=load_acquisition_record(tmp_path,expected_view_indices=ids)
    np.testing.assert_array_equal(record['tx_pos'],poses)
    assert json.loads(record['metadata_json'].item())['rift_antenna_selection']==acquisition


def test_single_pair_bounded_conversion_and_cache_resume(source,tmp_path,monkeypatch):
    import json
    import signal
    import numpy as np
    from rift.antenna_selection import selection
    from rift.rift_dataset import role_manifest
    from rift.radarsplat_b7873200_protocol import load_cache
    from scripts.prepare_radarsplat_b7873200_targets import main
    npz,manifest,original=source
    manifest.write_text(json.dumps(role_manifest('loader',40,selection(1,1))))
    reads=[]
    class Source:
        shape=(10000,16,16,1,600)
        dtype=np.dtype('complex64')
        def __getitem__(self,index):
            reads.append(index)
            return np.full((16,16,1,600),1+.5j,dtype=np.complex64)
    monkeypatch.setattr('rift.power_baseline_dataset._stored_member_memmap',lambda *a:Source())
    argv=['--npz-path',str(npz),'--role-manifest',str(manifest),'--cache-root',str(tmp_path/'cache'),
          '--max-train','1','--max-validation','1','--n-azimuth','3','--n-elevation','3','--n-range','33',
          '--device','cpu','--backend','direct']
    handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        main(argv)
        cache=load_cache(tmp_path/'cache')
        assert cache.identity['response_shape']==[10000,1,1,1,600]
        assert len(reads)==2
        assert reads==list(cache.train_indices+cache.validation_indices)
        main(argv)
        assert len(reads)==2
        assert load_cache(tmp_path/'cache').recipe==cache.recipe
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
