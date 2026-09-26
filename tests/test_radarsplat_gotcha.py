"""Native conversion, root routing, source lifecycle and sealed-role checks."""
import copy
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rift.gotcha_dataset import C, GOTCHADataset, NativeShardReader, sector_split
from rift import radarsplat_gotcha as backend
from tests.radarsplat_gotcha_fixtures import write_shard, tiny_region

CONFIG=dict(azimuth_samples=11,elevation_samples=3,point_chunk=2048,frequency_chunk=7)


@pytest.fixture
def dataset(tmp_path):
    write_shard(tmp_path/'data/New_Transfer/shards/pass1_hh.npz')
    return GOTCHADataset(tmp_path/'data',passes=[1],region=tiny_region())


def test_native_exact_all_pulses_ragged_frequencies_and_source_autofocus(tmp_path):
    for p,nf in ((1,11),(2,17)):
        for pol in ('hh','hv'):
            write_shard(tmp_path/f'New_Transfer/shards/pass{p}_{pol}.npz',p,pol,nf)
    ds=GOTCHADataset(tmp_path,passes=[1,2],polarizations=['hh','hv'],region=tiny_region())
    for pol,expected in (('hh',1.),('hv',4.)):
        cache=backend.GOTCHAPowerCache(ds,pol,tmp_path/f'cache_{pol}',CONFIG)
        for p in ds.passes:
            index=next(i for i,key in cache.view_keys.items() if key[0]==p)
            cal=copy.deepcopy(cache.calibration[index])
            # At the region origin, the exact adjoint returns known amplitude.
            cal.update(range_m=np.array([np.linalg.norm(cal['sensor_to_world'][:3,3])]),
                       azimuth_rad=np.array([np.pi]),elevation_rad=np.array([0.]))
            got=backend.sector_power(ds,*cache.view_keys[index],pol,cal,cache.config,device='cpu')
            assert got.item()==pytest.approx(expected,rel=2e-6)
            assert cal['native_pulse_count']==2
            assert len(ds.shards[p,pol].frequencies_hz)==(11 if p==1 else 17)
        assert sum(s.response_reads for (p,c),s in ds.shards.items() if c==pol)==4


def test_sector_power_matches_independent_direct_adjoint(dataset,tmp_path):
    from rift.radarsplat_fidelity import polar_world_points
    cache=backend.GOTCHAPowerCache(dataset,'hh',tmp_path/'cache',CONFIG)
    index=cache.train_indices[0]
    cal=copy.deepcopy(cache.calibration[index])
    cal.update(range_m=cal['range_m'][::16],azimuth_rad=cal['azimuth_rad'][::5],elevation_rad=cal['elevation_rad'])
    points=polar_world_points(cal)
    expected=np.zeros(len(points),dtype=np.complex128)
    for obs in dataset.observations(*cache.view_keys[index],'hh'):
        delay=np.linalg.norm(points-dataset.region.to_local(obs.position_m),axis=1)-obs.reference_range_m
        expected+=(np.exp((4j*np.pi/C)*delay[:,None]*obs.frequencies_hz[None,:])*obs.response[None,:]).mean(-1)/2
    expected=np.abs(expected.reshape(3,3,3))**2
    actual=backend.sector_power(dataset,*cache.view_keys[index],'hh',cal,cache.config,device='cpu')
    np.testing.assert_allclose(actual,expected.sum(0),rtol=2e-6,atol=1e-6)


def test_cache_normalization_resume_and_sealed_roles(dataset,tmp_path,monkeypatch):
    cache=backend.GOTCHAPowerCache(dataset,'hh',tmp_path/'cache',CONFIG)
    assert not dataset.summary()['response_payload_read']
    assert cache.prepare(device='cpu')
    reads=sum(s.response_reads for s in dataset.shards.values())
    assert reads==2*(250+55)
    trainmax=max(float(cache.read_target(i,'train')['radarsplat_mf_power'].max()) for i in cache.train_indices)
    assert trainmax==cache.train_peak_power
    from rift.radarsplat_release_training import read_view
    index=cache.validation_indices[0]
    arrays,_,target=read_view(cache,index,'validation','cpu')
    assert arrays['radarsplat_mf_power'].max()>cache.train_peak_power*8
    assert float(target.max())==1.
    monkeypatch.setattr(NativeShardReader,'read',lambda *_:pytest.fail('Unexpected native reread'))
    resumed=backend.GOTCHAPowerCache(dataset,'hh',cache.root,CONFIG)
    assert resumed.prepare(device='cpu') and resumed.train_peak_power==cache.train_peak_power
    with pytest.raises(PermissionError): resumed.read_target(index,'train')
    with pytest.raises(PermissionError): resumed.read_target(cache.train_indices[0],'test')
    with pytest.raises(ValueError,match='recipe mismatch'):
        backend.GOTCHAPowerCache(dataset,'hh',cache.root,{**CONFIG,'frequency_chunk':8})
    p=resumed.target_path(index); p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='bytes changed'): resumed.read_target(index,'validation')


def test_canonical_root_metadata_only_without_other_trainers(dataset,tmp_path,monkeypatch):
    import train_gotcha_dataset as owned
    import train_gotcha_dataset as shared
    monkeypatch.setattr(shared,'load_region',lambda *_:tiny_region())
    monkeypatch.setattr(NativeShardReader,'read',lambda *_:pytest.fail('metadata plan read response'))
    monkeypatch.setitem(sys.modules,'train_rift_dataset',None)
    monkeypatch.setitem(sys.modules,'train_sugavanam_ertin',None)
    args=owned.parse_args(['--method','rs','--dataset-root',str(dataset.root),'--passes','1','--output-root',str(tmp_path/'runs'),'--dry-run'])
    ds,combined=owned.make_plan(args)
    plan=combined['plans'][0]
    config=plan['config']
    assert not plan['response_payload_read'] and not ds.summary()['response_payload_read']
    assert plan['targets_per_head']['hh']['train']==250
    assert plan['recipe']['model_updates_per_head']==2000
    assert not (tmp_path/'runs').exists()
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='allocation'):
        backend.run_gotcha(dataset=ds,output_dir=tmp_path/'runs',config=config,device='cpu',resume=None)
    with pytest.raises(ValueError,match='model recipe is fixed'): backend.adapter_config({'steps':3})


def test_original_filter_crop_matches_full_circle_values_and_gradients(monkeypatch):
    import torch.nn.functional as F
    from rift import radarsplat_release as source
    from rift.radarsplat_b7873200 import RadarSplatGrid
    funcs=source.source_functions('gsplat/rendering.py',['spectral_leakage','azimuth_antenna_gain_projection'],dict(torch=torch,F=F))
    monkeypatch.setattr(torch.Tensor,'cuda',lambda t:t)
    calls=[]
    def raw(**kw):
        calls.append(kw['height'])
        k=kw['Ks'][0]
        rows=(torch.arange(kw['height'])+.5-k[1,2])/k[1,1]
        cols=(torch.arange(kw['width'])+.5-k[0,2])/k[0,0]
        p=(.4+.1*rows.sin()[:,None]+.05*cols.cos()[None,:])*kw['means'].sigmoid().mean()
        p=p[None,:,:,None]
        return p,p/2,None,None,None,None,{}
    model=SimpleNamespace(_radar_rasterization=raw,**{k:funcs[k] for k in ('spectral_leakage','azimuth_antenna_gain_projection')})
    splats,_=source.create_scene(scene_scale=100,scene_center=[0,0,0],device='cpu',num_points=2)
    grid=RadarSplatGrid(num_range_bins=13,range_resolution_m=.1,range_start_m=5.,azimuth_start_deg=173.5,
        azimuth_span_deg=13.,intermediate_azimuth_resolution_deg=.1,output_azimuth_resolution_deg=1.,
        azimuth_beamwidth_deg=1.8,spectral_leakage_width_m=1.)
    args=(splats,torch.eye(4),grid,5,torch.zeros(13,13))
    full=source.ReleasedRenderer(model,1.)(*args)[0]
    cropped=source.ReleasedRenderer(model,1.,local_azimuth=True)(*args)[0]
    torch.testing.assert_close(full,cropped,atol=2e-7,rtol=2e-6)
    a=torch.autograd.grad(full.sum(),splats['means'],retain_graph=True)[0]
    b=torch.autograd.grad(cropped.sum(),splats['means'])[0]
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    assert calls[1]<calls[0]/10


def test_preparation_interrupt_has_resume_marker_and_identity_gate(dataset,tmp_path,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','synthetic-test')
    monkeypatch.setattr(backend,'load_cuda_reference',lambda **_: (object(),object()))
    original=backend.sector_power
    def interrupted(*args,**kw):
        value=original(*args,**kw); signal.raise_signal(signal.SIGTERM); return value
    monkeypatch.setattr(backend,'sector_power',interrupted)
    output=tmp_path/'run'
    result=backend.run_gotcha(dataset=dataset,output_dir=output,config=CONFIG,device='cpu',resume=None)
    assert result['status']=='interrupted' and Path(result['resume']).is_file()
    assert sum(s.response_reads for s in dataset.shards.values())==2
    monkeypatch.setattr(NativeShardReader,'read',lambda *_:pytest.fail('identity failure read response'))
    with pytest.raises(ValueError,match='identity mismatch'):
        backend.run_gotcha(dataset=dataset,output_dir=output,config={**CONFIG,'frequency_chunk':8},device='cpu',resume=Path(result['resume']))


@pytest.mark.parametrize("profile", ["upstream", "budget48"])
def test_shared_source_training_resume_and_gotcha_readout(dataset,tmp_path,monkeypatch,profile):
    config = dict(CONFIG, fidelity_profile=profile)
    from rift import radarsplat_release as source
    from rift.radarsplat_fidelity import release_ssim_index
    import train_radarsplat as lifecycle
    monkeypatch.setenv('SLURM_JOB_ID','synthetic-test')
    original_to=torch.Tensor.to
    monkeypatch.setattr(torch.Tensor,'to',lambda t,*a,**kw:
        t if a and isinstance(a[0],str) and a[0]=='cuda' else original_to(t,*a,**kw))
    original_create=source.create_scene
    monkeypatch.setattr(backend.engine,'create_scene',lambda **kw:original_create(**{**kw,"num_points":2}))
    ssim=lambda a,b,padding:release_ssim_index(a[0,0],b[0,0])
    monkeypatch.setattr(backend,'load_cuda_reference',lambda **_:(object(),ssim))
    monkeypatch.setattr(backend.engine,'load_cuda_reference',lambda **_:(object(),ssim))
    control=dict(calls=0,stop=1)
    class Renderer:
        def __init__(self,rendering,units,**kw): self.units=units
        def __call__(self,splats,pose,grid,degree,bg):
            control['calls']+=1
            if control['calls']==control['stop']: signal.raise_signal(signal.SIGTERM)
            shape=(grid.output_azimuth_bins,grid.num_range_bins)
            return splats['sh0'].mean().sigmoid().expand(shape),splats['opacities'].sigmoid().mean().expand(shape)
    monkeypatch.setattr(backend.engine,'ReleasedRenderer',Renderer)
    output=tmp_path/'run'
    def run(root,stop,resume=None):
        control.update(calls=0,stop=stop)
        result=backend.run_gotcha(dataset=dataset,output_dir=root,config=config,device='cpu',resume=resume)
        assert result['status']=='interrupted'
        return lifecycle._load_checkpoint(root/'hh/checkpoints/checkpoint_latest.pt',torch.device('cpu'))
    run(output,1)
    resumed=run(output,1,output/backend.CONTROL_FILE)
    full=run(tmp_path/'full',2)
    assert resumed['step']==full['step']==2
    for key in ('splats','optimizers','position_scheduler','sampler'):
        assert lifecycle._directly_equal(resumed[key],full[key])
    # Identity/raw-state rejection happens before attempting native conversion.
    corrupt=copy.deepcopy(resumed); corrupt['splats']['means']=corrupt['splats']['means'].double()
    latest=output/'hh/checkpoints/checkpoint_latest.pt'; torch.save(corrupt,latest)
    monkeypatch.setattr(NativeShardReader,'read',lambda *_:pytest.fail('resume/readout read native response'))
    with pytest.raises(ValueError,match='incompatible'):
        backend.run_gotcha(dataset=dataset,output_dir=output,config=config,device='cpu',resume=output/backend.CONTROL_FILE)
    torch.save(resumed,latest)
    control.update(calls=0,stop=-1)
    from scripts.readout_radarsplat_gotcha import readout
    geometry=tmp_path/'geometry.npz'
    result=readout(run_root=output,polarization='hh',checkpoint_path=latest,device='cpu',geometry_path=geometry)
    assert result['step']==2 and result['metrics']['views']==55
    with np.load(geometry) as data:
        np.testing.assert_allclose(data['means'],resumed['splats']['means'].numpy()/result['identity']['adapter']['model_units_per_m'])
    with pytest.raises(ValueError,match='Polarization'):
        readout(run_root=output,polarization='hv',checkpoint_path=latest,device='cpu')
    import shutil
    relocated=tmp_path/'relocated'
    shutil.copytree(dataset.root,relocated)
    rebound=backend.cache_from_run(output,'hh',dataset_root=relocated)
    assert rebound.dataset.shard_root==relocated/'New_Transfer/shards'
    assert rebound.identity==backend.cache_from_run(output,'hh').identity
    # Construct a terminal-step fixture to exercise finalization/recovery
    # without running a 2000-update synthetic fit or changing production budget.
    terminal=copy.deepcopy(resumed); terminal['step']=2000
    lr=terminal['optimizers']['means']['param_groups'][0]['initial_lr']*.01
    terminal['optimizers']['means']['param_groups'][0]['lr']=lr
    terminal['position_scheduler'].update(last_epoch=2000,_step_count=2001,_last_lr=[lr])
    torch.save(terminal,latest)
    finished=backend.run_gotcha(dataset=dataset,output_dir=output,config=config,device='cpu',resume=output/backend.CONTROL_FILE)
    assert finished['status']=='complete' and finished['results']['hh']['step']==2000
    again=backend.run_gotcha(dataset=dataset,output_dir=output,config=config,device='cpu',resume=output/backend.CONTROL_FILE)
    assert again['results']['hh']['validation']==finished['results']['hh']['validation']
    final=output/'hh/checkpoints/checkpoint_final.pt'
    damaged=lifecycle._load_checkpoint(final,torch.device('cpu')); damaged['step']=1999
    torch.save(damaged,final)
    with pytest.raises(ValueError,match='disagree'):
        backend.run_gotcha(dataset=dataset,output_dir=output,config=config,device='cpu',resume=output/backend.CONTROL_FILE)


def test_all_selected_channels_route_to_same_source_recipe(tmp_path,monkeypatch):
    for pol in ('hh','hv','vh','vv'):
        write_shard(tmp_path/f'data/New_Transfer/shards/pass1_{pol}.npz',pol=pol)
    ds=GOTCHADataset(tmp_path/'data',passes=[1],polarizations=['hh','hv','vh','vv'],region=tiny_region())
    monkeypatch.setenv('SLURM_JOB_ID','synthetic-test')
    monkeypatch.setattr(backend,'load_cuda_reference',lambda **_:(object(),object()))
    # Dispatch-only test: native conversion and actual engine updates have their
    # own tests above. No model or fitting-budget override is introduced here.
    monkeypatch.setattr(backend.GOTCHAPowerCache,'prepare',lambda *_,**kw:True)
    seen=[]
    def train(cache,output,**kw):
        seen.append((cache.polarization,cache.identity,Path(output)))
        return dict(step=2000,validation={'views':55})
    monkeypatch.setattr(backend.engine,'train',train)
    result=backend.run_gotcha(dataset=ds,output_dir=tmp_path/'run',config=CONFIG,device='cpu',resume=None)
    assert result['status']=='complete' and [x[0] for x in seen]==['hh','hv','vh','vv']
    assert len({x[2] for x in seen})==4
    assert all(x[1]['dataset_contract']==ds.contract for x in seen)
