"""Source fidelity and both root bindings; synthetic fixtures only."""
from __future__ import annotations
import ast
import copy
import json
from pathlib import Path
import signal
import numpy as np
import pytest
import torch
from rift import geraf_source as source
from rift import geraf_source_training as runtime
from rift.geraf_source_ops import NativeAcquisition, bilinear_ray_sampler, source_amplitudes
from rift.geraf_source_data import GOTCHASourceData, RIFTSourceData, identity_hash
from rift.vendor.geraf_sens.rf_rendering import _compute_transmission
from tests.test_rift_dataset_core import source as collection_source

VENDOR = Path(source.__file__).parent / 'vendor' / 'geraf_sens'
SMALL = dict(steps=2, n_aperture=4, n_samples=4, n_samples_tgt=8, mf_grid=5,
             sdf_hidden_dim=80, sdf_layers=5, point_chunk=64, pair_chunk=2,
             validation_every=2, checkpoint_every=1)


def test_vendored_definitions_are_present_at_the_pinned_commit():
    """Provenance and required definitions only; AGENTS.md forbids content-hash gates on GitHub sources."""
    manifest = json.loads((VENDOR / 'source_manifest.json').read_text())
    assert manifest['commit'] == source.COMMIT
    for row in manifest['definitions']:
        file = VENDOR / row['destination']
        assert file.is_file(), row
        if 'name' in row:
            assert any(getattr(n, 'name', None) == row['name'] for n in ast.parse(file.read_text()).body), row


def test_source_network_shapes_activation_and_variance_parameterization():
    m = source.build_model(source.recipe_from_config({}, .15))
    assert m.sdf_network.lin0.in_features == 63
    assert m.sdf_network.lin3.out_features == 256 - 63
    linears = [layer for layer in m.signal_network.refelective_predictor if isinstance(layer, torch.nn.Linear)]
    assert [(x.in_features, x.out_features) for x in linears] == [(3,256), (256,256), (256,256), (256,1)]
    assert isinstance(m.signal_network.refelective_predictor[-1], torch.nn.Sigmoid)
    assert all(hasattr(x, 'weight_g') for x in linears)
    assert m.deviation_network.variance.item() == pytest.approx(.3)
    assert m.deviation_network(torch.zeros(1,3)).item() == pytest.approx(np.exp(3), rel=1e-6)
    assert m.signal_network.light_power.item() == 0.


def test_source_opacity_floor_transmission_and_cosine_are_not_repaired():
    m = source.build_model(source.recipe_from_config(SMALL, .15))
    sdf = torch.tensor([[0., -10., 10.]])
    actual = m._compute_alpha_from_sdf(sdf, torch.ones_like(sdf), torch.ones_like(sdf), torch.zeros_like(sdf))
    expected = 1e-5 / (sdf.sigmoid() + 1e-5)
    torch.testing.assert_close(actual, expected)
    assert (actual > 0).all()  # source floor intentionally retained
    alpha = torch.tensor([[.2,.3,.7]])
    torch.testing.assert_close(_compute_transmission(alpha), torch.tensor([[1.,.8000001,.8000001*.7000001]]))
    assert m.get_anneal_val(0) == 0
    m.update_step(123)
    assert source.source_step(source.recipe_from_config({}, .15), 123) == 0
    assert source.source_step(source.recipe_from_config({'model_step_policy':'advance'}, .15), 123) == 123


def test_source_scheduler_absolute_floor_is_not_changed_to_a_ratio():
    r = source.recipe_from_config({}, .15)
    model = source.build_model(r)
    optimizer, scheduler = runtime.make_optimizer({'scalar':model}, r)
    assert [g['lr'] for g in optimizer.param_groups] == [1e-4, 1e-3]
    scheduler.step(50000)
    assert [g['lr'] for g in optimizer.param_groups] == [5e-4, 5e-4]
    assert all(g['weight_decay'] == 0 for g in optimizer.param_groups)


def test_fallbacks_are_actual_release_values_and_runner_does_not_call_step_hook():
    from rift.vendor.geraf_sens import stage1_config_reference as upstream
    r=source.recipe_from_config({},.15)
    cfg=upstream.sample_cfg
    for key,source_key in [('n_aperture','N_aperture'),('n_samples','N_samples'),
                           ('n_samples_tgt','N_samples_tgt'),('bank_size','bank_size'),
                           ('anneal_end','anneal_end'),('freeze_inv_s_step','freeze_inv_s_step')]:
        assert r[key]==cfg[source_key]
    assert round((cfg['bound'][1]-cfg['bound'][0])/cfg['mf_res'][0])+1 == 601
    assert r['mf_grid'] == 48  # User-selected coarser target protocol.
    assert r['target_storage'] == 'lazy_trilinear_accumulated_only_v1'
    mask=next(t for t in upstream.train_pipeline if t['type']=='DynamicLossMask')
    assert (r['mask_current'],r['mask_accumulated'])==(mask['thres_rate'],mask['tot_thres_rate'])
    assert r['eta_min']==upstream.param_scheduler[0]['eta_min']
    assert r['grad_regression']==upstream.model['with_grad_regression']
    assert r['variance_init']==upstream.model['deviation_network']['init_val']
    calls=[n for n in ast.walk(ast.parse((VENDOR/'train_reference.py').read_text())) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr=='update_step' for n in calls)
    assert not any(isinstance(n.func,ast.Name) and n.func.id=='StepHook' for n in calls)


def test_ray_interpolation_cuda_bounds_and_denominator():
    z = torch.tensor([[0., 1e-7, 1.]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[-1., 0., 5e-8, 1., 2.]], dtype=torch.float64)
    n = torch.arange(9., dtype=torch.float64).reshape(1,3,3).requires_grad_()
    s = torch.tensor([[[1.,3.,7.]]], dtype=torch.float64, requires_grad=True)
    normals, sigmas, inside = bilinear_ray_sampler(n, s, z, q)
    torch.testing.assert_close(sigmas, torch.tensor([[[0.,1.,1.1,7.,0.]]],dtype=torch.float64))
    assert inside.tolist() == [[False, True, True, True, False]]
    (normals.sum() + sigmas.sum()).backward()
    assert z.grad is None
    assert n.grad is not None and s.grad is not None


def acquisition(uniform=False):
    tx = torch.tensor([[2.,-.04,0.],[2.,.04,0.]],dtype=torch.float64)
    rx = tx + torch.tensor([.1,0.,.01],dtype=torch.float64)
    f = torch.linspace(1e9,1.1e9,8,dtype=torch.float64)
    if not uniform:
        f[1::2] += 1323.
    return NativeAcquisition(tx,rx,f,torch.zeros(2,dtype=torch.float64),point_chunk=64,pair_chunk=2,uniform_rift=uniform)


@pytest.mark.parametrize('uniform', [False,True])
def test_native_forward_mf_and_gradients_against_independent_dense(uniform):
    op = acquisition(uniform)
    if not uniform:
        op.reference_path[:] = torch.tensor([3.9,4.1])
    xyz = torch.tensor([[0.,0.,0.],[.01,.02,0.],[-.02,.01,0.]],dtype=torch.float64)
    n = torch.tensor([[1.,0.,0.],[1.,.1,0.],[1.,0.,.1]],dtype=torch.float64,requires_grad=True)
    sigmas = torch.tensor([[.3,.4,.5],[.6,.7,.8]],dtype=torch.float64,requires_grad=True)
    inside = torch.tensor([True,False,True])
    actual = op.trace(n,sigmas,xyz,op.tx,op.rx,inside)
    terms = []
    for a in range(2):
        out = 0
        for j in range(3):
            inc, ret = xyz[j]-op.tx[a], op.rx[a]-xyz[j]
            distance = inc.norm()+ret.norm()
            inc, ret = inc/(inc.norm()+1e-7), ret/(ret.norm()+1e-7)
            dot = (inc*n[j]).sum()
            reflected = inc-2*dot*n[j]
            reflected = reflected/(reflected.norm()+1e-7)
            alignment = (ret*reflected).sum()
            amp = sigmas[a,j]*alignment/distance.square()/len(xyz) if dot <= 0 and alignment >= 1e-6 and inside[j] else sigmas[a,j]*0
            out = out + amp * torch.exp(-2j*torch.pi*op.frequencies/299792458. * (distance-op.reference_path[a]))
        terms.append(out)
    expected = torch.stack(terms)
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-11)
    actual_g = torch.autograd.grad(actual.abs().square().sum(), (n,sigmas), retain_graph=True)
    expected_g = torch.autograd.grad(expected.abs().square().sum(), (n,sigmas))
    # The released normal Jacobian uses an epsilon approximation, retained by
    # our custom backward; its tiny difference from exact AD is not a repair.
    for a,b in zip(actual_g,expected_g):
        torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-9)
    response = torch.complex(torch.arange(16.).reshape(2,8).double(), torch.ones(2,8).double())
    distance = (xyz[None]-op.tx[:,None]).norm(dim=-1)+(xyz[None]-op.rx[:,None]).norm(dim=-1)-op.reference_path[:,None]
    kernel = torch.exp(2j*torch.pi/299792458.*distance[...,None]*op.frequencies)
    torch.testing.assert_close(op.matched_filter(response,xyz), (kernel*response[:,None]).sum((0,2))/2, rtol=1e-8,atol=1e-7)
    # Source antenna bank reorders channels; actual reference follows the pair.
    torch.testing.assert_close(op.matched_filter(response.flip(0),xyz,op.tx.flip(0),op.rx.flip(0)), op.matched_filter(response,xyz))


def test_source_sampling_random_reproducible_and_full_ray_mask():
    op, r = acquisition(), source.recipe_from_config(SMALL,.15)
    cube = np.ones((5,5,5), np.float32)
    with source.fixed_numpy_seed(9):
        a = source.sample_frame(op,r,'a',cube,cube)
    with source.fixed_numpy_seed(9):
        b = source.sample_frame(op,r,'a',cube,cube)
    with source.fixed_numpy_seed(10):
        c = source.sample_frame(op,r,'a',cube,cube)
    torch.testing.assert_close(a['sampled_poses_norm'],b['sampled_poses_norm'],rtol=0,atol=0)
    assert a['sampled_poses_norm'].shape != c['sampled_poses_norm'].shape or not torch.equal(a['sampled_poses_norm'],c['sampled_poses_norm'])
    model = source.build_model(r)
    keep = torch.arange(len(a['ray_d'])) % 2 == 0
    frame = model._apply_loss_mask(model._extract_batch_inputs(a), keep)
    assert len(frame['sampled_poses_norm']) == int(keep.sum())
    assert len(frame['tgt_sampled_poses_norm']) == int(keep.sum())


class TinyData:
    heads=('scalar',)
    extent=.15
    contract=dict(test_fixture='GeRaF synthetic training lifecycle',role_ids=dict(train=[0,1],validation=[2]))
    identity=identity_hash(contract)
    def __init__(self):
        self.reads=[]
    def views(self, role):
        if role not in self.contract['role_ids']:
            raise PermissionError(role)
        return self.contract['role_ids'][role]
    def key(self,role,view,head):
        if head not in self.heads or view not in self.views(role):
            raise PermissionError('sealed')
        return f'{role}_{head}_{view}'
    def acquisition(self,role,view,head,recipe,device):
        self.key(role,view,head)
        return acquisition()
    def response(self,role,view,head,device):
        self.key(role,view,head); self.reads.append((role,view))
        return torch.ones(2,8,dtype=torch.complex128)


def compare_nested(a,b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: compare_nested(a[key],b[key])
    elif isinstance(a, (tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): compare_nested(x,y)
    else:
        assert a==b


@pytest.mark.parametrize('stop_update', [1, 2])
def test_real_source_loss_training_signal_stop_resume_and_selection(tmp_path, monkeypatch, stop_update):
    data=TinyData()
    uninterrupted=tmp_path/'uninterrupted'
    assert runtime.train(data=data,output_dir=uninterrupted,config=SMALL,device='cpu')['status']=='complete'
    interrupted=tmp_path/'interrupted'
    original=torch.optim.AdamW.step
    calls=[0]
    def stop_after_step(optimizer,*args,**kwargs):
        result=original(optimizer,*args,**kwargs)
        calls[0]+=1
        if calls[0]==stop_update:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
        return result
    with monkeypatch.context() as m:
        m.setattr(torch.optim.AdamW,'step',stop_after_step)
        assert runtime.train(data=data,output_dir=interrupted,config=SMALL,device='cpu')['status']=='interrupted'
    resumed=runtime.train(data=data,output_dir=interrupted,config=SMALL,device='cpu',resume='auto')
    assert resumed['status']=='complete'
    a=torch.load(uninterrupted/'checkpoint_latest.pth.tar',weights_only=False)
    b=torch.load(interrupted/'checkpoint_latest.pth.tar',weights_only=False)
    compare_nested(a['models'],b['models'])
    compare_nested(a['optimizer'],b['optimizer'])
    assert a['validation_history']==b['validation_history']
    assert set(a['exposures'].values())=={1}
    assert a['complete'] and a['validation_history'][-1]['masked'] is False
    assert a['models']['scalar']['ant_chunk_id']=={'train_scalar_0':1,'train_scalar_1':1}
    assert not any(v not in (0,1,2) for _,v in data.reads)
    recipe=source.recipe_from_config(SMALL,data.extent)
    for patch in ({'data_identity':'other'},{'step':1},{'complete':False}, {'pending_validation':2}):
        broken={**a,**patch}
        with pytest.raises(ValueError): runtime.validate_checkpoint(broken,data,recipe)
    broken=copy.deepcopy(a)
    broken['models']['scalar']['ant_real_list']['train_scalar_0'][0]=torch.zeros(1,7)
    with pytest.raises(ValueError,match='antenna tensor'):
        runtime.validate_checkpoint(broken,data,recipe)
    selected=torch.load(uninterrupted/'checkpoint_best.pth.tar',weights_only=False)
    models,row=runtime.load_selected_models(selected,data,'cpu')
    before=copy.deepcopy(models['scalar'].state_dict())
    targets=runtime.SourceTargets(uninterrupted/'source_targets',data,recipe)
    metrics=runtime.evaluate(models,data,recipe,targets,'cpu',lambda:False)
    assert metrics['mf_magnitude_mse']==row['mf_magnitude_mse']
    compare_nested(before,models['scalar'].state_dict())
    wrong=tmp_path/'wrong.pth'; torch.save({**a,'data_identity':'wrong'},wrong)
    data.response=lambda *a,**k: pytest.fail('response read before identity rejection')
    with pytest.raises(ValueError,match='mismatch'):
        runtime.train(data=data,output_dir=tmp_path/'wrongrun',config=SMALL,device='cpu',resume=wrong)


def test_cache_is_train_only_source_bound_and_tamper_evident(tmp_path):
    data=TinyData(); r=source.recipe_from_config(SMALL,data.extent)
    cache=runtime.SourceTargets(tmp_path,data,r); cache.start()
    accumulated=cache.accumulated('scalar','cpu',lambda:False)
    xyz = runtime.lattice_points(np.arange(r['mf_grid']**3), r['mf_grid'], data.extent, 'cpu')
    dense = runtime.measured_values(acquisition(), torch.ones(2,8,dtype=torch.complex128), xyz, r['trans_power']).reshape((r['mf_grid'],)*3)
    expected = np.add(dense, dense)
    np.testing.assert_array_equal(accumulated,expected)
    assert data.reads==[('train',0),('train',1)]
    assert [p.name for p in tmp_path.glob('*.npy')] == ['accumulated_train_scalar.npy']
    assert not list(tmp_path.glob('*.partial.pt'))
    with pytest.raises(PermissionError): cache.get('reserved_test',3,'scalar','cpu',lambda:False)
    file=tmp_path/'accumulated_train_scalar.npy'
    changed=np.load(file).copy(); changed.flat[0]+=1; np.save(file,changed)
    with pytest.raises(ValueError,match='checksum'): cache.accumulated('scalar','cpu',lambda:False)
    with pytest.raises(ValueError,match='identity'):
        runtime.SourceTargets(tmp_path,data,{**r,'extent_m':2}).start()


def test_lazy_targets_match_dense_frame_values_and_masks_without_view_files(tmp_path):
    data = TinyData(); r = source.recipe_from_config(SMALL, data.extent)
    cache = runtime.SourceTargets(tmp_path, data, r); cache.start()
    xyz = runtime.lattice_points(np.arange(r['mf_grid']**3), r['mf_grid'], data.extent, 'cpu')
    dense = runtime.measured_values(acquisition(), torch.ones(2,8,dtype=torch.complex128), xyz, r['trans_power']).reshape((r['mf_grid'],)*3)
    lazy = cache.get('train', 0, 'scalar', 'cpu', lambda: False)
    for seed in (3, 42, 71):
        with source.fixed_numpy_seed(seed):
            reference = source.sample_frame(acquisition(), r, 'same', dense, dense*2)
        with source.fixed_numpy_seed(seed):
            actual = source.sample_frame(acquisition(), r, 'same', lazy, dense*2)
        for key in actual:
            if key == 'mf_sampled_value':
                torch.testing.assert_close(actual[key], reference[key], rtol=2e-6, atol=5e-7)
            elif torch.is_tensor(actual[key]):
                torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
        # Validation reuses current-view targets for the unused mask input.
        with source.fixed_numpy_seed(seed):
            val = source.sample_frame(acquisition(), r, 'same', lazy, lazy)
        torch.testing.assert_close(val['mf_sampled_value'], reference['mf_sampled_value'], rtol=2e-6, atol=5e-7)
    assert not list(tmp_path.glob('*.npy'))
    assert data.reads == [('train', 0)]


@pytest.mark.parametrize('interrupt_mid_view', [False, True])
def test_accumulation_resume_commits_complete_views_only(tmp_path, monkeypatch, interrupt_mid_view):
    data=TinyData(); r=source.recipe_from_config(SMALL,data.extent)
    cache=runtime.SourceTargets(tmp_path,data,r); cache.start()
    save=cache._save_partial
    committed=[False]; post_commit_checks=[0]
    def tracked_save(*args):
        save(*args); committed[0]=True
    monkeypatch.setattr(cache,'_save_partial',tracked_save)
    def stopped():
        if not committed[0]: return False
        post_commit_checks[0]+=1
        # Beginning of next view vs after one of that view's lattice chunks.
        return post_commit_checks[0] >= (4 if interrupt_mid_view else 1)
    with pytest.raises(runtime.InterruptedPreparation):
        cache.accumulated('scalar','cpu',stopped)
    path=tmp_path/'accumulated_train_scalar.partial.pt'
    saved=torch.load(path,weights_only=True)
    assert saved['completed_views']==1
    assert not list(tmp_path.glob('*.npy'))
    resumed=TinyData()
    next_cache=runtime.SourceTargets(tmp_path,resumed,r);next_cache.start()
    actual=next_cache.accumulated('scalar','cpu',lambda:False)
    assert resumed.reads==[('train',1)]
    xyz=runtime.lattice_points(np.arange(r['mf_grid']**3),r['mf_grid'],data.extent,'cpu')
    dense=runtime.measured_values(acquisition(),torch.ones(2,8,dtype=torch.complex128),xyz,r['trans_power']).reshape(actual.shape)
    np.testing.assert_array_equal(actual,np.add(dense,dense))
    assert not path.exists()


def test_partial_accumulation_identity_and_integrity_gate_before_responses(tmp_path):
    data=TinyData();r=source.recipe_from_config(SMALL,data.extent)
    cache=runtime.SourceTargets(tmp_path,data,r);cache.start()
    name='accumulated_train_scalar'
    keys=[data.key('train',v,'scalar') for v in data.views('train')]
    cache._save_partial(name,keys,1,np.ones((r['mf_grid'],)*3,dtype=np.float32))
    path=cache._partial_path(name);saved=torch.load(path,weights_only=True)
    data.response=lambda *a,**k:pytest.fail('response read before partial validation')
    for patch in ({'identity':'wrong'}, {'completed_views':3}, {'train_keys':keys[::-1]}, {'sha256':'bad'}):
        torch.save({**saved,**patch},path)
        with pytest.raises(ValueError): cache.accumulated('scalar','cpu',lambda:False)


def test_old_dense_cache_and_checkpoint_recipes_are_rejected(tmp_path):
    data=TinyData();r=source.recipe_from_config(SMALL,data.extent)
    old={k:v for k,v in r.items() if k!='target_storage'}
    with pytest.raises(ValueError,match='lazy-storage'):
        runtime.SourceTargets(tmp_path,data,old)
    (tmp_path/'source_targets.json').write_text(json.dumps(dict(schema=source.SCHEMA+'_targets',contract=data.contract,recipe=old)))
    with pytest.raises(ValueError,match='identity'):
        runtime.SourceTargets(tmp_path,data,r).start()
    with pytest.raises(ValueError,match='mismatch'):
        runtime.validate_checkpoint(dict(schema=source.SCHEMA+'_checkpoint', data_identity=data.identity,
            contract=data.contract, recipe=old), data, r)


def test_prepare_only_writes_one_accumulator_and_never_reads_validation(tmp_path):
    data=TinyData()
    result=runtime.train(data=data,output_dir=tmp_path,config=SMALL,device='cpu',prepare_only=True)
    assert result['persistent_per_view_volumes']==0
    assert data.reads==[('train',0),('train',1)]
    assert [p.name for p in (tmp_path/'source_targets').glob('*.npy')]==['accumulated_train_scalar.npy']
    assert not list(tmp_path.glob('checkpoint*'))


@pytest.mark.parametrize("grid", [48, 101])
def test_selected_target_protocol_creates_one_small_accumulated_file(tmp_path, grid):
    class Data101(TinyData):
        def acquisition(self,role,view,head,recipe,device):
            self.key(role,view,head)
            op=acquisition(); op.point_chunk=65536
            return op
    protocol=json.loads((Path(__file__).resolve().parents[1]/f'protocols/geraf_mf{grid}.json').read_text())
    assert protocol=={'mf_grid':grid}
    r=source.recipe_from_config({**protocol,'point_chunk':65536},.15)
    data=Data101();cache=runtime.SourceTargets(tmp_path,data,r);cache.start()
    value=cache.accumulated('scalar','cpu',lambda:False)
    assert value.shape==(grid,grid,grid) and value.dtype==np.float32
    paths=list(tmp_path.glob('*.npy'))
    assert len(paths)==1 and paths[0].stat().st_size==4*grid**3+128
    assert not list(tmp_path.glob('*.partial.pt'))
    query=np.array([[0.,0.,0.],[1.,-.1,.2],[-1.01,.3,-.2]],dtype=np.float64)
    lazy=cache.get('validation',2,'scalar','cpu',lambda:False)
    from rift.vendor.geraf_sens.loading import _sample_volume
    # All fixture views use the same response; accumulated is exactly two copies.
    np.testing.assert_allclose(lazy.sample(query),_sample_volume(value/2,query,'bilinear',True),rtol=2e-6,atol=5e-7)
    assert list(tmp_path.glob('*.npy'))==paths


def test_recover_complete_partial_after_interrupted_final_commit(tmp_path, monkeypatch):
    data=TinyData();r=source.recipe_from_config(SMALL,data.extent)
    cache=runtime.SourceTargets(tmp_path,data,r);cache.start()
    def fail(*args): raise runtime.InterruptedPreparation()
    with monkeypatch.context() as m:
        m.setattr(cache,'_finish',fail)
        with pytest.raises(runtime.InterruptedPreparation):
            cache.accumulated('scalar','cpu',lambda:False)
    assert not list(tmp_path.glob('*.npy'))
    assert torch.load(cache._partial_path('accumulated_train_scalar'),weights_only=True)['completed_views']==2
    data.response=lambda *a,**k:pytest.fail('complete partial must not reread responses')
    assert cache.accumulated('scalar','cpu',lambda:False).shape==(5,5,5)
    assert not list(tmp_path.glob('*.partial.pt'))


@pytest.mark.parametrize('name',['a320','x59','firetruck','racecar','loader','b787'])
def test_collection_and_direct_root_default_to_source_without_legacy_preparer(name,tmp_path):
    import train_rift_dataset as collection
    import train_geraf
    commands=collection.commands_for(name,'geraf',dataset_root=tmp_path/'data',output_root=tmp_path/'out')
    assert len(commands)==1
    args=train_geraf.parse_args(commands[0][2:])
    assert args.implementation=='source_v1'
    assert 'source_v1' in str(args.checkpoint_dir)
    assert 'source_v1' in str(args.cache_root)
    assert not args.resume
    legacy=collection.commands_for(name,'geraf',dataset_root=tmp_path/'data',output_root=tmp_path/'out',geraf_implementation='legacy')
    assert len(legacy)==2


def test_gotcha_root_discovery_and_dispatch_forward_the_source_backend(tmp_path,monkeypatch):
    import train_gotcha_dataset as root
    import rift.geraf_gotcha as backend
    spec=root.backend_registry()['geraf']
    assert spec['status']=='available' and spec['native_frequency_policy']=='ragged_exact'
    data=object(); observed={}
    monkeypatch.setattr(backend,'GOTCHASourceData',lambda x: x)
    def run(**kw): observed.update(kw); return {'status':'synthetic_forwarding'}
    monkeypatch.setattr(backend,'train',run)
    result=root.dispatch(data,dict(method='geraf',backend=spec,output_dir=str(tmp_path),config=SMALL,resume=None),'cpu')
    assert result['status']=='synthetic_forwarding'
    assert observed['data'] is data and observed['config']==SMALL


def test_native_gotcha_adapter_preserves_pulses_ragged_frequencies_and_af(tmp_path,monkeypatch):
    from test_gotcha_dataset import write_shard,tiny_region
    from rift.gotcha_dataset import GOTCHADataset
    for p,h,nf in [(1,'hh',8),(2,'hh',9),(1,'hv',10),(2,'hv',11)]:
        write_shard(tmp_path/'New_Transfer'/'shards'/f'pass{p}_{h}.npz',p,h,nf)
    ds=GOTCHADataset(tmp_path,passes=(1,2),polarizations=('hh','hv'),region=tiny_region())
    data=GOTCHASourceData(ds); r=source.recipe_from_config(SMALL,data.extent)
    for head in data.heads:
        for p in (1,2):
            v=next(v for v in data.views('train') if v[0]==p)
            op=data.acquisition('train',v,head,r,'cpu')
            assert len(op.tx)==len(ds.shards[p,head].sector_rows[v[1]])
            assert len(op.frequencies)==ds.shards[p,head].shape[1]
            expected=list(ds.observations(*v,head))
            np.testing.assert_array_equal(op.reference_path.numpy(),2*np.array([x.reference_range_m for x in expected]))
            np.testing.assert_array_equal(data.response('train',v,head,'cpu').numpy(),np.stack([x.response for x in expected]))
    with pytest.raises(PermissionError): data.acquisition('test',(1,ds.split['test'][0]),'hh',r,'cpu')


def test_gotcha_actual_root_dispatch_runs_source_loss_with_all_sector_pulses(tmp_path,monkeypatch):
    from test_gotcha_dataset import write_shard,tiny_region
    from rift.gotcha_dataset import GOTCHADataset
    import train_gotcha_dataset as root
    def duplicate(a,meta):
        for key in list(a):
            if key != 'frequencies_hz':
                a[key]=np.repeat(a[key],2,axis=0)
        a['pulse_index'][1::2]=1
        a['y'][1::2]+=.002
    write_shard(tmp_path/'New_Transfer'/'shards'/'pass1_hh.npz',nf=8,mutate=duplicate)
    ds=GOTCHADataset(tmp_path,passes=(1,),region=tiny_region())
    original=ds.viewpoints
    monkeypatch.setattr(ds,'viewpoints',lambda role: original(role)[:2 if role=='train' else 1])
    spec=root.backend_registry()['geraf']
    result=root.dispatch(ds,dict(method='geraf',backend=spec,output_dir=str(tmp_path/'run'),config=SMALL,resume=None),'cpu')
    assert result['status']=='complete'
    ck=torch.load(tmp_path/'run'/'checkpoint_latest.pth.tar',weights_only=False)
    assert ck['recipe']['implementation']=='source_v1'
    assert ck['validation_history'][-1]['native_count']==16
    assert ck['contract']['native_gotcha_contract']==ds.contract
    assert sum(ck['exposures'].values())==2
    assert not any(s.response_reads and np.any(s.row_roles=='unused') for s in ds.shards.values())


def test_rift_adapter_preserves_cartesian_channel_order_and_lazy_roles(tmp_path,monkeypatch):
    import rift.rift_dataset as ingress
    from rift.geraf_source_data import RIFTSourceData
    from rift.geraf_signal_operator import bistatic_pair_positions
    from rift.npz_dataset import get_npz_response_view
    original=ingress._object_contract('loader')
    # Metadata validates through the real public ingress; response payload is
    # represented by a strict synthetic capability at its first read boundary.
    poses=np.zeros((10000,2,3),dtype=np.float64)
    poses[:,:,0]=2.
    poses[:,1,1]=.01
    arrays=dict(tx_pos=poses,rx_pos=poses+.003,meta=dict(radar_fc_hz=1e9,radar_bandwidth_hz=1e8,num_adc_samples=8))
    monkeypatch.setattr(ingress,'load_object_contract',lambda *a,**k:(arrays,original))
    archive=tmp_path/'fixture.npz'; archive.write_bytes(b'fixture metadata source')
    data=RIFTSourceData(archive,tmp_path/'roles.json')
    train=data.views('train')[0]
    raw=(np.arange(2*2*1*8).reshape(2,2,1,8)+1j).astype(np.complex64)
    reads=[]
    def read(a,v):
        assert v==train; reads.append(v); return raw
    monkeypatch.setattr('rift.npz_dataset.get_npz_response_view',read)
    op=data.acquisition('train',train,'scalar',source.recipe_from_config(SMALL,.15),'cpu')
    assert not reads
    tx,rx=bistatic_pair_positions(torch.tensor(poses[train]),torch.tensor(poses[train]+.003))
    torch.testing.assert_close(op.tx,tx); torch.testing.assert_close(op.rx,rx)
    np.testing.assert_array_equal(data.response('train',train,'scalar','cpu').numpy(),raw[:,:,0,:].reshape(4,8))
    with pytest.raises(PermissionError): data.response('reserved_test',original['role_ids']['reserved_test'][0],'scalar','cpu')


def test_root_rift_main_reaches_shared_source_lifecycle(tmp_path,monkeypatch):
    import train_geraf
    import rift.geraf_source_cli as cli
    config=tmp_path/'source.json'; config.write_text(json.dumps(SMALL))
    monkeypatch.setattr(cli,'RIFTSourceData',lambda *a,**k:TinyData())
    monkeypatch.setattr('sys.argv',['train_geraf.py','--object','a320','--checkpoint-dir',str(tmp_path/'run'),
        '--source-config',str(config),'--device','cpu','--no-resume'])
    result=train_geraf.main()
    assert result['status']=='complete'
    assert (tmp_path/'run'/'checkpoint_best.pth.tar').exists()


def test_source_geometry_load_is_metric_and_never_reads_response(tmp_path,monkeypatch):
    from scripts import eval_geraf_geometry as geometry
    from types import SimpleNamespace
    from rift.rift_dataset import _object_contract
    data=TinyData()
    contract=_object_contract('loader')
    contract['role_manifest_path']=str(tmp_path/'roles.json')
    data.contract={**data.contract,'dataset_identity':contract['dataset_identity']}
    data.identity=identity_hash(data.contract)
    runtime.train(data=data,output_dir=tmp_path/'run',config=SMALL,device='cpu')
    saved=torch.load(tmp_path/'run'/'checkpoint_best.pth.tar',weights_only=False)
    monkeypatch.setattr('rift.geraf_source_data.RIFTSourceData',lambda *a,**k:data)
    data.response=lambda *a,**k:pytest.fail('geometry read a response')
    model,row=geometry.checkpoint_model(saved,SimpleNamespace(arrays=SimpleNamespace(path='metadata-only')),contract,'cpu')
    assert model.extent==.15 and model.implementation=='source_v1'
    xyz=torch.tensor([[0.,0.,0.],[.03,.02,.01]])
    expected=model.sdf_network.network(xyz/.15)[...,0]*.15
    torch.testing.assert_close(model.sdf_network(xyz),expected,rtol=0,atol=0)
    assert row['step']==saved['step']
    foreign=_object_contract('race_car')
    foreign['role_manifest_path']=contract['role_manifest_path']
    with pytest.raises(ValueError,match='object identity'):
        geometry.checkpoint_model(saved,SimpleNamespace(arrays=SimpleNamespace(path='metadata-only')),
                                  foreign,'cpu')


def test_single_pair_bank_loss_gradients_resume_and_readout(tmp_path, monkeypatch):
    from dataclasses import replace
    from rift.antenna_selection import selection
    class OnePair(TinyData):
        contract={**TinyData.contract, 'experiment_contract': {'antenna_selection':selection(1,1)}}
        identity=identity_hash(contract)
        def acquisition(self,role,view,head,recipe,device):
            self.key(role,view,head)
            op=acquisition()
            return replace(op,tx=op.tx[:1],rx=op.rx[:1],reference_path=op.reference_path[:1],pair_chunk=1)
        def response(self,role,view,head,device):
            return super().response(role,view,head,device)[:1]
    data=OnePair()
    recipe=source.recipe_for_data(SMALL,data)
    assert recipe['bank_size']==1 and recipe['antenna_bank_adaptation']=='single_nonempty_bank_v1'
    assert source.recipe_from_config({},.15)['bank_size']==2
    with pytest.raises(ValueError,match='bank_size'): source.recipe_for_data({'bank_size':2},data)
    original=torch.optim.AdamW.step
    gradients=[]
    def check_gradients(optimizer,*args,**kwargs):
        g=[p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        assert g and all(torch.isfinite(v).all() for v in g)
        gradients.append(sum(v.abs().sum().item() for v in g))
        return original(optimizer,*args,**kwargs)
    monkeypatch.setattr(torch.optim.AdamW,'step',check_gradients)
    full=tmp_path/'full'; interrupted=tmp_path/'interrupted'
    assert runtime.train(data=data,output_dir=full,config=SMALL,device='cpu')['status']=='complete'
    calls=[0]
    def stop(optimizer,*args,**kwargs):
        result=check_gradients(optimizer,*args,**kwargs);calls[0]+=1
        if calls[0]==1: signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
        return result
    with monkeypatch.context() as m:
        m.setattr(torch.optim.AdamW,'step',stop)
        assert runtime.train(data=data,output_dir=interrupted,config=SMALL,device='cpu')['status']=='interrupted'
    assert runtime.train(data=data,output_dir=interrupted,config=SMALL,device='cpu',resume='auto')['status']=='complete'
    assert all(v>0 for v in gradients)
    a=torch.load(full/'checkpoint_latest.pth.tar',weights_only=False)
    b=torch.load(interrupted/'checkpoint_latest.pth.tar',weights_only=False)
    compare_nested(a['models'],b['models']);compare_nested(a['optimizer'],b['optimizer'])
    assert a['validation_history']==b['validation_history']
    assert set(a['models']['scalar']['ant_chunk_id'].values())=={0}
    for banks in a['models']['scalar']['ant_real_list'].values():
        assert len(banks)==1 and banks[0].shape==(1,8)
    runtime.validate_checkpoint(a,data,recipe)
    broken=copy.deepcopy(a)
    broken['models']['scalar']['ant_chunk_id']['train_scalar_0']=1
    with pytest.raises(ValueError): runtime.validate_checkpoint(broken,data,recipe)
    selected=torch.load(full/'checkpoint_best.pth.tar',weights_only=False)
    models,row=runtime.load_selected_models(selected,data,'cpu')
    assert isinstance(models['scalar'],source.SingleBankGeRaFStage1)


def test_selected_geometry_main_uses_public_ingress_not_frozen_mimo_source(collection_source,tmp_path,monkeypatch):
    from scripts import eval_geraf_geometry as geometry
    from rift.rift_dataset import role_manifest
    from rift.antenna_selection import selection
    npz,manifest,_=collection_source
    manifest.write_text(json.dumps(role_manifest('loader',2400,selection(1,1))))
    checkpoint=tmp_path/'selected.pt'
    torch.save({'schema':'rift_geraf_source_v1_checkpoint'},checkpoint)
    monkeypatch.setattr(geometry,'load_b7873200_metadata_source',lambda *a:pytest.fail('frozen full-MIMO loader'))
    class ReachedSelectedGeometry(Exception):pass
    def check(saved,source,contract,device):
        assert source.arrays.path==str(npz)
        assert contract['response_shape']==[10000,1,1,1,600]
        raise ReachedSelectedGeometry
    monkeypatch.setattr(geometry,'checkpoint_model',check)
    with pytest.raises(ReachedSelectedGeometry):
        geometry.main(['--object','loader','--dataset-root',str(tmp_path),'--role-manifest',str(manifest),
                       '--checkpoint',str(checkpoint),'--output-dir',str(tmp_path/'geometry')])
