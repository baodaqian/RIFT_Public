"""Selected physical channels, sealed roles and cross-method identity gates."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from rift import rift_dataset as collection
from rift.antenna_selection import selection, select_arrays
from rift.npz_dataset import get_npz_response_view, load_npz_arrays, restrict_npz_response_views
from tests.test_rift_dataset_core import source


@pytest.mark.parametrize('tx,rx', [([0],[0]), ([9,2],[3,11,0]), (list(range(16)),list(range(16)))])
def test_selected_reader_order_geometry_and_roles(tmp_path, tx, rx):
    shape=(3,16,16,1,8)
    raw=(np.arange(np.prod(shape)).reshape(shape)+1j).astype(np.complex64)
    poses=np.arange(3*16*3,dtype=np.float64).reshape(3,16,3)
    path=tmp_path/'synthetic.npz'
    np.savez(path,response=raw,tx_pos=poses,rx_pos=poses+.125,viewpoint_positions=np.ones((3,3)),
             metadata_json=json.dumps(dict(radar_fc_hz=1e10,radar_bandwidth_hz=3e9,num_adc_samples=8)))
    arrays=restrict_npz_response_views(load_npz_arrays(path,load_response=False),[1])
    chosen=select_arrays(arrays,selection(tx_indices=tx,rx_indices=rx))
    np.testing.assert_array_equal(get_npz_response_view(chosen,1),raw[1][np.ix_(tx,rx)])
    np.testing.assert_array_equal(chosen['tx_pos'],poses[:,tx])
    np.testing.assert_array_equal(chosen['rx_pos'],(poses+.125)[:,rx])
    if chosen.get('antenna_selection'):
        np.testing.assert_array_equal(chosen['source_tx_pos'], poses)
        np.testing.assert_array_equal(chosen['source_rx_pos'], poses+.125)
    for i in (0,2):
        with pytest.raises(PermissionError): get_npz_response_view(chosen,i)
    with pytest.raises(PermissionError): restrict_npz_response_views(chosen,[0,1])


@pytest.mark.parametrize('kwargs', [dict(num_tx=0),dict(num_rx=17),dict(num_tx=True),dict(num_tx=1.0),
    dict(tx_indices=[]),dict(tx_indices=[0,0]),dict(rx_indices=[16]),dict(tx_indices=[False]),
    dict(num_tx=2,tx_indices=[0])])
def test_invalid_source_selection(kwargs):
    with pytest.raises(ValueError): selection(**kwargs)


def test_collection_identity_and_resume_bind_indices_and_geometry(source):
    from train import _validate_saved_sealed_npz_protocol_contract
    npz,manifest,original=source
    parent=collection.load_object_contract(npz,manifest)[1]
    arrays,selected=collection.load_object_contract(npz,manifest,num_train=2400,num_tx=1,num_rx=1)
    assert selected['response_shape']==[10000,1,1,1,600]
    assert selected['source_response_shape']==parent['response_shape']
    assert selected['role_ids']['validation']==parent['role_ids']['validation']
    assert selected['role_ids']['reserved_test']==parent['role_ids']['reserved_test']
    assert collection.collection_contract(selected)['antenna_selection']==selection(1,1)
    _validate_saved_sealed_npz_protocol_contract(selected,copy.deepcopy(selected))
    alternate=collection.load_object_contract(npz,manifest,num_train=2400,tx_indices=[1],rx_indices=[0])[1]
    for wrong in (parent,alternate):
        with pytest.raises(ValueError): _validate_saved_sealed_npz_protocol_contract(wrong,selected)
    original['tx_pos'][0,0,1]+=.01
    changed=collection.load_object_contract(npz,manifest,num_train=2400,num_tx=1,num_rx=1)[1]
    with pytest.raises(ValueError): _validate_saved_sealed_npz_protocol_contract(changed,selected)
    manifest.write_text(json.dumps(collection.role_manifest('loader',2400,selection(1,1))))
    with pytest.raises(ValueError,match='conflict'):
        collection.load_object_contract(npz,manifest,num_tx=16,num_rx=16)


def test_six_collection_commands_share_selected_manifest_and_separate_outputs(source, tmp_path):
    import train_rift_dataset as root
    npz,parent,_=source
    args=root.parse_args(['--object','loader','--dataset-root',str(tmp_path),'--output-root',str(tmp_path/'out'),
        '--method','rift','spinr','radar_fields','geraf','radarsplat','sugavanam_ertin','--dry-run'])
    plan=root.make_plan(args)
    assert len(plan['plans'])==6 and plan['antenna_selection']==selection(1,1)
    assert next(p for p in plan['plans'] if p['method']=='geraf')['recipe']['bank_size']==1
    assert len({p['role_manifest_path'] for p in plan['plans']})==1
    assert not (tmp_path/'out').exists()
    for p in plan['plans']:
        assert p['dataset_identity']['response_shape']==[10000,1,1,1,600]
        assert '1t1r_' in p['output_dir']
    cmd=plan['plans'][0]['commands'][0]
    assert cmd[cmd.index('--num-tx')+1]==cmd[cmd.index('--num-rx')+1]=='1'
    args.num_tx=args.num_rx=16
    full=root.make_plan(args)
    assert full['antenna_selection'] is None
    assert full['plans'][0]['output_dir']!=plan['plans'][0]['output_dir']


def test_radar_fields_selected_reader_stats_and_reuse_gate(source,tmp_path,monkeypatch):
    from rift import radar_fields_dataset as rf
    npz,manifest,_=source
    public,contract=collection.load_object_contract(npz,manifest,num_train=40,num_tx=1,num_rx=1)
    arrays=rf.from_collection_arrays(public,contract)
    assert arrays.num_tx==arrays.num_rx==1
    stats=tmp_path/'stats.json'
    monkeypatch.setattr(rf,'estimate_power_peak',lambda *a,**k: 2.)
    value=rf.load_or_create_stats(str(stats),arrays,contract['role_ids']['train'],60.,sealed_protocol=True,
        dataset_identity=contract['dataset_identity'])
    assert value['acquisition_identity']==arrays.acquisition_identity
    wrong=copy.copy(arrays);wrong.acquisition_identity=None
    monkeypatch.setattr(rf,'estimate_power_peak',lambda *a,**k: pytest.fail('read before identity rejection'))
    with pytest.raises(ValueError,match='acquisition'):
        rf.load_or_create_stats(str(stats),wrong,contract['role_ids']['train'],60.,sealed_protocol=True,
            dataset_identity=contract['dataset_identity'])
    selected_manifest=tmp_path/'selected.json'
    selected_manifest.write_text(json.dumps(collection.role_manifest('loader',40,selection(1,1))))
    split=rf.load_radar_fields_sealed_split_manifest(str(selected_manifest),10000,
        response_shape=arrays.response_shape,response_dtype=arrays.response_dtype,
        expected_num_train=40,expected_num_val=1000,expected_num_test=1000)
    assert split.response_shape==(10000,1,1,1,600)


@pytest.mark.parametrize('tx,rx', [([0], [0]), ([7], [3]), ([9,2], [3,11,0]),
                                  ([1,7], [2,6]), (list(range(16)), list(range(16)))])
def test_radar_fields_source_frame_survives_antenna_selection(source, monkeypatch, tx, rx):
    import train_radar_fields as trainer
    from rift.radar_fields_dataset import from_collection_arrays, range_bin_centers
    from rift.radar_fields_native import prepare_bistatic_bins

    npz, manifest, original = source
    # Known simulator frame: boresight -X, Tx +Y, Rx -Z, rotated in world space.
    original['tx_pos'][:, :, 1] = np.linspace(-.1, .1, 16)
    original['rx_pos'][:, :, 2] = np.linspace(.1, -.1, 16)
    angle = .37
    world_rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                               [-np.sin(angle), 0., np.cos(angle)]])
    for key in ('viewpoint_positions', 'tx_pos', 'rx_pos'):
        original[key] = original[key] @ world_rotation.T
    public, contract = collection.load_object_contract(
        npz, manifest, num_train=40, tx_indices=tx, rx_indices=rx)
    arrays = from_collection_arrays(public, contract)
    view = contract['role_ids']['train'][0]
    # Only synthetic responses; the source fixture forbids archive payload reads.
    monkeypatch.setattr(arrays, 'response_view', lambda _: np.ones((len(tx), len(rx), 1, 600), np.complex64))
    args = trainer.parse_args(['--npz-path', str(npz), '--recipe', 'source-adapted-v3'])
    ranges = range_bin_centers(arrays.metadata, dtype=torch.float64)
    pairs = np.array([0, len(tx) * len(rx) - 1, 0])  # Released sampling permits duplicates.
    torch.manual_seed(9)
    actual = trainer.audited_view_tensors(
        None, arrays, view, pairs, ranges, dict(peak_power=1., dynamic_range_db=60.),
        args, 1., torch.device('cpu'), defer_render=True)
    expected_rotation = torch.as_tensor(world_rotation @ np.diag([-1., 1., -1.]))
    selected_tx = torch.as_tensor(original['tx_pos'][view, tx])[pairs // len(rx)]
    selected_rx = torch.as_tensor(original['rx_pos'][view, rx])[pairs % len(rx)]
    torch.manual_seed(9)
    expected = prepare_bistatic_bins(selected_tx, selected_rx, actual['ranges'], extent=args.extent,
                                    ray_samples=args.ray_samples, source_sampling=True,
                                    rotation=expected_rotation)
    geometry = actual['geometry']
    assert geometry['inside'].any()
    for key in ('inside', 'xyz', 'view'):
        torch.testing.assert_close(geometry[key], expected[key], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(geometry['view'].norm(dim=-1), torch.ones(len(geometry['view']), dtype=torch.float64))
    # Independently check that each retained point uses the selected bistatic pair.
    pair, bin_id, _ = geometry['inside'].nonzero(as_tuple=True)
    half_path = ((geometry['xyz'] - selected_tx[pair]).norm(dim=-1)
                 + (geometry['xyz'] - selected_rx[pair]).norm(dim=-1)) / 2
    torch.testing.assert_close(half_path, actual['ranges'][bin_id], rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('failure', ['missing', 'degenerate', 'nonorthogonal'])
def test_radar_fields_rejects_invalid_source_attitude_before_responses(source, monkeypatch, failure):
    import train_radar_fields as trainer
    from rift.radar_fields_dataset import from_collection_arrays

    npz, manifest, original = source
    original['tx_pos'][:, :, 1] = np.linspace(-.1, .1, 16)
    if failure == 'nonorthogonal':
        original['rx_pos'][:, :, 1] = np.linspace(-.1, .1, 16)
    public, contract = collection.load_object_contract(npz, manifest, num_tx=1, num_rx=1)
    if failure == 'missing':
        public.pop('source_tx_pos')
    arrays = from_collection_arrays(public, contract)
    monkeypatch.setattr(arrays, 'response_view', lambda _: pytest.fail('response read before attitude validation'))
    args = trainer.parse_args(['--npz-path', str(npz), '--recipe', 'source-adapted-v3'])
    with pytest.raises(ValueError, match='[Ss]ource.*(geometry|nondegenerate|orthonormal)'):
        trainer.audited_view_tensors(None, arrays, contract['role_ids']['train'][0], np.array([0]),
                                    None, None, args, 1., torch.device('cpu'), defer_render=True)


def test_spinr_selected_metadata_and_recipe(source):
    import train_spinr_style as spinr
    from rift.spinr_style import build_spinr_style_acquisition_identity, _validate_and_average_b787_raw_response
    npz,manifest,_=source
    arrays,contract=collection.load_object_contract(npz,manifest,num_train=40,tx_indices=[9,2],rx_indices=[3])
    identity=build_spinr_style_acquisition_identity(arrays,contract)
    assert identity['tx_pos_m'].shape==(1040,2,3) and identity['rx_pos_m'].shape==(1040,1,3)
    recipe=spinr._recipe_identity('paper-v1-direct',40,contract['antenna_selection'])
    assert recipe['batching']['all_pairs']==2
    raw=np.ones((2,1,1,600),dtype=np.complex64)
    assert _validate_and_average_b787_raw_response(raw,(2,1,1,600)).shape==(2,1,600)


def test_native_gotcha_rejects_nonexistent_antennas_before_shards(tmp_path):
    from rift.gotcha_dataset import GOTCHADataset
    import train_gotcha_dataset as root
    for kw in (dict(num_tx=2),dict(num_rx=2),dict(tx_indices=[1]),dict(rx_indices=[0,0])):
        with pytest.raises(ValueError): GOTCHADataset(tmp_path,**kw)
    for flags in (['--num-tx','2'],['--num-rx','0'],['--tx-indices','1']):
        with pytest.raises(SystemExit): root.parse_args(flags)


@pytest.mark.parametrize('nt,nr',[(1,1),(2,3),(16,16)])
def test_selected_range_forward_gradient_and_adjoint(nt,nr):
    from rift.range_operator import range_forward_operator, range_adjoint_operator
    from rift.forward_operator import get_kvector
    from rift.config import cc
    torch.manual_seed(1)
    f=torch.arange(24,dtype=torch.float64)*5e6+8.5e9
    tx=torch.tensor([[10.,.003*i,.01] for i in range(nt)],dtype=torch.float64)
    rx=torch.tensor([[10.,-.01,.004*i] for i in range(nr)],dtype=torch.float64)
    xyz=torch.rand(3,3,dtype=torch.float64)*.2-.1
    w=torch.randn(3,dtype=torch.complex128,requires_grad=True)
    rt=torch.linalg.vector_norm(xyz[:,None]-tx[None],dim=-1)
    rr=torch.linalg.vector_norm(xyz[:,None]-rx[None],dim=-1)
    distance=rr[:,:,None]+rt[:,None,:]
    kernel=torch.exp(-2j*torch.pi*f[:,None,None,None]*distance[None]/cc)/((4*torch.pi)**2*distance[None].square())
    direct=(kernel*w[None,:,None,None]).sum(1)
    kw=dict(phase_sign=-1.,compute_dtype=torch.float64,range_model='sum2',point_chunk=2,pair_chunk=min(4,nt*nr))
    predicted=range_forward_operator(f,get_kvector(f,cc),rx,tx,xyz,w,**kw)
    torch.testing.assert_close(predicted,direct,rtol=3e-6,atol=1e-10)
    y=torch.randn_like(predicted)
    loss=(predicted.conj()*y).real.sum()
    gradient,=torch.autograd.grad(loss,w)
    adjoint=range_adjoint_operator(f,get_kvector(f,cc),rx,tx,xyz,y,**kw)
    torch.testing.assert_close(adjoint,gradient,rtol=1e-10,atol=1e-12)
    expected,=torch.autograd.grad((direct.conj()*y).real.sum(),w)
    torch.testing.assert_close(gradient,expected,rtol=3e-6,atol=1e-10)


def test_selected_views_reach_rift_spinr_and_rf_without_full_channel_materialization(source,monkeypatch):
    from train import build_sealed_npz_dataloaders, _load_sealed_npz_protocol_contract
    from rift.spinr_style import build_sealed_raw_complex_views
    from rift.radar_fields_dataset import from_collection_arrays
    npz,manifest,_=source
    manifest.write_text(json.dumps(collection.role_manifest('loader',40,selection(tx_indices=[7],rx_indices=[3]))))
    raw=(np.arange(16*16*600).reshape(16,16,1,600)+1j).astype(np.complex64)
    reads=[]
    class Source:
        shape=(10000,16,16,1,600)
        dtype=np.dtype('complex64')
        def __getitem__(self,index):
            reads.append(index)
            return raw
    monkeypatch.setattr('rift.power_baseline_dataset._stored_member_memmap',lambda *a:Source())
    arrays,contract=_load_sealed_npz_protocol_contract(npz,manifest,num_train=40,num_val=1000,num_test=1000,
                                                      num_tx=1,num_rx=1)
    # Matching count-only flags must inherit ordered manifest indices, not reset them to zero.
    assert contract['antenna_selection']['tx_indices']==[7]
    train,val,test,_=build_sealed_npz_dataloaders(npz,manifest,num_train=40,num_val=1000,num_test=1000,
                                               num_tx=1,num_rx=1)
    assert test is None and len(train)==40 and len(val)==1000
    item=next(iter(train))
    assert item[3].shape==(1,600,1) and item[-1].shape==(1,1,3)
    views=build_sealed_raw_complex_views(arrays,contract)
    assert views.raw_training_mean_power()==pytest.approx(float((np.abs(raw[7,3]).astype(np.float64)**2).mean()),rel=1e-6)
    rf=from_collection_arrays(arrays,contract)
    np.testing.assert_array_equal(rf.response_view(contract['role_ids']['train'][0]),raw[7:8,3:4])
    assert set(reads)==set(contract['role_ids']['train']+contract['role_ids']['validation'])


def test_one_pair_point_spread_readout_peaks_at_synthetic_target():
    from rift.range_operator import range_forward_operator, range_adjoint_operator
    from rift.forward_operator import get_kvector
    from rift.config import cc
    f=torch.arange(600,dtype=torch.float64)*5e6+8.5e9
    tx=torch.tensor([[10.,-.1,.03]],dtype=torch.float64)
    rx=torch.tensor([[10.,.1,-.02]],dtype=torch.float64)
    xyz=torch.tensor([[0.,0.,0.]],dtype=torch.float64)
    k=get_kvector(f,cc)
    kw=dict(phase_sign=-1.,compute_dtype=torch.float64,range_model='sum2',point_chunk=32,pair_chunk=1)
    response=range_forward_operator(f,k,rx,tx,xyz,torch.ones(1,dtype=torch.complex128),**kw)
    points=torch.zeros(61,3,dtype=torch.float64);points[:,0]=torch.linspace(-.15,.15,61)
    support=range_adjoint_operator(f,k,rx,tx,points,response,**kw).abs()
    # Bounded localization check, not an unchanged-resolution assertion.
    assert abs(points[support.argmax(),0].item())<=.0051
    assert torch.isfinite(support).all() and support.max()>0
