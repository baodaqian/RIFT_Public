"""Synthetic acquisition/dispatch checks; no real training or test reads."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train_gotcha_dataset as cli
from rift.gotcha_dataset import (C, GOTCHADataset, NativeShardReader, Observation,
                                Region, SOURCE_SCHEMA, load_region, sector_split,
                                validate_checkpoint)
from rift.gotcha_training import (ChannelField, RangeReadout, native_forward,
                                 recipe_from_args, train, backproject)


def write_shard(path, pass_id=1, pol='hh', nf=32, mutate=None):
    """One native pulse per sector, with exact source-like metadata."""
    sector = np.arange(1,361,dtype=np.int16)
    angle = sector*np.pi/180
    positions = np.stack((20*np.cos(angle),20*np.sin(angle),np.full(360,pass_id*.1)),axis=1)
    r0 = np.linalg.norm(positions,axis=1)
    freqs = np.linspace(9e9,10e9,nf,dtype=np.float64)
    freqs[1::2] += 128  # genuinely nonuniform native grid
    co_pol = pol in ('hh','vv')
    meta=dict(schema=SOURCE_SCHEMA,pass_id=pass_id,polarization=pol,
              shard_id=f'pass{pass_id}_{pol}',corrections_applied=False,
              native_frequency_preserved=True,all_rows_role='train',source_file_count=360,
              all_available_sectors_used=True,evaluation_holdout=False,
              phase_reference=dict(frequency_unit='Hz',position_unit='m',range_unit='m',
                  reference_range_field='r0',frequency_values='native_stored_exact',
                  geometry_contract='paired_monostatic_tx_equals_rx_same_observation'),
              autofocus=dict(applied=False,official_available=co_pol,
                  mode='source_af_unapplied' if co_pol else 'official_af_absent',
                  source_shard_id=f'pass{pass_id}_{pol}' if co_pol else None))
    arrays=dict(frequencies_hz=freqs,x=positions[:,0],y=positions[:,1],z=positions[:,2],r0=r0,
                sector_id=sector,pulse_index=np.zeros(360,dtype=np.int32),
                pass_id=np.full(360,pass_id,dtype=np.int16),polarization=np.full(360,pol),
                role=np.full(360,'train'),r_correct_raw=np.full(360,.002) if co_pol else np.empty(0),
                ph_correct_raw=np.full(360,.13) if co_pol else np.empty(0),
                response=np.ones((360,nf),dtype=np.complex64))
    if mutate:
        mutate(arrays,meta)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,**arrays,metadata_json=json.dumps(meta))


def tiny_region():
    return Region('fixture','synthetic',(0.,0.,0.),((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),.03,'synthetic unit test')


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.shards=self.root/'New_Transfer'/'shards'
        write_shard(self.shards/'pass1_hh.npz')

    def tearDown(self):
        self.temp.cleanup()

    def test_split_exact_disjoint_shared_counts(self):
        split=sector_split()
        self.assertEqual([8*len(split[r]) for r in ('train','validation','test')],[2000,440,440])
        self.assertEqual(set().union(*map(set,split.values())),set(range(1,361)))
        self.assertFalse(set(split['train']) & set(split['validation']))
        self.assertFalse(set(split['test']) & (set(split['train']) | set(split['validation'])))
        self.assertEqual(split,sector_split())
        with self.assertRaises(ValueError): sector_split(43)

    def test_default_region_matches_historical_working_frame(self):
        from rift.gotcha_native_complex_roi_fit import CAMRY_PLACEMENT
        r=load_region()
        np.testing.assert_array_equal(r.translation_m,CAMRY_PLACEMENT.translation_m)
        np.testing.assert_array_equal(r.rotation_local_to_native,CAMRY_PLACEMENT.rotation)
        self.assertEqual(r.half_extent_m,5.)

    def test_custom_region_extension_and_override_rejection(self):
        r=tiny_region().as_dict()
        r.pop('name')
        p=self.root/'regions.json'
        p.write_text(json.dumps(dict(schema='rift_gotcha_regions_v1',regions={'fixture':r})))
        self.assertEqual(load_region('fixture',p),tiny_region())
        p.write_text(json.dumps(dict(schema='rift_gotcha_regions_v1',regions={'camry':r})))
        with self.assertRaises(ValueError): load_region('camry',p)
        with self.assertRaises(ValueError):
            Region('bad','x',(0,0,0),((2,0,0),(0,1,0),(0,0,1)),1,'fixture')

    def test_metadata_preflight_never_maps_or_loads_response(self):
        original=np.lib.npyio.NpzFile.__getitem__
        def guarded(source,key):
            if key in ('response','response.npy'):
                raise AssertionError('eager response read')
            return original(source,key)
        with patch.object(np.lib.npyio.NpzFile,'__getitem__',guarded), patch('numpy.memmap',side_effect=AssertionError('mapped response')):
            ds=GOTCHADataset(self.root,passes=(1,),region=tiny_region())
            self.assertFalse(ds.summary()['response_payload_read'])
            self.assertEqual(ds.summary()['viewpoints'],dict(train=250,validation=55,test=55))

    def test_test_denial_precedes_response_mapping(self):
        reader=NativeShardReader(self.shards/'pass1_hh.npz',1,'hh')
        row=int(reader.sector_rows[sector_split()['test'][0]][0])
        with patch('numpy.memmap',side_effect=AssertionError('mapped forbidden response')):
            with self.assertRaises(PermissionError): reader.read(row)
        self.assertEqual(reader.response_reads,0)
        with self.assertRaises(PermissionError):
            NativeShardReader(self.shards/'pass1_hh.npz',1,'hh',roles=('test',))

    def test_validation_only_reader_cannot_read_training(self):
        reader=NativeShardReader(self.shards/'pass1_hh.npz',1,'hh',roles=('validation',))
        with self.assertRaises(PermissionError): reader.read(int(reader.sector_rows[sector_split()['train'][0]][0]))

    def test_native_values_and_own_af_exactly_once(self):
        reader=NativeShardReader(self.shards/'pass1_hh.npz',1,'hh')
        row=int(reader.sector_rows[sector_split()['train'][0]][0])
        obs=reader.read(row)
        self.assertEqual(obs.reference_range_m,float(reader.arrays['r0'][row])+.002)
        np.testing.assert_allclose(obs.response,np.exp(.13j),atol=1e-15)
        np.testing.assert_array_equal(obs.frequencies_hz,reader.arrays['frequencies_hz'])
        self.assertEqual(obs.autofocus,'own_published_source_af_once')
        np.testing.assert_array_equal(obs.response,reader.read(row).response)

    def test_crosspol_is_raw_and_not_borrowed(self):
        write_shard(self.shards/'pass1_hv.npz',pol='hv',nf=35)
        reader=NativeShardReader(self.shards/'pass1_hv.npz',1,'hv')
        row=int(reader.sector_rows[sector_split()['train'][0]][0])
        obs=reader.read(row)
        self.assertEqual(len(obs.response),35)
        self.assertEqual(obs.reference_range_m,reader.arrays['r0'][row])
        np.testing.assert_array_equal(obs.response,np.ones(35))

    def test_all_passes_ragged_frequencies_and_polarization_socket(self):
        for p in range(1,9):
            for pol in ('hh','vv'):
                write_shard(self.shards/f'pass{p}_{pol}.npz',p,pol,nf=32+p)
        ds=GOTCHADataset(self.root,polarizations=('hh','vv'),region=tiny_region())
        self.assertEqual(ds.summary()['viewpoints'],dict(train=2000,validation=440,test=440))
        self.assertEqual(ds.shards[1,'hh'].shape[1],33)
        self.assertEqual(ds.shards[8,'hh'].shape[1],40)
        self.assertEqual(len(ds.viewpoints('train')),2000)
        for shard in ds.shards.values():
            self.assertEqual(set(shard.arrays['sector_id'][shard.row_roles=='test']),set(ds.split['test']))

    def test_malformed_metadata_and_rows_fail_before_response(self):
        mutations=[lambda a,m:m.update(pass_id=2),
                   lambda a,m:m.update(corrections_applied=True),
                   lambda a,m:a['sector_id'].__setitem__(0,2),
                   lambda a,m:a['r0'].__setitem__(0,np.nan),
                   lambda a,m:m['autofocus'].update(source_shard_id='pass2_hh')]
        for change in mutations:
            with self.subTest(change=change):
                write_shard(self.shards/'pass1_hh.npz',mutate=change)
                with patch('numpy.memmap',side_effect=AssertionError('mapped response')):
                    with self.assertRaises(ValueError): NativeShardReader(self.shards/'pass1_hh.npz',1,'hh')

    def test_resume_binds_source_region_split_and_recipe(self):
        ds=GOTCHADataset(self.root,passes=(1,),region=tiny_region())
        recipe={'method':'rift'}
        ck=dict(schema='rift_gotcha_checkpoint_v1',dataset_contract=ds.contract,dataset_identity=ds.identity,recipe=recipe)
        validate_checkpoint(ck,ds,recipe)
        for key,value in [('schema','historical'),('dataset_identity','foreign'),('recipe',{'method':'other'})]:
            bad=copy.deepcopy(ck);bad[key]=value
            with self.assertRaises(ValueError): validate_checkpoint(bad,ds,recipe)
        bad=copy.deepcopy(ck);bad['dataset_contract']['region']['half_extent_m']=2
        with self.assertRaises(ValueError): validate_checkpoint(bad,ds,recipe)
        self.assertFalse(ds.summary()['response_payload_read'])

    def test_archive_mutation_after_preflight_is_rejected(self):
        reader=NativeShardReader(self.shards/'pass1_hh.npz',1,'hh')
        write_shard(self.shards/'pass1_hh.npz',nf=36)
        with self.assertRaises(ValueError): reader.read(int(reader.sector_rows[sector_split()['train'][0]][0]))


class PhysicsTests(unittest.TestCase):
    def test_native_phase_and_position_gradient(self):
        x=torch.tensor([[.01,-.02,.03],[-.03,0,.01]],dtype=torch.float64,requires_grad=True)
        w=torch.tensor([1+.2j,-.1+.3j],dtype=torch.complex128,requires_grad=True)
        a=torch.tensor([20.,-10.,4.],dtype=torch.float64)
        f=torch.tensor([9e9,9.13e9,9.6e9],dtype=torch.float64)
        r0=21.5
        actual=native_forward(x,w,a,f,r0,point_chunk=1)
        expected=np.exp(-4j*np.pi/C*(np.linalg.norm(x.detach().numpy()-a.numpy(),axis=1)-r0)[:,None]*f.numpy())
        np.testing.assert_allclose(actual.detach().numpy(),w.detach().numpy()@expected,rtol=2e-10,atol=2e-10)
        self.assertTrue(torch.autograd.gradcheck(lambda points,weights:native_forward(points,weights,a,f,r0,point_chunk=1),(x,w),eps=1e-6,atol=1e-5,rtol=1e-4))

    def test_range_projection_orthogonal_and_retains_roi_point(self):
        region=tiny_region()
        f=np.linspace(9e9,10e9,64);f[1::2]+=128
        obs=Observation(1,'hh',2,0,np.array([20.,1.,2.]),f,20.,np.ones(64,dtype=complex),'synthetic')
        reader=RangeReadout(region)
        r=reader.for_observation(obs)
        q=r['q']
        torch.testing.assert_close(q.conj().T@q,torch.eye(q.shape[1],dtype=torch.complex128),atol=1e-12,rtol=1e-12)
        x=torch.tensor([[.01,-.01,.015]],dtype=torch.float64)
        y=native_forward(x,torch.ones(1,dtype=torch.complex128),r['antenna'],r['frequencies'],obs.reference_range_m)
        proj=reader.project(y,r)
        retained=float(proj.abs().square().sum()/y.abs().square().sum())
        self.assertGreater(retained,.99)
        recovered=reader.project(reader.lift(proj,r),r)
        torch.testing.assert_close(proj,recovered,atol=1e-12,rtol=1e-12)

    def test_native_field_controls_backward(self):
        args=cli.parse_args(['--granularity','2','--max-points','15','--sh-degree','1'])
        f=np.linspace(9e9,10e9,32)
        obs=Observation(1,'hh',2,0,np.array([20.,1.,2.]),f,20.,np.ones(32,dtype=complex),'synthetic')
        for method in ('rift','rift_grid','isotropic'):
            with self.subTest(method=method):
                recipe=recipe_from_args(args,method)
                field=ChannelField(method,tiny_region(),recipe,'cpu')
                r=RangeReadout(tiny_region()).for_observation(obs)
                y,_=field(obs,r)
                y.abs().square().mean().backward()
                self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in field.parameters()))


class DispatcherTests(unittest.TestCase):
    def test_defaults_match_user_selection(self):
        args=cli.parse_args([])
        self.assertEqual(args.passes,tuple(range(1,9)))
        self.assertEqual(args.polarizations,('hh',))
        self.assertEqual(args.region,'camry')

    def test_unavailable_methods_are_explicit(self):
        # A concurrent baseline owner may register a real hook at any time.
        with tempfile.TemporaryDirectory() as temp, patch.object(cli,'PROJECT_ROOT',Path(temp)):
            registry=cli.backend_registry()
        with self.assertRaisesRegex(ValueError,'native GOTCHA backend'):
            cli.resolve_methods(['spinr'],registry,['hh'])
        self.assertIn('rift',cli.resolve_methods(['all'],registry,['hh']))

    def test_opt_in_hook_discovery_is_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            descriptor=dict(schema=cli.BACKEND_SCHEMA,method='spinr',callable='run_gotcha',
                            selection_unit='pass_sector',joint_passes=True,
                            native_frequency_policy='ragged_exact',polarizations=['hh'],metric_domain='native_complex')
            (root/'train_spinr_style.py').write_text('raise RuntimeError("must not import during planning")\nGOTCHA_BACKEND = '+repr(descriptor)+'\ndef run_gotcha(*, dataset, output_dir, config, device, resume):\n    return {}\n')
            with patch.object(cli,'PROJECT_ROOT',root):
                registry=cli.backend_registry()
            self.assertEqual(registry['spinr']['status'],'available')
            self.assertIn('spinr',cli.resolve_methods(['all'],registry,['hh']))
            with self.assertRaises(ValueError):cli.resolve_methods(['spinr'],registry,['vv'])

    def test_dry_run_and_unallocated_run_do_not_write_or_read_responses(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_shard(root/'New_Transfer/shards/pass1_hh.npz')
            output=root/'outputs'
            argv=['--dataset-root',str(root),'--passes','1','--output-root',str(output)]
            with patch('numpy.memmap',side_effect=AssertionError('response read')),redirect_stdout(io.StringIO()):
                plan=cli.main(argv+['--dry-run'])
                self.assertEqual(plan['dataset']['viewpoints']['train'],250)
                with patch.dict('os.environ',{},clear=True):
                    with self.assertRaisesRegex(RuntimeError,'allocation'):cli.main(argv)
            self.assertFalse(output.exists())


class LifecycleTests(unittest.TestCase):
    def test_mid_epoch_resume_matches_uninterrupted_fit(self):
        """An interruption preserves order, optimizer and adaptive statistics."""
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_shard(root/'New_Transfer/shards/pass1_hh.npz')
            ds=GOTCHADataset(root,passes=(1,),region=tiny_region())
            original_viewpoints=ds.viewpoints
            ds.viewpoints=lambda role:original_viewpoints(role)[:2 if role=='train' else 1]
            args=cli.parse_args(['--epochs','2','--granularity','2','--max-points','15',
                                 '--sh-degree','1','--refine-every','2','--probe-every','1'])
            recipe=recipe_from_args(args,'rift')
            original_step=torch.optim.AdamW.step
            def interrupt_after_step(optimizer,*args,**kwargs):
                result=original_step(optimizer,*args,**kwargs)
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
                return result
            with redirect_stdout(io.StringIO()):
                train(ds,'rift',recipe,root/'reference',device='cpu')
                with patch.object(torch.optim.AdamW,'step',interrupt_after_step):
                    interrupted=train(ds,'rift',recipe,root/'resumed',device='cpu')
                self.assertEqual(interrupted['status'],'interrupted')
                self.assertEqual(interrupted['updates'],1)
                checkpoint=root/'resumed/checkpoint_latest.pt'
                ck=torch.load(checkpoint,weights_only=False)
                self.assertEqual(ck['cursor'],1)
                self.assertEqual(sorted(ck['order']),[0,1])
                # Reject damaged continuation before response reads or writes.
                before=sum(s.response_reads for s in ds.shards.values())
                bad_states=[]
                bad=copy.deepcopy(ck);bad['updates']+=1;bad_states.append(bad)
                bad=copy.deepcopy(ck);bad['order']=[0,0];bad_states.append(bad)
                bad=copy.deepcopy(ck);bad['training_statistics']['hh']['count']*=2;bad_states.append(bad)
                for i,bad in enumerate(bad_states):
                    path=root/f'bad{i}.pt';torch.save(bad,path)
                    with self.assertRaises(ValueError):
                        train(ds,'rift',recipe,root/f'bad_fit{i}',device='cpu',resume=path)
                    self.assertFalse((root/f'bad_fit{i}').exists())
                self.assertEqual(sum(s.response_reads for s in ds.shards.values()),before)
                train(ds,'rift',recipe,root/'resumed',device='cpu',resume=checkpoint)
            full=torch.load(root/'reference/checkpoint_final.pt',weights_only=False)
            resumed=torch.load(root/'resumed/checkpoint_final.pt',weights_only=False)
            self.assertEqual(full['history'],resumed['history'])
            for name,value in full['model_state_dict'].items():
                torch.testing.assert_close(value,resumed['model_state_dict'][name],rtol=0,atol=0)
            for key,state in full['optimizer_state_dict']['state'].items():
                for name,value in state.items():
                    torch.testing.assert_close(value,resumed['optimizer_state_dict']['state'][key][name],rtol=0,atol=0)

    def test_synthetic_training_resume_and_backprojection(self):
        """Two synthetic viewpoints exercise lifecycle; not a research run."""
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_shard(root/'New_Transfer/shards/pass1_hh.npz')
            ds=GOTCHADataset(root,passes=(1,),region=tiny_region())
            original_viewpoints=ds.viewpoints
            ds.viewpoints=lambda role:original_viewpoints(role)[:2 if role=='train' else 1]
            # The per-update refinement cadence exercised here is the earlier optimizer's.
            args=cli.parse_args(['--epochs','1','--granularity','2','--max-points','15',
                                 '--sh-degree','1','--refine-every','2','--probe-every','1','--optimizer','legacy'])
            recipe=recipe_from_args(args,'rift')
            with redirect_stdout(io.StringIO()):
                result=train(ds,'rift',recipe,root/'fit',device='cpu')
            self.assertEqual(result['status'],'complete')
            self.assertTrue((root/'fit/checkpoint_best.pt').is_file())
            ck=torch.load(root/'fit/checkpoint_latest.pt',weights_only=False)
            self.assertEqual(ck['updates'],2)
            self.assertEqual(ck['model_state_dict']['hh.field.refine_event_count'].item(),1)
            before=sum(s.response_reads for s in ds.shards.values())
            with redirect_stdout(io.StringIO()):
                result=train(ds,'rift',recipe,root/'fit',device='cpu',resume=root/'fit/checkpoint_latest.pt')
            self.assertEqual(sum(s.response_reads for s in ds.shards.values()),before)
            self.assertEqual(result['status'],'complete')
            wrong=dict(recipe,sh_degree=2)
            with self.assertRaises(ValueError):train(ds,'rift',wrong,root/'fit',device='cpu',resume=root/'fit/checkpoint_latest.pt')
            result=backproject(ds,recipe,root/'bp',device='cpu')
            self.assertTrue(result['geometry_only'])
            self.assertTrue((root/'bp/support.pt').is_file())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
