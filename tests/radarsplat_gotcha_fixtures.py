"""RadarSplat-owned tiny native shards; no dispatcher/other-baseline imports."""
import json
import numpy as np
from rift.gotcha_dataset import C, Region, SOURCE_SCHEMA, sector_split


def tiny_region():
    return Region('fixture','synthetic',(0.,0.,0.),((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),.03,'synthetic RadarSplat test')


def write_shard(path, pass_id=1, pol='hh', nf=11):
    sector=np.repeat(np.arange(1,361,dtype=np.int16),2)
    pulse=np.tile(np.arange(2,dtype=np.int32),360)
    angle=np.radians(sector+.05*pulse)
    xyz=np.stack((20*np.cos(angle),20*np.sin(angle),np.full(len(sector),.1*pass_id)),axis=-1)
    r0=np.linalg.norm(xyz,axis=1)
    freq=np.linspace(9e9,10e9,nf); freq[1::2]+=128_000
    co=pol in ('hh','vv')
    dr=.002*(pulse+1) if co else np.zeros(len(pulse))
    phase=.13*(pulse+1) if co else np.zeros(len(pulse))
    amplitude={'hh':1.,'hv':2.,'vh':3.,'vv':4.}[pol]*np.where(np.isin(sector,sector_split()['validation']),3.,1.)
    response=amplitude[:,None]*np.exp((4j*np.pi/C)*dr[:,None]*freq[None,:]-1j*phase[:,None])
    meta=dict(schema=SOURCE_SCHEMA,pass_id=pass_id,polarization=pol,shard_id=f'pass{pass_id}_{pol}',
        corrections_applied=False,native_frequency_preserved=True,all_rows_role='train',source_file_count=360,
        all_available_sectors_used=True,evaluation_holdout=False,
        phase_reference=dict(frequency_unit='Hz',position_unit='m',range_unit='m',reference_range_field='r0',
            frequency_values='native_stored_exact',geometry_contract='paired_monostatic_tx_equals_rx_same_observation'),
        autofocus=dict(applied=False,official_available=co,mode='source_af_unapplied' if co else 'official_af_absent',
            source_shard_id=f'pass{pass_id}_{pol}' if co else None))
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,metadata_json=json.dumps(meta),frequencies_hz=freq,x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],r0=r0,
        sector_id=sector,pulse_index=pulse,pass_id=np.full(len(sector),pass_id,dtype=np.int16),
        polarization=np.full(len(sector),pol),role=np.full(len(sector),'train'),
        r_correct_raw=dr if co else np.empty(0),ph_correct_raw=phase if co else np.empty(0),response=response.astype(np.complex64))
