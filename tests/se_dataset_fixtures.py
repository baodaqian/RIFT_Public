"""SE-owned synthetic shard fixtures; independent of other baseline test modules.

Copied from the native-ingress fixtures when isolating the SE entrypoints.
No real data, shared dispatcher imports, or source preprocessing changes.
"""
import json
import numpy as np
from rift.gotcha_dataset import Region, SOURCE_SCHEMA


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
