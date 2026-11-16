#!/usr/bin/env python3
"""Validate and combine experimental obstacle attempts, preserving failure labels."""
import argparse,hashlib,json
from pathlib import Path
import h5py
import numpy as np
from check_physics_record import check
from qualification import motor_current_from_joint_torque


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_trace(path,meta):
    trace=path.parent/meta['physics_trace']
    with np.load(trace) as p,h5py.File(path) as f:
        force=p['control_force'];n=len(force);rows=len(f['timestamp'])
        obs=(np.arange(rows)+1)*40-1
        # A frame observes the state after the previous physics step; its
        # four commands then execute over the following forty physics steps.
        starts=(np.arange(rows)+1)*40
        indices=starts[:,None,None]+np.arange(4)[None,:,None]*10+np.arange(10)[None,None,:]
        applied=force[indices,6:22]
        expected=f['action'][:].reshape(rows,4,1,16)
        checks=dict(trace_hash=sha(trace)==meta['physics_trace_sha256'],
            finite=all(np.isfinite(p[k]).all() for k in p),
            physics_timing=np.allclose(p['timestamp'],(np.arange(n)+1)*.0005,atol=1e-10),
            aligned_physics=all(len(p[k])==n for k in p),
            observation_before_action=np.allclose(f['sim_base_pos'][:],p['link_pos'][obs,0],atol=2e-7,rtol=0),
            actual_action_all_substeps=np.allclose(applied,expected,atol=1e-5,rtol=0),
            base_force_zero=float(np.max(np.abs(force[:,:6])))<1e-6,
            torque_limit_2khz=float(np.max(np.abs(force[:,6:16])))<=10.5+1e-4,
            motor_current_limit_2khz=float(np.max(np.abs(motor_current_from_joint_torque(force[:,6:16]))))<=30.+1e-3,
            contact_count_matches=int(np.count_nonzero(p['robot_box_contact_count']))==meta['robot_obstacle_contact_steps'])
    return dict(valid=bool(all(checks.values())),checks={k:bool(v) for k,v in checks.items()},physics_steps=n)


def run(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    target=out/'bruce_obstacle_attempts_20261115.h5'
    if target.exists():raise FileExistsError(target)
    paths=sorted(args.search.resolve().glob('recorded_*/attempt_*.h5'))
    reports=[];episodes=[]
    for path in paths:
        with h5py.File(path) as f:meta=json.loads(f.attrs['metadata'])
        report=check(path);report['trace']=verify_trace(path,meta)
        report['recording_valid']=report['recording_valid'] and report['trace']['valid']
        reports.append(report)
        if report['recording_valid']:
            episodes.append(dict(path=path,meta=meta,n=report['frames']))
    if not episodes:raise RuntimeError('No validated recordings')
    lengths=np.array([e['n'] for e in episodes]);offsets=np.r_[0,np.cumsum(lengths)[:-1]]
    total=int(lengths.sum())
    with h5py.File(episodes[0]['path']) as first:
        keys=[k for k in first if k not in ('ep_len','ep_offset','ep_idx','step_idx')]
        shapes={k:first[k].shape[1:] for k in keys};dtypes={k:first[k].dtype for k in keys}
    with h5py.File(target,'x') as f:
        for key in keys:f.create_dataset(key,shape=(total,*shapes[key]),dtype=dtypes[key],compression='gzip')
        for i,(ep,offset) in enumerate(zip(episodes,offsets)):
            sl=slice(int(offset),int(offset+ep['n']))
            with h5py.File(ep['path']) as source:
                for key in keys:f[key][sl]=source[key][:]
        f.create_dataset('ep_len',data=lengths);f.create_dataset('ep_offset',data=offsets)
        f.create_dataset('ep_idx',data=np.repeat(np.arange(len(episodes)),lengths))
        f.create_dataset('step_idx',data=np.concatenate([np.arange(n) for n in lengths]))
        f.attrs['purpose']='failed_obstacle_attempts_not_success_demonstrations'
        f.attrs['metadata']=json.dumps(dict(autonomous_crossing_success=False,qualified_success_episodes=0,
            episodes=[e['meta'] for e in episodes],record_hz=50,control_hz=200,physics_hz=2000,
            limitations=['Fixed external camera and simulator truth; no hardware sensor calibration.',
                'All episodes are unsuccessful obstacle attempts; do not treat them as success demonstrations.',
                'Legacy runs changed only robot friction; only episodes with material_friction have verified effective friction.']),ensure_ascii=False)
    scenarios=[];manifest=[]
    for i,ep in enumerate(episodes):
        m=ep['meta'];scenarios.append(dict(episode=i,source=str(ep['path']),frames=ep['n'],
            obstacle=m['obstacle'],robot_material_friction=m['friction'],
            effective_floor_robot_friction=m.get('effective_floor_robot_friction'),
            effective_friction_verified='material_friction' in m,
            stop=m['stop'],robot_obstacle_contact_steps=m['robot_obstacle_contact_steps'],
            autonomous_crossing_success=False))
        for path in (ep['path'],ep['path'].parent/m['physics_trace'],ep['path'].with_suffix('.mp4')):
            manifest.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
    summary=dict(episodes=len(episodes),frames=total,recorded_seconds=total/50,
        physics_steps=sum(r['trace']['physics_steps'] for r in reports if r['recording_valid']),
        qualified_success_episodes=0,excluded_recordings=[r['path'] for r in reports if not r['recording_valid']],
        dataset=str(target),dataset_sha256=sha(target),scenarios=scenarios)
    (out/'verification.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='scenarios'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--search',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
