#!/usr/bin/env python3
"""Check alignment and applied actuation in validate_physics.py recordings."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from qualification import motor_current_from_joint_torque


def check(path):
    with h5py.File(path) as f:
        meta = json.loads(f.attrs['metadata'])
        n = len(f['timestamp'])
        action = f['action'][:].reshape(n, 4, 16)
        force = f['sim_control_force'][:]
        checks = dict(
            nonempty=n > 0,
            aligned_columns=all(len(f[k]) == n for k in f if k not in ('ep_len','ep_offset')),
            pixels=f['pixels'].shape == (n, 64, 64, 3) and f['pixels'].dtype == np.uint8,
            proprio=f['proprio'].shape == (n, 36),
            timing=np.allclose(f['timestamp'][:], np.arange(n)*.02, atol=1e-10),
            episode=int(f['ep_len'][0]) == n and int(f['ep_offset'][0]) == 0,
            finite=all(np.isfinite(f[k][:]).all() for k in f),
            executed_action=np.allclose(action, force[..., 6:22], atol=1e-5, rtol=0),
            no_base_actuation=np.max(np.abs(force[..., :6])) < 1e-6,
            joint_torque_limit=np.max(np.abs(force[..., 6:16])) <= 10.5+1e-4,
            coupled_current_limit=np.max(np.abs(motor_current_from_joint_torque(
                force[..., 6:16]))) <= 30.+1e-3,
            reset_acceleration_cleared=np.linalg.norm(f['proprio'][0, 20:23]) < 100.,
            model_hash=hashlib.sha256(Path(meta['urdf']).read_bytes()).hexdigest() == meta['urdf_sha256'],
        )
        if meta['trial']=='support':
            checks['support_shapes']=all(f[k].shape==shape for k,shape in (
                ('sim_com',(n,3)),('sim_com_velocity',(n,3)),
                ('sim_support_cop',(n,2,3)),('sim_foot_force',(n,4))))
    return dict(path=str(path.resolve()), recording_valid=bool(all(checks.values())),
                checks={k: bool(v) for k, v in checks.items()},
                frames=n, autonomous_crossing_success=meta['autonomous_crossing_success'],
                note='Recording integrity does not imply successful obstacle crossing.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    report = check(args.path)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json:
        args.json.write_text(payload)
    raise SystemExit(0 if report['recording_valid'] else 1)
