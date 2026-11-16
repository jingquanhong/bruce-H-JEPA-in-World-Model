#!/usr/bin/env python3
"""Screen reference contact feasibility; this does not qualify a simulation rollout."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial import ConvexHull

import bruce_expert as bx
from support_geometry import SoleGeometry


def audit():
    trajectory, meta = bx.gen_step_over(h_obs=.08, stride=.28, T_prep=1.,
                                        T_lipm=1., lift_extra=.08)
    qs = [bx.row_to_pin(row) for row in trajectory]
    m = bx._PIN['model']; d = m.createData(); dt = bx.DT_TRAJ
    urdf = Path(bx.URDF)
    pin.forwardKinematics(m,d,qs[0]);pin.updateFramePlacements(m,d)
    level_poses=[d.oMf[m.getFrameId('ankle_pitch_link_'+s)].copy() for s in ('r','l')]
    geometry=SoleGeometry(urdf,level_poses)
    mass=pin.computeTotalMass(m)
    rows=[]
    for i in range(1,len(qs)-1):
        t=i*dt; q=qs[i]
        forward=pin.difference(m,q,qs[i+1]); backward=pin.difference(m,q,qs[i-1])
        v=(forward-backward)/(2*dt); a=(forward+backward)/dt**2
        dh=pin.computeCentroidalMomentumTimeVariation(m,d,q,v,a)
        com=d.com[0].copy()
        force=dh.linear-mass*m.gravity.linear
        moment=dh.angular+np.cross(com,force)
        if force[2]<=1.:continue
        zmp=np.array([-moment[1],moment[0]])/force[2]
        pin.forwardKinematics(m,d,q);pin.updateFramePlacements(m,d)
        supports=[]; ankle=[]
        for k,side in enumerate(('r','l')):
            frame=d.oMf[m.getFrameId('ankle_pitch_link_'+side)]
            ankle.append(frame.translation.copy())
            start,end=meta['swing'][side]
            if not start<t<end:
                supports.extend(geometry.polygon(k,frame))
        if len(supports)<3:continue
        hull=ConvexHull(np.asarray(supports))
        margin=float(np.min(-(hull.equations[:,:2]@zmp+hull.equations[:,2])))
        violation=float(max(np.max(m.lowerPositionLimit[7:]-q[7:]),np.max(q[7:]-m.upperPositionLimit[7:]),0.))
        rows.append(dict(t=t,com=com.tolist(),required_floor_cop=zmp.tolist(),
            support_polygon=np.asarray(supports)[hull.vertices].tolist(),
            floor_cop_margin_m=margin,joint_limit_violation_rad=violation,
            ankle_positions=[p.tolist() for p in ankle]))
    single=[r for r in rows if 1.05<=r['t']<=1.9]
    sample=min(rows,key=lambda r:abs(r['t']-1.2))
    return dict(reference='validate_physics box08 / gen_step_over',
        urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        assumptions=['Flat floor, no obstacle contact, URDF rigid-body centroidal dynamics.',
            'Finite-difference velocities/accelerations at 100 Hz; screening only.',
            'Sole polygons project the original mesh near-sole 0.1 mm band to a flat plane.',
            'This screen neither simulates contacts nor proves controller success.'],
        first_swing_min_floor_cop_margin_m=min(r['floor_cop_margin_m'] for r in single),
        first_swing_lateral_com_to_right_ankle_m=sample['com'][1]-sample['ankle_positions'][0][1],
        max_joint_limit_violation_rad=max(r['joint_limit_violation_rad'] for r in rows),
        reference_support_screen_passed=all(r['floor_cop_margin_m']>=-.005 for r in single),
        sample_1_2_seconds=sample,rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json',type=Path,required=True)
    args=parser.parse_args()
    if args.json.exists():raise FileExistsError(args.json)
    report=audit();args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
