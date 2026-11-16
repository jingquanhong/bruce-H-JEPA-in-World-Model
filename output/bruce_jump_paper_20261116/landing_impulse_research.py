"""Rigid sticking-impact snapshot on hypothetical flight data; no hardware IO.

No compliant contact, post-impact control, or impact force prediction. Reports
momentum impulses, not peak forces, and cannot validate real robot landing.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
import osqp
from scipy import sparse
from current_model import build

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flight',type=Path,default=HERE/'flight-retraction-research.json')
    parser.add_argument('--out',type=Path,default=HERE/'landing-impulse-research.json')
    args=parser.parse_args()
    model,_=build(HERE/'model-export-corrected.json')
    flight=json.loads(args.flight.read_text())
    frame_names=('right_toe','right_heel','left_toe','left_heel')
    frame_ids=[model.getFrameId(n) for n in frame_names]
    cases=[]
    for trajectory in flight['cases']:
        sample=trajectory['samples'][-1]
        if trajectory['ending'] is None or 'reached floor' not in trajectory['ending']['reason']:
            continue
        q=np.asarray(sample['model_configuration']);v=np.asarray(sample['model_velocity'])
        data=model.createData();H=pin.crba(model,data,q).copy();H=np.triu(H)+np.triu(H,1).T
        pin.computeJointJacobians(model,data,q);pin.updateFramePlacements(model,data)
        Jall=[pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED)[:3] for f in frame_ids]
        pin.centerOfMass(model,data,q,v);before_com_vz=float(data.vcom[0][2]);
        first=int(np.argmin(sample['foot_heights_m']))
        for name,indices in [('first_point',[first]),('first_point_sliding',[first]),('four_points',[0,1,2,3])]:
            J=np.vstack([Jall[i] for i in indices]);W=J@np.linalg.solve(H,J.T);before=J@v
            F=[]
            for i in range(len(indices)):
                row=np.zeros((5,3*len(indices)));j=3*i;row[0,j+2]=1
                row[1:5,j:j+3]=[[1,1,-.5],[1,-1,-.5],[-1,1,-.5],[-1,-1,-.5]];F.append(row)
            equality=W[2:3] if name=='first_point_sliding' else W
            target=-before[2:3] if name=='first_point_sliding' else -before
            A=np.vstack([equality,*F]);lo=np.r_[target,np.tile([0.,-np.inf,-np.inf,-np.inf,-np.inf],len(indices))]
            hi=np.r_[target,np.tile([np.inf,0.,0.,0.,0.],len(indices))]
            solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(W+np.eye(len(before))*1e-8),q=before,
                A=sparse.csc_matrix(A),l=lo,u=hi,verbose=False,eps_abs=1e-6,eps_rel=1e-6,max_iter=30000,polishing=False)
            result=solver.solve()
            case={'foot_clearance_reference_m':trajectory['target_foot_height_m'],'impact_contact_assumption':name,
                  'contact_points':[frame_names[i] for i in indices],'solver_status':result.info.status,
                  'preimpact_com_vz_m_s':before_com_vz,'preimpact_contact_velocity_m_s':before.tolist()}
            if result.info.status=='solved':
                impulse=result.x;vplus=v+np.linalg.solve(H,J.T@impulse)
                pin.centerOfMass(model,data,q,vplus)
                case.update(contact_impulse_N_s=impulse.reshape(-1,3).tolist(),
                            postimpact_com_vz_m_s=float(data.vcom[0][2]),
                            postimpact_contact_velocity_max_m_s=float(abs(J@vplus).max()),
                            postimpact_contact_velocities_m_s=(J@vplus).reshape(-1,3).tolist(),
                            kinetic_energy_dissipated_J=float(.5*v@H@v-.5*vplus@H@vplus),
                            model_configuration=q.tolist(),postimpact_model_velocity=vplus.tolist(),
                            constraint_violation=float(max(np.max(lo-A@impulse),np.max(A@impulse-hi),0.)))
            cases.append(case)
    report={'status':'IDEALIZED_IMPACT_SNAPSHOT_ONLY' if cases else 'NOT_EVALUATED_NO_LANDING_CANDIDATE',
            'source_flight_report':str(args.flight),'physical_jump_validated':False,'hardware_actuation':False,
            'landing_controller_validated':False,'cases':cases,
            'limitations':['Assumed perfectly inelastic rigid impact and friction coefficient; single-point sliding is only an impulse snapshot',
                           'No physical impact duration, force peak, compliance, motor dynamics, terrain or control',
                           'Unmeasured actual mass/inertia and approximate simultaneous contacts',
                           'Impact impulses are not peak forces and cannot certify safe landing']}
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(cases,indent=2))


if __name__=='__main__':main()
