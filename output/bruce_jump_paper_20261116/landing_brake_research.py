"""Idealized post-impact support braking screen. No hardware or serial IO.

Starts after an ideal perfectly inelastic four-point impact, assumes rigid
contacts and direct torque control, and stops when downward CoM speed is small.
No impact forces, motor bandwidth, whole recovery, or real safety validation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation
from current_model import build
from integrate_push_research import installed_joint_velocity_map

HERE=Path(__file__).resolve().parent


def run(impact,az,dt=.001):
    model,desc=build(HERE/'model-export-corrected.json')
    M=np.asarray(json.loads((HERE/'khadas-model-screen.json').read_text())['current_map_A_per_Nm'])
    velocity_map,_=installed_joint_velocity_map()
    q=np.asarray(impact['model_configuration']);v=np.asarray(impact['postimpact_model_velocity'])
    names=('right_toe','right_heel','left_toe','left_heel');ids=[model.getFrameId(n) for n in names]
    data=model.createData();pin.framesForwardKinematics(model,data,q)
    origin=np.array([data.oMf[f].translation.copy() for f in ids]);rows=[];failure=None;stopped=False
    for k in range(301):
        t=k*dt;data=model.createData()
        H=pin.crba(model,data,q).copy();H=np.triu(H)+np.triu(H,1).T;h=pin.nonLinearEffects(model,data,q,v).copy()
        pin.forwardKinematics(model,data,q,v,np.zeros(model.nv));pin.computeJointJacobians(model,data,q);pin.updateFramePlacements(model,data)
        J=np.vstack([pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED)[:3] for f in ids])
        bias=np.hstack([pin.getFrameClassicalAcceleration(model,data,f,pin.LOCAL_WORLD_ALIGNED).linear.copy() for f in ids])
        feet=np.hstack([data.oMf[f].translation.copy() for f in ids]);drift=feet-origin.ravel()
        Jcom=pin.jacobianCenterOfMass(model,data,q).copy();pin.centerOfMass(model,data,q,v,np.zeros(model.nv))
        com=data.com[0].copy();vc=data.vcom[0].copy();bias_com=data.acom[0].copy()
        if vc[2]>=-.02:stopped=True;break
        B=np.c_[H[6:],-J[:,6:].T];C=M@B
        A0=np.c_[H[:6],-J[:,:6].T];Afoot=np.c_[J,np.zeros((12,12))];Acom=np.c_[Jcom,np.zeros((3,12))]
        fr=[]
        for j in range(4):
            F=np.zeros((5,28));i=16+j*3;F[0,i+2]=1
            F[1:5,i:i+3]=[[1,1,-.5],[1,-1,-.5],[-1,1,-.5],[-1,-1,-.5]];fr.append(F)
        A=np.vstack([A0,Afoot,Acom,B,C,*fr])
        foot_goal=-100*drift-20*(J@v)-bias
        desired=np.array([0.,0.,az])-bias_com
        lo=np.r_[-h[:6],foot_goal,desired,-np.full(10,desc['torque_max'])-h[6:],
                 -np.full(10,desc['current_max'])-M@h[6:],np.tile([0.,-np.inf,-np.inf,-np.inf,-np.inf],4)]
        hi=np.r_[-h[:6],foot_goal,desired,np.full(10,desc['torque_max'])-h[6:],
                 np.full(10,desc['current_max'])-M@h[6:],np.tile([100.,0.,0.,0.,0.],4)]
        P=np.eye(28)*1e-5;P[:16,:16]+=np.eye(16)*1e-3;P[3,3]+=.1;P[4,4]+=.1;P[5,5]+=.1
        P+=.01*B.T@M.T@M@B;linear=.01*B.T@M.T@M@h[6:]
        solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(P),q=linear,A=sparse.csc_matrix(A),l=lo,u=hi,
            verbose=False,eps_abs=1e-5,eps_rel=1e-5,max_iter=10000,polishing=False)
        sol=solver.solve()
        if sol.info.status!='solved':failure={'time_s':t,'reason':'QP '+sol.info.status};break
        x=sol.x;tau=h[6:]+B@x;constraint=float(max(np.max(lo-A@x),np.max(A@x-hi),0.))
        error=float(np.linalg.norm(drift.reshape(4,3),axis=1).max())
        rows.append({'time_s':t,'com_z_m':float(com[2]),'com_vz_m_s':float(vc[2]),
                     'body_tilt_deg':float(np.degrees(np.arccos(np.clip(Rotation.from_quat(q[3:7]).as_matrix()[2,2],-1,1)))),
                     'max_joint_speed_rad_s':float(abs(v[6:]).max()),'max_joint_acceleration_rad_s2':float(abs(x[6:16]).max()),
                     'motor_speeds_rad_s':(velocity_map@v[6:]).tolist(),
                     'max_motor_acceleration_rad_s2':float(abs(velocity_map@x[6:16]).max()),
                     'max_joint_torque_Nm':float(abs(tau).max()),'max_motor_current_A':float(abs(M@tau).max()),
                     'max_foot_drift_m':error,'constraint_violation':constraint,'joint_positions':q[7:].tolist()})
        if rows[-1]['body_tilt_deg']>8:failure={'time_s':t,'reason':'model torso tilt exceeded 8-degree research bound'};break
        if constraint>.002 or error>.003:failure={'time_s':t,'reason':'contact drift or solver tolerance'};break
        lower=np.asarray(desc['joint_lower']);upper=np.asarray(desc['joint_upper'])
        if np.any(q[7:]<lower+.01) or np.any(q[7:]>upper-.01):failure={'time_s':t,'reason':'joint position margin'};break
        vn=v+x[:16]*dt;q=pin.integrate(model,q,vn*dt);v=vn
    return {'target_acceleration_m_s2':az,'model_downward_speed_reduced':stopped,
            'failure':failure,'rows':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--impact',type=Path,default=HERE/'landing-impulse-research.json')
    parser.add_argument('--out',type=Path,default=HERE/'landing-brake-research.json')
    args=parser.parse_args()
    data=json.loads(args.impact.read_text())
    config=json.loads((HERE/'actuator-config-readonly.json').read_text())
    motor_speed_limits=[float(r['readbacks']['get_limit_velocity_max'][0][0][0]) for r in config['motor_reads']]
    impacts=[x for x in data['cases'] if x['solver_status']=='solved' and x['impact_contact_assumption']=='four_points']
    cases=[]
    for impact in impacts:
        for az in [3.,5.]:
            case=run(impact,az);case['foot_clearance_reference_m']=impact['foot_clearance_reference_m']
            case['motor_ids_exceeding_velocity_register']=[i+1 for i in range(10) if any(abs(r['motor_speeds_rad_s'][i])>motor_speed_limits[i] for r in case['rows'])]
            cases.append(case)
    report={'status':'IDEALIZED_LANDING_BRAKING_SCREEN' if cases else 'NOT_EVALUATED_NO_LANDING_CANDIDATE',
            'source_impact_report':str(args.impact),'physical_jump_validated':False,'hardware_actuation':False,
            'standing_recovery_simulated':False,'cases':cases,
            'limitations':['Ideal simultaneous sticking impact; real contacts may differ',
             'Velocity register is effective in modes 1 and 2, not direct force mode 3; not a physical speed rating',
             'No impact duration, peak force, compliance, contact debounce, actuator bandwidth or PD tracking',
             'Constant commanded acceleration transitions are not smooth, not a full landing controller',
             'Friction and physical mass/inertia assumed; no thermal model or disturbance robustness',
             'No demonstration that actual robot can safely follow these motions']}
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps([{'foot_height_m':c['foot_clearance_reference_m'],'acceleration':c['target_acceleration_m_s2'],
                       'downward_speed_reduced':c['model_downward_speed_reduced'],'failure':c['failure'],
                       'samples':len(c['rows']),'final_com_vz':c['rows'][-1]['com_vz_m_s'] if c['rows'] else None,
                       'max_current_A':max([r['max_motor_current_A'] for r in c['rows']],default=None),
                       'motor_ids_exceeding_velocity_register':c['motor_ids_exceeding_velocity_register'],
                       'max_joint_speed_rad_s':max([r['max_joint_speed_rad_s'] for r in c['rows']],default=None)} for c in cases],indent=2))


if __name__=='__main__':main()
