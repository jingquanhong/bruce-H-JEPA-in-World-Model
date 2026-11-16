"""Offline inverse-dynamics screen of illustrative fixed-foot push trajectories.

Not an executable trajectory, controller, contact simulation or safety proof.
No robot IO or motor commands. A rejected example cannot exclude other jumps.
"""
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
import osqp
from scipy import sparse
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from current_model import build

HERE=Path(__file__).resolve().parent


def main():
    source=json.loads((HERE/'crouch-6cm-geometry.json').read_text())
    model,desc=build(HERE/'model-export-corrected.json')
    current_map=np.asarray(json.loads((HERE/'khadas-model-screen.json').read_text())['current_map_A_per_Nm'])
    state=source['anchor_state'];R=np.asarray(state['estimated']['body_rot_matrix'])
    depths=-np.asarray([r['body_offset_m'] for r in source['records']])
    samples=np.asarray([r['q'] for r in source['records']])
    joint=CubicSpline(depths,samples,axis=0)
    com=[]
    for depth,qj in zip(depths,samples):
        q=pin.neutral(model);q[:3]=np.asarray(state['estimated']['body_position'])+[0,0,-depth]
        q[3:7]=Rotation.from_matrix(R).as_quat();q[7:]=qj
        com.append(pin.centerOfMass(model,model.createData(),q)[2])
    com=CubicSpline(depths,com)
    frames=[model.getFrameId(n) for n in ('right_toe','right_heel','left_toe','left_heel')]
    takeoff_depth=.015
    vz_com=np.sqrt(2*9.81*.01)
    body_v=vz_com/(-float(com(takeoff_depth,1)))
    # com(z) = com(depth=-z); enforce free-fall vertical CoM acceleration at takeoff.
    body_a=(-9.81-float(com(takeoff_depth,2))*body_v**2)/(-float(com(takeoff_depth,1)))
    cases=[]
    for T in [.12,.16,.20,.24,.30,.40,.50,.60]:
        # z(u)=-.06 + a3*u^3+a4*u^4+a5*u^5, u=t/T.
        coeff=np.linalg.solve([[1,1,1],[3,4,5],[6,12,20]],
                              [.06-takeoff_depth,body_v*T,body_a*T*T])
        peak_tau=peak_current=peak_joint_speed=0.;max_contact_velocity=0.;failed=None
        depths_seen=[];normal_forces=[]
        for u in np.linspace(0,1,101):
            z=-.06+np.dot(coeff,[u**3,u**4,u**5])
            zdot=np.dot(coeff,[3*u**2,4*u**3,5*u**4])/T
            zddot=np.dot(coeff,[6*u,12*u**2,20*u**3])/T**2
            depth=-z;depths_seen.append(float(depth))
            if not -.0001<=depth<=.0601:
                failed={'reason':'outside screened geometry','u':float(u),'depth_m':float(depth)};break
            qj=joint(depth);dqj=joint(depth,1)*(-zdot)
            ddqj=joint(depth,2)*zdot**2-joint(depth,1)*zddot
            q=pin.neutral(model);q[:3]=np.asarray(state['estimated']['body_position'])+[0,0,z]
            q[3:7]=Rotation.from_matrix(R).as_quat();q[7:]=qj
            v=np.r_[R.T@np.array([0,0,zdot]),np.zeros(3),dqj]
            a=np.r_[R.T@np.array([0,0,zddot]),np.zeros(3),ddqj]
            data=model.createData();h=pin.rnea(model,data,q,v,a).copy()
            pin.computeJointJacobians(model,data,q);pin.updateFramePlacements(model,data)
            J=np.vstack([pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED)[:3] for f in frames])
            max_contact_velocity=max(max_contact_velocity,float(abs(J@v).max()))
            B=-J[:,6:].T;C=current_map@B;c=current_map@h[6:]
            blocks=[J[:,:6].T,B,C];lower=[h[:6],np.full(10,-desc['torque_max'])-h[6:],np.full(10,-desc['current_max'])-c]
            upper=[h[:6],np.full(10,desc['torque_max'])-h[6:],np.full(10,desc['current_max'])-c]
            for foot in range(4):
                F=np.zeros((5,12));j=foot*3;F[0,j+2]=1
                F[1:5,j:j+3]=[[1,1,-.5],[1,-1,-.5],[-1,1,-.5],[-1,-1,-.5]]
                blocks.append(F);lower.append(np.r_[0.,np.full(4,-np.inf)]);upper.append(np.r_[100.,np.zeros(4)])
            A=np.vstack(blocks);lo=np.concatenate(lower);hi=np.concatenate(upper)
            solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(C.T@C+.001*np.eye(12)),q=C.T@c,
                A=sparse.csc_matrix(A),l=lo,u=hi,verbose=False,eps_abs=1e-6,eps_rel=1e-6,
                max_iter=10000,polishing=False)
            sol=solver.solve();peak_joint_speed=max(peak_joint_speed,float(abs(dqj).max()))
            if sol.info.status!='solved':
                failed={'reason':'contact force and actuator constraint QP '+sol.info.status,'u':float(u),'depth_m':float(depth)};break
            tau=h[6:]+B@sol.x;i=current_map@tau
            peak_tau=max(peak_tau,float(abs(tau).max()));peak_current=max(peak_current,float(abs(i).max()))
            normal_forces.append(float(sol.x.reshape(4,3)[:,2].sum()))
        cases.append({'push_duration_s':T,'status':'REJECTED' if failed else 'SOLVED_SCREEN_ONLY',
                      'failure':failed,'peak_joint_speed_rad_s_before_failure':peak_joint_speed,
                      'peak_joint_torque_Nm_before_failure':peak_tau,'peak_motor_current_A_before_failure':peak_current,
                      'max_model_contact_velocity_m_s':max_contact_velocity,
                      'normal_force_min_N':min(normal_forces) if normal_forces else None,
                      'samples_evaluated':len(depths_seen)})
    report={'status':'ILLUSTRATIVE_MODEL_ONLY','model_mass_kg':sum(desc['mass']),'physical_mass_measured':False,
            'takeoff_body_depth_m':takeoff_depth,'assumed_flight_apex_above_takeoff_m':.01,
            'required_com_takeoff_speed_m_s':float(vz_com),'required_body_takeoff_speed_m_s':float(body_v),
            'target_body_takeoff_acceleration_m_s2':float(body_a),'cases':cases,
            'hardware_actuation':False,'jump_trajectory_validated':False,
            'limitations':['One fixed-foot path family and quintic push, not an optimizer or complete jump',
                'Foot motion and contact acceleration are approximations; actual terrain and impact not modeled',
                'Model omits measured arm removal/camera mass changes, motor PD and transient current',
                'No certified joint velocity or actuator power/thermal limits included',
                'Infeasibility of an example does not establish infeasibility of another trajectory']}
    (HERE/'jump-push-candidate-screen.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
