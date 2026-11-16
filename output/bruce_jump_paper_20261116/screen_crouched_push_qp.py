"""Offline isolated crouched-pose acceleration feasibility; no hardware IO.

Each QP checks one instant with zero initial velocity. This is not a time-domain
trajectory, controller, jump or landing test and excludes motor PD transients.
"""
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation
from current_model import build

HERE=Path(__file__).resolve().parent


def main():
    src=json.loads((HERE/'crouch-6cm-geometry.json').read_text())
    model,desc=build(HERE/'model-export-corrected.json')
    M=np.asarray(json.loads((HERE/'khadas-model-screen.json').read_text())['current_map_A_per_Nm'])
    R=np.asarray(src['anchor_state']['estimated']['body_rot_matrix'])
    base=np.asarray(src['anchor_state']['estimated']['body_position'])
    depth_grid=-np.asarray([r['body_offset_m'] for r in src['records']])
    qgrid=np.asarray([r['q'] for r in src['records']])
    frames=[model.getFrameId(n) for n in ('right_toe','right_heel','left_toe','left_heel')]
    cases=[]
    for depth in [.03,.045,.06]:
        q=pin.neutral(model);q[:3]=base+[0,0,-depth];q[3:7]=Rotation.from_matrix(R).as_quat()
        q[7:]=[np.interp(depth,depth_grid,qgrid[:,i]) for i in range(10)]
        data=model.createData();H=pin.crba(model,data,q).copy();H=np.triu(H)+np.triu(H,1).T
        h=pin.nonLinearEffects(model,data,q,np.zeros(model.nv)).copy()
        pin.computeJointJacobians(model,data,q);pin.updateFramePlacements(model,data)
        J=np.vstack([pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED)[:3] for f in frames])
        Jcom=pin.jacobianCenterOfMass(model,data,q).copy()
        # z=[generalized accelerations(16), contact forces(12)]; Pin convention
        # linear base velocity then angular, contact and COM vectors world-aligned.
        B=np.c_[H[6:],-J[:,6:].T];C=M@B
        dyn=np.c_[H[:6],-J[:,:6].T]
        foot=np.c_[J,np.zeros((12,12))]
        com=np.c_[Jcom,np.zeros((3,12))]
        orientation=np.zeros((2,28));orientation[0,3]=1;orientation[1,5]=1
        frictions=[]
        for i in range(4):
            f=np.zeros((5,28));j=16+3*i;f[0,j+2]=1
            f[1:5,j:j+3]=[[1,1,-.5],[1,-1,-.5],[-1,1,-.5],[-1,-1,-.5]]
            frictions.append(f)
        A=np.vstack([dyn,foot,com,orientation,B,C,*frictions]);n=28
        # Small accelerations/contact forces and current are preferred, with
        # an extra base-pitch penalty. This penalty is not a tracking guarantee.
        P=np.eye(n)*1e-5;P[:16,:16]+=np.eye(16)*1e-3;P[4,4]+=.1
        P+=B.T@M.T@M@B*.01
        linear=B.T@M.T@M@h[6:]*.01
        for az in [0.,1.,3.,5.]:
            lo=np.r_[-h[:6],np.zeros(12),[0,0,az],np.zeros(2),
                     -np.full(10,desc['torque_max'])-h[6:],
                     -np.full(10,desc['current_max'])-M@h[6:],
                     np.tile([0.,-np.inf,-np.inf,-np.inf,-np.inf],4)]
            hi=np.r_[-h[:6],np.zeros(12),[0,0,az],np.zeros(2),
                     np.full(10,desc['torque_max'])-h[6:],
                     np.full(10,desc['current_max'])-M@h[6:],
                     np.tile([100.,0.,0.,0.,0.],4)]
            solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(P),q=linear,A=sparse.csc_matrix(A),l=lo,u=hi,
                                          verbose=False,eps_abs=1e-6,eps_rel=1e-6,max_iter=20000,polishing=True)
            sol=solver.solve();case={'body_depth_m':depth,'target_com_acceleration_m_s2':az,
                                    'solver_status':sol.info.status}
            if sol.info.status=='solved':
                z=sol.x;tau=h[6:]+B@z;actual=Jcom@z[:16];normal=z[16:].reshape(4,3)[:,2]
                case.update(max_constraint_violation=float(max(np.max(lo-A@z),np.max(A@z-hi),0.)),
                            actual_com_acceleration_m_s2=actual.tolist(),
                            contact_acceleration_max_m_s2=float(abs(J@z[:16]).max()),
                            body_pitch_acceleration_rad_s2=float(z[4]),
                            max_joint_acceleration_rad_s2=float(abs(z[6:16]).max()),
                            max_joint_torque_Nm=float(abs(tau).max()),
                            max_motor_current_A=float(abs(M@tau).max()),
                            contact_normal_forces_N=normal.tolist())
            cases.append(case)
    report={'status':'OFFLINE_ISOLATED_ACCELERATION_SCREEN_ONLY','model_mass_kg':sum(desc['mass']),
            'physical_mass_measured':False,'initial_joint_and_body_velocity':'zero',
            'hardware_actuation':False,'physical_jump_validated':False,'cases':cases,
            'limitations':['Independent static-pose snapshots; not integrated and no takeoff or landing',
              'Joint acceleration, motor speed, power, thermal and PD transient limits not characterized',
              'Contact friction and physical mass/inertia are assumed, not measured',
              'Pitch acceleration is allowed and is not a planned orientation trajectory',
              'Solver feasibility is not evidence that the robot can perform a jump']}
    (HERE/'crouched-push-qp-screen.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
