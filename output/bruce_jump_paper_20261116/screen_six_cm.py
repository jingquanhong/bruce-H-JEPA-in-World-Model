"""Recorded-anchor crouch geometry and inverse-dynamics screen. No hardware IO."""
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation
from current_model import build

HERE = Path(__file__).resolve().parent


def main():
    source = json.loads((HERE/'crouch-6cm-geometry.json').read_text())
    model, description = build(HERE/'model-export-corrected.json')
    current_map = np.asarray(json.loads((HERE/'khadas-model-screen.json').read_text())['current_map_A_per_Nm'])
    state = source['anchor_state']
    rotation = np.asarray(state['estimated']['body_rot_matrix'])
    q0 = np.asarray(state['joint_positions'])
    samples = np.asarray([r['q'] for r in source['records']])
    depths = -np.asarray([r['body_offset_m'] for r in source['records']])
    times = np.linspace(0., 14.5, 726)
    ramp = 6.
    down = np.clip(times/ramp, 0, 1)
    up = np.clip((times-ramp-.5)/ramp, 0, 1)
    smooth = lambda u: u**3*(10-15*u+6*u*u)
    depth = .06*(smooth(down)-smooth(up))
    joints = np.column_stack([np.interp(depth, depths, samples[:,i]) for i in range(10)])
    velocity = np.gradient(joints, times, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, times, axis=0, edge_order=2)
    vz = -np.gradient(depth, times, edge_order=2)
    az = np.gradient(vz, times, edge_order=2)
    limits = np.asarray(description['joint_lower']), np.asarray(description['joint_upper'])
    frames = [model.getFrameId(n) for n in ['right_toe','right_heel','left_toe','left_heel']]
    cases = []
    max_tau = max_current = max_violation = 0.
    min_normal = float('inf')
    initial_com = None
    for i in range(0, len(times), 5):
        data = model.createData()
        q = pin.neutral(model)
        q[:3] = np.asarray(state['estimated']['body_position']) + [0,0,-depth[i]]
        q[3:7] = Rotation.from_matrix(rotation).as_quat()
        q[7:] = joints[i]
        v = np.r_[rotation.T@np.array([0,0,vz[i]]),np.zeros(3),velocity[i]]
        a = np.r_[rotation.T@np.array([0,0,az[i]]),np.zeros(3),acceleration[i]]
        h = pin.rnea(model,data,q,v,a).copy()
        pin.computeJointJacobians(model,data,q)
        pin.updateFramePlacements(model,data)
        jac = np.vstack([pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED)[:3] for f in frames])
        torque_map = -jac[:,6:].T
        current_force_map = current_map@torque_map
        constant_current = current_map@h[6:]
        # Contact forces, base equilibrium, friction, motor-current and joint-torque limits.
        blocks=[jac[:,:6].T, torque_map, current_force_map]
        lower=[h[:6],np.full(10,-description['torque_max'])-h[6:],np.full(10,-description['current_max'])-constant_current]
        upper=[h[:6],np.full(10,description['torque_max'])-h[6:],np.full(10,description['current_max'])-constant_current]
        for foot in range(4):
            f=np.zeros((5,12));j=foot*3
            f[0,j+2]=1
            f[1,j]=1;f[1,j+2]=-.5
            f[2,j]=-1;f[2,j+2]=-.5
            f[3,j+1]=1;f[3,j+2]=-.5
            f[4,j+1]=-1;f[4,j+2]=-.5
            blocks.append(f);lower.append(np.array([.5,-np.inf,-np.inf,-np.inf,-np.inf]));upper.append(np.array([100,0,0,0,0]))
        A=np.vstack(blocks);lo=np.concatenate(lower);hi=np.concatenate(upper)
        P=current_force_map.T@current_force_map+.001*np.eye(12)
        c=current_force_map.T@constant_current
        solver=osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(P),q=c,A=sparse.csc_matrix(A),l=lo,u=hi,verbose=False,eps_abs=1e-6,eps_rel=1e-6,polishing=True)
        result=solver.solve()
        if result.info.status!='solved':raise RuntimeError((float(times[i]),result.info.status))
        force=result.x;torque=h[6:]+torque_map@force;current=current_map@torque
        violation=float(max(np.max(lo-A@force),np.max(A@force-hi),0))
        com=pin.centerOfMass(model,data,q).copy()
        if initial_com is None:initial_com=com.copy()
        max_tau=max(max_tau,float(abs(torque).max()))
        max_current=max(max_current,float(abs(current).max()))
        max_violation=max(max_violation,violation)
        min_normal=min(min_normal,float(force.reshape(4,3)[:,2].min()))
        cases.append(dict(time_s=float(times[i]),body_depth_m=float(depth[i]),q=joints[i].tolist(),com=com.tolist(),torque_Nm=torque.tolist(),motor_current_A=current.tolist(),forces_N=force.reshape(4,3).tolist()))
    report=dict(status='PASS_RECORDED_MODEL_SCREEN_ONLY',requested_body_crouch_m=.06,lower_s=6,hold_s=.5,raise_s=6,settle_s=2,
                max_joint_delta_deg=float(np.degrees(abs(joints-q0).max())),max_joint_velocity_rad_s=float(abs(velocity).max()),
                minimum_joint_limit_margin_rad=float(min(np.min(joints-limits[0]),np.min(limits[1]-joints))),
                maximum_ankle_geometry_error_m=max(r['ankle_error_m'] for r in source['records']),
                max_joint_torque_Nm=max_tau,max_motor_current_A=max_current,minimum_contact_normal_N=min_normal,
                max_constraint_violation=max_violation,model_mass_kg=sum(description['mass']),physical_mass_measured=False,
                estimated_model_com_excursion_m=float(initial_com[2]-min(c['com'][2] for c in cases)),
                hardware_commands_changed=False,real_jump_attempted=False,cases=cases,
                limitations=['Recorded anchor and model only, not physical validation','Arms removed and camera mass not measured','Driver PD transient/current not represented','Actual floor friction not measured'])
    (HERE/'crouch-6cm-model-screen.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
