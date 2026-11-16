"""Constrained offline push integration from recorded model. No robot IO.

Exploratory forward integration only: no flight or impact integration, actuator
bandwidth/temperature, closed-loop tracking, or real-time implementation.
Results must not be executed on hardware.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation
from current_model import build

HERE=Path(__file__).resolve().parent


def installed_joint_velocity_map():
    """Read only the installed driver's verified arithmetic velocity map."""
    path=HERE/'installed_run_bear_snapshot.py'
    tree=ast.parse(path.read_text())
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)
               and node.name=='joint_to_motor_velocity']
    if len(functions)!=1:raise ValueError('Installed velocity conversion missing')
    fn=functions[0];fn.decorator_list=[]
    allowed=(ast.FunctionDef,ast.arguments,ast.arg,ast.Expr,ast.Constant,ast.Assign,
             ast.Name,ast.Store,ast.Load,ast.BinOp,ast.UnaryOp,ast.Add,ast.Sub,
             ast.Mult,ast.Div,ast.USub,ast.UAdd,ast.Return,ast.List,ast.Tuple)
    if any(not isinstance(node,allowed) for node in ast.walk(fn)):
        raise ValueError('Velocity conversion is no longer pure arithmetic')
    namespace={'__builtins__':{}}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    convert=namespace[fn.name]
    matrix=np.column_stack([np.asarray(convert(*x)).ravel() for x in np.eye(10)])
    np.testing.assert_allclose(np.asarray(convert(*np.arange(10.))).ravel(),matrix@np.arange(10.))
    return matrix,hashlib.sha256(path.read_bytes()).hexdigest()


def run(accel_target, push_s, taper_s, dt=.002, ramp_s=.02, initial_depth_m=.06, body_damping_weight=0., angular_momentum_weight=0.):
    if not (0 < ramp_s < push_s and taper_s > 0 and dt > 0 and .005<=initial_depth_m<=.06 and 0<=body_damping_weight<=1000 and 0<=angular_momentum_weight<=1000):
        raise ValueError('Invalid push timing')
    source=json.loads((HERE/'crouch-6cm-geometry.json').read_text())
    model,desc=build(HERE/'model-export-corrected.json')
    M=np.asarray(json.loads((HERE/'khadas-model-screen.json').read_text())['current_map_A_per_Nm'])
    velocity_map,_=installed_joint_velocity_map()
    s=source['anchor_state'];R=np.asarray(s['estimated']['body_rot_matrix'])
    q=pin.neutral(model);q[:3]=np.asarray(s['estimated']['body_position'])+[0,0,-initial_depth_m]
    depth_grid=-np.asarray([r['body_offset_m'] for r in source['records']])
    joint_grid=np.asarray([r['q'] for r in source['records']])
    q[3:7]=Rotation.from_matrix(R).as_quat()
    q[7:]=[np.interp(initial_depth_m,depth_grid,joint_grid[:,i]) for i in range(10)]
    v=np.zeros(model.nv);data=model.createData();frames=[model.getFrameId(n) for n in ('right_toe','right_heel','left_toe','left_heel')]
    pin.framesForwardKinematics(model,data,q)
    initial_feet=np.array([data.oMf[f].translation.copy() for f in frames])
    pin.centerOfMass(model,data,q);initial_com=data.com[0].copy()
    rows=[];failure=None
    for k in range(int(round((push_s+taper_s)/dt))+1):
        t=k*dt
        smooth=lambda u: (lambda x:x*x*x*(10-15*x+6*x*x))(min(1.,max(0.,u)))
        if t<ramp_s:target=accel_target*smooth(t/ramp_s)
        elif t<=push_s:target=accel_target
        else:target=accel_target+(-9.81-accel_target)*smooth((t-push_s)/taper_s)
        data=model.createData()
        H=pin.crba(model,data,q).copy();H=np.triu(H)+np.triu(H,1).T
        h=pin.nonLinearEffects(model,data,q,v).copy()
        pin.forwardKinematics(model,data,q,v,np.zeros(model.nv));pin.computeJointJacobians(model,data,q)
        pin.updateFramePlacements(model,data)
        J=np.vstack([pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED)[:3] for f in frames])
        bias=np.hstack([pin.getFrameClassicalAcceleration(model,data,f,pin.LOCAL_WORLD_ALIGNED).linear.copy() for f in frames])
        feet=np.hstack([data.oMf[f].translation.copy() for f in frames])
        desired_foot=-100*(feet-initial_feet.ravel())-20*(J@v)
        Jcom=pin.jacobianCenterOfMass(model,data,q).copy()
        pin.centerOfMass(model,data,q,v,np.zeros(model.nv));com=data.com[0].copy();vc=data.vcom[0].copy();ac_bias=data.acom[0].copy()
        dAg=pin.computeCentroidalMapTimeVariation(model,data,q,v).copy()
        Ag=data.Ag.copy();angular_momentum=Ag[3:,:]@v
        B=np.c_[H[6:],-J[:,6:].T];C=M@B
        dyn=np.c_[H[:6],-J[:,:6].T];foot=np.c_[J,np.zeros((12,12))];comA=np.c_[Jcom,np.zeros((3,12))]
        fr=[]
        for j in range(4):
            F=np.zeros((5,28));i=16+3*j;F[0,i+2]=1
            F[1:5,i:i+3]=[[1,1,-.5],[1,-1,-.5],[-1,1,-.5],[-1,-1,-.5]];fr.append(F)
        A=np.vstack([dyn,foot,comA,B,C,*fr])
        lo=np.r_[-h[:6],desired_foot-bias,np.array([0.,0.,target])-ac_bias,
                 -np.full(10,desc['torque_max'])-h[6:],-np.full(10,desc['current_max'])-M@h[6:],
                 np.tile([0.,-np.inf,-np.inf,-np.inf,-np.inf],4)]
        hi=np.r_[-h[:6],desired_foot-bias,np.array([0.,0.,target])-ac_bias,
                 np.full(10,desc['torque_max'])-h[6:],np.full(10,desc['current_max'])-M@h[6:],
                 np.tile([100.,0.,0.,0.,0.],4)]
        P=np.eye(28)*1e-5;P[:16,:16]+=np.eye(16)*1e-3;P[3,3]+=.1;P[4,4]+=.1;P[5,5]+=.1
        P+=.01*B.T@M.T@M@B;linear=.01*B.T@M.T@M@h[6:]
        if body_damping_weight:
            error=pin.log3(R.T@Rotation.from_quat(q[3:7]).as_matrix())
            target_ang_acc=-200*error-20*v[3:6]
            P[3:6,3:6]+=body_damping_weight*np.eye(3)
            linear[3:6]-=body_damping_weight*target_ang_acc
        if angular_momentum_weight:
            angular_map=Ag[3:,:];angular_bias=dAg[3:,:]@v+10*angular_momentum
            P[:16,:16]+=angular_momentum_weight*angular_map.T@angular_map
            linear[:16]+=angular_momentum_weight*angular_map.T@angular_bias
        solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(P),q=linear,A=sparse.csc_matrix(A),l=lo,u=hi,
               verbose=False,eps_abs=1e-5,eps_rel=1e-5,max_iter=100000,polishing=False)
        solution=solver.solve()
        if solution.info.status!='solved':failure={'time_s':t,'reason':'QP '+solution.info.status};break
        x=solution.x;ddq=x[:16];tau=h[6:]+B@x;forces=x[16:].reshape(4,3)
        error=float(np.linalg.norm((feet-initial_feet.ravel()).reshape(4,3),axis=1).max())
        violation=float(max(np.max(lo-A@x),np.max(A@x-hi),0.))
        rows.append({'time_s':t,'target_com_acceleration_m_s2':target,'com_z_m':float(com[2]),
             'com_vz_m_s':float(vc[2]),'max_joint_speed_rad_s':float(abs(v[6:]).max()),
             'angular_momentum_kg_m2_s':angular_momentum.tolist(),
             'body_tilt_deg':float(np.degrees(np.arccos(np.clip(Rotation.from_quat(q[3:7]).as_matrix()[2,2],-1,1)))),
             'max_joint_acceleration_rad_s2':float(abs(ddq[6:]).max()),'max_joint_torque_Nm':float(abs(tau).max()),
             'max_motor_current_A':float(abs(M@tau).max()),'ground_normal_force_N':float(forces[:,2].sum()),
             'max_motor_speed_rad_s':float(abs(velocity_map@v[6:]).max()),
             'motor_speeds_rad_s':(velocity_map@v[6:]).tolist(),
             'max_motor_acceleration_rad_s2':float(abs(velocity_map@ddq[6:]).max()),
             'max_foot_drift_m':error,'constraint_violation':violation,'joint_positions':q[7:].tolist(),
             'model_configuration':q.tolist(),'model_velocity':v.tolist()})
        if violation>2e-3 or error>.003:
            failure={'time_s':t,'reason':'contact drift or numerical constraint violation'};break
        if rows[-1]['body_tilt_deg']>8:
            failure={'time_s':t,'reason':'model torso tilt exceeded 8-degree research bound'};break
        lower=np.asarray(desc['joint_lower']);upper=np.asarray(desc['joint_upper'])
        if np.any(q[7:]<lower+.01) or np.any(q[7:]>upper-.01):
            failure={'time_s':t,'reason':'joint position margin violated'};break
        if k<int(round((push_s+taper_s)/dt)):
            v_next=v+ddq*dt;q=pin.integrate(model,q,v_next*dt);v=v_next
    return {'status':'REJECTED' if failure else 'INTEGRATED_MODEL_ONLY',
            'acceleration_m_s2':accel_target,'push_s':push_s,'taper_s':taper_s,'ramp_s':ramp_s,'dt_s':dt,'initial_depth_m':initial_depth_m,'body_damping_weight':body_damping_weight,'angular_momentum_weight':angular_momentum_weight,
            'failure':failure,'initial_com_z_m':float(initial_com[2]),'samples':rows}


def main():
    cases=[run(4.,.10,.03),run(5.,.09,.03),run(6.,.08,.03)]
    _,mapping_hash=installed_joint_velocity_map()
    report={'status':'OFFLINE_PUSH_INTEGRATION_ONLY','hardware_actuation':False,'flight_simulated':False,
            'landing_simulated':False,'physical_jump_validated':False,'cases':cases,'installed_driver_snapshot_sha256':mapping_hash,
            'limitations':['Assumed robot mass, inertia, friction and fixed toe/heel contacts',
                           'No measured actuator torque-speed envelope, thermal, saturation or PD dynamics',
                           'Forward numerical integration with approximate stabilization, not real hardware',
                           'No tracking of a full torso orientation or landing motion',
                           'Illustrative candidate trajectories cannot be executed on the robot']}
    (HERE/'integrated-push-research.json').write_text(json.dumps(report,indent=2))
    print(json.dumps([{**{k:v for k,v in c.items() if k!='samples'},'sample_count':len(c['samples']),
                       'last_sample':c['samples'][-1] if c['samples'] else None} for c in cases],indent=2))


if __name__=='__main__':main()
