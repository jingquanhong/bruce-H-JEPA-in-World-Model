"""Exploratory ideal-flight foot-retraction QP. Never controls hardware.

Assumes unconstrained free flight after a model-only push. No actuator dynamics,
collision geometry, contact impacts or real-time guarantees are modeled.
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
from integrate_push_research import run, installed_joint_velocity_map

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--push-accel',type=float,default=6.)
    parser.add_argument('--push-time',type=float,default=.10288333333333333)
    parser.add_argument('--taper',type=float,default=.06)
    parser.add_argument('--depth',type=float,default=.06)
    parser.add_argument('--flight-period',type=float,default=.09)
    parser.add_argument('--landing-speed',type=float,default=.1,help='Research-only downward foot speed at nominal touchdown, m/s')
    parser.add_argument('--body-weight',type=float,default=1.,help='Relative idealized torso orientation task weight')
    parser.add_argument('--push-body-weight',type=float,default=0.)
    parser.add_argument('--push-momentum-weight',type=float,default=0.)
    parser.add_argument('--heights',type=float,nargs='+',default=[.003,.006,.010])
    parser.add_argument('--out',type=Path,default=HERE/'flight-retraction-research.json')
    args=parser.parse_args()
    if not .02<=args.flight_period<=.3 or not .01<=args.landing_speed<=.3 or not 0<=args.body_weight<=100 or any(not .0005<=h<=.02 for h in args.heights):
        parser.error('flight period or foot heights outside research bounds')
    model,description=build(HERE/'model-export-corrected.json')
    mapping=np.asarray(json.loads((HERE/'khadas-model-screen.json').read_text())['current_map_A_per_Nm'])
    velocity_map,_=installed_joint_velocity_map()
    config=json.loads((HERE/'actuator-config-readonly.json').read_text())
    speed_reference=np.asarray([r['readbacks']['get_limit_velocity_max'][0][0][0] for r in config['motor_reads']])
    push=run(args.push_accel,args.push_time,args.taper,.001,initial_depth_m=args.depth,
             body_damping_weight=args.push_body_weight,angular_momentum_weight=args.push_momentum_weight)
    if push['status']!='INTEGRATED_MODEL_ONLY':raise RuntimeError('Push integration rejected')
    q0=np.asarray(push['samples'][0]['model_configuration']);end=push['samples'][-1]
    reference_rotation=Rotation.from_quat(q0[3:7]).as_matrix()
    frames=[model.getFrameId(n) for n in ('right_toe','right_heel','left_toe','left_heel')]
    initial=model.createData();pin.framesForwardKinematics(model,initial,q0)
    origin=np.array([initial.oMf[f].translation.copy() for f in frames]);ground=origin[:,2]
    cases=[]
    for target_height in args.heights:
        q=np.asarray(end['model_configuration']).copy();v=np.asarray(end['model_velocity']).copy()
        rows=[];failure=None;clear=False;period=args.flight_period;dt=.001
        for k in range(151):
            t=k*dt;data=model.createData()
            H=pin.crba(model,data,q).copy();H=np.triu(H)+np.triu(H,1).T
            h=pin.nonLinearEffects(model,data,q,v).copy()
            pin.forwardKinematics(model,data,q,v,np.zeros(model.nv));pin.computeJointJacobians(model,data,q);pin.updateFramePlacements(model,data)
            J=np.vstack([pin.getFrameJacobian(model,data,f,pin.LOCAL_WORLD_ALIGNED)[:3] for f in frames])
            bias=np.hstack([pin.getFrameClassicalAcceleration(model,data,f,pin.LOCAL_WORLD_ALIGNED).linear.copy() for f in frames])
            feet=np.array([data.oMf[f].translation.copy() for f in frames]);heights=feet[:,2]-ground
            foot_velocity=(J@v).reshape(4,3);footv=foot_velocity[:,2]
            S=np.vstack([np.zeros((6,10)),np.eye(10)])
            a0=np.linalg.solve(H,-h);T=np.linalg.solve(H,S)
            foot0=J@a0+bias;footT=J@T
            u=min(t/period,1.)
            if t<period:
                ref=target_height*np.sin(np.pi*u)**2+args.landing_speed*period*u*u*(1-u)
                refv=target_height*np.pi/period*np.sin(2*np.pi*u)+args.landing_speed*(2*u-3*u*u)
                refa=2*target_height*(np.pi/period)**2*np.cos(2*np.pi*u)+args.landing_speed/period*(2-6*u)
            else:
                ref=-args.landing_speed*(t-period);refv=-args.landing_speed;refa=0.
            desired=np.zeros((4,3))
            desired[:,:2]=250*(origin[:,:2]-feet[:,:2])-30*foot_velocity[:,:2]
            desired[:,2]=refa+400*(ref-heights)+30*(refv-footv)
            desired=desired.ravel();weights=np.tile([.5,.5,1.],4)
            P=footT.T@(weights[:,None]*footT)+.05*np.eye(10)
            linear=footT.T@(weights*(foot0-desired))
            current_rotation=Rotation.from_quat(q[3:7]).as_matrix()
            orientation_error=pin.log3(reference_rotation.T@current_rotation)
            body_desired=-200*orientation_error-20*v[3:6]
            bodyT=T[3:6,:];body0=a0[3:6]
            P+=args.body_weight*bodyT.T@bodyT
            linear+=args.body_weight*bodyT.T@(body0-body_desired)
            A=np.vstack([np.eye(10),mapping]);lo=np.r_[-np.full(10,description['torque_max']),-np.full(10,description['current_max'])];hi=-lo
            solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(P),q=linear,A=sparse.csc_matrix(A),l=lo,u=hi,
                verbose=False,eps_abs=1e-6,eps_rel=1e-6,max_iter=10000,polishing=False)
            result=solver.solve()
            if result.info.status!='solved':failure={'time_s':t,'reason':'QP '+result.info.status};break
            tau=result.x;acc=a0+T@tau;actual=foot0+footT@tau
            pin.centerOfMass(model,data,q,v);comz=float(data.com[0][2]);comvz=float(data.vcom[0][2])
            rows.append({'time_s':t,'reference_foot_height_m':float(ref),'foot_heights_m':heights.tolist(),
                         'foot_vertical_velocity_m_s':footv.tolist(),'com_z_m':comz,'com_vz_m_s':comvz,
                         'body_tilt_deg':float(np.degrees(np.arccos(np.clip(Rotation.from_quat(q[3:7]).as_matrix()[2,2],-1,1)))),
                         'model_configuration':q.tolist(),'model_velocity':v.tolist(),
                         'motor_speeds_rad_s':(velocity_map@v[6:]).tolist(),
                         'max_torque_Nm':float(abs(tau).max()),'max_current_A':float(abs(mapping@tau).max()),
                         'max_foot_acceleration_tracking_error_m_s2':float(abs(actual-desired).max())})
            if rows[-1]['body_tilt_deg']>8:
                failure={'time_s':t,'reason':'model torso tilt exceeded 8-degree research bound'};break
            lower=np.asarray(description['joint_lower']);upper=np.asarray(description['joint_upper'])
            if np.any(q[7:]<lower+.01) or np.any(q[7:]>upper-.01):
                failure={'time_s':t,'reason':'model joint position margin'};break
            if np.any(abs(velocity_map@v[6:])>speed_reference):
                failure={'time_s':t,'reason':'exceeded conservative velocity-register comparison (not force-mode limit)'};break
            if heights.min()>.001:clear=True
            if heights.min()<-.002:
                failure={'time_s':t,'reason':'model foot penetrated floor plane >2mm'};break
            if clear and comvz<0 and heights.min()<=0:
                failure={'time_s':t,'reason':'model foot reached floor; contact and impact not modeled'};break
            v_next=v+acc*dt;q=pin.integrate(model,q,v_next*dt);v=v_next
        cases.append({'target_foot_height_m':target_height,'achieved_both_feet_over_1mm':clear,'ending':failure,'samples':rows})
    report={'status':'IDEAL_FLIGHT_RETRACTION_QP_ONLY','hardware_actuation':False,'physical_jump_validated':False,
            'landing_simulated':False,'body_orientation_weight':args.body_weight,
            'push_configuration':{'accel_m_s2':args.push_accel,'push_s':args.push_time,'taper_s':args.taper,
                'depth_m':args.depth,'body_damping_weight':args.push_body_weight,'angular_momentum_weight':args.push_momentum_weight},
            'nominal_flight_period_s':args.flight_period,'nominal_touchdown_foot_speed_m_s':args.landing_speed,'cases':cases,
            'limitations':['No motor bandwidth, torque-speed curve, thermals, foot collision geometry or control latency',
                           'Configured velocity-register comparisons are conservative research filters; they are not force-mode hardware limits',
                           'Flight beginning is inherited from unvalidated model-only push',
                           'Unmeasured real mass and inertia; simplified rigid ground plane',
                           'No physical impact or controlled landing modeled',
                           'Constraint solver feasibility does not imply real robot safety']}
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps([{'target_height_m':c['target_foot_height_m'],'both_feet_over_1mm':c['achieved_both_feet_over_1mm'],
                       'ending':c['ending'],'samples':len(c['samples']),
                       'max_min_foot_height_m':max(min(r['foot_heights_m']) for r in c['samples']),
                       'max_current_A':max(r['max_current_A'] for r in c['samples'])} for c in cases],indent=2))


if __name__=='__main__':main()
