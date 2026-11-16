#!/usr/bin/env python3
"""Unassisted two-foot jump experiments. Screened candidates are not qualified successes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

import bruce_env as be
import bruce_expert as bx
from qualification import motor_current_from_joint_torque
from support_reference import smooth
from support_wbc import SupportController
from flight_wbc import FlightController


def limit(command):
    command=np.clip(command,-np.r_[np.full(10,10.5),np.ones(6)],np.r_[np.full(10,10.5),np.ones(6)])
    peak=np.max(np.abs(motor_current_from_joint_torque(command[:10])))
    if peak>29.99: command[:10]*=29.99/peak
    return command


def floor_forces(scene):
    result=np.zeros(2);contacts=scene.get_contacts()
    if contacts is None:return result
    for i,(a,b) in enumerate(zip(contacts['geom_a'],contacts['geom_b'])):
        if int(a)==0: geom=int(b);force=contacts['force_b'][i]
        elif int(b)==0: geom=int(a);force=contacts['force_a'][i]
        else:continue
        name=scene.geom_link.get(geom)
        if name in be.ANKLE_LINKS:result[be.ANKLE_LINKS.index(name)]+=max(float(force[2]),0.)
    return result


def flight_target(state,initial,elapsed,landing_x,tuck_height):
    row=initial.copy()
    if state['base_vel'][2]>.1:
        row=bx._pose({'r':(0,.025),'l':(0,.025)},-.03,tuck_height)
    else:
        R=be.quat_to_mat(state['base_quat'])
        pitch=float(np.arctan2(R[0,2],R[2,2]))
        row=initial.copy()
        for side,offset,y in [('r',6,-bx.FOOT_Y),('l',11,bx.FOOT_Y)]:
            local=R.T@(np.array([landing_x,y,.025])-state['base_pos'])
            hip,knee,ankle=bx.ik(local[0]-bx.HIP_X,local[2]-bx.HIP_Z)
            row[offset:offset+5]=[0.,hip,0.,knee,ankle-pitch]
    return row[6:]


def flight_command(state,target,kp=65.,kd=1.2):
    error=target-state['dof_pos'][6:]
    gains=np.r_[np.full(10,kp),np.full(6,8.)]
    damping=np.r_[np.full(10,kd),np.full(6,.4)]
    return limit(gains*error-damping*state['dof_vel'][6:])


def run(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError(out/'results.json')
    initial=bx._pose({'r':(0,.025),'l':(0,.025)},-.03,args.height)
    scene=be.BruceScene(backend=args.backend,box_specs=[dict(half=(.025,.15,.04),pos=(.15,0,.04))])
    scene.set_friction(1.);scene.set_gains(np.zeros(22),np.zeros(22))
    bounds=np.r_[np.zeros(6),np.full(10,10.5),np.ones(6)]
    scene.robot.set_dofs_force_range(-bounds,bounds)
    results=[]
    for ax in args.ax:
      for az in args.az:
        scene.reset_robot(initial[2],initial[6:],initial[0]);scene.reset_box(scene.boxes[0],(.15,0,.04))
        c=SupportController(initial,attitude_weight=100.,com_weight=3000.,solver_backend=args.solver)
        fc=FlightController(initial)
        phase='prepare';takeoff=None;touchdown=None;stop='duration_complete';history=[]
        airborne=None;ticks=0
        base_peak=torque_peak=current_peak=box_motion=0.;min_up=1.;apex=0.
        landing_x=.28;landing_feet=None;push_com=None
        for i in range(int(args.duration/.005)):
            t=i*.005;state=scene.read_state();R=be.quat_to_mat(state['base_quat'])
            ground_forces=floor_forces(scene)
            force=float(sum(ground_forces));feet=np.array([scene.get_link_pos(n) for n in be.ANKLE_LINKS])
            q=fc.q0.copy();q[:3]=state['base_pos'];q[3:7]=np.r_[state['base_quat'][1:],state['base_quat'][0]]
            v=np.zeros(fc.model.nv);v[:3]=R.T@state['base_vel'];v[3:6]=R.T@state['base_ang']
            for iq,iv,slot in fc.mapping:q[iq]=state['dof_pos'][6+slot];v[iv]=state['dof_vel'][6+slot]
            pin.centerOfMass(fc.model,fc.data,q,v)
            actual_com=fc.data.com[0].copy();actual_com_velocity=fc.data.vcom[0].copy()
            actual_momentum=pin.computeCentroidalMomentum(fc.model,fc.data,q,v).angular.copy()
            if phase=='flight' and airborne is None and force<1. and np.min(feet[:,2])>.03:airborne=t
            if phase=='prepare' and t>=.8:
                phase='push';push_com=np.asarray(c.debug['com']);push_start=t
                c.hard_com=args.hard_com;c.attitude_weight=args.push_attitude_weight
                c.momentum_target[1]=args.launch_momentum
                c.momentum_weight=args.momentum_weight
            if phase=='push' and t-push_start>.06 and (force<2. or np.min(feet[:,2])>.045 or
                    state['base_pos'][2]>args.release_height):
                phase='flight';takeoff=t
                landing_x=max(.24,float(c.debug['com'][0])+max(.25,float(state['base_vel'][0]))*.38)
                print('TAKEOFF',ax,az,t,state['base_pos'],state['base_vel'],'landing',landing_x,flush=True)
            if phase=='flight' and t-takeoff>.12 and force>4.:
                phase='landing';touchdown=t;landing_feet=feet.copy();landing_feet[:,2]=.025
                print('TOUCHDOWN',t,feet,state['base_vel'],flush=True)
            try:
                if phase=='prepare':
                    com=c.com0.copy();com[0]+=(args.pre_x-com[0])*smooth((t-.2)/.6)
                    command=c.command(state,com,np.array([p.translation for p in c.feet0]),[0,1])
                elif phase=='push':
                    s=t-push_start;acc=np.array([ax,0.,az])
                    com=push_com+.5*acc*s*s;vel=acc*s
                    command=c.command(state,com,np.array([p.translation for p in c.feet0]),[0,1],vel,acc)
                    if s>args.push_timeout:
                        stop='no_takeoff';break
                elif phase=='flight':
                    flight_state=state.copy();flight_state['base_vel']=state['base_vel'].copy()
                    flight_state['base_vel'][2]=actual_com_velocity[2]
                    target=flight_target(flight_state,initial,t-takeoff,landing_x,args.tuck_height)
                    command=(fc.command_flight(state,target) if args.flight_mode=='wbc'
                             else flight_command(state,target))
                else:
                    com=np.array([np.mean(landing_feet[:,0])+.005,0.,c.com0[2]])
                    command=c.command(state,com,landing_feet,[0,1])
            except RuntimeError as exc:
                stop=str(exc);break
            scene.robot.control_dofs_force(command,dofs_idx_local=np.arange(6,22))
            for _ in range(10):
                scene.step();actual=be.to_np(scene.robot.get_dofs_control_force())
                base_peak=max(base_peak,float(np.abs(actual[:6]).max()))
                torque_peak=max(torque_peak,float(np.abs(actual[6:16]).max()))
                current_peak=max(current_peak,float(np.abs(motor_current_from_joint_torque(actual[6:16])).max()))
                box_motion=max(box_motion,float(np.linalg.norm(scene.box_pos(scene.boxes[0])-[.15,0.,.04])))
            ticks+=1
            apex=max(apex,float(state['base_pos'][2]));min_up=min(min_up,float(R[2,2]))
            if i%5==0:
                history.append(dict(t=t,phase=phase,base=state['base_pos'].tolist(),
                    quat=state['base_quat'].tolist(),velocity=state['base_vel'].tolist(),
                    angular_velocity=state['base_ang'].tolist(),feet=feet.tolist(),force=state['foot_force'].tolist(),
                    q=state['dof_pos'][6:].tolist(),command=command.tolist(),
                    com=actual_com.tolist(),com_velocity=actual_com_velocity.tolist(),
                    angular_momentum=actual_momentum.tolist(),
                    floor_forces=ground_forces.tolist(),
                    flight_debug=fc.debug if phase=='flight' and args.flight_mode=='wbc' else None))
            if i%100==0:print('JUMP',ax,az,t,phase,state['base_pos'],state['base_vel'],flush=True)
            end=scene.read_state()
            if end['base_pos'][2]<.18 or (phase!='flight' and be.quat_to_mat(end['base_quat'])[2,2]<.5):
                stop='loss_of_balance';break
        end=scene.read_state();final_feet=np.array([scene.get_link_pos(n) for n in be.ANKLE_LINKS])
        candidate=bool(stop=='duration_complete' and touchdown is not None and
            args.duration-touchdown>1. and np.min(final_feet[:,0])>.22 and
            np.linalg.norm(end['base_vel'])<.1 and be.quat_to_mat(end['base_quat'])[2,2]>.98 and box_motion<.001)
        result=dict(ax=ax,az=az,stop=stop,seconds=ticks*.005,release_time=takeoff,first_airborne_time=airborne,touchdown=touchdown,
            landing_x=landing_x,apex=apex,min_up=min_up,base_force_peak=base_peak,
            torque_peak=torque_peak,current_peak=current_peak,box_motion=box_motion,
            final_feet=final_feet.tolist(),candidate_screen_passed=candidate,
            autonomous_crossing_success=False,history=history)
        results.append(result)
        (out/'results.json').write_text(json.dumps(dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},trials=results),indent=2))
        print('RESULT',json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backend',choices=('cpu','gpu'),default='gpu')
    p.add_argument('--ax',type=float,nargs='+',default=[4.,6.])
    p.add_argument('--az',type=float,nargs='+',default=[14.,18.])
    p.add_argument('--height',type=float,default=.35)
    p.add_argument('--pre-x',type=float,default=.045)
    p.add_argument('--tuck-height',type=float,default=.30)
    p.add_argument('--push-timeout',type=float,default=.25)
    p.add_argument('--release-height',type=float,default=.43)
    p.add_argument('--duration',type=float,default=3.)
    p.add_argument('--solver',choices=('osqp','clarabel'),default='clarabel')
    p.add_argument('--flight-mode',choices=('wbc','pd'),default='wbc')
    p.add_argument('--momentum-weight',type=float,default=0.)
    p.add_argument('--hard-com',action='store_true')
    p.add_argument('--push-attitude-weight',type=float,default=100.)
    p.add_argument('--launch-momentum',type=float,default=0.)
    run(p.parse_args())
