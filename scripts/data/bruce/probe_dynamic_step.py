#!/usr/bin/env python3
"""Explore obstacle steps; candidate screening does not qualify success data."""
import argparse,json
import os
import hashlib,shutil,importlib.metadata
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from pathlib import Path
import numpy as np
import bruce_env as be
import bruce_expert as bx
from dynamic_step import DynamicStepPolicy
from static_step import StaticStepPolicy
from qualification import motor_current_from_joint_torque
from collision_audit import CollisionAudit,robot_box_contacts


def run(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError(out)
    source=out/'source';source.mkdir(exist_ok=True)
    manifest={}
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path,source/path.name)
        manifest[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    model=Path(be.DEFAULT_URDF)
    for path in [model,*sorted((model.parent.parent/'meshes').glob('*.STL'))]:
        manifest[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    payload=json.dumps(manifest,sort_keys=True,indent=2)
    (source/'manifest.json').write_text(payload)
    provenance=dict(source_manifest='source/manifest.json',source_manifest_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        runtime={name:importlib.metadata.version(name) for name in ('genesis-world','pin','clarabel','numpy','scipy','h5py')})
    if args.record:args.audit=True
    scene=be.BruceScene(backend='gpu',camera=args.record,box_specs=[dict(half=(args.thickness/2,args.width/2,h/2),pos=(args.center_x,0,h/2)) for h in args.heights])
    scene.set_friction(args.friction);scene.set_gains(np.zeros(22),np.zeros(22))
    scene.scene.rigid_solver.geoms[0].set_friction(args.friction)
    for box in scene.boxes:scene.set_box_friction(box,args.friction)
    materials=dict(floor=float(be.to_np(scene.scene.rigid_solver.geoms[0].get_friction())),
        robot=float(be.to_np(scene.scene.rigid_solver.geoms[scene.robot.geom_start].get_friction())),
        boxes=[float(be.to_np(scene.scene.rigid_solver.geoms[b.geom_start].get_friction())) for b in scene.boxes],
        mixing_rule='Genesis 1.4.4 max(friction_a,friction_b,0.01)')
    bounds=np.r_[np.zeros(6),np.full(10,10.5),np.ones(6)];scene.robot.set_dofs_force_range(-bounds,bounds)
    row=bx._pose({'r':(0,.025),'l':(0,.025)},args.base_x,args.body_height)
    results=[]
    audit=CollisionAudit(scene)
    for k,h,T,repeat in [(k,h,T,r) for k,h in enumerate(args.heights) for T in args.swing_time for r in range(args.repeats)]:
        scene.park_boxes(scene.boxes,keep=k);scene.reset_box(scene.boxes[k],(args.center_x,0,h/2))
        initial_joints=row[6:].copy()
        trial_seed=args.seed+len(results)
        if args.initial_joint_noise:
            initial_joints[:10]+=np.random.default_rng(trial_seed).normal(0,args.initial_joint_noise,10)
        scene.reset_robot(row[2],initial_joints,row[0])
        mesh_transform_error=audit.check_transform(scene)
        initial_intersections=audit.intersections(scene,scene.boxes[k],np.array([args.thickness,args.width,h])/2)
        if args.policy=='ppo':
            from learned_crossing_policy import LearnedCrossingPolicy
            policy_class=LearnedCrossingPolicy
        else:policy_class=StaticStepPolicy if args.policy=='static' else DynamicStepPolicy
        policy=policy_class(row,height=h,swing_time=T,second_time=args.second_time,
            pulse_acc=args.pulse_acc,lateral=args.lateral,swing_weight=args.swing_weight,
            attitude_weight=args.attitude_weight,capture_mode=args.capture_mode,momentum_weight=args.momentum_weight,
            swing_profile=args.swing_profile,line_single=args.line_single,
            **(dict(checkpoint=args.checkpoint,center=args.center_x,thickness=args.thickness,
                stochastic=args.stochastic,seed=trial_seed) if args.policy=='ppo' else {}))
        policy.landing_min=args.landing_min or args.center_x+args.thickness/2+bx.HEEL+.02
        policy.controller.attitude_kp=args.attitude_kp;policy.controller.attitude_kd=2*np.sqrt(args.attitude_kp)
        policy.controller.foot_kp=args.foot_kp;policy.controller.foot_kd=2*np.sqrt(args.foot_kp)
        policy.controller.track_swing_roll=args.track_swing_roll
        policy.capture_bias=args.capture_bias;policy.capture_wait=args.capture_wait
        policy.second_y=args.second_y
        policy.landing_z=args.landing_z
        policy.soft_single_com=args.soft_single_com
        policy.second_profile=args.second_profile
        policy.controller.arm_posture_weight=args.arm_weight;policy.controller.limit_horizon=args.limit_horizon
        policy.controller.momentum_weight=args.momentum_weight
        policy.controller.point_contact=args.point_contact
        policy.controller.friction=min(.7,args.friction)
        policy.line_single=args.line_single
        history=[];stop='duration_complete';ticks=0;max_base=max_tau=max_current=box_motion=0.
        contact_steps=0;first_contact_time=None;sampled_intersections=[]
        recorder=None
        if args.record and not initial_intersections:
            from crossing_recorder import CrossingRecorder
            recorder=CrossingRecorder(out,len(results),scene,scene.boxes[k])
        for i in range(int(args.duration/.005)):
            t=i*.005;state=scene.read_state()
            if initial_intersections:stop='initial_mesh_obstacle_intersection';break
            if recorder and i%4==0:recorder.observation(t,state,policy.phase)
            if args.audit and i%4==0:
                hits=audit.intersections(scene,scene.boxes[k],np.array([args.thickness,args.width,h])/2)
                if hits:sampled_intersections.append(dict(t=t,links=hits))
            try:command=policy.command(scene,t,state)
            except RuntimeError as exc:stop=str(exc);break
            scene.robot.control_dofs_force(command,dofs_idx_local=np.arange(6,22))
            for _ in range(10):
                scene.step();actual=be.to_np(scene.robot.get_dofs_control_force())
                contact_count=robot_box_contacts(scene,scene.boxes[k]) if args.audit else 0
                if contact_count:
                    contact_steps+=1
                    if first_contact_time is None:first_contact_time=t+(_+1)*scene.sim_dt
                if recorder:recorder.physics_step(t+(_+1)*scene.sim_dt,actual,contact_count)
                max_base=max(max_base,float(np.abs(actual[:6]).max()));max_tau=max(max_tau,float(np.abs(actual[6:16]).max()))
                max_current=max(max_current,float(np.abs(motor_current_from_joint_torque(actual[6:16])).max()))
                box_motion=max(box_motion,float(np.linalg.norm(scene.box_pos(scene.boxes[k])-[args.center_x,0,h/2])))
            ticks+=1
            if recorder:recorder.action(command,actual)
            if i%5==0:history.append(dict(t=t,base=state['base_pos'].tolist(),quat=state['base_quat'].tolist(),
                q=state['dof_pos'][6:].tolist(),**policy.debug))
            if i%100==0:print('STEP',h,T,t,policy.phase,state['base_pos'],flush=True)
            end=scene.read_state()
            if end['base_pos'][2]<.22 or be.quat_to_mat(end['base_quat'])[2,2]<.5:
                stop='loss_of_balance';break
        end=scene.read_state();feet=np.array([scene.get_link_pos(n) for n in be.ANKLE_LINKS])
        candidate=bool(stop=='duration_complete' and policy.final_start is not None and
            args.duration-policy.final_start>1. and np.min(feet[:,0])>args.center_x+args.thickness/2+bx.HEEL+.005 and
            np.linalg.norm(end['base_vel'])<.1 and be.quat_to_mat(end['base_quat'])[2,2]>.98 and box_motion<.001
            and not initial_intersections and not contact_steps and not sampled_intersections)
        result=dict(height=h,swing_time=T,stop=stop,seconds=ticks*.005,
            body_height=args.body_height,second_time=args.second_time,trial_seed=trial_seed,
            initial_joints=initial_joints.tolist(),learned_policy=getattr(policy,'provenance',None),
            config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            provenance=provenance,
            obstacle=dict(center_x=args.center_x,height=h,thickness=args.thickness,width=args.width),friction=args.friction,
            material_friction=materials,effective_floor_robot_friction=max(materials['floor'],materials['robot']),
            base_force_peak=max_base,torque_peak=max_tau,current_peak=max_current,box_motion=box_motion,
            initial_mesh_intersections=initial_intersections,contact_audit_at_2000hz=args.audit,
            collision_mesh_transform_peak_error_m=mesh_transform_error,
            robot_obstacle_contact_steps=contact_steps,first_robot_obstacle_contact_s=first_contact_time,
            sampled_mesh_intersections=sampled_intersections,
            final_feet=feet.tolist(),events=policy.events,candidate_screen_passed=candidate,
            autonomous_crossing_success=False,history=history)
        results.append(result);(out/'results.json').write_text(json.dumps(dict(trials=results),indent=2))
        if recorder:
            result['recording']=recorder.finish(result)
            (out/'results.json').write_text(json.dumps(dict(trials=results),indent=2))
        print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ('history','events')}),flush=True)
    return 0 if any(r['autonomous_crossing_success'] for r in results) else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--heights',type=float,nargs='+',default=[.02,.04,.08])
    p.add_argument('--swing-time',type=float,nargs='+',default=[.22,.26])
    p.add_argument('--second-time',type=float,default=.28)
    p.add_argument('--pulse-acc',type=float,default=1.3)
    p.add_argument('--lateral',type=float,default=-.03)
    p.add_argument('--swing-weight',type=float,default=1000.)
    p.add_argument('--attitude-weight',type=float,default=50.)
    p.add_argument('--attitude-kp',type=float,default=40.)
    p.add_argument('--foot-kp',type=float,default=200.)
    p.add_argument('--momentum-weight',type=float,default=0.)
    p.add_argument('--capture-mode',choices=['planned','measured','quintic','recapture'],default='planned')
    p.add_argument('--landing-min',type=float)
    p.add_argument('--swing-profile',choices=['plateau','arch'],default='plateau')
    p.add_argument('--line-single',action='store_true')
    p.add_argument('--audit',action='store_true')
    p.add_argument('--track-swing-roll',action='store_true')
    p.add_argument('--capture-bias',type=float,default=0.)
    p.add_argument('--capture-wait',type=float,default=.6)
    p.add_argument('--second-y',type=float)
    p.add_argument('--arm-weight',type=float,default=100.)
    p.add_argument('--limit-horizon',type=float,default=.1)
    p.add_argument('--landing-z',type=float,default=.025)
    p.add_argument('--soft-single-com',action='store_true')
    p.add_argument('--record',action='store_true')
    p.add_argument('--second-profile',choices=['plateau','arch'])
    p.add_argument('--policy',choices=['dynamic','static','ppo'],default='dynamic')
    p.add_argument('--point-contact',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--initial-joint-noise',type=float,default=0.)
    p.add_argument('--stochastic',action='store_true')
    p.add_argument('--duration',type=float,default=6.)
    p.add_argument('--body-height',type=float,default=.40)
    p.add_argument('--base-x',type=float,default=-.03)
    p.add_argument('--center-x',type=float,default=.15)
    p.add_argument('--thickness',type=float,default=.05)
    p.add_argument('--width',type=float,default=.30)
    p.add_argument('--friction',type=float,default=1.)
    raise SystemExit(run(p.parse_args()))
