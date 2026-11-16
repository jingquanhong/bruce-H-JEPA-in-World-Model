#!/usr/bin/env python3
"""Explore obstacle steps; candidate screening does not qualify success data."""
import argparse,json
from pathlib import Path
import numpy as np
import bruce_env as be
import bruce_expert as bx
from dynamic_step import DynamicStepPolicy
from qualification import motor_current_from_joint_torque


def run(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError(out)
    scene=be.BruceScene(backend='gpu',box_specs=[dict(half=(args.thickness/2,args.width/2,h/2),pos=(args.center_x,0,h/2)) for h in args.heights])
    scene.set_friction(args.friction);scene.set_gains(np.zeros(22),np.zeros(22))
    bounds=np.r_[np.zeros(6),np.full(10,10.5),np.ones(6)];scene.robot.set_dofs_force_range(-bounds,bounds)
    row=bx._pose({'r':(0,.025),'l':(0,.025)},-.03,args.body_height)
    results=[]
    for k,h in enumerate(args.heights):
      for T in args.swing_time:
        scene.park_boxes(scene.boxes,keep=k);scene.reset_box(scene.boxes[k],(args.center_x,0,h/2))
        scene.reset_robot(row[2],row[6:],row[0])
        policy=DynamicStepPolicy(row,height=h,swing_time=T,second_time=args.second_time,
            pulse_acc=args.pulse_acc,lateral=args.lateral,swing_weight=args.swing_weight)
        policy.landing_min=args.center_x+args.thickness/2+bx.HEEL+.02
        history=[];stop='duration_complete';ticks=0;max_base=max_tau=max_current=box_motion=0.
        for i in range(int(args.duration/.005)):
            t=i*.005;state=scene.read_state()
            try:command=policy.command(scene,t,state)
            except RuntimeError as exc:stop=str(exc);break
            scene.robot.control_dofs_force(command,dofs_idx_local=np.arange(6,22))
            for _ in range(10):
                scene.step();actual=be.to_np(scene.robot.get_dofs_control_force())
                max_base=max(max_base,float(np.abs(actual[:6]).max()));max_tau=max(max_tau,float(np.abs(actual[6:16]).max()))
                max_current=max(max_current,float(np.abs(motor_current_from_joint_torque(actual[6:16])).max()))
                box_motion=max(box_motion,float(np.linalg.norm(scene.box_pos(scene.boxes[k])-[args.center_x,0,h/2])))
            ticks+=1
            if i%5==0:history.append(dict(t=t,base=state['base_pos'].tolist(),quat=state['base_quat'].tolist(),
                q=state['dof_pos'][6:].tolist(),**policy.debug))
            if i%100==0:print('STEP',h,T,t,policy.phase,state['base_pos'],flush=True)
            end=scene.read_state()
            if end['base_pos'][2]<.22 or be.quat_to_mat(end['base_quat'])[2,2]<.5:
                stop='loss_of_balance';break
        end=scene.read_state();feet=np.array([scene.get_link_pos(n) for n in be.ANKLE_LINKS])
        candidate=bool(stop=='duration_complete' and policy.final_start is not None and
            args.duration-policy.final_start>1. and np.min(feet[:,0])>args.center_x+args.thickness/2+bx.HEEL+.005 and
            np.linalg.norm(end['base_vel'])<.1 and be.quat_to_mat(end['base_quat'])[2,2]>.98 and box_motion<.001)
        result=dict(height=h,swing_time=T,stop=stop,seconds=ticks*.005,
            body_height=args.body_height,second_time=args.second_time,
            obstacle=dict(center_x=args.center_x,height=h,thickness=args.thickness,width=args.width),friction=args.friction,
            base_force_peak=max_base,torque_peak=max_tau,current_peak=max_current,box_motion=box_motion,
            final_feet=feet.tolist(),events=policy.events,candidate_screen_passed=candidate,
            autonomous_crossing_success=False,history=history)
        results.append(result);(out/'results.json').write_text(json.dumps(dict(trials=results),indent=2))
        print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ('history','events')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--heights',type=float,nargs='+',default=[.02,.04,.08])
    p.add_argument('--swing-time',type=float,nargs='+',default=[.22,.26])
    p.add_argument('--second-time',type=float,default=.28)
    p.add_argument('--pulse-acc',type=float,default=1.3)
    p.add_argument('--lateral',type=float,default=-.03)
    p.add_argument('--swing-weight',type=float,default=1000.)
    p.add_argument('--duration',type=float,default=6.)
    p.add_argument('--body-height',type=float,default=.40)
    p.add_argument('--center-x',type=float,default=.15)
    p.add_argument('--thickness',type=float,default=.05)
    p.add_argument('--width',type=float,default=.30)
    p.add_argument('--friction',type=float,default=1.)
    run(p.parse_args())
