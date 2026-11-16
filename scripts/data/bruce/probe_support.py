#!/usr/bin/env python3
"""Short unassisted COM transfer and single-foot-lift diagnostics, not training data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import bruce_env as be
import bruce_expert as bx
from qualification import motor_current_from_joint_torque
from support_wbc import SupportController


from support_reference import plan, smooth, support_cop


def run(args):
    out = args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():
        raise FileExistsError('Choose a new output directory')
    initial = bx._pose({'r':(0,.025),'l':(0,.025)},-.03,args.height)
    scene = be.BruceScene(backend=args.backend,
        box_specs=[dict(half=(.025,.15,.04),pos=(.15,0,.04))])
    scene.set_friction(1.)
    scene.set_gains(np.zeros(22),np.zeros(22))
    scene.robot.set_dofs_force_range(-np.r_[np.zeros(6),np.full(10,10.5),np.ones(6)],
                                     np.r_[np.zeros(6),np.full(10,10.5),np.ones(6)])
    results=[]
    for attitude in args.attitude:
        scene.reset_robot(initial[2],initial[6:],initial[0])
        scene.reset_box(scene.boxes[0],(.15,0,.04))
        controller = SupportController(initial,attitude_weight=attitude,line_contact=args.line_contact,
            com_weight=args.com_weight,swing_weight=args.swing_weight)
        history=[]; stop='duration_complete'; peak_base=peak_leg=peak_current=0.
        ticks=0;end=scene.read_state()
        for i in range(int((args.shift_time+4.)/.005)):
            t=i*.005; state=scene.read_state()
            state['support_cop']=support_cop(scene)
            com,feet,stance=plan(t,controller,args.lift,args.shift_time,args.support_offset)
            cm,fm,_=plan(t-.005,controller,args.lift,args.shift_time,args.support_offset)
            cp,fp,_=plan(t+.005,controller,args.lift,args.shift_time,args.support_offset)
            if args.track_cop:
                target_y=state['support_cop'][0][1]+args.support_offset
                com[1]=target_y*smooth((t-.5)/args.shift_time)
                cm[1]=target_y*smooth((t-.505)/args.shift_time)
                cp[1]=target_y*smooth((t-.495)/args.shift_time)
            try:
                command=controller.command(state,com,feet,stance,
                    (cp-cm)/.01,(cp-2*com+cm)/.005**2,
                    (fp-fm)/.01,(fp-2*feet+fm)/.005**2)
            except RuntimeError as exc:
                stop=str(exc);break
            scene.robot.control_dofs_force(command,dofs_idx_local=np.arange(6,22))
            for _ in range(10):
                scene.step()
                actual=be.to_np(scene.robot.get_dofs_control_force())
                peak_base=max(peak_base,float(np.abs(actual[:6]).max()))
                peak_leg=max(peak_leg,float(np.abs(actual[6:16]).max()))
                peak_current=max(peak_current,float(np.abs(motor_current_from_joint_torque(actual[6:16])).max()))
            if i%10==0:
                history.append(dict(t=t,base_pos=state['base_pos'].tolist(),
                    base_quat=state['base_quat'].tolist(),q=state['dof_pos'][6:].tolist(),
                    dq=state['dof_vel'][6:].tolist(),foot_force=state['foot_force'].tolist(),**controller.debug))
                history[-1]['support_cop']=state['support_cop']
            if i%100==0:
                print('SUPPORT',attitude,round(t,3),'com',controller.debug['com'],
                      'feet',controller.debug['feet'],'forces',state['foot_force'].tolist(),flush=True)
            end=scene.read_state()
            ticks+=1
            if end['base_pos'][2]<.22 or be.quat_to_mat(end['base_quat'])[2,2]<.65:
                stop='loss_of_balance';break
        result=dict(attitude_weight=attitude,stop_reason=stop,seconds=ticks*.005,
            base_force_peak=peak_base,leg_torque_peak=peak_leg,motor_current_peak=peak_current,
            final_base=end['base_pos'].tolist(),history=history,
            autonomous_crossing_success=False,purpose='support_debug_not_training_data')
        results.append(result)
        (out/'results.json').write_text(json.dumps(dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            source_sha256={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (__file__,str(Path(__file__).with_name('support_wbc.py')))},
            trials=results),indent=2))
        print('RESULT',json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backend',choices=('cpu','gpu'),default='gpu')
    p.add_argument('--height',type=float,default=.40)
    p.add_argument('--attitude',type=float,nargs='+',default=[.1,1.,10.])
    p.add_argument('--lift',type=float,default=.10)
    p.add_argument('--shift-time',type=float,default=2.)
    p.add_argument('--support-offset',type=float,default=0.)
    p.add_argument('--line-contact',action='store_true')
    p.add_argument('--com-weight',type=float,default=1000.)
    p.add_argument('--swing-weight',type=float,default=100.)
    p.add_argument('--track-cop',action='store_true')
    run(p.parse_args())
