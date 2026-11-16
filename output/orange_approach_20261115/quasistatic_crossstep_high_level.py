"""Supported single-pair cross-step diagnostic using the original full-body QP."""
import time
import os,json
from pathlib import Path
import numpy as np
import Startups.memory_manager as MM
import Settings.BRUCE_data as RDS
from Util.quasistatic_crossstep import CrossStep,base_plan,center,yaw_rotation,smooth


def load_crossstep_tuning():
    values=dict(support_lean_deg=2.,foot_advance_m=.04,tracking_integral_gain=1.5)
    origin=os.environ.get('BRUCE_STATIONARY_TUNING_FILE')
    if origin:
        path=Path(origin).parent/'quasistatic-tuning.json'
        if path.exists():values=json.loads(path.read_text())
        else:
            temp=path.with_suffix('.new');temp.write_text(json.dumps(values));os.replace(temp,path)
    if set(values)!=set(('support_lean_deg','foot_advance_m','tracking_integral_gain')):
        raise ValueError('Unexpected cross-step tuning fields')
    lean=float(values['support_lean_deg']);distance=float(values['foot_advance_m'])
    gain=float(values['tracking_integral_gain'])
    if not np.isfinite([lean,distance,gain]).all() or not 0<=lean<=3 or not .025<=distance<=.04 or not 0<=gain<=2:
        raise ValueError('Cross-step tuning exceeds 0..3 degrees / 2.5..4 cm / integral 0..2 per second')
    return dict(support_lean_deg=lean,distance=distance,tracking_integral_gain=gain)


def snapshot(robot):
    robot.update_robot_status()
    return dict(com=robot.p_wg,velocity=robot.v_wg,rotation=robot.R_wb,
        right_ankle=robot.p_wa_r,left_ankle=robot.p_wa_l,
        right_foot=robot.p_wf_r,left_foot=robot.p_wf_l,contacts=robot.foot_contacts,
        right_rotation=robot.R_wf_r,left_rotation=robot.R_wf_l)


def main_loop():
    if RDS.safe_input('Start supported single-pair cross-step controller? (y/n) ')!='y':return
    robot=RDS.BRUCE();state=snapshot(robot);rotation=yaw_rotation(state['rotation'])
    target=center(state);initial=np.array(state['com'],copy=True);start=time.monotonic()
    sequence=None;armed=True;event_index=0
    MM.THREAD_STATE.set({'high_level':np.ones(1)},opt='only')
    while True:
        loop=time.monotonic()
        if robot.thread_error():robot.stop_threading()
        state=snapshot(robot);command=MM.USER_COMMAND.get()
        mode=float(command['mode'][0]);velocity=np.asarray(command['com_xy_velocity'])
        if not np.isfinite(velocity).all() or abs(velocity[0])>.000001:
            raise RuntimeError('Cross-step diagnostic only accepts pure lateral commands')
        walking=bool(mode==1 and abs(velocity[1])>=.005)
        if not walking and sequence is None:armed=True
        if walking and sequence is None and armed:
            armed=False
            try:
                tuning=load_crossstep_tuning()
                sequence=CrossStep(state,1 if velocity[1]>0 else -1,loop,**tuning)
            except (ValueError,OSError,TypeError,KeyError) as error:
                print('QUASISTATIC_REQUEST_REJECTED',str(error),flush=True)
            else:
                event_index=0
                print('QUASISTATIC_CROSSSTEP_BEGIN',sequence.direction,'FOOT_ADVANCE_M',sequence.distance,
                      'SUPPORT_LEAN_DEG',tuning['support_lean_deg'],
                      'TRACKING_INTEGRAL_GAIN',tuning['tracking_integral_gain'],flush=True)
        if sequence is not None:
            plan=sequence.sample(state,loop,walking)
            for event in sequence.events[event_index:]:print('QUASISTATIC_EVENT',event,flush=True)
            event_index=len(sequence.events)
            if sequence.done and not walking:
                target=np.array(plan['com_position'],copy=True)
                print('QUASISTATIC_PAIR_FINISHED','aborted='+str(sequence.aborted),flush=True)
                sequence=None;armed=True
        else:
            plan=base_plan(state,rotation)
            s,ds=smooth((loop-start)/.8)
            plan['com_position']=(1.-s)*initial+s*target
            plan['com_velocity']=ds/.8*(target-initial)
        MM.PLANNER_COMMAND.set(plan)
        remaining=.002-(time.monotonic()-loop)
        if remaining>0:time.sleep(remaining)


if __name__=='__main__':
    RDS.BRUCE().clear_thread_state('high_level')
    try:main_loop()
    except (KeyboardInterrupt,RDS.ThreadStopException):
        MM.THREAD_STATE.set({'high_level':np.zeros(1)},opt='only')
    except Exception:
        import traceback
        traceback.print_exc();MM.THREAD_STATE.set({'high_level':np.array([2.])},opt='only')
