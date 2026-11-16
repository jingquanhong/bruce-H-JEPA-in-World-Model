"""One slow 6 cm double-support preparation request; never requests a jump."""
import json
import math
import os
from pathlib import Path
import sys
import time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
sys.path.insert(0,'/home/khadas/BRUCE/BRUCE-OP')
from preparation_feedback_guard import FeedbackGuard


def main():
    import Startups.memory_manager as mm
    folder=Path(sys.argv[1])
    records=[];good=None;requested=False;request_wall=None
    started=time.monotonic()
    guard=FeedbackGuard()
    try:
        while time.monotonic()-started<30:
            now=time.monotonic()
            state=json.loads((folder/'state.json').read_text())
            motor=json.loads((folder/'motor-latest.json').read_text())
            sense=mm.SENSE_STATE.get();leg=mm.LEG_STATE.get();est=mm.ESTIMATOR_STATE.get()
            now=time.monotonic()
            ages=guard.check(now,time.time(),float(sense['time_stamp'][0]),
                             float(leg['time_stamp'][0]),motor['monotonic'],float(est['time_stamp'][0]))
            rotation_zz=float(est['body_rot_matrix'][2,2])
            if not math.isfinite(rotation_zz):raise RuntimeError('Invalid fused orientation')
            tilt=math.degrees(math.acos(max(-1.,min(1.,rotation_zz))))
            if tilt is None or not math.isfinite(tilt):raise RuntimeError('Invalid fused tilt')
            v=min(x[0][3] for x in motor['values'] if x is not None)
            codes=motor['codes']
            contacts=sense['foot_contacts'].tolist()
            if tilt>8 or v<14.5 or any(c!=128 for c in codes):
                raise RuntimeError('Preparation extra tilt/voltage/motor guard')
            records.append({'observed_monotonic':now,'state':state,'motor':motor,
                            'critical_feedback_ages':ages,'live_tilt_deg':tilt,
                            'live_contacts':contacts,'live_com_z':float(est['com_position'][2])})
            if not requested:
                okay=(state['phase']=='holding_balance' and tilt<3 and
                      any(contacts[:2]) and any(contacts[2:]) and v>15 and
                      sum(float(x)**2 for x in est['com_velocity'])<.03**2)
                good=(now if good is None else good) if okay else None
                if good is not None and now-good>3:
                    path=folder/'preparation-request.json'
                    if path.exists() or(folder/'step-request.json').exists():
                        raise RuntimeError('Another request pending')
                    temp=path.with_suffix('.new')
                    temp.write_text(json.dumps({'depth_m':.06}))
                    request_wall=time.time();os.replace(temp,path);requested=True
                    print('REQUESTED_SLOW_CROUCH_6CM_NOT_A_JUMP',flush=True)
                elif now-started>10:raise RuntimeError('Preconditions did not settle')
            else:
                result=folder/'preparation-result.json'
                if result.exists() and result.stat().st_mtime>=request_wall:
                    print('RESULT',result.read_text(),flush=True)
                    break
            time.sleep(.04)
        else:raise RuntimeError('Preparation observation timed out')
    except BaseException as error:
        if requested and folder.exists():
            report={'observer':'observe_six_cm_crouch','time':time.time(),
                    'error':repr(error),'last_observation':records[-1] if records else None}
            (folder/'observer-stop-reason.json').write_text(json.dumps(report))
            (folder/'operator-stop').touch()
        raise
    finally:
        output=Path(__file__).with_name('six-cm-observation-'+str(time.time_ns())+'.json')
        output.write_text(json.dumps({'requested':requested,'real_jump_attempted':False,'records':records}))
        print('SAVED',output,flush=True)


if __name__=='__main__':main()
