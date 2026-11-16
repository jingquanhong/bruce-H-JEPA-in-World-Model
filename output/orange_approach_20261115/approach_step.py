"""One conservative, range-gated approach bout using the verified supervisor."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time


def validate(measurement, state, motor, now, seconds):
    if not math.isfinite(seconds) or not .5 <= seconds <= 4:
        raise ValueError('Approach bouts must be 0.5..4 seconds')
    if not measurement.get('range_accepted'):
        raise ValueError('Target range not accepted')
    distance = measurement['distance_m']
    lower = measurement['lower_range_m']
    age = now-measurement['last_frame_monotonic_ns']/1e9
    if not all(math.isfinite(v) for v in (distance,lower,age)) or not 0 <= age <= 15:
        raise ValueError('Invalid or stale target measurement')
    if lower > distance or lower < .4 or distance > 4:
        raise ValueError('Invalid distance interval')
    # Conservatively allow twice the commanded 0.05 m/s, with 5 cm range
    # allowance already subtracted in lower_range_m. This is a step budget,
    # not a claim that real speed is calibrated.
    if lower-.10*seconds < 1.0:
        raise ValueError('This bout could exceed the 1 m stopping budget')
    if state['phase'] != 'holding_balance' or not 0 <= now-state['monotonic'] < 3:
        raise ValueError('Standing state is stale or not holding')
    if (not 0 <= now-motor['monotonic'] < .2 or len(motor['codes']) != 10
            or any(code != 128 for code in motor['codes'])):
        raise ValueError('Motor feedback is stale or abnormal')
    if motor['command']['BEAR_enable'] != [1.]:
        raise ValueError('Motor command is not enabled')
    return dict(seconds=seconds,forward_velocity_m_s=.05,
                measured_range_m=distance,conservative_lower_range_m=lower,
                maximum_budgeted_travel_m=.10*seconds,measurement_age_s=age)


def atomic(path, data):
    temporary=path.with_suffix('.new')
    temporary.write_text(json.dumps(data))
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurement',type=Path)
    parser.add_argument('--runtime',type=Path,default=Path('/dev/shm/bruce-step-4914'))
    parser.add_argument('--seconds',type=float,default=4)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    measurement=json.loads(args.measurement.read_text())
    state=json.loads((args.runtime/'state.json').read_text())
    motor=json.loads((args.runtime/'motor-latest.json').read_text())
    plan=validate(measurement,state,motor,time.monotonic(),args.seconds)
    print('RANGE_GATED_PLAN',json.dumps(plan),flush=True)
    if not args.run:
        print('CHECK ONLY; no request or tuning written',flush=True)
        return
    if (args.runtime/'step-request.json').exists():
        raise RuntimeError('Another motion request is pending')
    saved=json.loads(Path('/home/khadas/BRUCE/session-save-20261115/validated-motion-profiles-20261114.json').read_text())
    common=saved['common']
    tuning=args.runtime/'planner-tuning.json'
    def set_bias(bias):
        atomic(tuning,dict(forward_bias_m=bias,minimum_width_m=common['minimum_width_m'],lateral_bias_m=common['lateral_bias_m']))
    started=time.time()
    try:
        set_bias(saved['profiles']['forward']['forward_bias_m'])
        subprocess.run(['nice','-n','10','taskset','-c','1','python3',
            '/home/khadas/BRUCE/BRUCE-OP/Util/observe_supported_step.py',str(args.runtime),
            '--request','--seconds',str(args.seconds),'--forward-velocity','.05',
            '--capture',str(args.seconds+10)],check=True)
    finally:
        deadline=time.monotonic()+10
        while args.runtime.exists() and not (args.runtime/'finished').exists():
            current=json.loads((args.runtime/'state.json').read_text())
            if current['phase']=='holding_balance':
                set_bias(saved['profiles']['stationary']['forward_bias_m'])
                print('RESTORED_STATIONARY_BIAS',flush=True)
                break
            if time.monotonic()>deadline:
                raise RuntimeError('No standing confirmation; inspect supervisor before any further request')
            time.sleep(.1)
    motion=args.runtime/'last-motion-request.json'
    actually_started=motion.exists() and motion.stat().st_mtime>=started
    report=dict(plan=plan,actually_started=actually_started,measurement=str(args.measurement),
                finished_at=time.time(),current_state=json.loads((args.runtime/'state.json').read_text()) if args.runtime.exists() else None)
    output=Path(__file__).resolve().parent/('approach-motion-'+str(time.time_ns())+'.json')
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('MOTION_RECORD',output,'ACTUALLY_STARTED',actually_started,flush=True)


if __name__=='__main__':
    main()
