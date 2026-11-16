"""Observe a supervised diagnostic; optionally request one bounded stepping bout."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--request', action='store_true')
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--capture', type=float)
    parser.add_argument('--forward-velocity', type=float, default=0., help='Optional small forward/backward command in m/s')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not .5 <= args.seconds <= 30:
        parser.error('seconds must be 0.5..30')
    capture = args.capture if args.capture is not None else args.seconds + 20
    if not 1 <= capture <= 60:
        parser.error('capture must be 1..60 seconds')
    folder = args.runtime
    profile = json.loads((folder / 'profile.json').read_text())
    tuning_path = folder / 'planner-tuning.json'
    if tuning_path.exists():
        profile['planner_tuning'] = json.loads(tuning_path.read_text())
    from Util import stationary_imu_heading_speed_test_v2 as speed_policy
    speed_policy.FORWARD_STAGE_FILE=folder/'forward-stage.json'
    checked_forward_velocity=speed_policy.checked_forward_velocity
    try:
        args.forward_velocity = checked_forward_velocity(args.forward_velocity, args.seconds, profile.get('allow_small_motion',False), profile.get('max_motion_seconds',4.))
    except ValueError as error:
        parser.error(str(error))
    if (folder / 'translation-tuning.json').exists():
        profile['translation_tuning'] = json.loads((folder / 'translation-tuning.json').read_text())
    if args.request and args.seconds > profile.get('max_step_seconds', 5):
        parser.error('request exceeds this supervisor duration ceiling')
    import Startups.memory_manager as mm
    rows = []
    good = low = None
    requested = False
    last = None
    started = time.monotonic()
    try:
        while folder.exists() and time.monotonic() - started < capture:
            now = time.monotonic()
            state = json.loads((folder / 'state.json').read_text())
            motor = json.loads((folder / 'motor-latest.json').read_text())
            sense = mm.SENSE_STATE.get()
            est = mm.ESTIMATOR_STATE.get()
            plan = mm.PLANNER_COMMAND.get()
            contacts = sense['foot_contacts'].tolist()
            tilt = math.degrees(math.acos(max(-1., min(1., est['body_rot_matrix'][2, 2]))))
            voltage = min(x[0][3] for x in motor['values'] if x is not None)
            if motor['monotonic'] != last:
                spacing = est['body_rot_matrix'].T @ (est['left_foot_position'] - est['right_foot_position'])
                rows.append({'t': now, 'state': state, 'motor': motor, 'contacts': contacts,
                             'planner_phase': float(plan['phase'][0]), 'tilt': tilt,
                             'com': est['com_position'].tolist(), 'des_com': plan['com_position'].tolist(),
                             'imu_yaw_rad': math.atan2(est['body_rot_matrix'][1,0],est['body_rot_matrix'][0,0]),
                             'planned_yaw_rad': math.atan2(plan['body_rot_matrix'][1,0],plan['body_rot_matrix'][0,0]),
                             'lateral_spacing_m': float(spacing[1]),
                             'estimated': {k: est[k].tolist() for k in ('body_rot_matrix','body_velocity','com_velocity','right_foot_position','left_foot_position')},
                             'planned': {k: plan[k].tolist() for k in ('com_position','com_velocity','right_foot_position','left_foot_position','right_foot_velocity','left_foot_velocity')}})
                last = motor['monotonic']
            if args.request:
                if tilt>8:
                    (folder/'operator-stop').touch()
                    print('STOP_ADDITIONAL_TILT_GUARD',tilt,flush=True)
                    break
                low = (now if low is None else low) if voltage < 14.5 else None
                if low is not None and now-low > .1:
                    (folder / 'operator-stop').touch()
                    print('STOP_UNDERVOLTAGE', voltage, flush=True)
                    break
                okay = (state['phase'] == 'holding_balance' and now-state['monotonic'] < 3
                        and now-motor['monotonic'] < .2 and time.time()-sense['time_stamp'][0] < .15
                        and all(x == 128 for x in motor['codes']) and any(contacts[:2])
                        and any(contacts[2:]) and tilt < 5 and voltage > 15)
                if not requested:
                    good = (now if good is None else good) if okay else None
                    if good is not None and now-good > 3:
                        path = folder / 'step-request.json'
                        if path.exists():
                            raise RuntimeError('A stepping request is already pending')
                        temp = path.with_suffix('.new')
                        if args.forward_velocity!=0:
                            if profile.get('imu_heading_hold_revision')!=1:
                                raise RuntimeError('Not the IMU heading diagnostic runtime')
                            tuning=json.loads(tuning_path.read_text())
                            tuning['forward_bias_m']=.015 if args.forward_velocity>0 else -.015
                            update=tuning_path.with_suffix('.new');update.write_text(json.dumps(tuning));os.replace(update,tuning_path)
                        temp.write_text(json.dumps({'seconds': args.seconds, 'forward_velocity_m_s': args.forward_velocity}))
                        os.replace(temp, path)
                        requested = True
                        print('REQUESTED_SECONDS', args.seconds, 'FORWARD_VELOCITY_M_S', args.forward_velocity, flush=True)
                    elif now-started > 10:
                        print('PRECONDITIONS_NOT_SUSTAINED_NO_STEP', flush=True)
                        break
            time.sleep(.01)
    except FileNotFoundError:
        print('SUPERVISOR_FINISHED', flush=True)
    finally:
        if tuning_path.exists() and (folder/'state.json').exists():
            current=json.loads((folder/'state.json').read_text())
            if current['phase']=='holding_balance':
                tuning=json.loads(tuning_path.read_text());tuning['forward_bias_m']=-.002
                update=tuning_path.with_suffix('.new');update.write_text(json.dumps(tuning));os.replace(update,tuning_path)
                print('STATIONARY_BIAS_RESTORED',flush=True)
        output = ROOT.parent / 'diagnostics' / ('imu-speed-observation-' + str(time.time_ns()) + '.json')
        output.write_text(json.dumps({'profile': profile, 'requested': requested, 'seconds': args.seconds, 'forward_velocity_m_s': args.forward_velocity, 'records': rows}))
        print('SAVED', str(output), 'samples', len(rows), flush=True)
        if rows:
            print('SUMMARY', json.dumps({'min_voltage': min(v[0][3] for r in rows for v in r['motor']['values'] if v is not None),
                                       'max_abs_iq': max(abs(v[0][2]) for r in rows for v in r['motor']['values'] if v is not None),
                                       'motor_statuses': sorted({tuple(r['motor']['codes']) for r in rows}),
                                       'max_tilt': max(r['tilt'] for r in rows),
                                       'last_spacing_m': rows[-1]['lateral_spacing_m']}), flush=True)


if __name__ == '__main__':
    main()
