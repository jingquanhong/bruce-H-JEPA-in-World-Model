"""Observe and request one bounded native-Y bout through its sole supervisor."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,time,sys
from pathlib import Path
import numpy as np
from lateral_feedback_health import feedback_checks
from lateral_tuning_trial import LateralTuningTrial


def atomic(path,data):
    temp=path.with_suffix('.new');temp.write_text(json.dumps(data));os.replace(temp,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime',type=Path)
    parser.add_argument('--direction',choices=('left','right'),required=True)
    parser.add_argument('--seconds',type=float,default=2.)
    parser.add_argument('--speed',type=float,default=.01)
    parser.add_argument('--lateral-bias-offset',type=float,default=0.,
                        help='Temporary change relative to current stationary bias, +/-0.006 m')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    tuning_trial=LateralTuningTrial(args.runtime/'planner-tuning.json',args.lateral_bias_offset)
    if not .5<=args.seconds<=6 or not 0<args.speed<=.05:parser.error('Lateral bout requires 0.5..6 seconds and 0..0.05 m/s')
    profile=json.loads((args.runtime/'profile.json').read_text())
    if not (profile.get('allow_lateral_motion') and profile.get('allow_small_motion')):
        raise RuntimeError('This supervisor has not explicitly enabled lateral motion')
    limit=profile.get('lateral_velocity_limit_m_s',.01)
    if not isinstance(limit,(int,float)) or not 0<limit<=.05 or args.speed>limit:
        raise ValueError('Requested speed exceeds this supervisor profile')
    if args.speed>.03 and not profile.get('explicit_sidestep_geometry'):
        raise ValueError('Speeds above 0.03 m/s require explicit side-step geometry')
    if args.speed>.01 and not profile.get('planar_tracking'):
        raise ValueError('Speeds above 0.01 m/s require the opt-in planar controller')
    if profile.get('planar_tracking') and tuning_trial.offset!=0.:
        raise ValueError('Planar tracking trials keep stationary lateral trim fixed')
    if args.seconds>profile['max_motion_seconds']:raise ValueError('Configured motion ceiling exceeded')
    if args.seconds>profile.get('lateral_bout_limit_s',0):raise ValueError('Supervisor lateral ceiling exceeded')
    velocity=args.speed if args.direction=='left' else -args.speed
    request=dict(seconds=args.seconds,forward_velocity_m_s=0.,lateral_velocity_m_s=velocity)
    print('LATERAL_REQUEST_CHECK',json.dumps(request),flush=True)
    print('LATERAL_BIAS_OFFSET_M',tuning_trial.offset,flush=True)
    if not args.run:
        print('CHECK ONLY: no request written; no actuator interfaces opened.');return
    sys.path.insert(0,'/home/khadas/BRUCE/BRUCE-OP')
    import Startups.memory_manager as mm
    output=Path(__file__).resolve().parent/('lateral-'+args.direction+'-'+time.strftime('%H%M%S'))
    output.mkdir();atomic(output/'request.json',request)
    records=[];ready_since=None;sent=None;finished=None;checkpoint_before=None;reason='not_started'
    start=time.monotonic()
    estimator_stamp=None;estimator_progress_at=start
    try:
        while True:
            sample_start=time.monotonic()
            if sent is None and sample_start-start>=35:
                reason='preflight_timeout';break
            if not args.runtime.exists() or (args.runtime/'finished').exists():
                reason='supervisor_stopped';break
            try:
                state=json.loads((args.runtime/'state.json').read_text())
                motor=json.loads((args.runtime/'motor-latest.json').read_text())
            except FileNotFoundError:
                reason='supervisor_stopped';break
            estimate=mm.ESTIMATOR_STATE.get();sense=mm.SENSE_STATE.get();plan=mm.PLANNER_COMMAND.get()
            current_estimator_stamp=float(estimate['time_stamp'][0])
            rotation=estimate['body_rot_matrix']
            rotation_valid=bool(np.isfinite(rotation).all() and np.linalg.norm(rotation.T@rotation-np.eye(3))<.01)
            tilt=float(np.degrees(np.arccos(np.clip(rotation[2,2],-1,1)))) if rotation_valid else None
            contacts=sense['foot_contacts'].tolist()
            values=motor.get('values')
            values_valid=bool(values is not None and len(values)==10
                and all(v is not None and v[0] is not None and len(v[0])>=4
                        and np.isfinite(v[0]).all() for v in values))
            voltage=min(v[0][3] for v in values) if values_valid else None
            current=max(abs(v[0][2]) for v in values) if values_valid else None
            # Read the clock AFTER the snapshots. A concurrent publisher can
            # legitimately return data newer than the loop's starting time.
            now=time.monotonic();wall_now=time.time()
            if np.isfinite(current_estimator_stamp) and current_estimator_stamp!=estimator_stamp:
                estimator_stamp=current_estimator_stamp;estimator_progress_at=now
            checks=feedback_checks(now,wall_now,motor['monotonic'],float(sense['time_stamp'][0]),
                now-estimator_progress_at,current_estimator_stamp,motor['codes'],
                rotation_valid,voltage,values_valid)
            good=all(checks.values())
            row=dict(monotonic=now,phase=state['phase'],tilt_deg=tilt,contacts=contacts,
                codes=motor['codes'],minimum_voltage_V=voltage,
                maximum_abs_current_A=current,feedback_checks=checks,
                failed_checks=[k for k,v in checks.items() if not v],sample_duration_s=now-sample_start,
                motor_feedback_age_s=now-motor['monotonic'],sensor_feedback_age_s=wall_now-float(sense['time_stamp'][0]),
                estimator_progress_age_s=now-estimator_progress_at,estimator_relative_stamp=current_estimator_stamp,
                motor_positions_rad=[v[0][0] for v in values] if values_valid else None,
                joint_goal_positions_rad=motor.get('command',{}).get('goal_positions'),
                estimated_com_position=estimate['com_position'].tolist(),
                estimated_com_velocity=estimate['com_velocity'].tolist(),
                estimated_right_foot_position=estimate['right_foot_position'].tolist(),
                estimated_left_foot_position=estimate['left_foot_position'].tolist(),
                planned_right_foot_position=plan['right_foot_position'].tolist(),
                planned_left_foot_position=plan['left_foot_position'].tolist(),
                body_position=estimate['body_position'].tolist(),body_velocity=estimate['body_velocity'].tolist(),
                body_rotation=rotation.tolist(),planned_com_velocity=plan['com_velocity'].tolist(),
                planned_com_position=plan['com_position'].tolist(),planned_phase=float(plan['phase'][0]))
            records.append(row)
            if sent is None:
                eligible=(good and tilt<5 and state['phase']=='holding_balance'
                          and now-state['monotonic']<3 and any(contacts[:2]) and any(contacts[2:]))
                ready_since=(now if ready_since is None else ready_since) if eligible else None
                if ready_since is not None and now-ready_since>=3:
                    request_path=args.runtime/'step-request.json'
                    if request_path.exists():raise RuntimeError('Another request is pending')
                    checkpoint=args.runtime/'checkpoint.json'
                    checkpoint_before=checkpoint.stat().st_mtime_ns if checkpoint.exists() else 0
                    tuning_trial.apply();atomic(output/'tuning-trial.json',tuning_trial.record())
                    atomic(request_path,request);sent=now;reason='requested'
                    print('LATERAL_BOUT_REQUESTED',json.dumps(request),flush=True)
            else:
                # The original, faster supervisor owns all motor/sensor/serial
                # fault stops. Avoid a second competing freshness stop here.
                if tilt is not None and tilt>8:
                    (args.runtime/'operator-stop').touch();reason='additional_tilt_guard';break
                checkpoint=args.runtime/'checkpoint.json'
                if finished is None and checkpoint.exists() and checkpoint.stat().st_mtime_ns!=checkpoint_before:
                    result=json.loads(checkpoint.read_text());finished=now;reason='returned_to_balance'
                    tuning_trial.restore();atomic(output/'tuning-trial.json',tuning_trial.record())
                    atomic(output/'checkpoint.json',result)
                    print('LATERAL_BOUT_RETURNED',json.dumps(result),flush=True)
                if finished is not None and now-finished>=3:break
                if finished is None and now-sent>args.seconds+8:
                    (args.runtime/'operator-stop').touch();reason='bout_completion_timeout';break
            time.sleep(.05)
    finally:
        try:
            tuning_trial.restore()
        except (OSError,ValueError,RuntimeError) as error:
            reason='tuning_restore_error: '+str(error)
        atomic(output/'tuning-trial.json',tuning_trial.record())
        atomic(output/'observations.json',records)
        result=dict(reason=reason,request=request,request_sent=sent is not None,
            request_sent_monotonic=sent,completion_monotonic=finished,
            completion_observed=finished is not None,samples=len(records),
            tuning_trial=tuning_trial.record(),
            actual_lateral_motion_verified=False,
            note='Native-Y command and estimator data; real displacement/direction/stability require onsite observation.')
        valid_tilts=[r['tilt_deg'] for r in records if r['tilt_deg'] is not None]
        if records:
            result.update(maximum_tilt_deg=max(valid_tilts) if valid_tilts else None,
                minimum_voltage_V=min((r['minimum_voltage_V'] for r in records if r['minimum_voltage_V'] is not None),default=None),
                maximum_abs_current_A=max((r['maximum_abs_current_A'] for r in records if r['maximum_abs_current_A'] is not None),default=None))
        moving=[r for r in records if sent is not None and r['monotonic']>=sent]
        if moving:
            result['motion_maximum_tilt_deg']=max((r['tilt_deg'] for r in moving if r['tilt_deg'] is not None),default=None)
            result['planned_com_velocity_axis_ranges_m_s']=[[min(r['planned_com_velocity'][i] for r in moving),max(r['planned_com_velocity'][i] for r in moving)] for i in range(3)]
        if sent is not None and finished is not None:
            before=[r for r in records if sent-.8<=r['monotonic']<sent]
            after=[r for r in records if r['monotonic']>=finished+2]
            if before and after:
                delta=np.mean([r['body_position'] for r in after],axis=0)-np.mean([r['body_position'] for r in before],axis=0)
                orientation=np.asarray(before[0]['body_rotation'])
                result['estimated_body_displacement_m']=(orientation.T@delta).tolist()
        atomic(output/'summary.json',result);print('LATERAL_RESULT',output,json.dumps(result),flush=True)


if __name__=='__main__':main()
