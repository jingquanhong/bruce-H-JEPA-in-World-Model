"""Live registered RGB-D tracking + range PID. Default is observation only."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import math
from pathlib import Path
import secrets
import struct
import time

import numpy as np
from PIL import Image
from scipy import ndimage

from measure_pair import component_near,surface_range
from vision_range_pid import RangeFilter,RangePID
from lateral_image_control import BearingTrimPD


def atomic(path,data):
    temporary=path.with_suffix('.new')
    temporary.write_text(json.dumps(data,allow_nan=False))
    os.replace(temporary,path)


def decode(blob):
    if len(blob)<28:
        raise ValueError('Incomplete image header')
    sequence,stamp,width,height,scale=struct.unpack_from('<QdIIf',blob)
    if (width,height)!=(640,480) or len(blob)!=28+width*height*5 or not .01<=scale<=10:
        raise ValueError('Invalid registered RGB-D frame')
    rgb=np.frombuffer(blob,np.uint8,count=width*height*3,offset=28).reshape(height,width,3)
    depth=np.frombuffer(blob,'<u2',count=width*height,offset=28+width*height*3).reshape(height,width)
    return sequence,stamp,rgb,depth.astype(float)*scale/1000


def detect(rgb,depth,intrinsic,fruit_seed,reference_seed):
    # Process only the local tracking window on the small CPU core. Full-frame
    # morphology can otherwise consume most of the measurement freshness budget.
    x0=max(0,int(min(fruit_seed[0],reference_seed[0]))-55)
    x1=min(rgb.shape[1],int(max(fruit_seed[0],reference_seed[0]))+56)
    y0=max(0,int(min(fruit_seed[1],reference_seed[1]))-55)
    y1=min(rgb.shape[0],int(max(fruit_seed[1],reference_seed[1]))+56)
    if x0>=x1 or y0>=y1:
        raise ValueError('Tracking seeds outside image')
    rgb=rgb[y0:y1,x0:x1];depth=depth[y0:y1,x0:x1]
    offset=np.array([x0,y0])
    local_intrinsic=dict(intrinsic,cx=intrinsic['cx']-x0,cy=intrinsic['cy']-y0)
    hsv=np.asarray(Image.fromarray(rgb).convert('HSV'))
    mask=(((hsv[...,0]<30)|(hsv[...,0]>245))&(hsv[...,1]>90)&(hsv[...,2]>100))
    allowed=np.zeros(mask.shape,bool)
    for x,y in (np.asarray(fruit_seed)-offset,np.asarray(reference_seed)-offset):
        allowed[max(0,int(y)-55):min(mask.shape[0],int(y)+56),
                max(0,int(x)-55):min(mask.shape[1],int(x)+56)]=True
    mask &= allowed
    mask=ndimage.binary_closing(mask,structure=np.ones((3,3)))
    labels,_=ndimage.label(mask)
    fruit_component=component_near(labels,np.asarray(fruit_seed)-offset,radius=25)
    reference_component=component_near(labels,np.asarray(reference_seed)-offset,radius=25)
    if fruit_component is not None:
        # The adjacent orange print is concave; never silently assign its range
        # to the round fruit if the camera moves away from the initial seed.
        _,fruit_mask,_,box=fruit_component
        width,height=box[2]-box[0],box[3]-box[1]
        if not (.70<=width/height<=1.40 and fruit_mask.sum()/(width*height)>=.65):
            fruit_component=None
    distinct=(fruit_component is not None and reference_component is not None
              and np.linalg.norm(fruit_component[2]-reference_component[2])>12
              and not np.any(fruit_component[1]&reference_component[1]))
    fruit=surface_range(fruit_component,depth,local_intrinsic)
    reference=surface_range(reference_component,depth,local_intrinsic)
    paired=(distinct and fruit['valid'] and reference['valid']
            and abs(fruit['optical_depth_m']-reference['optical_depth_m'])<=.10)
    for result in (fruit,reference):
        if 'center_xy' in result:
            result['center_xy']=(np.asarray(result['center_xy'])+offset).tolist()
            result['bbox_xyxy']=(np.asarray(result['bbox_xyxy'])+np.tile(offset,2)).tolist()
    return fruit,reference,bool(distinct),bool(paired)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frames',type=Path)
    parser.add_argument('--runtime',type=Path,default=Path('/dev/shm/bruce-step-4914'))
    parser.add_argument('--seconds',type=float,default=20)
    parser.add_argument('--bout-seconds',type=float,default=10.,help='Each motion bout is limited to 0.5..10 seconds.')
    parser.add_argument('--max-bouts',type=int,default=0,help='Stop after this many bouts; 0 uses session time limit.')
    parser.add_argument('--lateral-trim',action='store_true',help='Opt in to image-driven lateral landing trim, limited to +/-2 mm from the validated baseline.')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--camera-only',action='store_true',help='Observation only with motors off; no controller files required.')
    parser.add_argument('--fruit-seed',nargs=2,type=float,default=[315,268])
    parser.add_argument('--reference-seed',nargs=2,type=float,default=[286,264])
    args=parser.parse_args()
    if not 1<=args.seconds<=120:
        parser.error('seconds must be 1..120')
    if not .5<=args.bout_seconds<=10 or not 0<=args.max_bouts<=20:
        parser.error('bout-seconds must be 0.5..10 and max-bouts 0..20')
    if args.camera_only and args.run:
        parser.error('camera-only cannot enable motion')
    profile={} if args.camera_only else json.loads((args.runtime/'profile.json').read_text())
    if args.run and not profile.get('allow_vision_control',False):
        raise RuntimeError('Supervisor has not explicitly enabled vision control')
    session=secrets.token_hex(12)
    output=Path(__file__).resolve().parent/('live-pid-'+time.strftime('%H%M%S'))
    output.mkdir()
    config=dict(run=args.run,session_id=session,frames=str(args.frames),runtime=str(args.runtime),
                target_range_m=1.,stop_threshold_m=1.05,kp=.12,ki=0.,kd=.025,
                max_velocity_m_s=.05,range_ttl_s=.35,command_ttl_s=.30,
                maximum_target_bearing_deg=8.,
                lateral_image_trim=args.lateral_trim,lateral_trim_limit_m=.002,lateral_trim_slew_m_s=.001,
                maximum_bout_seconds=args.bout_seconds,maximum_bouts=args.max_bouts,maximum_session_seconds=args.seconds,
                camera_only=args.camera_only,
                note='Visible fruit surface range; center offset not calibrated; hardware frame sync unavailable.')
    atomic(output/'config.json',config)
    started=time.monotonic();last_sequence=None;last_snapshot=0.;last_print=0.
    fruit_seed,reference_seed=args.fruit_seed,args.reference_seed
    filter_=RangeFilter();pid=RangePID();side= BearingTrimPD();lateral_trim=0.
    busy=False;request_sent=0.;ready_since=None;last_good=None;last_measurement=0.
    reached=False;reason='session_timeout';rows=0;valid_rows=0;bouts=0
    common=json.loads(Path('/home/khadas/BRUCE/session-save-20261115/validated-motion-profiles-20261114.json').read_text())
    def bias(value,side_delta=0.):
        atomic(args.runtime/'planner-tuning.json',dict(forward_bias_m=value,
            minimum_width_m=common['common']['minimum_width_m'],lateral_bias_m=common['common']['lateral_bias_m']+side_delta))
    def moving_bias(speed):
        stationary=common['profiles']['stationary']['forward_bias_m']
        forward=common['profiles']['forward']['forward_bias_m']
        # Previous ground tests showed that this robot's translation depends on
        # the landing trim as well as velocity. Fade BOTH as the target nears.
        return stationary+(forward-stationary)*max(0.,min(1.,speed/.05))
    def command(mode,speed,why,measurement):
        if args.run:
            atomic(args.runtime/'vision-command.json',dict(session_id=session,monotonic=time.monotonic(),
                measurement_monotonic=measurement,forward_velocity_m_s=speed,mode=mode,reason=why))
    log=(output/'observations.jsonl').open('w')
    try:
        command('balance',0.,'initializing',0.)
        while time.monotonic()-started<args.seconds:
            now=time.monotonic()
            if not args.camera_only and (not args.runtime.exists() or (args.runtime/'finished').exists()):
                reason='supervisor_finished';break
            try:
                state={'phase':'motor_state_not_read_camera_only'} if args.camera_only else json.loads((args.runtime/'state.json').read_text())
                motor=None if args.camera_only else json.loads((args.runtime/'motor-latest.json').read_text())
            except (OSError,ValueError):
                command('balance',0.,'state_unavailable',last_measurement);time.sleep(.02);continue
            if not args.camera_only and (now-motor['monotonic']>.2 or len(motor['codes'])!=10 or any(c!=128 for c in motor['codes'])
                    or min(v[0][3] for v in motor['values'] if v is not None)<14.5):
                reason='motor_feedback_or_voltage';break
            try:
                calibration=json.loads((args.frames/'calibration.json').read_text())
                if calibration['rgb']!=calibration['depth']:
                    raise ValueError('Aligned intrinsics differ')
                if args.run and calibration.get('frame_timestamp_source')!='oldest_host_receive':
                    raise ValueError('Motion requires host-reception timestamped frames')
                sequence,stamp,rgb,depth=decode((args.frames/'live-frame.bin').read_bytes())
            except (OSError,ValueError):
                command('balance',0.,'camera_unavailable',last_measurement)
                if now-started>8:reason='camera_unavailable';break
                time.sleep(.02);continue
            if now-stamp>.30 or now<stamp:
                command('balance',0.,'camera_stale',last_measurement)
                if now-stamp>3:reason='camera_stale';break
                time.sleep(.02);continue
            if sequence!=last_sequence:
                last_sequence=sequence
                fruit,reference,distinct,paired=detect(rgb,depth,calibration['rgb'],fruit_seed,reference_seed)
                now=time.monotonic()
                if distinct:
                    fruit_seed=fruit['center_xy'];reference_seed=reference['center_xy']
                if paired:
                    filter_.add(sequence,stamp,fruit['range_m'],now)
                filtered=filter_.read(now)
                if filtered is not None and distinct:
                    valid_rows+=1
                    last_measurement=filtered[1]
                bearing=(math.degrees(math.atan2(fruit['center_xy'][0]-calibration['rgb']['cx'],
                                                 calibration['rgb']['fx'])) if distinct else None)
                if filtered is None or not distinct:
                    side.reset();lateral_trim=0.
                    pid.reset();decision=dict(mode='balance',speed=0.,reason='range_not_reliable',reached=False)
                elif abs(bearing)>8:
                    side.reset();lateral_trim=0.
                    pid.reset();decision=dict(mode='balance',speed=0.,reason='target_not_ahead',reached=False)
                    last_good=now
                else:
                    distance,last_measurement=filtered
                    decision=pid.update(distance,now);last_good=now
                    lateral_trim=side.update(bearing,now) if args.lateral_trim else 0.
                    if decision['reached']:
                        reached=True;reason='stop_band_reached'
                command(decision['mode'],decision['speed'],decision['reason'],last_measurement)
                if args.run and busy and decision['mode']=='walk' and state['phase']=='stepping':
                    bias(moving_bias(decision['speed']),lateral_trim)
                row=dict(monotonic=now,frame=sequence,frame_age_s=now-stamp,fruit=fruit,reference=reference,
                         paired=paired,filtered_range_m=filtered[0] if filtered else None,target_bearing_deg=bearing,
                         lateral_trim_m=lateral_trim,
                         decision=decision,supervisor_phase=state['phase'])
                log.write(json.dumps(row)+'\n');rows+=1
                atomic(output/'latest.json',row)
                if now-last_snapshot>1:
                    # Compression on the small camera CPU used to block this
                    # command producer for >0.5 s. An uncompressed PNG keeps the
                    # preview lossless without consuming the freshness budget.
                    temporary=output/'latest-rgb.new.png'
                    Image.fromarray(rgb).save(temporary,compress_level=0)
                    os.replace(temporary,output/'latest-rgb.png')
                    last_snapshot=now;log.flush()
                if now-last_print>1:
                    print('VISION_PID',json.dumps(dict(range_m=row['filtered_range_m'],decision=decision,
                          phase=state['phase'],frame_age_s=row['frame_age_s'])),flush=True);last_print=now
            else:
                filtered=filter_.read(now)
                if filtered is None:
                    command('balance',0.,'range_stale',last_measurement)
                else:
                    # Refresh only the last decision, never the measurement timestamp.
                    command(decision['mode'],decision['speed'],decision['reason'],last_measurement)
            if reached:
                command('balance',0.,'target_reached',last_measurement);break
            if last_good is not None and now-last_good>5:
                reason='target_lost_for_5_seconds';break
            if last_good is None and now-started>10:
                reason='target_not_acquired';break
            if args.run:
                result=args.runtime/'vision-bout-result.json'
                if busy and result.exists():
                    finished=json.loads(result.read_text())
                    if finished['vision_session_id']==session and finished['started_monotonic']>=request_sent:
                        busy=False;ready_since=None
                        if args.max_bouts and bouts>=args.max_bouts:
                            reason='bout_limit_reached';break
                standing=(state['phase']=='holding_balance' and now-state['monotonic']<3
                          and any(state['contacts'][:2]) and any(state['contacts'][2:])
                          and (state.get('fused_tilt_deg') or 0)<5
                          and min(v[0][3] for v in motor['values'] if v is not None)>15)
                eligible=standing and not busy and filtered is not None and decision['mode']=='walk'
                ready_since=(now if ready_since is None else ready_since) if eligible else None
                if ready_since is not None and now-ready_since>3:
                    if (args.runtime/'step-request.json').exists():
                        reason='another_request_pending';break
                    bias(moving_bias(decision['speed']),lateral_trim)
                    command(decision['mode'],decision['speed'],decision['reason'],last_measurement)
                    request_sent=time.monotonic()
                    atomic(args.runtime/'step-request.json',dict(seconds=args.bout_seconds,forward_velocity_m_s=.05,vision_session_id=session))
                    busy=True;ready_since=None;bouts+=1
                    print('VISION_BOUT_REQUESTED',session,flush=True)
            time.sleep(.02)
    finally:
        # A hardware fault can remove the runtime before we get here. Preserve
        # the diagnostic summary even when the supervisor has already shut down.
        try:command('balance',0.,'vision_session_ended',last_measurement)
        except OSError:pass
        # Keep refreshing the stop request while the existing controller lands.
        deadline=time.monotonic()+8
        while args.run and args.runtime.exists() and time.monotonic()<deadline:
            try:
                command('balance',0.,'vision_session_ended',last_measurement)
                state=json.loads((args.runtime/'state.json').read_text())
            except (OSError,ValueError):break
            result=args.runtime/'vision-bout-result.json'
            landed=not busy
            if result.exists():
                finished=json.loads(result.read_text())
                landed=landed or (finished['vision_session_id']==session and finished['started_monotonic']>=request_sent)
            if landed and state['phase']=='holding_balance' and not (args.runtime/'step-request.json').exists():
                bias(common['profiles']['stationary']['forward_bias_m']);break
            time.sleep(.05)
        log.close()
        summary=dict(reached=reached,reason=reason,frames=rows,accepted_filtered_frames=valid_rows,
                     actual_motion_enabled=args.run,motion_bouts_requested=bouts,
                     final_fruit_seed=fruit_seed,final_reference_seed=reference_seed)
        atomic(output/'summary.json',summary)
        print('VISION_RESULT',output,json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
