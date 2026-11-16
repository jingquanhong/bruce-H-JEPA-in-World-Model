"""Bounded, supported standing/stepping diagnostic with optional bounded forward/backward motion.

Default is a non-actuating check. --run requires physical support and an
operator with an immediately accessible motor emergency stop.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import datetime
import importlib
import json
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections import deque
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
from Play.Walking.walking_macros import PARAMETER_DEFAULT, PARAMETER_MIN, PARAMETER_MAX, FOOT_CLEARANCE

def atomic_json(path,value):
    temp=path.with_suffix('.new')
    temp.write_text(json.dumps(value))
    os.replace(temp,path)

def check_tilt(tilt,now,started):
    # Match the original controller's 45 degree / 0.3 second tilt condition.
    if tilt<=45:return None
    if started is None:return now
    if now-started>.3:raise RuntimeError('Tilt >45 degrees continuously for 0.3 seconds')
    return started

def step_request_seconds(value,maximum=5.):
    seconds=float(value)
    if not np.isfinite(maximum) or not 5<=maximum<=30:raise ValueError('Maximum stepping duration must be 5..30 seconds')
    if not np.isfinite(seconds) or not .5<=seconds<=maximum:
        raise ValueError('Each stepping request must be 0.5..'+str(maximum)+' seconds')
    return seconds

FORWARD_STAGE_FILE = None


def checked_forward_velocity(value, seconds, allowed=False, maximum=4.):
    velocity=float(value)
    ceiling=.05
    if FORWARD_STAGE_FILE is not None:
        stage=json.loads(FORWARD_STAGE_FILE.read_text())
        if set(stage)!=set(('speed_limit_m_s','maximum_seconds')):raise ValueError('Invalid speed-stage fields')
        ceiling=float(stage['speed_limit_m_s']);duration=float(stage['maximum_seconds'])
        if not np.isfinite([ceiling,duration]).all() or ceiling not in (.05,.10,.15,.20,.25) or not .5<=duration<=12:
            raise ValueError('Speed stage must be finite, 0.05/0.10/0.15/0.20/0.25 m/s and at most 12 seconds')
        maximum=min(maximum,duration)
    if not np.isfinite(velocity) or abs(velocity)>ceiling:
        raise ValueError('Requested velocity exceeds the current speed stage')
    if velocity != 0.:
        if not allowed:raise ValueError('This supervisor only permits stationary steps')
        if not np.isfinite(maximum) or not .5<=maximum<=30.:raise ValueError('Motion duration ceiling must be 0.5..30 seconds')
        if not .5<=seconds<=maximum:raise ValueError('Moving bout exceeds configured duration ceiling')
    return velocity

def checked_foot_spacing(value):
    value=float(value)
    if not np.isfinite(value) or not PARAMETER_MIN[FOOT_CLEARANCE]<=value<=PARAMETER_MAX[FOOT_CLEARANCE]:
        raise ValueError('Foot spacing must stay within the original walking parameter limits')
    return value

def motor_worker(folder):
    from pybear import Manager
    import Startups.memory_manager as mm
    original=Manager.BEAR.bulk_read_write
    torque_write=Manager.BEAR.set_torque_enable
    last=[0.,None]
    recent=deque(maxlen=75)
    def write(self,*args,**kwargs):
        record={'monotonic':time.monotonic(),'kind':'software_torque_write','values':args,'stack':traceback.format_stack(limit=7)}
        with (folder/'command-origin.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print('SOFTWARE_TORQUE_WRITE',record['monotonic'],args,flush=True)
        return torque_write(self,*args,**kwargs)
    def exchange(self,*args,**kwargs):
        call_args=list(args)
        if len(call_args)>1:
            call_args[1]=list(call_args[1])
            for field in ('present_iq','input_voltage'):
                if field not in call_args[1]:call_args[1].append(field)
        result=original(self,*call_args,**kwargs)
        now=time.monotonic()
        codes=[x[1] if x is not None else None for x in result] if result is not None else None
        changed=codes!=last[1]
        if changed or now-last[0]>.04:
            command=mm.LEG_COMMAND.get()
            snapshot={'monotonic':now,'codes':codes,'read_registers':call_args[1],'values':result,'command':{k:command[k].tolist() for k in ('BEAR_enable','BEAR_mode','damping','goal_positions','goal_velocities','goal_torques')}}
            recent.append(snapshot)
            # Publish fault history before the supervisor can observe the fault
            # and interrupt this worker during cleanup.
            if codes and any(c is not None and c>=128 and c!=128 for c in codes):atomic_json(folder/'motor-before-fault.json',list(recent))
            atomic_json(folder/'motor-latest.json',snapshot)
            last[0]=now
        if changed:
            with (folder/'motor-status.jsonl').open('a') as stream:
                stream.write(json.dumps({'monotonic':now,'codes':codes})+'\n')
            print('MOTOR_STATUS',now,codes,flush=True)
            last[1]=codes
        return result
    Manager.BEAR.set_torque_enable=write
    Manager.BEAR.bulk_read_write=exchange
    runpy.run_module('Startups.run_bear',run_name='__main__')

def guard_worker(folder):
    start=time.monotonic()
    maximum=json.loads((folder/'limits.json').read_text())['maximum_seconds']
    while True:
        if (folder/'finished').exists():return
        expired=(maximum is not None and time.monotonic()-start>maximum) or time.time()-(folder/'heartbeat').stat().st_mtime>1.
        if expired:
            print('GUARD_STOP',time.monotonic(),'heartbeat_age',time.time()-(folder/'heartbeat').stat().st_mtime,'maximum_seconds',maximum,flush=True)
            try:children=json.loads((folder/'children.json').read_text())
            except (OSError,ValueError):children={}
            for key in ['bear','low_level','high_level','estimation','sense']:
                if key in children:
                    try:os.kill(children[key],signal.SIGINT)
                    except ProcessLookupError:pass
            return
        time.sleep(.05)


def preload_worker(folder,module,key):
    height=os.environ.get('BRUCE_STATIONARY_SWING_HEIGHT')
    if height is not None:
        height=float(height)
        if not np.isfinite(height) or not .01<=height<=.035:
            raise ValueError('Diagnostic swing height must be 0.01..0.035 m')
        import Play.Walking.walking_macros as walking
        walking.zm_l=height;walking.zm_r=height
        print('STATIONARY_SWING_HEIGHT_METERS',height,flush=True)
    importlib.import_module(module)
    import Settings.BRUCE_data as RDS
    from Util import math_function as mf
    mf.exp_filter(0.0,0.0,0.0)
    for name in ('Rx','Ry','Rz'):
        if hasattr(mf,name):getattr(mf,name)(0.0)
    mf.hat(np.zeros(3));mf.hatexp(np.zeros(3));mf.logvee(np.eye(3));mf.norm(np.ones(3))
    if key=='low_level':
        import faulthandler
        faulthandler.register(signal.SIGUSR1,all_threads=True)
        fixture=ROOT.parent/'diagnostics'/'control-warmup-fixture.json'
        if fixture.exists():
            from Util.control_preflight import warm_low_level
            count=warm_low_level(sys.modules[module],json.loads(fixture.read_text()))
            print('WBC_PRIVATE_MEMORY_PREFLIGHT_PASSED',count,flush=True)
    RDS.BRUCE()
    (folder/('warm_'+key)).touch()
    while not (folder/('start_'+key)).exists():time.sleep(.02)
    subprocess.run(['sudo','-n','chrt','-f','--pid','50',str(os.getpid())],check=True)
    if key=='sense' and (folder/'passive_check').exists():
        # Sensor-only diagnosis; the supervisor asserts BEAR_enable remains zero.
        sensor_module=sys.modules[module]
        from Settings.BRUCE_macros import PICO_port,PICO_baudrate
        sensor_module.sm=sensor_module.sense_manager.SenseManager(port=PICO_port,baudrate=PICO_baudrate)
        sensor_module.sm.send_data(pico_mode='nominal')
        try:
            sensor_module.IMU_check()
            sensor_module.main_loop()
        except KeyboardInterrupt:
            pass
        finally:
            sensor_module.sm.send_data(pico_mode='idle');sensor_module.sm.close_port()
            sensor_module.MM.THREAD_STATE.set({'sense':np.zeros(1)},opt='only')
        return
    sys.modules.pop(module,None)
    runpy.run_module(module,run_name='__main__')

class Probe:
    def __init__(self,args):
        import Startups.memory_manager as mm
        self.mm=mm;self.args=args
        self.archive=ROOT.parent/'diagnostics'/('step-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.folder=Path('/dev/shm')/('bruce-step-'+str(os.getpid()))
        self.folder.mkdir(parents=True)
        global FORWARD_STAGE_FILE
        FORWARD_STAGE_FILE=self.folder/'forward-stage.json'
        atomic_json(FORWARD_STAGE_FILE,{'speed_limit_m_s':.10,'maximum_seconds':4.})
        self.children={};self.logs={};self.guard=None;self.sensor=None
        self.phase='setup';self.enabled=False;self.records=[];self.last_record=0.;self.last_print=0.
        self.last_sensor=time.monotonic();self.est_progress=None;self.est_changed=time.monotonic();self.seen=set()
        self.comm_failure=None;self.stop_reason='not completed'
        self.cpu_map={'bear':4,'sense':2,'estimation':6,'low_level':7,'high_level':5}
        self.allowed_cpus=os.sched_getaffinity(0)
        required_cpus=set(self.cpu_map.values())
        if not required_cpus.issubset(self.allowed_cpus):
            raise RuntimeError('Launch supervisor without taskset; controller CPU cores are unavailable')
        os.sched_setaffinity(0,{min(self.allowed_cpus)})
        self.tilt_started=None
        self.step_observations={}
        self.yaw_drift_started=None
    def save_children(self):atomic_json(self.folder/'children.json',{k:p.pid for k,p in self.children.items()})
    def spawn(self,key,module=None,answer=False):
        if key in self.children:
            (self.folder/('start_'+key)).touch()
            return self.children[key]
        f=(self.folder/(key+'.log')).open('w');self.logs[key]=f
        command=[sys.executable,'-u','-m',module] if module else [sys.executable,'-u',str(Path(__file__)), '--worker','bear','--folder',str(self.folder)]
        p=subprocess.Popen([str(ROOT/'Startups/rt_run.sh'),*command],cwd=ROOT,stdin=subprocess.PIPE if answer else subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        if key in self.cpu_map and self.cpu_map[key] in self.allowed_cpus:os.sched_setaffinity(p.pid,{self.cpu_map[key]})
        self.children[key]=p;self.save_children()
        if answer:p.stdin.write(b'y\n');p.stdin.flush();p.stdin.close()
        return p
    def preload(self,key,module,answer=False):
        f=(self.folder/(key+'.log')).open('w');self.logs[key]=f
        command=[sys.executable,'-u',str(Path(__file__)),'--worker','preload','--folder',str(self.folder),'--module',module,'--key',key]
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE if answer else subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        if key in self.cpu_map and self.cpu_map[key] in self.allowed_cpus:os.sched_setaffinity(p.pid,{self.cpu_map[key]})
        self.children[key]=p;self.save_children()
        if answer:p.stdin.write(b'y\n');p.stdin.flush();p.stdin.close()
    def watch(self):
        now=time.monotonic();(self.folder/'heartbeat').touch()
        if (self.folder/'operator-stop').exists():raise KeyboardInterrupt('Operator requested stop')
        states=self.mm.THREAD_STATE.get()
        for key,p in self.children.items():
            if p.poll() is not None:raise RuntimeError(key+' process exited')
            value=states[key][0]
            if value>=2 or (key in self.seen and value!=1):raise RuntimeError(key+' thread state '+str(value))
            if value==1:self.seen.add(key)
        if self.sensor:
            if self.sensor.read_data():self.last_sensor=now
            if now-self.last_sensor>.25:raise RuntimeError('IMU frames stale')
            accel=self.sensor.accel;contacts=self.sensor.foot_contact
        elif 'sense' in self.seen:
            data=self.mm.SENSE_STATE.get();accel=data['imu_acceleration'];contacts=data['foot_contacts']
            if time.time()-data['time_stamp'][0]>.15:raise RuntimeError('SENSE timestamp stale')
        else:accel=np.array([0.,0.,9.81]);contacts=np.zeros(4)
        norm=np.linalg.norm(accel)
        tilt=float(np.degrees(np.arccos(np.clip(accel[2]/norm,-1,1)))) if norm>1 else 0.
        fused_tilt=None;est=None
        if self.enabled:
            leg=self.mm.LEG_STATE.get()
            if time.time()-leg['time_stamp'][0]>.15:raise RuntimeError('Leg feedback stale')
            if self.args.hold_hip_yaw and 'low_level' in self.seen:
                yaw_error=np.max(np.abs(leg['joint_positions'][[0,5]]+np.pi/2))
                self.yaw_drift_started=(now if self.yaw_drift_started is None else self.yaw_drift_started) if yaw_error>.35 else None
                if self.yaw_drift_started is not None and now-self.yaw_drift_started>.2:
                    raise RuntimeError('Stationary hip yaw deviated >20 degrees for 0.2 seconds')
            p=self.folder/'motor-latest.json'
            if p.exists():
                status=json.loads(p.read_text());codes=status['codes']
                if codes and any(c is not None and c>=128 and c!=128 for c in codes):raise RuntimeError('Motor fault '+str(codes))
                valid=codes is not None and len(codes)==10 and all(c==128 for c in codes)
                if valid:self.comm_failure=None
                elif self.comm_failure is None:self.comm_failure=now
                elif now-self.comm_failure>.05:raise RuntimeError('Continuous corrupt/missing motor responses')
                if now-status['monotonic']>.2:raise RuntimeError('Motor telemetry stale')
        if 'estimation' in self.seen:
            est=self.mm.ESTIMATOR_STATE.get()
            if not all(np.isfinite(v).all() for v in est.values()):raise RuntimeError('Nonfinite estimator output')
            progress=float(est['time_stamp'][0])
            if progress!=self.est_progress:self.est_progress=progress;self.est_changed=now
            elif now-self.est_changed>.15:raise RuntimeError('Estimator stopped updating')
            fused_tilt=float(np.degrees(np.arccos(np.clip(est['body_rot_matrix'][2,2],-1,1))))
        # Acceleration includes walking impacts. Once estimation is running,
        # use the same fused orientation as the original WBC safety check.
        self.tilt_started=check_tilt(fused_tilt if fused_tilt is not None else tilt,now,self.tilt_started)
        plan=self.mm.PLANNER_COMMAND.get()
        if now-self.last_record>.04:
            record={'monotonic':now,'phase':self.phase,'tilt_deg':tilt,'fused_tilt_deg':fused_tilt,'contacts':contacts.tolist(),'planner_phase':float(plan['phase'][0]),'planner_mode':float(plan['mode'][0]),'joint_positions':self.mm.LEG_STATE.get()['joint_positions'].tolist()}
            record['imu_acceleration']=accel.tolist()
            if self.sensor:record['imu_ang_rate']=self.sensor.omega.tolist()
            elif 'sense' in self.seen:record['imu_ang_rate']=data['imu_ang_rate'].tolist()
            if self.enabled:record['joint_velocities']=leg['joint_velocities'].tolist()
            if est is not None:
                record['estimated']={k:est[k].tolist() for k in ('body_position','body_velocity','body_rot_matrix','body_ang_rate','com_position','com_velocity','right_foot_position','left_foot_position')}
                record['planned']={k:plan[k].tolist() for k in ('body_rot_matrix','com_position','com_velocity','right_foot_position','left_foot_position','right_foot_velocity','left_foot_velocity','right_foot_rot_matrix','left_foot_rot_matrix')}
            self.records.append(record)
            self.last_record=now
        if now-self.last_print>2:
            print('STATE',self.phase,'accel_angle=%.1f'%tilt,'fused_tilt='+str(fused_tilt),'contacts='+str(contacts.tolist()),'planner_phase='+str(plan['phase'][0]),flush=True)
            atomic_json(self.folder/'state.json',self.records[-1]);self.last_print=now
        return tilt,contacts,plan
    def wait_ready(self,key,timeout=15):
        end=time.monotonic()+timeout
        diagnosed=False
        while key not in self.seen:
            self.watch()
            if key=='low_level' and not diagnosed and time.monotonic()>end-timeout+4:
                self.children[key].send_signal(signal.SIGUSR1);diagnosed=True
            if time.monotonic()>end:raise RuntimeError(key+' startup timeout')
            time.sleep(.01)
        print('THREAD_READY',key,flush=True)
    def command(self,mode,forward_velocity=0.,yaw_rate=0.):
        self.mm.USER_COMMAND.set({'time_stamp':np.array([time.time()]),'mode':np.array([float(mode)]),'com_xy_velocity':np.array([forward_velocity if mode==1 else 0.,0.]),'yaw_rate':np.array([yaw_rate if mode==1 else 0.]),'com_position_change_scaled':np.zeros(3),'body_euler_angle_change':np.zeros(3),'right_foot_yaw_angle_change':np.zeros(1),'left_foot_yaw_angle_change':np.zeros(1),'foot_clearance':np.array([self.args.foot_spacing]),'cooling_speed':np.zeros(1),'dcm_offset_compensation':np.zeros(2),'com_offset_compensation':np.zeros(1)},opt='only')
        self.mm.THREAD_STATE.set({'top_level':np.ones(1)},opt='only')
    def checkpoint(self):
        atomic_json(self.folder/'checkpoint.json',{'result':self.stop_reason,'step_contact_patterns':{k:[list(x) for x in v] for k,v in self.step_observations.items()}})
    def perform_steps(self,seconds,forward_velocity=0.):
        seconds=step_request_seconds(seconds,self.args.max_step_seconds)
        forward_velocity=checked_forward_velocity(forward_velocity,seconds,self.args.allow_small_motion,self.args.max_motion_seconds)
        atomic_json(self.folder/'last-motion-request.json',{'seconds':seconds,'forward_velocity_m_s':forward_velocity})
        self.step_observations={};observed=set();actual_swing=set()
        self.phase='stepping';print('SUPPORTED_STEPPING_START',seconds,'FORWARD_VELOCITY_M_S',forward_velocity,flush=True);end=time.monotonic()+seconds
        while time.monotonic()<end:
            self.command(1,forward_velocity);_,contacts,plan=self.watch();phase=int(plan['phase'][0]);observed.add(phase)
            pattern=tuple(int(x) for x in contacts)
            self.step_observations.setdefault(str(phase),set()).add(pattern)
            if phase==1 and np.any(contacts[:2]) and not np.any(contacts[2:]):actual_swing.add(1)
            if phase==2 and np.any(contacts[2:]) and not np.any(contacts[:2]):actual_swing.add(2)
            time.sleep(.01)
        self.phase='return_to_balance';end=time.monotonic()+4;stable_since=None
        while True:
            self.command(0);_,contacts,plan=self.watch();now=time.monotonic()
            balanced=plan['phase'][0]==0 and np.any(contacts[:2]) and np.any(contacts[2:])
            stable_since=(stable_since if stable_since is not None else now) if balanced else None
            if stable_since is not None and now-stable_since>1:break
            if now>end:raise RuntimeError('Did not return to double-support balance')
            time.sleep(.01)
        complete={1,2}.issubset(observed) and actual_swing=={1,2}
        self.stop_reason='PASS_SHORT_SUPPORTED_STEP_TEST' if complete else 'INCOMPLETE_STEP_CONTACT_EVIDENCE'
        self.phase='holding_balance';self.checkpoint()
        print('STEP_FINISHED_KEEPING_BALANCE',self.stop_reason,flush=True)
    def perform_turn(self,degrees):
        from Util.imu_turn_policy import checked_turn_degrees,wrap,turn_rate
        from Util.imu_heading_hold import yaw_and_rate
        degrees=checked_turn_degrees(degrees)
        if not self.args.allow_small_motion:raise ValueError("Turning not allowed by this supervisor")
        imu=self.mm.ESTIMATOR_STATE.get();start_yaw,_=yaw_and_rate(imu['body_rot_matrix'],imu['body_ang_rate'])
        anchor=(imu['right_foot_position'][:2]+imu['left_foot_position'][:2])/2
        target=wrap(start_yaw+np.deg2rad(degrees));deadline=time.monotonic()+12;settled=None
        atomic_json(self.folder/'last-turn-request.json',{'degrees':degrees,'start_yaw_rad':start_yaw,'target_yaw_rad':target})
        self.phase='turning';outcome='ANGLE_NOT_REACHED'
        filtered=start_yaw;filtered_rate=0.;last_sample=time.monotonic();last_log=0.
        print('NATIVE_IMU_TURN_START',degrees,flush=True)
        while time.monotonic()<deadline:
            imu=self.mm.ESTIMATOR_STATE.get();yaw,rate=yaw_and_rate(imu['body_rot_matrix'],imu['body_ang_rate'])
            now=time.monotonic();dt=max(0.,min(.1,now-last_sample));last_sample=now
            filtered=wrap(filtered+dt/(.30+dt)*wrap(yaw-filtered));filtered_rate+=dt/(.20+dt)*(rate-filtered_rate)
            error=wrap(target-filtered)
            midpoint=(imu['right_foot_position'][:2]+imu['left_foot_position'][:2])/2
            if np.linalg.norm(midpoint-anchor)>.12:outcome='TURN_DRIFT_LIMIT';break
            if now-last_log>.2:
                with(self.folder/'native-turn-feedback.jsonl').open('a') as stream:stream.write(json.dumps({'t':now,'yaw_rad':yaw,'filtered_yaw_rad':filtered,'target_rad':target,'error_rad':error})+'\n')
                last_log=now
            settled=(now if settled is None else settled) if abs(error)<np.deg2rad(1.) else None
            if settled is not None and now-settled>.35:outcome='IMU_ANGLE_REACHED';break
            self.command(1,0.,turn_rate(error,filtered_rate));self.watch();time.sleep(.01)
        self.phase='return_to_balance';deadline=time.monotonic()+4;settled=None
        while True:
            self.command(0);_,contacts,plan=self.watch();now=time.monotonic()
            balanced=plan['phase'][0]==0 and np.any(contacts[:2]) and np.any(contacts[2:])
            settled=(now if settled is None else settled) if balanced else None
            if settled is not None and now-settled>1:break
            if now>deadline:raise RuntimeError('Turn did not return to double-support balance')
            time.sleep(.01)
        imu=self.mm.ESTIMATOR_STATE.get();yaw,_=yaw_and_rate(imu['body_rot_matrix'],imu['body_ang_rate'])
        result={'result':outcome,'requested_degrees':degrees,'measured_delta_degrees':float(np.rad2deg(wrap(yaw-start_yaw))),'final_error_degrees':float(np.rad2deg(wrap(target-yaw))),'ground_truth':False}
        atomic_json(self.folder/'turn-result.json',result);self.stop_reason=outcome;self.phase='holding_balance'
        print('NATIVE_IMU_TURN_FINISHED_KEEPING_BALANCE',result,flush=True)
    def run(self):
        from Library.BRUCE_SENSE.Manager import SenseManager
        import Library.ROBOT_MODEL.BRUCE_kinematics as kin
        for port in ('/dev/ttyUSB1','/dev/ttyACM0'):
            if subprocess.run(['fuser',port],capture_output=True).returncode==0:raise RuntimeError('Serial port in use')
        if self.mm.LEG_COMMAND.get()['BEAR_enable'][0]!=0:raise RuntimeError('Existing enable command')
        self.mm.THREAD_STATE.set({k:np.zeros(1) for k in self.mm.THREAD_STATE.get()})
        self.mm.LEG_COMMAND.set({'BEAR_enable':np.zeros(1),'damping':np.zeros(1)},opt='only')
        (self.folder/'heartbeat').touch();self.save_children()
        atomic_json(self.folder/'limits.json',{'maximum_seconds':None if self.args.keep_standing else min(360,90+self.args.ground_wait+self.args.seconds+self.args.stand_seconds)})
        if self.args.passive_check:(self.folder/'passive_check').touch()
        gf=(self.folder/'guard.log').open('w');self.logs['guard']=gf
        self.guard=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker','guard','--folder',str(self.folder)],stdout=gf,stderr=subprocess.STDOUT,start_new_session=True)
        self.phase='warming_up'
        if self.args.hold_hip_yaw:
            os.environ['BRUCE_STATIONARY_HIP_YAW_HOLD']='1'
            print('STATIONARY_HIP_YAW_HOLD_PROFILE_ENABLED',flush=True)
        else:os.environ.pop('BRUCE_STATIONARY_HIP_YAW_HOLD',None)
        if self.args.swing_height is not None:
            os.environ['BRUCE_STATIONARY_SWING_HEIGHT']=str(self.args.swing_height)
        else:os.environ.pop('BRUCE_STATIONARY_SWING_HEIGHT',None)
        os.environ['BRUCE_SUPPORTED_TRANSLATION_TRACKING']='1' if self.args.translation_tracking else '0'
        translation_file=self.folder/'translation-tuning.json'
        atomic_json(translation_file,{'forward_weight':3.})
        os.environ['BRUCE_SUPPORTED_TRANSLATION_TUNING_FILE']=str(translation_file)
        os.environ['BRUCE_STATIONARY_ATTITUDE_SCALE']=str(self.args.attitude_scale)
        os.environ['BRUCE_STATIONARY_SMOOTH_STOP']='1' if self.args.smooth_stop else '0'
        tuning_file=self.folder/'planner-tuning.json'
        atomic_json(tuning_file,{'forward_bias_m':self.args.forward_bias,'minimum_width_m':self.args.minimum_step_width,'lateral_bias_m':self.args.lateral_bias})
        os.environ['BRUCE_STATIONARY_TUNING_FILE']=str(tuning_file)
        atomic_json(self.folder/'profile.json',{'imu_heading_hold_revision':1,'native_imu_turn_revision':1,'turn_angle_limit_deg':10.,'turn_rate_limit_deg_s':3.,'forward_speed_stage_enabled':True,'forward_speed_stage_file':'forward-stage.json','maximum_stage_speed_m_s':.25,'initial_stage_speed_m_s':.10,'heading_correction_rate_limit_deg_s':3.,'heading_offset_limit_deg':5.,'stationary_behavior_unchanged':True,'max_motion_seconds':self.args.max_motion_seconds,'translation_tracking':self.args.translation_tracking,'allow_small_motion':self.args.allow_small_motion,'hold_hip_yaw':self.args.hold_hip_yaw,'attitude_scale':self.args.attitude_scale,'smooth_stop':self.args.smooth_stop,'swing_height_m':self.args.swing_height if self.args.swing_height is not None else .035,'foot_spacing_command_m':self.args.foot_spacing,'max_step_seconds':self.args.max_step_seconds})
        for key,module,answer in [('sense','Startups.run_sense',False),('estimation','Startups.run_estimation',False),('low_level','Util.imu_turn_low_level',True),('high_level','Util.imu_heading_high_level',True)]:
            self.preload(key,module,answer)
        deadline=time.monotonic()+180
        while not all((self.folder/('warm_'+key)).exists() for key in self.children):
            self.watch()
            if time.monotonic()>deadline:raise RuntimeError('Controller warmup timeout')
            time.sleep(.02)
        print('ALL_CONTROLLERS_WARMED_BEFORE_ENABLING',flush=True)
        self.spawn('bear');self.wait_ready('bear')
        if self.args.passive_check:
            self.phase='passive_sense';self.spawn('sense','Startups.run_sense');self.wait_ready('sense')
            self.phase='passive_estimator';self.spawn('estimation','Startups.run_estimation');self.wait_ready('estimation')
            self.phase='passive_stack';end=time.monotonic()+10
            while time.monotonic()<end:
                if self.mm.LEG_COMMAND.get()['BEAR_enable'][0]!=0:raise RuntimeError('Unexpected enable during passive check')
                self.watch();time.sleep(.01)
            self.stop_reason='PASS_PASSIVE_STACK_NO_MOTOR_ENABLE';return
        self.sensor=SenseManager(port='/dev/ttyACM0');self.sensor.send_data(pico_mode='nominal');self.last_sensor=time.monotonic()
        q0=self.mm.LEG_STATE.get()['joint_positions'].copy()
        target=np.array([*kin.legIK_foot(np.array([.074,-.07,-.44]),np.array([1.,0.,0.]),1.),*kin.legIK_foot(np.array([.074,.07,-.44]),np.array([1.,0.,0.]),-1.)])
        if not np.isfinite(q0).all():raise RuntimeError('Invalid initial joint positions')
        duration=max(3.,float(np.max(np.abs(target-q0)))*15)
        self.phase='initializing';start=time.monotonic()
        while time.monotonic()-start<duration:
            elapsed=time.monotonic()-start;u=min(elapsed/duration,1.);q=q0+(target-q0)*(3*u*u-2*u*u*u)
            self.mm.LEG_COMMAND.set({'BEAR_enable':np.ones(1),'BEAR_mode':np.array([2.]),'goal_positions':q,'goal_velocities':np.zeros(10),'goal_torques':np.zeros(10)},opt='only');self.enabled=True
            self.watch()
            if np.max(np.abs(self.mm.LEG_STATE.get()['joint_positions']-q))>.25:raise RuntimeError('Initialization tracking error >0.25 rad')
            time.sleep(.01)
        self.mm.LEG_COMMAND.set({'goal_positions':target},opt='only')
        self.phase='await_ground';print('POSE_READY_PLACE_BOTH_FEET_ON_GROUND_NOW',str(self.folder),flush=True)
        end=time.monotonic()+self.args.ground_wait;ground_since=None
        while True:
            tilt,contacts,_=self.watch();now=time.monotonic()
            grounded=np.any(contacts[:2]) and np.any(contacts[2:]) and tilt<15 and (not self.args.manual_landing or (self.folder/'operator-ready').exists())
            ground_since=(ground_since if ground_since is not None else now) if grounded else None
            if ground_since is not None and now-ground_since>1:break
            if now>end and not self.args.keep_standing:raise RuntimeError('Ground/vertical-pose confirmation timed out')
            time.sleep(.01)
        print('GROUND_VERIFIED',flush=True)
        self.sensor.close_port();self.sensor=None
        self.phase='starting_sense';self.spawn('sense','Startups.run_sense');self.wait_ready('sense')
        self.phase='starting_estimator';self.spawn('estimation','Startups.run_estimation');self.wait_ready('estimation')
        self.phase='starting_balance';self.spawn('low_level','Util.imu_turn_low_level',True);self.wait_ready('low_level')
        self.command(0);self.spawn('high_level','Util.imu_heading_high_level',True);self.wait_ready('high_level')
        self.phase='standing';print('STANDING_BALANCE_ACTIVE',flush=True);end=time.monotonic()+self.args.stand_seconds
        while time.monotonic()<end:self.command(0);self.watch();time.sleep(.01)
        print('STANDING_CHECK_PASSED',flush=True)
        if self.args.seconds>0:
            self.perform_steps(self.args.seconds)
        else:self.stop_reason='PASS_SUPPORTED_STANDING_TEST'
        if self.args.keep_standing:
            self.phase='holding_balance'
            self.checkpoint()
            print('KEEPING_STANDING_NO_NORMAL_TIME_LIMIT',str(self.folder),flush=True)
            while not (self.folder/'operator-stop').exists():
                self.command(0);self.watch();time.sleep(.01)
                spacing_request=self.folder/'foot-spacing-request.json'
                if spacing_request.exists():
                    try:
                        spacing=checked_foot_spacing(json.loads(spacing_request.read_text())['meters'])
                    except (OSError,ValueError,KeyError,TypeError) as error:
                        print('FOOT_SPACING_REQUEST_REJECTED',str(error),flush=True)
                    else:
                        self.args.foot_spacing=spacing
                        profile=json.loads((self.folder/'profile.json').read_text());profile['foot_spacing_command_m']=spacing
                        atomic_json(self.folder/'profile.json',profile)
                        print('FOOT_SPACING_FOR_NEXT_STEP',spacing,flush=True)
                    spacing_request.unlink(missing_ok=True)
                turn_request=self.folder/'turn-request.json'
                if turn_request.exists():
                    try:
                        from Util.imu_turn_policy import checked_turn_degrees
                        angle=checked_turn_degrees(json.loads(turn_request.read_text())['degrees'])
                    except (OSError,ValueError,KeyError,TypeError) as error:
                        print('TURN_REQUEST_REJECTED',str(error),flush=True);turn_request.unlink(missing_ok=True);continue
                    turn_request.unlink();self.perform_turn(angle)
                request=self.folder/'step-request.json'
                if request.exists():
                    try:
                        data=json.loads(request.read_text())
                        seconds=step_request_seconds(data['seconds'],self.args.max_step_seconds)
                        velocity=checked_forward_velocity(data.get('forward_velocity_m_s',0.),seconds,self.args.allow_small_motion,self.args.max_motion_seconds)
                    except (OSError,ValueError,KeyError,TypeError) as error:
                        print('STEP_REQUEST_REJECTED',str(error),flush=True);request.unlink(missing_ok=True);continue
                    request.unlink();self.perform_steps(seconds,velocity)
            print('OPERATOR_REQUESTED_STOP',flush=True)
    def close(self):
        self.mm.LEG_COMMAND.set({'BEAR_enable':np.zeros(1)},opt='only')
        for key in ('bear','low_level','high_level','estimation','sense'):
            p=self.children.get(key)
            if p and p.poll() is None:p.send_signal(signal.SIGINT)
        for p in self.children.values():
            try:p.wait(timeout=4)
            except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=2)
        if self.sensor:self.sensor.send_data(pico_mode='idle');self.sensor.close_port()
        self.mm.THREAD_STATE.set({'top_level':np.zeros(1)},opt='only')
        (self.folder/'finished').touch()
        if self.guard:
            try:self.guard.wait(timeout=2)
            except subprocess.TimeoutExpired:self.guard.terminate();self.guard.wait(timeout=2)
        for f in self.logs.values():f.close()
        (self.folder/'result.json').write_text(json.dumps({'result':self.stop_reason,'records':self.records},indent=2))
        print('RESULT',self.stop_reason,'LOG_DIR',self.folder,flush=True)
        for key in self.children:
            print('LOG_TAIL',key,'\n'+'\n'.join((self.folder/(key+'.log')).read_text(errors='replace').splitlines()[-12:]),flush=True)
        shutil.copytree(self.folder,self.archive)
        # All motors are stopped; persist diagnostics before a power-off.
        for path in self.archive.iterdir():
            if path.is_file():
                with path.open('rb') as stream:os.fsync(stream.fileno())
        for directory in (self.archive,self.archive.parent):
            descriptor=os.open(str(directory),os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        print('ARCHIVE_DIR',self.archive,flush=True)
        shutil.rmtree(self.folder)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--keep-standing',action='store_true',help='Keep supported standing after the test until operator-stop, a fault, or supervisor interruption; heartbeat watchdog remains active.');p.add_argument('--manual-landing',action='store_true');p.add_argument('--stand-seconds',type=float,default=5.);p.add_argument('--passive-check',action='store_true');p.add_argument('--seconds',type=float,default=2.);p.add_argument('--ground-wait',type=float,default=75.);p.add_argument('--worker',choices=['bear','guard','preload']);p.add_argument('--folder',type=Path);p.add_argument('--module');p.add_argument('--key')
    p.add_argument('--max-motion-seconds',type=float,default=4.,help='Moving-bout ceiling, 0.5..30 seconds, also bounded by max-step-seconds.')
    p.add_argument('--translation-tracking',action='store_true',help='Opt-in bounded forward tracking only during moving single support.')
    p.add_argument('--allow-small-motion',action='store_true',help='Permit explicitly requested forward/backward bouts up to 0.05 m/s within the configured duration ceiling.')
    p.add_argument('--hold-hip-yaw',action='store_true',help='Supervised zero-yaw profile: anchor stance hip yaw with unchanged gains and limits.')
    p.add_argument('--swing-height',type=float,help='Optional zero-velocity diagnostic swing apex in meters, 0.01..0.035; ordinary startup is unchanged.')
    p.add_argument('--attitude-scale',type=float,default=1.,help='Supported diagnostic roll/pitch feedback scale, 0.25..1; does not scale gravity compensation.')
    p.add_argument('--smooth-stop',action='store_true',help='Use continuous 0.8-second CoM/orientation settling in the diagnostic profile.')
    p.add_argument('--foot-spacing',type=float,default=PARAMETER_DEFAULT[FOOT_CLEARANCE],help='Lateral landing-spacing command, within original walking parameter bounds.')
    p.add_argument('--max-step-seconds',type=float,default=5.,help='Supervised per-request duration ceiling, 5..30 seconds; default remains 5.')
    p.add_argument('--forward-bias',type=float,default=.010,help='DCM fore-aft offset in meters, -0.02..0.02; original default is +0.010.')
    p.add_argument('--minimum-step-width',type=float,default=.04,help='Minimum lateral foothold separation in meters, 0.04..0.12; original default is 0.04.')
    p.add_argument('--lateral-bias',type=float,default=0.,help='Independent lateral DCM trim in meters, -0.02..0.02.')
    args=p.parse_args()
    from Util.stationary_planner_tuning import checked_tuning
    try:args.forward_bias,args.minimum_step_width,args.lateral_bias=checked_tuning(args.forward_bias,args.minimum_step_width,args.lateral_bias)
    except ValueError as error:p.error(str(error))
    try:args.foot_spacing=checked_foot_spacing(args.foot_spacing)
    except ValueError as error:p.error(str(error))
    if not np.isfinite(args.max_step_seconds) or not 5<=args.max_step_seconds<=30:p.error('max-step-seconds must be 5..30')
    if not np.isfinite(args.max_motion_seconds) or not .5<=args.max_motion_seconds<=args.max_step_seconds:p.error('max-motion-seconds must be 0.5..max-step-seconds')
    if not np.isfinite(args.attitude_scale) or not .25<=args.attitude_scale<=1.:p.error('attitude-scale must be 0.25..1')
    if args.swing_height is not None and (not np.isfinite(args.swing_height) or not .01<=args.swing_height<=.035):p.error('swing-height must be 0.01..0.035 meters')
    if args.worker:
        if args.worker=='preload':return preload_worker(args.folder,args.module,args.key)
        return motor_worker(args.folder) if args.worker=='bear' else guard_worker(args.folder)
    if not (0<=args.seconds<=args.max_step_seconds and 5<=args.ground_wait<=300 and 3<=args.stand_seconds<=60):p.error('Bounded test requires stepping within max-step-seconds, 5..300 ground-wait seconds, 3..60 standing seconds')
    if not args.run:
        for name in ['Library.ROBOT_MODEL.BRUCE_kinematics','Library.ROBOT_MODEL.BRUCE_dynamics','Library.STATE_ESTIMATION.BRUCE_estimation_KF']:importlib.import_module(name)
        print('Non-actuating dependency check passed. --run starts a bounded supported test.')
        return
    def interrupt(sig,frame):raise KeyboardInterrupt('Supervisor interrupted')
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupt)
    probe=Probe(args)
    try:probe.run()
    except BaseException as e:probe.stop_reason=type(e).__name__+': '+str(e);traceback.print_exc()
    finally:probe.close()

if __name__=='__main__':main()
