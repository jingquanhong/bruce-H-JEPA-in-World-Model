#!/usr/bin/env python3
"""Saved, supported IMU forward/back gait; no automatic walking on reboot."""
import argparse,fcntl,json,math,os,subprocess,sys,time
from pathlib import Path
BASE=Path('/home/khadas/BRUCE')
SAVED=BASE/'session-save-20261116-imu-straight'
REPO=BASE/'BRUCE-OP'

def verified():
 result=subprocess.run([sys.executable,str(SAVED/'launch_imu_straight.py')],capture_output=True,text=True)
 if result.returncode:raise RuntimeError(result.stdout+result.stderr)

def runtime():
 found=[]
 for folder in Path('/dev/shm').glob('bruce-step-*'):
  try:
   pid=int(folder.name.rsplit('-',1)[1]);cmd=Path('/proc',str(pid),'cmdline').read_bytes().replace(b'\0',b' ')
   if b'stationary_imu_heading_speed_test.py' not in cmd:continue
   state=json.loads((folder/'state.json').read_text())
   if not 0<=time.monotonic()-state['monotonic']<3:continue
   if (folder/'finished').exists() or (folder/'operator-stop').exists():continue
   profile=json.loads((folder/'profile.json').read_text())
   if profile.get('imu_heading_hold_revision')!=1:continue
   found.append((folder,state))
  except (OSError,ValueError,KeyError):continue
 if len(found)!=1:raise RuntimeError('Expected exactly one live saved IMU supervisor; found '+str(len(found)))
 return found[0]

def atomic(path,data):
 temp=path.with_suffix('.new');temp.write_text(json.dumps(data));os.replace(temp,path)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('action',choices=['check','start','status','ground','forward','backward'],nargs='?',default='check')
 parser.add_argument('--seconds',type=float,default=12.)
 args=parser.parse_args()
 if not math.isfinite(args.seconds) or not .5<=args.seconds<=12:parser.error('seconds must be 0.5..12')
 if args.action=='status':
  folder,state=runtime();print(json.dumps({'runtime':str(folder),'phase':state['phase'],'tilt_deg':state.get('tilt_deg'),'contacts':state.get('contacts')},ensure_ascii=False,indent=2));return
 verified()
 if args.action=='check':print('Saved forward/back source verified. No motor enabled. Default: command 0.08 m/s, 12 s.');return
 if args.action=='start':
  print('Start only with support fully bearing weight, feet slightly airborne, accessible emergency stop. Place feet on ground after POSE_READY, then run ground in another terminal.',flush=True)
  return subprocess.call([sys.executable,str(SAVED/'launch_imu_straight.py'),'--run'])
 folder,state=runtime()
 if args.action=='ground':
  if state['phase']!='await_ground':raise RuntimeError('ground is accepted only during await_ground')
  print('Ground confirmation: both feet placed, torso upright, support prevents falls.',flush=True)
  (folder/'operator-ready').touch();return
 if state['phase']!='holding_balance':raise RuntimeError('Wait for holding_balance before motion')
 with (folder/'validated-motion.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('Another saved gait observation is running')
  if (folder/'step-request.json').exists():raise RuntimeError('Another request is pending')
  atomic(folder/'forward-stage.json',{'speed_limit_m_s':.10,'maximum_seconds':12.})
  speed=.08 if args.action=='forward' else -.08
  command=['nice','-n','10','taskset','-c','3',sys.executable,'-u','Util/observe_imu_straight_trim_bout.py',str(folder),'--request','--seconds',str(args.seconds),'--capture',str(args.seconds+11),'--forward-velocity',str(speed),'--forward-lateral-bias','0']
  print('Supported '+args.action+': command '+str(speed)+' m/s for '+str(args.seconds)+' s, IMU heading hold, stationary trims restored afterward.',flush=True)
  return subprocess.call(command,cwd=REPO)

if __name__=='__main__':
 try:raise SystemExit(main())
 except (OSError,ValueError,RuntimeError) as error:print('REFUSED:',error,file=sys.stderr);raise SystemExit(1)
