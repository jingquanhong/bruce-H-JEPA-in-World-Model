"""Request one bounded native IMU turn; log and supervise extra tilt/voltage guards."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,json,math,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from Util.imu_turn_policy import checked_turn_degrees

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('runtime',type=Path);parser.add_argument('--degrees',type=float,required=True);args=parser.parse_args();degrees=checked_turn_degrees(args.degrees);folder=args.runtime
 profile=json.loads((folder/'profile.json').read_text());assert profile.get('native_imu_turn_revision')==1,'Wrong supervisor'
 import Startups.memory_manager as mm
 rows=[];requested=False;good=low=done=None;started=time.monotonic();request_wall=None
 try:
  while time.monotonic()-started<30:
   if not folder.exists():print('SUPERVISOR_ENDED',flush=True);break
   now=time.monotonic();state=json.loads((folder/'state.json').read_text());motor=json.loads((folder/'motor-latest.json').read_text());sense=mm.SENSE_STATE.get();est=mm.ESTIMATOR_STATE.get();plan=mm.PLANNER_COMMAND.get();contacts=sense['foot_contacts'].tolist();tilt=math.degrees(math.acos(max(-1.,min(1.,est['body_rot_matrix'][2,2]))));voltage=min(v[0][3] for v in motor['values'] if v is not None)
   rows.append({'t':now,'phase':state['phase'],'tilt_deg':tilt,'contacts':contacts,'imu_yaw_rad':math.atan2(est['body_rot_matrix'][1,0],est['body_rot_matrix'][0,0]),'motor':motor,'estimated':{k:est[k].tolist() for k in ('com_position','com_velocity','right_foot_position','left_foot_position')},'planner_phase':float(plan['phase'][0])})
   if tilt>8:(folder/'operator-stop').touch();print('STOP_TILT',tilt,flush=True);break
   low=(now if low is None else low) if voltage<14.5 else None
   if low is not None and now-low>.1:(folder/'operator-stop').touch();print('STOP_UNDERVOLTAGE',voltage,flush=True);break
   if not requested:
    okay=(state['phase']=='holding_balance' and now-state['monotonic']<3 and now-motor['monotonic']<.2 and time.time()-sense['time_stamp'][0]<.15 and all(c==128 for c in motor['codes']) and any(contacts[:2]) and any(contacts[2:]) and tilt<5 and voltage>15)
    good=(now if good is None else good) if okay else None
    if good is not None and now-good>3:
     path=folder/'turn-request.json'
     if path.exists() or(folder/'step-request.json').exists():raise RuntimeError('Another motion pending')
     temp=path.with_suffix('.new');temp.write_text(json.dumps({'degrees':degrees}));request_wall=time.time();os.replace(temp,path);requested=True;print('REQUESTED_TURN_DEGREES',degrees,flush=True)
    elif now-started>10:print('PRECONDITIONS_NOT_MET_NO_TURN',flush=True);break
   else:
    result=folder/'turn-result.json'
    complete=state['phase']=='holding_balance' and result.exists() and result.stat().st_mtime>=request_wall
    done=(now if done is None else done) if complete else None
    if done is not None and now-done>2:print('TURN_RESULT',result.read_text(),flush=True);break
   time.sleep(.04)
  else:
   if requested:(folder/'operator-stop').touch()
   print('OBSERVER_TIMEOUT',flush=True)
 except (OSError,ValueError,KeyError,RuntimeError):
  if requested and folder.exists():(folder/'operator-stop').touch()
  raise
 finally:
  output=ROOT.parent/'diagnostics'/('imu-native-turn-'+str(time.time_ns())+'.json');output.write_text(json.dumps({'degrees':degrees,'requested':requested,'records':rows}));print('SAVED',output,flush=True)
  if rows:print('SUMMARY',json.dumps({'max_tilt_deg':max(r['tilt_deg'] for r in rows),'motor_codes':sorted({tuple(r['motor']['codes']) for r in rows}),'min_voltage':min(v[0][3] for r in rows for v in r['motor']['values'] if v is not None)}),flush=True)
if __name__=='__main__':main()
