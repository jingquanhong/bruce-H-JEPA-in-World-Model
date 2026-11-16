from pathlib import Path
import argparse,json,hashlib,shutil,os,math,datetime
import numpy as np
p=argparse.ArgumentParser();p.add_argument('observation',type=Path);p.add_argument('--feedback',default='pending');args=p.parse_args()
base=Path('/home/khadas/BRUCE');repo=base/'BRUCE-OP';diag=base/'diagnostics/orange-approach-20261115';save=base/'session-save-20261116-imu-lateral-candidate';save.mkdir(exist_ok=True)
trial=json.loads(args.observation.read_text());rows=trial['records'];active=[r for r in rows if r['state']['phase']=='stepping']
summary={'observation':str(args.observation),'command_vx_m_s':trial['forward_velocity_m_s'],'command_vy_m_s':trial['lateral_velocity_m_s'],'seconds':trial['seconds'],'request_sent':trial['requested'],'operator_feedback':args.feedback,'ground_displacement_verified':False,'all_motor_status_sets':[list(v) for v in sorted({tuple(r['motor']['codes']) for r in rows})],'max_fused_tilt_deg':max(r['tilt'] for r in rows)}
if active:
 yaw=rows[0]['imu_yaw_rad'];c,s=math.cos(yaw),math.sin(yaw);rot=np.array([[c,s],[-s,c]])
 velocities=np.array([r['estimated']['com_velocity'][:2] for r in active])@rot.T
 def feet(r):return(np.array(r['estimated']['right_foot_position'])+np.array(r['estimated']['left_foot_position']))[:2]/2
 delta=rot@(feet(rows[-1])-feet(rows[0]));summary.update({'estimated_mean_forward_velocity_m_s':float(velocities[:,0].mean()),'estimated_mean_left_velocity_m_s':float(velocities[:,1].mean()),'estimated_foot_midpoint_delta_forward_m':float(delta[0]),'estimated_foot_midpoint_delta_left_m':float(delta[1]),'odometry_warning':'Encoder/contact/IMU estimate only; not independent ground-truth displacement'})
manifest=[]
for name in ['imu_heading_hold.py','imu_direction_high_level.py','stationary_imu_direction_test.py','observe_imu_direction_bout.py']:
 src=repo/'Util'/name;shutil.copy2(src,save/name);manifest.append({'path':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
shutil.copy2(args.observation,save/args.observation.name)
(save/(args.observation.stem+'-summary.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(save/'source-manifest.json').write_text(json.dumps({'files':manifest},indent=2))
runtime=Path('/dev/shm/bruce-step-16048')
for name in ['profile.json','checkpoint.json','planner-tuning.json']:
 if(runtime/name).exists():shutil.copy2(runtime/name,save/('snapshot-'+name))
status={'saved_at':datetime.datetime.now().astimezone().isoformat(),'candidate':'native fast stepping with directional IMU heading lock; no crossstep state machine','last_trial':summary,'runtime':str(runtime),'requires_live_state_check':True,'default_forward_backward_unchanged':str(base/'session-save-20261116-imu-straight'),'autostart':False}
(save/'validation-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
for file in save.iterdir():
 if file.is_file():
  with file.open('rb') as stream:os.fsync(stream.fileno())
print(json.dumps(summary,ensure_ascii=False,indent=2))
