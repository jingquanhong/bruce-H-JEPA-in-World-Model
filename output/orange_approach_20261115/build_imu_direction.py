from pathlib import Path
import ast
root=Path(__file__).resolve().parent
high=(root/'imu_heading_high_level.py').read_text()
old='abs(Bruce.cmd_vxy_wg[0])>=.005 and abs(Bruce.cmd_vxy_wg[1])<.001'
assert high.count(old)==1
high=high.replace(old,'np.linalg.norm(Bruce.cmd_vxy_wg)>=.005')
(root/'imu_direction_high_level.py').write_text(high)
parent=(root/'stationary_imu_heading_speed_test.py').read_text()
parent=parent.replace('Util.imu_heading_high_level','Util.imu_direction_high_level')
checker='''
def checked_direction_velocity(forward, lateral, seconds, allowed=False, maximum=4.):
    forward=checked_forward_velocity(forward,seconds,allowed,maximum)
    lateral=float(lateral)
    if not np.isfinite(lateral) or abs(lateral)>.03:
        raise ValueError("Lateral test is limited to 0.03 m/s")
    if lateral!=0.:
        if not allowed or forward!=0. or not .5<=seconds<=min(6.,maximum):
            raise ValueError("Lateral test requires permission, pure lateral command and at most 6 seconds")
    return forward,lateral

'''
parent=parent.replace('def checked_foot_spacing(value):',checker+'def checked_foot_spacing(value):')
parent=parent.replace('def command(self,mode,forward_velocity=0.):','def command(self,mode,forward_velocity=0.,lateral_velocity=0.):')
parent=parent.replace('np.array([forward_velocity if mode==1 else 0.,0.])','np.array([forward_velocity,lateral_velocity]) if mode==1 else np.zeros(2)')
parent=parent.replace('def perform_steps(self,seconds,forward_velocity=0.):','def perform_steps(self,seconds,forward_velocity=0.,lateral_velocity=0.):')
parent=parent.replace('forward_velocity=checked_forward_velocity(forward_velocity,seconds,self.args.allow_small_motion,self.args.max_motion_seconds)','forward_velocity,lateral_velocity=checked_direction_velocity(forward_velocity,lateral_velocity,seconds,self.args.allow_small_motion,self.args.max_motion_seconds)')
parent=parent.replace("{'seconds':seconds,'forward_velocity_m_s':forward_velocity}","{'seconds':seconds,'forward_velocity_m_s':forward_velocity,'lateral_velocity_m_s':lateral_velocity}")
parent=parent.replace('self.command(1,forward_velocity);','self.command(1,forward_velocity,lateral_velocity);')
parent=parent.replace("velocity=checked_forward_velocity(data.get('forward_velocity_m_s',0.),seconds,self.args.allow_small_motion,self.args.max_motion_seconds)","velocity,lateral=checked_direction_velocity(data.get('forward_velocity_m_s',0.),data.get('lateral_velocity_m_s',0.),seconds,self.args.allow_small_motion,self.args.max_motion_seconds)")
parent=parent.replace('request.unlink();self.perform_steps(seconds,velocity)','request.unlink();self.perform_steps(seconds,velocity,lateral)')
parent=parent.replace("{'imu_heading_hold_revision':1,","{'imu_heading_hold_revision':1,'directional_heading_revision':1,'lateral_speed_limit_m_s':.03,'lateral_duration_limit_s':6.,")
(root/'stationary_imu_direction_test.py').write_text(parent)
observer=(root/'observe_imu_straight_trim_bout.py').read_text()
observer=observer.replace("parser.add_argument('--forward-lateral-bias'","parser.add_argument('--lateral-velocity',type=float,default=0.)\n    parser.add_argument('--forward-lateral-bias'")
observer=observer.replace('stationary_imu_heading_speed_test as speed_policy','stationary_imu_direction_test as speed_policy')
observer=observer.replace("args.forward_velocity = checked_forward_velocity(args.forward_velocity, args.seconds, profile.get('allow_small_motion',False), profile.get('max_motion_seconds',4.))","args.forward_velocity,args.lateral_velocity=speed_policy.checked_direction_velocity(args.forward_velocity,args.lateral_velocity,args.seconds,profile.get('allow_small_motion',False),profile.get('max_motion_seconds',4.))")
observer=observer.replace('if args.forward_velocity!=0:', 'if args.forward_velocity!=0 or args.lateral_velocity!=0:')
observer=observer.replace("tuning['forward_bias_m']=.015 if args.forward_velocity>0 else -.015", "tuning['forward_bias_m']=.015 if args.forward_velocity>0 else (-.015 if args.forward_velocity<0 else 0.)")
observer=observer.replace("{'seconds': args.seconds, 'forward_velocity_m_s': args.forward_velocity}","{'seconds':args.seconds,'forward_velocity_m_s':args.forward_velocity,'lateral_velocity_m_s':args.lateral_velocity}")
observer=observer.replace("'FORWARD_VELOCITY_M_S', args.forward_velocity,", "'FORWARD_VELOCITY_M_S', args.forward_velocity,'LATERAL_VELOCITY_M_S',args.lateral_velocity,")
observer=observer.replace("'forward_velocity_m_s': args.forward_velocity, 'records': rows", "'forward_velocity_m_s':args.forward_velocity,'lateral_velocity_m_s':args.lateral_velocity,'records':rows")
observer=observer.replace('imu-straight-trim-observation-','imu-direction-observation-')
(root/'observe_imu_direction_bout.py').write_text(observer)
for name in ['imu_direction_high_level.py','stationary_imu_direction_test.py','observe_imu_direction_bout.py']:
 compile((root/name).read_text(),name,'exec')
def nodes(text):
 tree=ast.parse(text);return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
a=nodes((root/'stationary_imu_heading_speed_test.py').read_text());b=nodes(parent)
for name in ['motor_worker','guard_worker','watch','check_tilt','close']:
 if name in a:assert a[name]==b[name],name
print('BUILT direction candidate; original motor, watchdog, watch, tilt and close unchanged')
