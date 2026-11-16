from pathlib import Path
import ast
p=Path(__file__).resolve().parent;old=p.parent/'orange_approach_20261115'
s=(p/'low_level_original.py').read_text()
s=s.replace('hip_yaw_reference = None','hip_yaw_reference = None\n    hip_yaw_gate = None')
s=s.replace("        print('STATIONARY_HIP_YAW_REFERENCE', hip_yaw_reference.tolist(), flush=True)","        print('STATIONARY_HIP_YAW_REFERENCE', hip_yaw_reference.tolist(), flush=True)\n        from Util.turning_hip_yaw_reference import TurningHipYawReference\n        hip_yaw_gate = TurningHipYawReference(hip_yaw_reference)")
s=s.replace('hold_stance_hip_yaw(q_des, dq_des, Bs, hip_yaw_reference)',"hip_yaw_gate.apply(q_des,dq_des,Bs,np.array([Bruce.joint[j]['q'] for j in LEG_JOINT_LIST]),float(MM.USER_COMMAND.get()['yaw_rate'][0]))")
(p/'imu_turn_low_level.py').write_text(s)
s=(old/'stationary_imu_heading_speed_test.py').read_text()
s=s.replace("'Play.Walking.low_level'","'Util.imu_turn_low_level'")
s=s.replace('def command(self,mode,forward_velocity=0.):','def command(self,mode,forward_velocity=0.,yaw_rate=0.):')
s=s.replace("'yaw_rate':np.zeros(1)","'yaw_rate':np.array([yaw_rate if mode==1 else 0.])")
s=s.replace("{'imu_heading_hold_revision':1,","{'imu_heading_hold_revision':1,'native_imu_turn_revision':1,'turn_angle_limit_deg':10.,'turn_rate_limit_deg_s':3.,")
method='''    def perform_turn(self,degrees):
        from Util.imu_turn_policy import checked_turn_degrees,wrap,turn_rate
        from Util.imu_heading_hold import yaw_and_rate
        degrees=checked_turn_degrees(degrees)
        if not self.args.allow_small_motion:raise ValueError("Turning not allowed by this supervisor")
        imu=self.mm.ESTIMATOR_STATE.get();start_yaw,_=yaw_and_rate(imu['body_rot_matrix'],imu['body_ang_rate'])
        target=wrap(start_yaw+np.deg2rad(degrees));deadline=time.monotonic()+12;settled=None
        atomic_json(self.folder/'last-turn-request.json',{'degrees':degrees,'start_yaw_rad':start_yaw,'target_yaw_rad':target})
        self.phase='turning';outcome='ANGLE_NOT_REACHED'
        print('NATIVE_IMU_TURN_START',degrees,flush=True)
        while time.monotonic()<deadline:
            imu=self.mm.ESTIMATOR_STATE.get();yaw,rate=yaw_and_rate(imu['body_rot_matrix'],imu['body_ang_rate'])
            error=wrap(target-yaw);now=time.monotonic()
            settled=(now if settled is None else settled) if abs(error)<np.deg2rad(1.) else None
            if settled is not None and now-settled>.35:outcome='IMU_ANGLE_REACHED';break
            self.command(1,0.,turn_rate(error,rate));self.watch();time.sleep(.01)
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
'''
s=s.replace('    def run(self):',method+'    def run(self):')
turn_request='''                turn_request=self.folder/'turn-request.json'
                if turn_request.exists():
                    try:
                        from Util.imu_turn_policy import checked_turn_degrees
                        angle=checked_turn_degrees(json.loads(turn_request.read_text())['degrees'])
                    except (OSError,ValueError,KeyError,TypeError) as error:
                        print('TURN_REQUEST_REJECTED',str(error),flush=True);turn_request.unlink(missing_ok=True);continue
                    turn_request.unlink();self.perform_turn(angle)
'''
s=s.replace("                request=self.folder/'step-request.json'",turn_request+"                request=self.folder/'step-request.json'")
(p/'stationary_imu_turn_test.py').write_text(s)
def functions(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef)}
a=functions((old/'stationary_imu_heading_speed_test.py').read_text());b=functions(s)
for name in ('motor_worker','guard_worker','check_tilt','watch','close','perform_steps','checked_forward_velocity'):
 assert a[name]==b[name],name
for name in ('imu_turn_low_level.py','stationary_imu_turn_test.py','imu_turn_policy.py','turning_hip_yaw_reference.py'):compile((p/name).read_text(),name,'exec')
print('BUILT independent native-turn candidate; motor/guards/forward-loop unchanged')
