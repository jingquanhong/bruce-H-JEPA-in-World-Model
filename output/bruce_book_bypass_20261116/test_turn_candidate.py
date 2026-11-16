import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[name]='1'
import sys,json,copy,math,contextlib
from pathlib import Path
sys.path.insert(0,'/home/khadas/BRUCE/BRUCE-OP')
import numpy as np
import Settings.BRUCE_data as rds
from Util.imu_turn_policy import checked_turn_degrees,turn_rate,wrap
from Util.turning_hip_yaw_reference import TurningHipYawReference
from Util.hip_yaw_reference import hold_stance_hip_yaw
for value in (0.,10.1,float('nan'),float('inf')):
 try:checked_turn_degrees(value)
 except ValueError:pass
 else:raise AssertionError('Invalid angle accepted')
for sign in (-1,1):
 assert math.copysign(1,turn_rate(sign*.3,0))==sign
 assert abs(turn_rate(sign*100,0))<=math.radians(3)
assert abs(wrap(math.radians(181))-math.radians(-179))<1e-12
ref=np.array([-1.57,-1.57]);measured=np.zeros(10);measured[[0,5]]=ref
for phase in (0,1,2):
 a=np.arange(10,dtype=float);b=a.copy();av=a.copy();bv=av.copy();hold_stance_hip_yaw(a,av,phase,ref);TurningHipYawReference(ref).apply(b,bv,phase,measured,0.)
 assert np.array_equal(a,b) and np.array_equal(av,bv)
gate=TurningHipYawReference(ref);q=np.arange(10,dtype=float);dq=q.copy();before=q.copy();gate.apply(q,dq,1,measured,.05);assert np.array_equal(q,before)
measured[[0,5]]=[-1.55,-1.59];gate.apply(q,dq,0,measured,0.);assert np.array_equal(q[[0,5]],measured[[0,5]])
from Util.control_preflight import warm_low_level
import Play.Walking.low_level as baseline
from Util import imu_turn_low_level as candidate
original_warm=warm_low_level
def deterministic_warm(module,fixture,capture):
 previous=rds.BRUCE.get_time;clock=[0.]
 def tick(robot):clock[0]+=.01;return clock[0]
 try:
  rds.BRUCE.get_time=tick
  return original_warm(module,fixture,capture)
 finally:rds.BRUCE.get_time=previous
warm_low_level=deterministic_warm
fixture=json.loads(Path('/home/khadas/BRUCE/diagnostics/control-warmup-fixture.json').read_text());root=Path(__file__).resolve().parent;reports=[]
with(root/'turn-private-check.log').open('w') as log,contextlib.redirect_stdout(log):
 for phase in (0,1,2):
  f=copy.deepcopy(fixture);f['PLANNER_COMMAND']['phase']=[phase]
  os.environ['BRUCE_STATIONARY_HIP_YAW_HOLD']='1';os.environ['BRUCE_STATIONARY_ATTITUDE_SCALE']='.5';os.environ['BRUCE_SUPPORTED_TRANSLATION_TRACKING']='1'
  a=[];b=[];warm_low_level(baseline,f,a);warm_low_level(candidate,f,b)
  assert len(a)==len(b)==3
  for x,y in zip(a,b):
   for key in x:assert np.allclose(x[key],y[key],atol=1e-12), (phase,key)
  reports.append({'phase':phase,'zero_yaw_output_unchanged':True})
  if phase:
   for sign in (-1,1):
    g=copy.deepcopy(f);g['USER_COMMAND']['yaw_rate']=[sign*math.radians(3)];capture=[];warm_low_level(candidate,g,capture)
    assert len(capture)==3
    reports.append({'phase':phase,'turn_sign':sign,'finite_native_IK_outputs':len(capture)})
(root/'turn-private-check-results.json').write_text(json.dumps({'result':'PASS_PRIVATE_SOFTWARE_ONLY','cases':reports,'hardware_actuation':False,'physical_turning_validated':False},indent=2))
print('PASS turn limits/sign/wrap, hip-yaw mode switching and actual WBC zero-yaw equality in 3 phases; turning outputs finite. Private software only.')
