"""Read-only motor communication check while capturing camera data. Never enables motors."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,subprocess,sys,time
from pathlib import Path
import numpy as np
REPO=Path('/home/khadas/BRUCE/BRUCE-OP');sys.path.insert(0,str(REPO))
from Library.ACTUATORS.BEAR_controller import TimedBEAR
from Settings.BRUCE_macros import BEAR_port,BEAR_baudrate
import Startups.memory_manager as mm
root=Path(__file__).resolve().parent
assert mm.LEG_COMMAND.get()['BEAR_enable'][0]==0
assert subprocess.run(['fuser',BEAR_port],capture_output=True).returncode!=0
b=TimedBEAR(port=BEAR_port,baudrate=BEAR_baudrate,timeout=.005,bulk_timeout=.010)
rows=[];child=None;log=None
try:
 enables=[b.get_torque_enable(i)[0][0] for i in range(1,11)]
 assert all(x in (0,3) for x in enables),('Motor already active',enables)
 output=root/'capture-02';output.mkdir(exist_ok=True)
 started=time.monotonic();after=None
 while True:
  t=time.monotonic();elapsed=t-started
  if child is None and elapsed>=3:
   env=dict(os.environ);env['LD_LIBRARY_PATH']=str(root/'sdk/OrbbecSDK_v1.10.37/SDK/lib');log=(output/'capture.log').open('w')
   child=subprocess.Popen(['taskset','-c','0,1',str(root/'camera_probe'),str(output)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
  if child is not None and child.poll() is not None and after is None:after=t
  phase='baseline' if child is None else 'camera_running' if after is None else 'after_camera'
  begin=time.monotonic();result=b.bulk_read(list(range(1,11)),['present_position','present_velocity','present_iq','input_voltage']);dt=time.monotonic()-begin
  okay=result is not None and len(result)==10 and all(x is not None and x[0] is not None for x in result)
  rows.append({'elapsed':elapsed,'phase':phase,'duration_ms':dt*1000,'okay':okay,'codes':[x[1] for x in result] if okay else None,'voltage':[x[0][3] for x in result] if okay else None})
  assert mm.LEG_COMMAND.get()['BEAR_enable'][0]==0
  if after is not None and t-after>3:break
  if elapsed>25:raise RuntimeError('Camera capture check timeout')
  time.sleep(max(0.,.005-(time.monotonic()-t)))
 summary={}
 for phase in ('baseline','camera_running','after_camera'):
  r=[x for x in rows if x['phase']==phase];dur=[x['duration_ms'] for x in r]
  summary[phase]={'reads':len(r),'failed':sum(not x['okay'] for x in r),'duration_ms_p50_p99_max':np.percentile(dur,[50,99,100]).tolist(),'codes':sorted({tuple(x['codes']) for x in r if x['okay']})}
 summary['torque_enable_before']=enables;summary['torque_enable_after']=[b.get_torque_enable(i)[0][0] for i in range(1,11)];summary['camera_exit_code']=child.returncode
 (root/'passive-camera-bus-check.json').write_text(json.dumps({'summary':summary,'records':rows},indent=2));print(json.dumps(summary,indent=2))
finally:
 b.ser.close()
 if child is not None and child.poll() is None:child.terminate();child.wait(timeout=5)
 if log:log.close()
