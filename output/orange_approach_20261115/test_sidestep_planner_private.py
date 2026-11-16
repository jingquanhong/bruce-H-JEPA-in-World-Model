"""Exercise actual planner entry/stance selection using private recorded state."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,types,contextlib
from pathlib import Path
sys.path.insert(0,'/home/khadas/BRUCE/BRUCE-OP')
import numpy as np
import Settings.BRUCE_data as rds
import Play.Walking.high_level as baseline
from Util import sidestep_high_level as candidate

root=Path(__file__).resolve().parent
fixture=json.loads(Path('/home/khadas/BRUCE/diagnostics/control-warmup-fixture.json').read_text())

class Completed(Exception):pass
class Clock:
    def __init__(self):self.now=0.;self.calls=0
    def tick(self,robot):
        self.calls+=1
        if self.calls>20000:raise RuntimeError('Private planner exceeded time budget')
        self.now+=.002
        return self.now

reports=[]
for module,label in ((baseline,'baseline'),(candidate,'sidestep')):
    for sign in (-1.,1.):
        clock=Clock();commands=[]
        class Segment:
            def __init__(self,name,values):
                self.name=name;self.data={k:np.asarray(v,dtype=float).copy() for k,v in values.items()};self.reads=0
            def get(self):
                data={k:v.copy() for k,v in self.data.items()}
                if self.name=='USER_COMMAND':
                    self.reads+=1
                    data['mode']=np.array([0. if self.reads<=2 else 1.])
                    data['com_xy_velocity']=np.array([0.,sign*.03])
                    data['foot_clearance']=np.array([.08])
                return data
            def set(self,data,opt=None):
                self.data.update({k:np.asarray(v).copy() for k,v in data.items()})
                if self.name=='PLANNER_COMMAND' and self.data['phase'][0] in (1,2):
                    assert all(np.isfinite(v).all() for v in self.data.values())
                    commands.append({k:v.tolist() for k,v in self.data.items()})
                    if len(commands)>=3:raise Completed()
        private=types.SimpleNamespace(**{k:Segment(k,v) for k,v in fixture.items()})
        private.THREAD_STATE.data={k:np.ones(1) for k in private.THREAD_STATE.data}
        previous=(module.MM,rds.MM,rds.safe_input,rds.BRUCE.get_time,module.time)
        try:
            module.MM=private;rds.MM=private;rds.safe_input=lambda p:'y';rds.BRUCE.get_time=lambda robot:clock.tick(robot)
            module.time=types.SimpleNamespace(sleep=lambda seconds:None)
            with (root/'sidestep-private-planner.log').open('a') as log:
                with contextlib.redirect_stdout(log):
                    try:module.main_loop()
                    except Completed:pass
            assert len(commands)==3
            actual_phase=int(commands[0]['phase'][0])
            expected_phase=(1 if sign>0 else 2) if label=='sidestep' else 2
            assert actual_phase==expected_phase,(label,sign,actual_phase,expected_phase)
            reports.append(dict(profile=label,requested_vy=sign*.03,
                first_published_support_phase=actual_phase,expected_phase=expected_phase,
                first_command=commands[0],hardware_actuation=False))
        finally:
            module.MM,rds.MM,rds.safe_input,rds.BRUCE.get_time,module.time=previous

(root/'sidestep-private-planner-results.json').write_text(json.dumps(dict(
    result='PASS_PRIVATE_PLANNER_ONLY',cases=reports,
    note='Fixed recorded-state software check; not a physical dynamics simulation.'),indent=2))
print('PASS actual private planner: left leads with left foot; right leads with right foot; finite commands.')
