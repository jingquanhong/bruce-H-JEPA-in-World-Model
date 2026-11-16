"""Real high-level entry with private memory and ideal tracking, no dynamics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,types,contextlib
from pathlib import Path
sys.path.insert(0,'/home/khadas/BRUCE/BRUCE-OP')
import numpy as np
import Settings.BRUCE_data as rds
from Util import quasistatic_crossstep_high_level as high

class Complete(Exception):pass
root=Path(__file__).resolve().parent
fixture=json.loads(Path('/home/khadas/BRUCE/diagnostics/control-warmup-fixture.json').read_text())
reports=[]
for direction in (-1,1):
    clock=[0.];commands=[];phases=set()
    class Segment:
        def __init__(self,name,data):self.name=name;self.data={k:np.asarray(v,dtype=float).copy() for k,v in data.items()}
        def get(self):
            data={k:v.copy() for k,v in self.data.items()}
            if self.name=='USER_COMMAND':
                walking=.9<=clock[0]<6.9
                data['mode']=np.array([float(walking)])
                data['com_xy_velocity']=np.array([0.,direction*.05 if walking else 0.])
            return data
        def set(self,data,opt=None):
            self.data.update({k:np.asarray(v).copy() for k,v in data.items()})
            if self.name=='PLANNER_COMMAND':
                assert all(np.isfinite(v).all() for v in self.data.values())
                phase=int(self.data['phase'][0]);phases.add(phase)
                est=private.ESTIMATOR_STATE.data
                est['com_position']=self.data['com_position'].copy()
                est['com_velocity']=self.data['com_velocity'].copy()
                for side,moving in (('right',phase==2),('left',phase==1)):
                    if moving:
                        ankle=self.data[side+'_foot_position'].copy()
                        ankle[2]=max(floors[side],ankle[2])
                        est[side+'_ankle_position']=ankle
                        est[side+'_foot_position']=ankle-offsets[side]
                    contact=est[side+'_ankle_position'][2]<floors[side]+.004
                    if side=='right':est['foot_contacts'][:2]=contact
                    else:est['foot_contacts'][2:]=contact
                commands.append(phase)
                if clock[0]>7.1:raise Complete()
    private=types.SimpleNamespace(**{k:Segment(k,v) for k,v in fixture.items()})
    private.THREAD_STATE.data={k:np.ones(1) for k in private.THREAD_STATE.data}
    est=private.ESTIMATOR_STATE.data
    origins={s:est[s+'_ankle_position'].copy() for s in ('right','left')}
    floors={s:est[s+'_ankle_position'][2] for s in origins}
    offsets={s:est[s+'_ankle_position']-est[s+'_foot_position'] for s in origins}
    old=high.MM,rds.MM,rds.safe_input,rds.BRUCE.get_time,high.time
    try:
        high.MM=private;rds.MM=private;rds.safe_input=lambda _:'y'
        rds.BRUCE.get_time=lambda robot:clock[0]
        high.time=types.SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda s:clock.__setitem__(0,clock[0]+s))
        with (root/'quasistatic-private-high.log').open('a') as stream,contextlib.redirect_stdout(stream):
            try:high.main_loop()
            except Complete:pass
        assert phases=={0,1,2},phases
        assert commands[-1]==0 and all(est['foot_contacts'])
        rotation=high.yaw_rotation(est['body_rot_matrix'])
        changes={s:float(rotation[:,1]@(est[s+'_ankle_position']-origins[s])) for s in origins}
        assert all(.035<=direction*v<=.040001 for v in changes.values()),changes
        assert abs(changes['right']-changes['left'])<.002
        reports.append(dict(direction=direction,commands=len(commands),phases=sorted(phases),
            foot_lateral_change_m=changes,hardware_actuation=False))
    finally:high.MM,rds.MM,rds.safe_input,rds.BRUCE.get_time,high.time=old
(root/'quasistatic-private-high-results.json').write_text(json.dumps(dict(
    result='PASS_REAL_HIGH_ENTRY_PRIVATE_IDEAL_TRACKER',cases=reports,
    note='Interface/state-machine check, not a physics simulation or evidence of real balance.'),indent=2))
print('PASS real high-level entry in private memory, both 4 cm foot paths and return to balance.')
