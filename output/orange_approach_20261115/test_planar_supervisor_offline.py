"""Run actual lateral command/stepping methods with fake memory and time only."""
import ast,json,types,tempfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
tree=ast.parse((ROOT/'stationary_planar_test.py').read_text())
baseline=ast.parse((ROOT/'stationary_step_test_vision.py').read_text())
def probe_node(t):return next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Probe')
def method(t,name):return next(n for n in probe_node(t).body if isinstance(n,ast.FunctionDef) and n.name==name)
for name in ('watch','close','spawn','preload'):
    assert ast.dump(method(tree,name))==ast.dump(method(baseline,name)),name+' protection changed'
for name in ('motor_worker','guard_worker','check_tilt'):
    get=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(get(tree))==ast.dump(get(baseline)),name+' protection changed'
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
           ('step_request_seconds','checked_forward_velocity','checked_lateral_velocity')]
methods=[method(tree,n) for n in ('command','perform_steps')]

class Clock:
    def __init__(self):self.now=0.
    def monotonic(self):return self.now
    def time(self):return self.now
    def sleep(self,seconds):self.now+=seconds

class Memory:
    def __init__(self):self.rows=[]
    def set(self,data,opt=None):self.rows.append({k:v.copy() for k,v in data.items()})

def scenario(vx,vy,seconds=2.,enabled=True,maximum=10.,vision=None):
    clock=Clock();ns=dict(np=np,time=clock,json=json,atomic_json=lambda p,d:p.write_text(json.dumps(d)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions+methods,type_ignores=[])),'lateral_candidate','exec'),ns)
    with tempfile.TemporaryDirectory() as directory:
        user=Memory();threads=Memory()
        dummy=types.SimpleNamespace(args=types.SimpleNamespace(max_step_seconds=30,allow_small_motion=True,
            max_motion_seconds=maximum,allow_lateral_motion=enabled,foot_spacing=.08),
            mm=types.SimpleNamespace(USER_COMMAND=user,THREAD_STATE=threads),folder=Path(directory),
            step_observations={},phase='holding_balance',checkpoint=lambda:None)
        dummy.command=types.MethodType(ns['command'],dummy)
        def watch():
            mode=user.rows[-1]['mode'][0]
            phase=(1 if int(clock.now*4)%2==0 else 2) if mode==1 else 0
            contacts=np.array([1,1,0,0]) if phase==1 else np.array([0,0,1,1]) if phase==2 else np.ones(4)
            return 0.,contacts,{'phase':np.array([phase])}
        dummy.watch=watch
        ns['perform_steps'](dummy,seconds,vx,vision,vy)
        walking=[r for r in user.rows if r['mode'][0]==1]
        standing=[r for r in user.rows if r['mode'][0]==0]
        assert walking and standing and dummy.phase=='holding_balance'
        assert all(np.array_equal(r['com_xy_velocity'],[vx,vy]) for r in walking)
        assert all(np.array_equal(r['com_xy_velocity'],[0,0]) for r in standing)
        saved=json.loads((dummy.folder/'last-motion-request.json').read_text())
        assert saved['lateral_velocity_m_s']==vy and saved['forward_velocity_m_s']==vx

for vx,vy in ((0,.03),(0,-.03),(.03,0),(0,0)):scenario(vx,vy)
scenario(0,.03,seconds=6.)
for args,kwargs in [((0,.031),{}),((0,float('nan')),{}),((0,float('inf')),{}),
                    ((0,.01),dict(enabled=False)),((0,.01),dict(seconds=6.01)),
                    ((0,.01),dict(maximum=1)),((.01,.01),{}),
                    ((0,.01),dict(vision='testsession'))]:
    try:scenario(*args,**kwargs)
    except ValueError:pass
    else:raise AssertionError('Invalid lateral request accepted')
ns=dict(np=np)
exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),'lateral_guards','exec'),ns)
try:ns['checked_lateral_velocity'](.01,2.,True,False,3.)
except ValueError:pass
else:raise AssertionError('Lateral request bypassed allow-small-motion')
print('Lateral offline checks passed: both signs reach native Y; stance zeros both axes; invalid, disabled, mixed and excessive requests rejected; original safety methods unchanged.')
