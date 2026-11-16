"""Exercise the actual candidate stepping method with fake memory/time only."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types

import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('Util.vision_range_pid',ROOT/'vision_range_pid.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
sys.modules['Util']=types.ModuleType('Util')
sys.modules['Util.vision_range_pid']=policy
source=ast.parse((ROOT/'stationary_step_test_vision.py').read_text())
functions=[node for node in source.body if isinstance(node,ast.FunctionDef)
           and node.name in ('step_request_seconds','checked_forward_velocity')]
probe=next(node for node in source.body if isinstance(node,ast.ClassDef) and node.name=='Probe')
method=next(node for node in probe.body if isinstance(node,ast.FunctionDef) and node.name=='perform_steps')


class Clock:
    def __init__(self):self.now=0.
    def monotonic(self):return self.now
    def sleep(self,seconds):self.now+=seconds


def scenario(message,session='testsession'):
    clock=Clock()
    def atomic(path,value):path.write_text(json.dumps(value))
    namespace=dict(np=np,time=clock,json=json,atomic_json=atomic)
    module=ast.fix_missing_locations(ast.Module(body=functions+[method],type_ignores=[]))
    exec(compile(module,'candidate_method','exec'),namespace)
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory)
        if message is not None:(folder/'vision-command.json').write_text(json.dumps(message))
        dummy=types.SimpleNamespace(args=types.SimpleNamespace(max_step_seconds=30,allow_small_motion=True,max_motion_seconds=10),
                                    folder=folder,step_observations={},phase='holding_balance')
        commands=[]
        dummy.command=lambda mode,velocity=0:commands.append((mode,velocity,clock.now))
        dummy.watch=lambda:(0.,np.ones(4),{'phase':np.array([0.])})
        dummy.checkpoint=lambda:None
        namespace['perform_steps'](dummy,2.,.05,session)
        assert dummy.phase=='holding_balance'
        assert commands[-1][0]==0
        assert all(mode in (0,1) and 0<=speed<=.05 for mode,speed,_ in commands)
        return commands,clock.now


base=dict(session_id='testsession',monotonic=0.,measurement_monotonic=0.,forward_velocity_m_s=.03,mode='walk')
commands,elapsed=scenario(base)
walking=[row for row in commands if row[0]==1]
assert walking and max(row[2] for row in walking)<=.31 and elapsed<2.
for message in [None,dict(base,mode='balance',forward_velocity_m_s=0),
                dict(base,session_id='wrongsession'),dict(base,forward_velocity_m_s=float('nan'))]:
    commands,_=scenario(message)
    assert not any(mode==1 for mode,_,_ in commands)
commands,elapsed=scenario(None,None)
assert any(mode==1 for mode,_,_ in commands) and elapsed>=3.
print('6 offline supervisor paths passed: stale vision brakes into balance; missing/balance/wrong-session/nonfinite commands never walk; legacy timed stepping still runs. No shared-memory or serial interfaces imported.')
