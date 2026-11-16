"""Geometry/state-machine checks only: perfect tracking is NOT robot physics."""
import sys,copy,json,ast
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from quasistatic_crossstep import CrossStep,yaw_rotation,base_plan


def initial():
    return dict(com=np.array([0.,0.,.365]),velocity=np.zeros(3),rotation=np.eye(3),
        right_ankle=np.array([0.,-.07,.025]),left_ankle=np.array([0.,.07,.025]),
        right_foot=np.array([0.,-.07,0.]),left_foot=np.array([0.,.07,0.]),contacts=np.ones(4))


def follow(snapshot,plan):
    snapshot['com']=plan['com_position'].copy();snapshot['velocity']=plan['com_velocity'].copy()
    phase=int(plan['phase'][0])
    for side,moving in [('right',phase==2),('left',phase==1)]:
        if moving:
            position=plan[side+'_foot_position'].copy();position[2]=max(.025,position[2])
            snapshot[side+'_ankle']=position
            snapshot[side+'_foot']=position-np.array([0.,0.,.025])
        contact=snapshot[side+'_ankle'][2]<.029
        if side=='right':snapshot['contacts'][:2]=contact
        else:snapshot['contacts'][2:]=contact


reports=[]
for direction in (-1,1):
    state=initial();sequence=CrossStep(state,direction,0.)
    phases=set()
    for i in range(4001):
        t=i*.002;plan=sequence.sample(state,t,True)
        assert np.degrees(np.arccos(np.clip(plan['body_rot_matrix'][2,2],-1,1))) <= 2.000001
        phases.add(int(plan['phase'][0]));follow(state,plan)
        width=state['left_ankle'][1]-state['right_ankle'][1]
        assert .10<=width<=.198,(t,width)
        if sequence.done:break
    assert sequence.done and not sequence.aborted,(sequence.stage,sequence.aborted)
    assert phases=={0,1,2},phases
    changes=[state['right_ankle'][1]+.07,state['left_ankle'][1]-.07]
    assert all(.035<=direction*d<=.040001 for d in changes),changes
    assert abs(changes[0]-changes[1])<.002
    assert all(state['contacts'])
    assert t<6.,t
    reports.append(dict(direction=direction,completion_s=t,phases=sorted(phases),
        nominal_foot_advance_m=direction*.04,contact_limited_foot_changes_m=changes,
        hardware_actuation=False))

# Stop during a swing must land before returning to double support.
state=initial();sequence=CrossStep(state,-1,0.);stopped=False
for i in range(4001):
    t=i*.002
    if sequence.stage=='swing' and t-sequence.t0>.1:stopped=True
    plan=sequence.sample(state,t,not stopped);follow(state,plan)
    if sequence.done:break
assert stopped and sequence.done and sequence.leg_index==0 and all(state['contacts'])

# No weight shift: abort to double-support standing, never lift.
state=initial();sequence=CrossStep(state,-1,0.);seen=set()
for i in range(1801):
    plan=sequence.sample(state,i*.002,True);seen.add(int(plan['phase'][0]))
    if sequence.done:break
assert sequence.done and sequence.aborted=='weight_transfer_not_reached' and seen=={0}
assert np.allclose(plan['body_rot_matrix'],np.eye(3),atol=.00001)

# Refuse to remain hanging when timely landing never occurs.
state=initial();sequence=CrossStep(state,-1,0.)
state['com']=np.array([0.,.07,.365]);sequence.sample(state,1.201)
state['contacts'][:2]=0
sequence.sample(state,1.30);sequence.sample(state,1.442)
try:sequence.sample(state,1.485)
except RuntimeError:pass
else:raise AssertionError('Late touchdown accepted')

# Common-mode odometry offsets cancel in support-relative references.
state=initial();sequence=CrossStep(state,-1,0.)
plan=sequence.sample(state,.3)
shifted=copy.deepcopy(state);offset=np.array([.7,-.4,.1])
for key in ('com','right_ankle','left_ankle','right_foot','left_foot'):shifted[key]+=offset
other=sequence.sample(shifted,.3)
assert np.allclose(other['com_position']-plan['com_position'],offset)

# Lost stance contact is a fault, not an invitation to keep lifting.
state=initial();sequence=CrossStep(state,-1,0.);state['com']=np.array([0.,.07,.365])
sequence.sample(state,1.201);assert sequence.stage=='swing'
state['contacts'][2:]=0
sequence.sample(state,1.25)
try:sequence.sample(state,1.36)
except RuntimeError:pass
else:raise AssertionError('Lost support contact accepted')

# Regression: the unloaded future swing foot may release its pressure switch
# during double-support transfer; the loaded support remains mandatory.
state=initial();sequence=CrossStep(state,-1,0.)
state['contacts'][:2]=0;state['com']=np.array([0.,.07,.365])
sequence.sample(state,.75);sequence.sample(state,1.201)
assert sequence.stage=='swing'
state=initial();sequence=CrossStep(state,-1,0.)
state['contacts'][:]=0
sequence.sample(state,.1)
try:sequence.sample(state,.21)
except RuntimeError:pass
else:raise AssertionError('Loss of all floor loading accepted')

# Lean is toward the loaded foot, mirrored by support side, and bounded.
for direction in (-1,1):
    state=initial();sequence=CrossStep(state,direction,0.,support_lean_deg=2.)
    plan=sequence.sample(state,1.201)
    assert direction*plan['body_rot_matrix'][1,2] < 0
for invalid in (-.1,3.01,float('nan')):
    try:CrossStep(initial(),-1,0.,support_lean_deg=invalid)
    except ValueError:pass
    else:raise AssertionError('Invalid support lean accepted')

# A biased ideal tracker checks integral arithmetic, not robot dynamics.
for gain in (0.,1.5):
    state=initial();sequence=CrossStep(state,-1,0.,tracking_integral_gain=gain)
    for i in range(1001):
        t=i*.002;plan=sequence.sample(state,t,True)
        state['com']=plan['com_position']-np.array([0.,.02,0.])
        state['velocity']=plan['com_velocity'].copy()
        assert abs(sequence.integral_offset)<=.020000001
        if sequence.stage!='transfer':break
    if gain==0.:
        assert sequence.aborted=='weight_transfer_not_reached'
    else:
        assert sequence.stage=='swing' and not sequence.aborted
        assert abs(state['com'][1]-.07)<.008
    reports.append(dict(biased_tracker_gain=gain,stage=sequence.stage,time_s=t,
                        hardware_actuation=False))
for invalid in (-.1,2.01,float('nan')):
    try:CrossStep(initial(),-1,0.,tracking_integral_gain=invalid)
    except ValueError:pass
    else:raise AssertionError('Invalid integral gain accepted')

# Original fast motor/sensor/tilt/watchdog and shutdown paths are unchanged.
root=Path(__file__).parent
def functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
a,b=functions(root/'stationary_sidestep_test.py'),functions(root/'stationary_quasistatic_test.py')
for key in ('motor_worker','guard_worker','check_tilt','checked_lateral_velocity'):
    assert a[key]==b[key],key
def methods(path):
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef))
    return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
a,b=methods(root/'stationary_sidestep_test.py'),methods(root/'stationary_quasistatic_test.py')
for key in ('watch','command','perform_steps','close'):assert a[key]==b[key],key
(root/'quasistatic-state-machine-results.json').write_text(json.dumps(dict(
    result='PASS_PERFECT_TRACKING_GEOMETRY_ONLY',cases=reports,
    checks=['two directions','stop and land','weight transfer refusal','lost support refusal',
            'support-relative odometry','original guards unchanged','bounded biased tracker correction'],
    note='Not a dynamics simulation; does not prove real balance.'),indent=2))
print('PASS: one 4 cm pair in either direction; landing, contact and original guard checks.')
