"""Validate both side-step signs against unchanged planner reachability."""
import ast
from pathlib import Path
from sidestep_geometry import initial_support,nominal_width

for sign in (-1,1):
    support=initial_support(0.,sign*.03,-sign)
    assert support==sign
    width=nominal_width(.08,.10,.20,0.,[sign*.03]*3,.28)
    assert abs(width-.1134)<1e-12
    for time in (.20,.24,.28):
        shift=sign*.03*time
        for stance in (-1,1):
            lateral=width*stance+shift
            assert .10<=abs(lateral)<=.20
        # Two alternating steps produce a signed displacement, with restored
        # stance width. Nominal references do not guarantee physical motion.
        assert abs((width+shift)+(-width+shift)-2*shift)<1e-12
    assert nominal_width(.08,.10,.20,0.,[0.,0.,0.],.28)==.10
    assert nominal_width(.08,.10,.20,.03,[0.,0.,0.],.28)==.10
    assert initial_support(.03,0.,-sign)==-sign
    assert initial_support(0.,0.,-sign)==-sign

for sign in (-1,1):
    width=nominal_width(.08,.10,.20,0.,[sign*.05]*3,.28)
    assert .10<=width-.05*.28 and width+.05*.28<=.20

for bad in ([float('nan')],[float('inf')]):
    try:nominal_width(.08,.10,.20,0.,bad,.28)
    except ValueError:pass
    else:raise AssertionError('Nonfinite geometry accepted')
try:nominal_width(.08,.19,.20,0.,[.03],.28)
except ValueError:pass
else:raise AssertionError('Unreachable pair accepted')

root=Path(__file__).resolve().parent
old=ast.parse((root/'high_level_baseline.py').read_text())
new=ast.parse((root/'sidestep_high_level.py').read_text())
for name in ('constraint_update','cost_update','get_swing_traj','get_swing_ref','get_DCM_nom'):
    node=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(node(old))==ast.dump(node(new)),name+' modified'
print('PASS: left/right leading foot, nonclipped closing/outward nominal steps,')
print('      unchanged stationary/forward geometry and original DCM/reachability/swing functions.')
