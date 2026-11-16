"""Preparation bounds/contact/timing checks, without actuator access."""
import math
from jump_preparation_profile import PreparationCycle, checked_depth, offset_at

for d in [0.,-.01,.016,float('nan'),float('inf')]:
    try:checked_depth(d)
    except ValueError:pass
    else:raise AssertionError('Invalid depth accepted')
for t in [0.,2.,2.5,4.5,6.5]:
    eps=1e-5
    if t>0:
        assert abs(offset_at(t+eps,.01)-offset_at(t-eps,.01))<1e-8
assert offset_at(0,.01)==0 and offset_at(2,.01)==-.01 and offset_at(4.5,.01)==0
c=PreparationCycle(10.,.01,.35)
assert c.update(10.,.35,1.,[1,1,1,1])==(0.,False)
c.update(12.3,.341,2.,[1,1,1,1])
assert c.update(16.5,.35,1.,[1,1,1,1])==(0.,True)
assert c.result(.35)['result']=='PREPARATION_RESPONSE_SEEN'
assert c.result(.3437)['result']=='PREPARATION_RESPONSE_INCOMPLETE'
c=PreparationCycle(0,.01,.35);c.update(1.,.35,1.,[0,0,1,1])
try:c.update(1.101,.35,1.,[0,0,1,1])
except RuntimeError:pass
else:raise AssertionError('Lost support accepted')
try:PreparationCycle(0,.01,.35).update(.1,.35,8.1,[1,1,1,1])
except RuntimeError:pass
else:raise AssertionError('Large tilt accepted')
print('PASS slow preparation bounds, smooth endpoints, support/tilt feedback and completion; no motor enabled')
