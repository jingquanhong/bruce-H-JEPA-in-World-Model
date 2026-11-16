"""Crouch timing, magnitude, continuity and feedback protections; no hardware IO."""
import math
from six_cm_crouch_profile import checked_depth, duration, offset_at, offset_velocity, PreparationCycle

for invalid in [0.,-.01,.06001,float('nan'),float('inf')]:
    try:checked_depth(invalid)
    except ValueError:pass
    else:raise AssertionError('Invalid depth accepted')
for depth in [.005,.01,.03,.06]:
    total=duration(depth)
    offsets=[offset_at(total*i/2000,depth) for i in range(2001)]
    speeds=[offset_velocity(total*i/2000,depth) for i in range(2001)]
    assert max(offsets)<=0 and min(offsets)>=-depth-1e-12
    assert abs(min(offsets)+depth)<1e-12
    assert max(map(abs,speeds))<=.018751
    assert offset_at(0,depth)==offset_at(total,depth)==0
    for t in [0,2,2.5,4.5,6,6.5,12.5,total]:
        if t>0:
            assert abs(offset_at(t+1e-6,depth)-offset_at(t-1e-6,depth))<4e-8
assert duration(.06)==14.5
c=PreparationCycle(100,.06,.31)
assert c.update(106.2,.258,1,[1,1,1,1])==(-.06,False)
assert c.update(114.5,.3105,1,[1,1,1,1])==(0.,True)
assert c.result(.3105)['result']=='CROUCH_RESPONSE_SEEN'
assert c.result(.258)['result']=='CROUCH_RESPONSE_INCOMPLETE'
for bad in ['tilt','contact','nan','time']:
    c=PreparationCycle(0,.06,.31)
    try:
        if bad=='tilt':c.update(1,.31,8.01,[1,1,1,1])
        if bad=='contact':
            c.update(1,.31,1,[0,0,1,1]);c.update(1.101,.31,1,[0,0,1,1])
        if bad=='nan':c.update(1,float('nan'),1,[1,1,1,1])
        if bad=='time':c.update(-1,.31,1,[1,1,1,1])
    except RuntimeError:pass
    else:raise AssertionError('Unsafe feedback accepted: '+bad)
print('PASS six cm trajectory magnitude/timing, smooth endpoints, support/tilt/invalid-feedback guards; no motor IO')
