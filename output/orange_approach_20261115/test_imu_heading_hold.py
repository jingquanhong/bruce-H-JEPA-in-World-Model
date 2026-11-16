"""Controller sign, wrap and limit checks; not a robot dynamics simulation."""
import math
from imu_heading_hold import HeadingHold,wrap

def R(yaw):
 c,s=math.cos(yaw),math.sin(yaw)
 return [[c,-s,0.],[s,c,0.],[0.,0.,1.]]

for sign in (-1,1):
 h=HeadingHold();h.reset(R(0.))
 desired,rate,info=h.sample(R(sign*math.radians(5)),[0.,0.,0.],.24)
 assert sign*desired<0 and sign*rate<0
 for _ in range(100):
  desired,rate,info=h.sample(R(sign*math.radians(30)),[0.,0.,0.],.24)
  assert abs(rate)<=math.radians(3)+1e-12
  assert abs(h.offset)<=math.radians(5)+1e-12
h=HeadingHold();h.reset(R(math.radians(179)))
_,rate,info=h.sample(R(math.radians(-179)),[0.,0.,0.],.24)
assert abs(math.degrees(info['error_rad'])+2)<1e-8 and rate<0
h=HeadingHold();h.reset(R(0.));_,rate,_=h.sample(R(0.),[0.,0.,.1],.24);assert rate<0
h=HeadingHold();h.reset(R(0.));yaw=math.radians(5);velocity=0.
for _ in range(80):
 desired,_,_=h.sample(R(yaw),[0.,0.,velocity],.24)
 velocity=2.*wrap(desired-yaw);yaw+=velocity*.24
assert abs(math.degrees(yaw))<1.5
try:h.sample(R(float('nan')),[0.,0.,0.],.24)
except ValueError:pass
else:raise AssertionError('Nonfinite feedback accepted')
print('PASS heading feedback sign, gyro damping, wrap, saturation and simple tracker convergence; not physics.')
