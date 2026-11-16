from directional_drift_feedback import DriftFeedback
import math
for yaw in (0.,math.pi/2,-math.pi/2):
 f=DriftFeedback(yaw,[0.,0.],.03);c,s=math.cos(yaw),math.sin(yaw)
 # Backward/right drift in heading frame: both biases must increase.
 velocity=[c*(-.03)-s*(-.01),s*(-.03)+c*(-.01)]
 updates=[]
 for i in range(100):
  r=f.sample(i*.04,velocity,[0.,0.])
  if r:updates.append(r)
 assert updates and updates[-1]['forward_bias_m']>0 and updates[-1]['lateral_bias_m']>0
 previous=[0.,0.]
 for r in updates:
  for j,key in enumerate(('forward_bias_m','lateral_bias_m')):
   value=r[key];assert abs(value)<=.008+1e-12
   assert abs(value-previous[j])<=.003*r['mean_window_s']+1e-12;previous[j]=value
f=DriftFeedback(0.,[0.,0.],-.03)
for i in range(100):r=f.sample(i*.04,[0.,-.03],[0.,0.])
assert f.bias==[0.,0.]
for value in [float('nan'),float('inf')]:
 try:DriftFeedback(value,[0.,0.],.03)
 except ValueError:pass
 else:raise AssertionError('nonfinite accepted')
try:f.sample(4.,[float('nan'),0.],[0.,0.])
except ValueError:pass
else:raise AssertionError('nonfinite measurement accepted')
print('PASS sign in rotated frames, perfect-tracking neutrality, bias/rate bounds and nonfinite rejection; no physical dynamics tested')
